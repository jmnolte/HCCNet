from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
import argparse
import os
import time
import copy
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    SpatialPadd,
    CropForegroundd,
    ConcatItemsd,
    ResampleToMatchd,
    EnsureTyped,
    DeleteItemsd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    CopyItemsd,
    KeepLargestConnectedComponentd,
    Lambdad,
    NormalizeIntensityd
)
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    set_track_meta,
    partition_dataset
)
from monai.utils.misc import ensure_tuple_rep
from monai.utils import set_determinism
from data.transforms import PercentileSpatialCropd, YeoJohnsond, SoftClipIntensityd, ResampleToMatchFirstd, RandSelectChanneld
from data.splits import GroupStratifiedSplit
from data.utils import DatasetPreprocessor, convert_to_dict
from models.convnext3d import convnext3d_femto, convnext3d_pico, convnext3d_nano, convnext3d_tiny
from models.dinohead import DINOHead
from losses.dinoloss import DINOLoss
from utils.utils import cancel_gradients_last_layer, cosine_scheduler, get_params_groups, scale_learning_rate, MultiCropWrapper

class Pretrainer:

    def __init__(
            self,
            student: nn.Module,
            teacher: nn.Module,
            loss_fn: nn.Module,
            dataloaders: dict,
            optimizer: optim,
            scheduler: List[np.array],
            num_steps: int = 1000,
            amp: bool = True,
            suffix: str | None = None,
            output_dir: str | None = None
        ) -> None:

        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.student = student
        self.teacher = teacher
        self.dataloaders = dataloaders
        self.num_steps = num_steps
        self.num_folds = 1
        self.amp = amp
        self.suffix = suffix
        if self.suffix is None:
            raise ValueError('Please specify a unique suffix for results storage.')
        self.output_dir = output_dir
        if self.output_dir is None:
            raise ValueError('Please specify a path to the data directory.')
        self.scaler = GradScaler(enabled=amp)

        self.dino_loss = loss_fn.to(self.gpu_id)
        self.optim = optimizer
        self.lr_schedule, self.wd_schedule, self.m_schedule = scheduler[0], scheduler[1], scheduler[2]
        self.results_dict = {dataset: {metric: [] for metric in ['loss']} for dataset in ['train']}

    def save_output(
            self, 
            output_dict: dict, 
            output_type: str,
            fold: int
        ) -> None:

        '''
        Save the model's output.

        Args:
            output_dict (dict): Dictionary containing the model outputs.
            output_type (str): Type of output. Can be 'weights', 'history', or 'preds'.
        '''
        try: 
            assert any(output_type == output_item for output_item in ['weights','history','preds'])
        except AssertionError:
            print('Invalid Input. Please choose from: weights, history, or preds')
            exit(1)
        
        if output_type == 'weights':
            folder_name = f'weights_fold{fold}_' + self.suffix + '.pth'
        elif output_type == 'history':
            folder_name = f'hist_fold{fold}_' + self.suffix + '.npy'
        elif output_type == 'preds':
            folder_name = f'preds_fold{fold}_' + self.suffix + '.npy'
        folder_path = os.path.join(self.output_dir, 'model_' + output_type, folder_name)
        folder_path_root = os.path.join(self.output_dir, 'model_' + output_type)

        if os.path.exists(folder_path):
            os.remove(folder_path)
        elif not os.path.exists(folder_path_root):
            os.makedirs(folder_path_root)

        if output_type == 'weights':
            torch.save(output_dict, folder_path)
        else:
            np.save(folder_path, output_dict)

    def log_dict(
            self,
            phase: str,
            keys: str | List[str],
            values: float | List[float]
        ) -> None:

        if not isinstance(keys, list):
            keys = [keys]
        if not isinstance(values, list):
            values = [values]
        for key, value in zip(keys, values):
            self.results_dict[phase][key].append(value)

    def pretraining_step(
            self,
            batch: dict,
            step: int,
            accum_steps: int
        ) -> float:

        self.student.train()
        self.teacher.train()
        gv1, gv2, lv1, lv2 = batch['gv1'], batch['gv2'], batch['lv1'], batch['lv2']
        views = [view.to(self.gpu_id) for view in [gv1, gv2, lv1, lv2]]

        with autocast(enabled=self.amp):
            student_logits = self.student(views)
            teacher_logits = self.teacher(views[:2])
            loss = self.dino_loss(step, student_logits, teacher_logits)
            loss /= accum_steps
        self.scaler.scale(loss).backward()

        return loss.item()

    def accumulation_step(
            self,
            step: int,
            warmup_steps: int,
            clip_grad: bool = True
        ) -> None:

        for i, param_group in enumerate(self.optim.param_groups):
            param_group['lr'] = self.lr_schedule[step]
            if i == 0:  # only the first group is regularized
                param_group['weight_decay'] = self.wd_schedule[step]

        if clip_grad:
            self.scaler.unscale_(self.optim)
            nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0, norm_type=2)
        cancel_gradients_last_layer(self.student, step=step, warmup_steps=warmup_steps)

        self.scaler.step(self.optim)
        self.scaler.update()
        self.optim.zero_grad(set_to_none=True)

    @torch.no_grad()
    def update_teacher(
            self,
            step: int
        ) -> None:

        m = self.m_schedule[step]
        for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def pretrain(
            self,
            accum_steps: int,
            warmup_steps: int,
            log_every: int = 10,
            checkpoints: list = [2000, 4000, 8000, 16000, 32000]
        ) -> None:

        accum_loss = 0.0
        running_loss = 0.0
        start_time = time.time()
        self.optim.zero_grad(set_to_none=True)

        for epoch in range(self.num_steps * accum_steps // len(self.dataloaders['train']) + 1):
            for idx, batch in enumerate(self.dataloaders['train']):

                step = epoch * len(self.dataloaders['train']) + idx
                update_step = step // accum_steps
                if self.gpu_id == 0 and step % (accum_steps * log_every) == 0:
                    print('-' * 15)
                    print(f'Step {update_step}/{self.num_steps}')
                    print('-' * 15)

                accum_loss += self.pretraining_step(batch, update_step, accum_steps)

                if (step + 1) % accum_steps == 0:
                    self.accumulation_step(update_step, warmup_steps, clip_grad=False)
                    if self.gpu_id == 0:
                        print(f"Step Loss: {accum_loss:.4f}")
                    running_loss += accum_loss
                    accum_loss = 0.0
                    self.update_teacher(update_step)

                if (step + 1) % (accum_steps * log_every) == 0:
                    loss = torch.Tensor([running_loss / log_every])
                    running_loss = 0.0
                    dist.all_reduce(loss.to(self.gpu_id), op=dist.ReduceOp.AVG)

                    if self.gpu_id == 0:
                        self.log_dict(phase='train', keys='loss', values=loss)
                        print(f"[GPU {self.gpu_id}] Step {update_step}/{self.num_steps}, Loss: {loss.item():.4f}")

                if (step + 1) / accum_steps in checkpoints:
                    dist.barrier()
                    if self.gpu_id == 0:
                        self.save_output(self.teacher.module.state_dict(), 'weights', fold=int(step + 1))
                    dist.barrier()

                if (step + 1) == self.num_steps * accum_steps:
                    break

        if self.gpu_id == 0:
            time_elapsed = time.time() - start_time
            print(f'Pretraining finished in {time_elapsed // 60:.0f}min {time_elapsed % 60:.0f}sec')
            self.save_output(self.results_dict, 'history', fold=0)
        dist.barrier()

    def visualize_training(
            self, 
            phases: str | List[str],
            log_type: str
        ) -> None:

        '''
        Visualize the training and validation history.

        Args:
            metric (str): String specifying the metric to be visualized. Can be 'loss' or 'f1score'.
        '''

        if log_type == 'loss':
            axis_label = 'Loss'
        elif log_type == 'auprc':
            axis_label = 'AUPRC'
        elif log_type == 'auroc':
            axis_label = 'AUROC'
        plot_name = log_type + '_' + self.suffix + '.png' if self.suffix is not None else log_type + '.png'
        phases = [phases] if isinstance(phases, str) else phases

        for dataset in phases:
            log_book = []
            for fold in range(self.num_folds):
                file_name = f'hist_fold{fold}_' + self.suffix + '.npy'
                fold_log = np.load(os.path.join(self.output_dir, 'model_history', file_name), allow_pickle='TRUE').item()
                log_book.append(fold_log[dataset][log_type])
                plt.plot(fold_log[dataset][log_type], color=('blue' if dataset == 'train' else 'orange'), alpha=0.2)
            log_df = pd.DataFrame(log_book)
            mean_log = log_df.mean(axis=0).tolist()
            plt.plot(mean_log, color=('blue' if dataset == 'train' else 'orange'), label=('Training' if dataset == 'train' else 'Validation'), alpha=1.0)
            
        plt.ylabel(axis_label, fontsize=20, labelpad=10)
        plt.xlabel('Training Epochs', fontsize=20, labelpad=10)
        plt.legend(loc='lower right')
        file_path = os.path.join(self.output_dir, 'model_diagnostics/learning_curves', plot_name)
        file_path_root, _ = os.path.split(file_path)
        if os.path.exists(file_path):
            os.remove(file_path)
        elif not os.path.exists(file_path_root):
            os.makedirs(file_path_root)
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()

def parse_args() -> argparse.Namespace:

    '''
    Parse command line arguments.

    Returns:
        argparse.Namespace: Command line arguments.
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-steps", default=100, type=int, 
                        help="Number of steps to train for. Defaults to 100.")
    parser.add_argument("--warmup-steps", default=10, type=int, 
                        help="Number of warmup steps. Defaults to 10.")
    parser.add_argument("--log-every", default=10, type=int,
                        help="Interval to log model results. Defaults to 10.")
    parser.add_argument("--batch-size", default=4, type=int, 
                        help="Batch size to use for training. Defaults to 4.")
    parser.add_argument("--effective-batch-size", default=32, type=int,
                        help="Total batch size: batch size x number of devices x number of accumulations steps.")
    parser.add_argument("--min-learning-rate", default=1e-6, type=float, 
                        help="Final learning rate. Defaults to 1e-6.")
    parser.add_argument("--weight-decay", default=0.05, type=float, 
                        help="Initial weight decay value. Defaults to 0.05.")
    parser.add_argument("--max-weight-decay", default=0.5, type=float, 
                        help="Final weight decay value. Defaults to 0.5.")
    parser.add_argument("--stochastic-depth", default=0, type=float, 
                        help="Stochastic depth rate to use for training. Defaults to 0.")
    parser.add_argument("--epsilon", default=1e-5, type=float, 
                        help="Epsilon value to use for norm layers. Defaults to 1e-5.")
    parser.add_argument("--global-crop-size", default=72, type=int, 
                        help="Global crop size to use. Defaults to 72.")
    parser.add_argument("--local-crop-size", default=48, type=int, 
                        help="Local crop size to use. Defaults to 48.")

    parser.add_argument("--arch", type=str, default='femto',
                        help="ConvNeXT model architecture. Choices are femto, pico, nano, tiny. Defaults to femto.")
    parser.add_argument("--use-v2", action='store_true',
                        help="Whether to use ConvNext v1 or v2.")
    parser.add_argument("--kernel-size", default=3, type=int,
                        help="Kernel size of convolutional downsampling layers. Defaults to 2.")
    parser.add_argument("--out-dim", default=4096, type=int,
                        help="Output dimensionality of projection head. Defaults to 4096")
    parser.add_argument("--norm-last-layer", action='store_true',
                        help="Whether to weight normalize the last layer of the DINO head.")
    parser.add_argument("--teacher-momentum", default=0.9995, type=float, 
                        help="Inital momentum value to update teacher network with EMA. Defaults to 0.9995.")
    parser.add_argument("--teacher-temp", default=0.04, type=float, 
                        help="Final (i.e., after warmup) teacher temperature value. Defaults to 0.04")
    parser.add_argument("--teacher-warmup-temp", default=0.04, type=float, 
                        help="Initial teacher temperature value. Defaults to 0.04")

    parser.add_argument("--mod-list", default=MOD_LIST, nargs='+', 
                        help="List of modalities to use for training")
    parser.add_argument("--distributed", action='store_true',
                        help="Whether to enable distributed training.")
    parser.add_argument("--amp", action='store_true',
                        help="Whether to enable automated mixed precision training.")
    parser.add_argument("--seed", default=1234, type=int, 
                        help="Seed to use for reproducibility")
    parser.add_argument("--num-workers", default=8, type=int,
                        help="Number of workers to use for training. Defaults to 8.")
    parser.add_argument("--suffix", default='MedNet', type=str, 
                        help="File suffix for identification")
    parser.add_argument("--data-dir", default=DATA_DIR, type=str, 
                        help="Path to data directory")
    parser.add_argument("--results-dir", default=RESULTS_DIR, type=str, 
                        help="Path to results directory")
    return parser.parse_args()

def ssl_transforms(
        modalities: list,
        device: torch.device,
        global_crop_size: tuple = (72, 72, 72),
        local_crop_size: tuple = (48, 48, 48),
        image_spacing: tuple = (1.5, 1.5, 1.5)
    ) -> monai.transforms:
    '''
    Perform data transformations on image and image labels.

    Args:
        dataset (str): Dataset to apply transformations on. Can be 'train', 'val' or 'test'.

    Returns:
        transforms (monai.transforms): Data transformations to be applied.
    '''
    prep = [
        LoadImaged(keys=modalities, image_only=True, allow_missing_keys=True),
        EnsureChannelFirstd(keys=modalities, allow_missing_keys=True),
        Orientationd(keys=modalities, axcodes='PLI', allow_missing_keys=True),
        Spacingd(keys=modalities, pixdim=image_spacing, mode=3, allow_missing_keys=True),
        ResampleToMatchFirstd(keys=modalities, mode=3, allow_missing_keys=True),
        ConcatItemsd(keys=modalities, name='image', allow_missing_keys=True),
        PercentileSpatialCropd(
            keys='image',
            roi_center=(0.5, 0.5, 0.5),
            roi_size=(0.85, 0.8, 0.99),
            min_size=global_crop_size),
        CopyItemsd(keys='image', names='mask'),
        Lambdad(keys='mask', func=lambda x: x[:1]),
        Lambdad(keys='mask', func=lambda x: torch.where(x > torch.mean(x), 1, 0)),
        KeepLargestConnectedComponentd(keys='mask', connectivity=1),
        CropForegroundd(
            keys='image',
            source_key='mask',
            select_fn=lambda x: x > 0,
            k_divisible=1,
            allow_smaller=False),
        YeoJohnsond(keys='image', lmbda=0.1, channel_wise=True),
        NormalizeIntensityd(keys='image', channel_wise=True),
        PercentileSpatialCropd(
            keys='image',
            roi_center=(0.45, 0.25, 0.3),
            roi_size=(0.6, 0.45, 0.4),
            min_size=global_crop_size),
        SoftClipIntensityd(keys='image', min_value=-3.0, max_value=3.0, channel_wise=True),
        CopyItemsd(keys='image', times=4, names=['gv1','gv2','lv1','lv2']),
        DeleteItemsd(keys=modalities + ['image','mask']),
        EnsureTyped(keys=['gv1','gv2','lv1','lv2'], track_meta=False, device=device, dtype=torch.float),
    ]

    global_crop = [
        RandSpatialCropd(keys=['gv1','gv2'], roi_size=global_crop_size, random_size=False),
        SpatialPadd(keys=['gv1','gv2'], spatial_size=global_crop_size, method='symmetric')
    ]

    local_crop = [
        RandSpatialCropd(keys=['lv1','lv2'], roi_size=local_crop_size, random_size=False),
        SpatialPadd(keys=['lv1','lv2'], spatial_size=local_crop_size, method='symmetric')
    ]

    post = [
        RandFlipd(keys=['gv1','gv2','lv1','lv2'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['gv1','gv2','lv1','lv2'], prob=0.5, spatial_axis=1),
        RandFlipd(keys=['gv1','gv2','lv1','lv2'], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=['gv1','gv2','lv1','lv2'], prob=0.5),
        RandSelectChanneld(keys=['gv1','gv2','lv1','lv2'], num_channels=1),
        NormalizeIntensityd(keys=['gv1','gv2','lv1','lv2'], subtrahend=0.4376, divisor=0.6323)
    ]
    return Compose(prep + global_crop + local_crop + post)

def load_pretrain_data(
        args: argsparse.Namespace,
        device: torch.device
    ) -> tuple:
    
    phases = ['train']
    data_dict, label_df = DatasetPreprocessor(data_dir=args.data_dir).load_data(modalities=args.mod_list, keys=['label','age'], file_name='labels.csv', verbose=False)
    ssl_dict, ssl_df = DatasetPreprocessor(data_dir=args.data_dir, partial=True).load_data(modalities=args.mod_list, keys=['age'], file_name='nolabels.csv')
    dev, test = GroupStratifiedSplit(split_ratio=0.75).split_dataset(label_df)
    ssl_test = ssl_df[ssl_df['patient_id'].isin(test['patient_id'])]
    ssl_dev = ssl_df[-ssl_df['patient_id'].isin(ssl_test['patient_id'])]
    split_dict = convert_to_dict([ssl_dev], data_dict=ssl_dict, split_names=phases, verbose=False)

    split_dict = {x: partition_dataset(
        data=split_dict[x],
        num_partitions=dist.get_world_size(),
        shuffle=True,
        even_divisible=(True if x == 'train' else False)
        )[dist.get_rank()] for x in phases}
    datasets = {x: CacheDataset(
        data=split_dict[x], 
        transform=ssl_transforms(
            modalities=args.mod_list, 
            device=device,
            global_crop_size=ensure_tuple_rep(args.global_crop_size, 3),
            local_crop_size=ensure_tuple_rep(args.local_crop_size, 3)),
        cache_rate=1.0,
        num_workers=8,
        copy_cache=False
        ) for x in phases}
    dataloader = {x: ThreadDataLoader(
        dataset=datasets[x], 
        batch_size=(args.batch_size if x == 'train' else 1), 
        shuffle=(True if x == 'train' else False),   
        num_workers=0,
        drop_last=(True if x == 'train' else False)
        ) for x in phases}

    return dataloader

def load_pretrain_objs(
        args: argparse.Namespace,
        student: nn.Module,
        learning_rate: float = 1e-4
    ) -> tuple:

    loss_fn = DINOLoss(
        out_dim=args.out_dim,
        num_crops=4,
        num_steps=args.num_steps,
        teacher_temp=args.teacher_temp,
        teacher_warmup_temp=args.teacher_warmup_temp,
        teacher_warmup_steps=args.warmup_steps)
    params = get_params_groups(student)
    optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=args.weight_decay)
    lr_schedule = cosine_scheduler(
        base_value=learning_rate,
        final_value=args.min_learning_rate,
        steps=args.num_steps,
        warmup_steps=args.warmup_steps)
    wd_schedule = cosine_scheduler(
        base_value=args.weight_decay,
        final_value=args.max_weight_decay,
        steps=args.num_steps)
    m_schedule = cosine_scheduler(
        base_value=args.teacher_momentum,
        final_value=1,
        steps=args.num_steps)
    schedules = [lr_schedule, wd_schedule, m_schedule]
    return loss_fn, optimizer, schedules

def load_models(
        args: argparse.Namespace
    ) -> Tuple[nn.Module]:

    if args.arch == 'femto':
        student = convnext3d_femto(
            in_chans=1, kernel_size=args.kernel_size, drop_path_rate=args.stochastic_depth, use_v2=args.use_v2, eps=args.epsilon)
        teacher = convnext3d_femto(
            in_chans=1, kernel_size=args.kernel_size, use_v2=args.use_v2, eps=args.epsilon)
    elif args.arch == 'pico':
        student = convnext3d_pico(
            in_chans=1, kernel_size=args.kernel_size, drop_path_rate=args.stochastic_depth, use_v2=args.use_v2, eps=args.epsilon)
        teacher = convnext3d_pico(
            in_chans=1, kernel_size=args.kernel_size, use_v2=args.use_v2, eps=args.epsilon)
    elif args.arch == 'nano':
        student = convnext3d_nano(
            in_chans=1, kernel_size=args.kernel_size, drop_path_rate=args.stochastic_depth, use_v2=args.use_v2, eps=args.epsilon)
        teacher = convnext3d_nano(
            in_chans=1, kernel_size=args.kernel_size, use_v2=args.use_v2, eps=args.epsilon)
    elif args.arch == 'tiny':
        student = convnext3d_tiny(
            in_chans=1, kernel_size=args.kernel_size, drop_path_rate=args.stochastic_depth, use_v2=args.use_v2, eps=args.epsilon)
        teacher = convnext3d_tiny(
            in_chans=1, kernel_size=args.kernel_size, use_v2=args.use_v2, eps=args.epsilon)
    return student, teacher

def setup() -> None:

    '''
    Setup distributed training.
    '''
    dist.init_process_group(backend="nccl")

def cleanup() -> None:

    '''
    Cleanup distributed training.
    '''
    dist.destroy_process_group()

def main(
        args: argparse.Namespace
    ) -> None:

    '''
    Main function. The function loads the dataloader, model, and metric, and trains the model. After
    training, the function plots the training and validation loss and F1 score. It saves the updated
    model weights, the training and validation history (i.e., loss and F1 score), and the corresponding
    plots.

    Args:
        args (argparse.Namespace): Command line arguments.
    '''
    set_determinism(seed=args.seed)
    if args.distributed:
        setup()
    rank = dist.get_rank()
    num_devices = torch.cuda.device_count()
    device_id = rank % num_devices
    accum_steps = args.effective_batch_size // args.batch_size // num_devices
    learning_rate = scale_learning_rate(args.effective_batch_size)
    version = 'v2' if args.use_v2 else 'v1'

    dataloader = load_pretrain_data(args, device_id)
    set_track_meta(False)
    student, teacher = load_models(args)
    embed_dim = student.head.in_features
    student = MultiCropWrapper(
        student,
        DINOHead(embed_dim, args.out_dim, norm_last_layer=args.norm_last_layer))
    teacher = MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim))
    student, teacher = student.to(device_id), teacher.to(device_id)
    if args.distributed:
        student = nn.parallel.DistributedDataParallel(student, device_ids=[device_id])
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[device_id])
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    loss_fn, optimizer, schedules = load_pretrain_objs(args, student, learning_rate)

    pretrainer = Pretrainer(
        student=student,
        teacher=teacher,
        loss_fn=loss_fn,
        dataloaders=dataloader,
        optimizer=optimizer,
        scheduler=schedules,
        num_steps=args.num_steps,
        amp=args.amp,
        suffix=args.suffix,
        output_dir=args.results_dir)

    if rank == 0:
        print('-' * 15)
        print(f'Model pretraining is initialized using the following parameters:')
        print(f'- AdamW optimizer with cosine learning rate ({learning_rate} to {args.min_learning_rate}), weight decay ({args.weight_decay} to {args.max_weight_decay}), and momentum ({args.teacher_momentum} to {1.0}) decay.')
        print(f'- Model is ConvNext{version} {args.arch} using a {args.kernel_size}^3 downsampling kernel and projection head of dimensionality {args.out_dim}.') 
        print(f'- The teacher temperature ranges from ({args.teacher_warmup_temp} to {args.teacher_temp} in {args.warmup_steps} steps), stochastic depth rate is set to {args.stochastic_depth}, and epislon to {args.epsilon}.')
        print(f'- Pretraining is set to {args.num_steps} steps with {args.warmup_steps} warm-up steps, a batch size of {batch_size_gpu} per GPU and a effective batch size of {args.effective_batch_size}.')
        print(f'Starting DINO pretraining...')

    pretrainer.pretrain(accum_steps, args.warmup_steps // 10)
    pretrainer.visualize_training('train', 'loss')
    print('Script finished.')

RESULTS_DIR = '/Users/noltinho/thesis/results'
DATA_DIR = '/Users/noltinho/thesis/sensitive_data'
MOD_LIST = ['DWI_b0','DWI_b150','DWI_b400','DWI_b800']
WEIGHTS_DIR = '/Users/noltinho/MedicalNet/pytorch_files/pretrain'

if __name__ == '__main__':
    args = parse_args()
    main(args)
    

                    
