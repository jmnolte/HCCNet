from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torchmetrics.classification import BinaryAUROC
import argparse
import os
import time
import copy
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monai.data import set_track_meta
from monai.utils.misc import ensure_tuple_rep
from monai.utils import set_determinism
from models.dinohead import DINOHead, MultiCropWrapper
from models.mednet import MedNet
from losses.dinoloss import DINOLoss
from utils.transforms import transforms, dino_transforms
from utils.preprocessing import load_backbone, load_data, load_objs
from utils.config import parse_args
from utils.utils import (
    cancel_gradients_last_layer, 
    scale_learning_rate, 
    prep_batch)

class Pretrainer:

    def __init__(
            self,
            model: nn.Module | List[nn.Module],
            loss_fn: nn.Module,
            dataloaders: dict,
            optimizer: optim,
            scheduler: List[np.array],
            num_steps: int = 1000,
            amp: bool = True,
            suffix: str | None = None,
            output_dir: str | None = None
        ) -> None:

        '''
        Args:
            model (nn.Module): Pytorch module object for Transformer pretraining or list of pytorch module objects for CNN backbone pretraining. 
            loss_fn (nn.Module): Loss function.
            dataloaders (dict): Dataloader objects. Have to be provided as a dictionary, where the the entries are 'train' and 'val'. 
            optimizer (optim): Pytorch optimizer.
            scheduler (List[np.array]): List of learing rate, weight decay, and momentum schedules. Has to be of length 2 or 3.
            num_steps (int): Number of training steps. Defaults to 1000.
            amp (bool): Boolean flag to enable automatic mixed precision training. Defaults to true.
            suffix (str | None): Unique string under which model results are stored.
            output_dir (str | None): Directory to store model outputs.
        '''

        self.gpu_id = int(os.environ['LOCAL_RANK'])
        if isinstance(model, list):
            self.student = model[0]
            self.teacher = model[1]
            self.backbone_only = True
        else:
            self.model = model
            self.backbone_only = False
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

        if isinstance(loss_fn, list):
            self.loss_fn = loss_fn[0].to(self.gpu_id)
        else:
            self.loss_fn = loss_fn.to(self.gpu_id)

        self.optim = optimizer
        if self.backbone_only:
            self.lr_schedule, self.wd_schedule, self.m_schedule = scheduler[0], scheduler[1], scheduler[2]
            self.params = self.student.parameters()
        else:
            self.lr_schedule, self.wd_schedule = scheduler[0], scheduler[1]
            self.params = self.model.parameters()
        self.results_dict = {dataset: {metric: [] for metric in ['loss']} for dataset in ['train']}
        self.auroc = BinaryAUROC()

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
            fold (int): Current training step.
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

        '''
        Args:
            phase (str): String specifying the training phase. Can be 'train' or 'val'.
            keys (str | List[str]): Metric name or list of metric names that should be logged.
            values (float | List[float]): Metric value or list of metric values corresponding to their keys. 
        '''

        if not isinstance(keys, list):
            keys = [keys]
        if not isinstance(values, list):
            values = [values]
        for key, value in zip(keys, values):
            self.results_dict[phase][key].append(value)

    def encoder_step(
            self,
            batch: dict,
            step: int,
            accum_steps: int
        ) -> float:

        '''
        Args:
            batch (dict): Batch obtained from a Pytorch dataloader.
            step (int): Current training step.
            accum_steps (int): Number of steps to accumulate before updating the gradients.
        '''

        self.student.train()
        self.teacher.train()
        gv1, gv2, lv1, lv2 = batch['gv1'], batch['gv2'], batch['lv1'], batch['lv2']
        views = [view.to(self.gpu_id) for view in [gv1, gv2, lv1, lv2]]

        with autocast(enabled=self.amp):
            student_logits = self.student(views)
            teacher_logits = self.teacher(views[:2])
            loss = self.loss_fn(step, student_logits, teacher_logits)
            loss /= accum_steps
        self.scaler.scale(loss).backward()
        return loss.item()

    def decoder_step(
            self,
            batch: dict,
            batch_size: int,
            accum_steps: int
        ) -> float:

        '''
        Args:
            batch (dict): Batch obtained from a Pytorch dataloader.
            batch_size (int): Number of unique observations in the batch.
            accum_steps (int): Number of steps to accumulate before updating the gradients.
        '''

        self.model.train()
        inputs, _, delta, padding_mask = prep_batch(batch, batch_size=batch_size, device=self.gpu_id, pretrain=True)

        with autocast(enabled=self.amp):
            logits, labels = self.model(inputs, pad_mask=padding_mask, pos=delta)
            loss = self.loss_fn(logits.squeeze(-1), labels.float())
            loss /= accum_steps
        self.scaler.scale(loss).backward()

        preds = F.sigmoid(logits.squeeze(-1))
        self.auroc.update(preds, labels.int())
        return loss.item()

    def accumulation_step(
            self,
            step: int,
            warmup_steps: int,
            clip_grad: bool = True
        ) -> None:

        '''
        Args:
            step (int): Current training step.
            warmup_steps (int): Number of steps to wait before updating last layer.
            clip_grad (bool): Boolean flag to clip parameter gradients.
        '''

        for i, param_group in enumerate(self.optim.param_groups):
            param_group['lr'] = self.lr_schedule[step]
            if i == 0:  # only the first group is regularized
                param_group['weight_decay'] = self.wd_schedule[step]

        if clip_grad:
            self.scaler.unscale_(self.optim)
            nn.utils.clip_grad_norm_(self.params, max_norm=1.0, norm_type=2)
        if self.backbone_only:
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

        '''
        Args:
            step (int): Current training step.
        '''

    def pretrain(
            self,
            batch_size: int,
            accum_steps: int,
            warmup_steps: int,
            log_every: int = 10
        ) -> None:

        '''
        Args:
            batch_size (int): Number of unique observations in the batch.
            accum_steps (int): Number of steps to accumulate before updating parameter gradients.
            warmup_steps (int): Number of steps to wait before updating last layer.
            log_every (int): Number of steps to wait before storing current model weights.
        '''

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

                if self.backbone_only:
                    accum_loss += self.encoder_step(batch, update_step, accum_steps)
                else:
                    accum_loss += self.decoder_step(batch, batch_size, accum_steps)

                if (step + 1) % accum_steps == 0:
                    self.accumulation_step(update_step, warmup_steps, clip_grad=True)
                    if self.gpu_id == 0:
                        print(f"Step Loss: {accum_loss:.4f}")
                    running_loss += accum_loss
                    accum_loss = 0.0
                    if self.backbone_only:
                        self.update_teacher(update_step)

                if (step + 1) % (accum_steps * log_every) == 0:
                    loss = torch.Tensor([running_loss / log_every])
                    running_loss = 0.0
                    dist.all_reduce(loss.to(self.gpu_id), op=dist.ReduceOp.AVG)
                    if not self.backbone_only:
                        auroc = self.auroc.compute()
                        self.auroc.reset()
                        if self.gpu_id == 0:
                            print(f"[GPU {self.gpu_id}] Step {update_step}/{self.num_steps}, AUROC: {auroc:.4f}")

                    if self.gpu_id == 0:
                        self.log_dict(phase='train', keys='loss', values=loss)
                        print(f"[GPU {self.gpu_id}] Step {update_step}/{self.num_steps}, Loss: {loss.item():.4f}")

                if (step + 1) / accum_steps in [2000, 4000, 8000, 16000, 32000]:
                    if self.gpu_id == 0:
                        model_weights = self.teacher.module.state_dict() if self.backbone_only else self.model.module.state_dict()
                        self.save_output(model_weights, 'weights', fold=int(update_step + 1))

                if (step + 1) / accum_steps == self.num_steps:
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
        Args:
            phase (str | List[str]): String or list of strings. Should be 'train' and 'val'.
            log_type (str): String specifying the metric that should be visualized.
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

def load_weights(
        args: argparse.Namespace,
        weights_path: str
    ) -> dict:

    '''
    Args:
        args (argparse.Namespace): Command line arguments.
        weights_path (str): Path to weights directory.
    '''

    weights = torch.load(weights_path, map_location='cpu')
    weights['backbone.downsample_layers.0.0.weight'] = weights['backbone.downsample_layers.0.0.weight'].repeat(1, len(args.mod_list), 1, 1, 1)
    return weights

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
    backbone_only = True if args.loss_fn == 'dino' else False
    modality = args.suffix.split('_')[0]

    dataloader, _ = load_data(args, device_id, phase='pretrain', partial=True if backbone_only else False)
    dataloader = {x: dataloader[x][0] for x in ['train']}
    set_track_meta(False)
    if backbone_only:
        student, teacher = load_backbone(args, args.arch, dino_pretraining=True)
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
        loss_fn, optimizer, schedules = load_objs(args, student, learning_rate)
        model = [student, teacher]
    else:
        backbone = load_backbone(args, args.arch)
        model = MedNet(
            backbone=backbone, 
            num_classes=1,
            pretrain=True,
            max_len=12,
            num_layers=4 if any(args.arch in x for x in ['femto', 'pico']) else 6,
            dropout=args.dropout,
            eps=args.epsilon)
        weights = load_weights(args, os.path.join(args.results_dir, f'model_weights/weights_fold32000_{modality}_{args.arch}.pth'))
        model.load_state_dict(weights, strict=False)
        model = model.to(device_id)
        for p in model.backbone.parameters():
            p.requires_grad = False
        if args.distributed:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id])
        loss_fn, optimizer, schedules = load_objs(args, model, learning_rate, pos_weight=None)

    pretrainer = Pretrainer(
        model=model,
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
        print(f'- Pretraining is set to {args.num_steps} steps with {args.warmup_steps} warm-up steps, an effective batch size of {args.effective_batch_size}, and a world size of {num_devices}.')
        if backbone_only:
            print(f'- Model is ConvNeXt{version} {args.arch} using a {args.kernel_size}^3 downsampling kernel and projection head of dimensionality {args.out_dim}.') 
            print(f'- The teacher temperature ranges from ({args.teacher_warmup_temp} to {args.teacher_temp} in {args.warmup_steps} steps), stochastic depth rate is set to {args.stochastic_depth}, and epsilon to {args.epsilon}.')
            print(f'Starting DINO pretraining...')
        else:
            print(f'- Model consists of ConvNeXt{version} {args.arch} encoder and Transformer decoder.')
            print(f'- The stochastic depth rate is set to {args.stochastic_depth}, dropout rate to {args.dropout}, and epsilon to {args.epsilon}.')
            print(f'Starting Decoder pretraining...')

    pretrainer.pretrain(args.batch_size, accum_steps, args.warmup_steps // 10)
    pretrainer.visualize_training('train', 'loss')
    print('Script finished.')

if __name__ == '__main__':
    args = parse_args()
    main(args)
    

                    
