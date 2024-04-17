from __future__ import annotations

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAveragePrecision, BinaryAUROC, MulticlassAveragePrecision, MulticlassAUROC
from sklearn.model_selection import StratifiedGroupKFold
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
    CropForegroundd,
    ConcatItemsd,
    CenterSpatialCropd,
    ResampleToMatchd,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
    DeleteItemsd,
    RandSpatialCropd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandKSpaceSpikeNoised,
    RandGibbsNoised,
    RandCoarseShuffled,
    RandCoarseDropoutd,
    RandZoomd,
    CopyItemsd,
    KeepLargestConnectedComponentd,
    Lambdad,
    NormalizeIntensityd,
    SpatialPadd
)
from monai import transforms
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    set_track_meta,
    partition_dataset_classes,
    list_data_collate
)
from monai.utils.misc import ensure_tuple_rep
from monai.utils import set_determinism
from data.transforms import PercentileSpatialCropd, YeoJohnsond, SoftClipIntensityd, RobustNormalized
from data.splits import GroupStratifiedSplit
from data.datasets import CacheSeqDataset
from data.utils import DatasetPreprocessor, convert_to_dict, convert_to_seqdict, SequenceBatchCollater
from models.mednet import MedNet
from models.convnext3d import convnext3d_atto, convnext3d_femto, convnext3d_pico, convnext3d_nano, convnext3d_tiny
from losses.focalloss import FocalLoss
from losses.binaryceloss import BinaryCELoss
from utils.utils import cosine_scheduler, get_params_groups, scale_learning_rate, prep_batch


class Trainer:
    
    def __init__(
            self, 
            model: nn.Module, 
            loss_fn: List[nn.Module],
            dataloaders: dict, 
            optimizer: optim,
            scheduler: List[np.array],
            num_folds: int = 1,
            num_steps: int = 1000,
            amp: bool = True,
            backbone_only: bool = False,
            suffix: str | None = None,
            output_dir: str | None = None
        ) -> None:

        '''
        Initialize the training class.

        Args:
            model (torch.nn): Model to train.
            backbone (str): Backbone architecture to use. Has to be 'resnet50', or 'densenet121'.
            amp (bool): Flag to use automated mixed precision.
            dataloaders (dict): Dataloader objects. Have to be provided as a dictionary, where the the entries are 'train', 'val', and/or 'test'. 
            learning_rate (float): Float specifiying the learning rate.
            weight_decay (float): Float specifying the Weight decay.
            accum_steps (int): Number of accumulations steps.
            weights (torch.Tensor): Tensor to rescale the weights given to each class C. Has to be of size C.
            smoothing_coef (float): Float to specify the amount of smoothing applied to the class labels. Has to be in range [0.0 to 1.0].
            output_dir (str): Directory to store model outputs.
        '''
        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.model = model
        self.dataloaders = dataloaders
        self.num_folds = num_folds
        self.num_steps = num_steps
        self.amp = amp
        self.backbone_only = backbone_only
        self.suffix = suffix
        if self.suffix is None:
            raise ValueError('Please specify a unique suffix for results storage.')
        self.output_dir = output_dir
        if self.output_dir is None:
            raise ValueError('Please specify a path to the data directory.')
        self.scaler = GradScaler(enabled=amp)

        self.train_loss = loss_fn[0].to(self.gpu_id)
        self.val_loss = loss_fn[1].to(self.gpu_id)
        self.optim = optimizer
        self.lr_schedule, self.wd_schedule = scheduler[0], scheduler[1]
        self.params = self.model.parameters()

        if backbone_only:
            metrics = MetricCollection([MulticlassAveragePrecision(num_classes=3), MulticlassAUROC(num_classes=3)])
            self.train_metric_str = ['train_MulticlassAveragePrecision', 'train_MulticlassAUROC']
            self.val_metric_str = ['val_MulticlassAveragePrecision', 'val_MulticlassAUROC']
        else:
            metrics = MetricCollection([BinaryAveragePrecision(), BinaryAUROC()])
            self.train_metric_str = ['train_BinaryAveragePrecision', 'train_BinaryAUROC']
            self.val_metric_str = ['val_BinaryAveragePrecision', 'val_BinaryAUROC']
        self.train_metrics = metrics.clone(prefix='train_').to(self.gpu_id)
        self.val_metrics = metrics.clone(prefix='val_').to(self.gpu_id)
        self.results_dict = {dataset: {metric: [] for metric in ['loss','auprc','auroc']} for dataset in ['train','val']}

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

    def training_step(
            self,
            batch,
            batch_size: int,
            accum_steps: int
        ) -> float:

        self.model.train()
        if self.backbone_only:
            inputs, labels = batch['image'].to(self.gpu_id), batch['lirads'].to(self.gpu_id)
        else:
            inputs, labels, pt_info, padding_mask = prep_batch(batch, batch_size=batch_size)
            inputs, labels, padding_mask = inputs.to(self.gpu_id), labels.to(self.gpu_id), padding_mask.to(self.gpu_id)
            pt_info = [info.to(self.gpu_id) for info in pt_info]

        with autocast(enabled=self.amp):
            logits = self.model(inputs) if self.backbone_only else self.model(inputs, pad_mask=padding_mask, pt_info=pt_info)
            logits = logits.squeeze(-1) if not self.backbone_only else logits
            loss = self.train_loss(logits, labels.long() if self.backbone_only else labels.float())
            loss /= accum_steps

        self.scaler.scale(loss).backward()
        preds = F.softmax(logits, dim=-1) if self.backbone_only else F.sigmoid(logits)
        self.train_metrics.update(preds, labels.int())

        return loss.item()

    def accumulation_step(
            self,
            step: int,
            clip_grad: bool = True
        ) -> None:

        for i, param_group in enumerate(self.optim.param_groups):
            param_group['lr'] = self.lr_schedule[step]
            if i == 0:  # only the first group is regularized
                param_group['weight_decay'] = self.wd_schedule[step]

        if clip_grad:
            self.scaler.unscale_(self.optim)
            nn.utils.clip_grad_norm_(self.params, max_norm=1.0, norm_type=2)

        self.scaler.step(self.optim)
        self.scaler.update()
        self.optim.zero_grad(set_to_none=True)

    @torch.no_grad()
    def validation_step(
            self, 
            batch,
            batch_size: int = 1
        ) -> float:

        '''
        Validate the model.

        Args:
            epoch (int): Current epoch.
        
        Returns:
            tuple: Tuple containing the loss and F1-score.
        '''
        self.model.eval()
        if self.backbone_only:
            inputs, labels = batch['image'].to(self.gpu_id), batch['lirads'].to(self.gpu_id)
        else:
            inputs, labels, pt_info, padding_mask = prep_batch(batch, batch_size=batch_size)
            inputs, labels, padding_mask = inputs.to(self.gpu_id), labels.to(self.gpu_id), padding_mask.to(self.gpu_id)
            pt_info = [info.to(self.gpu_id) for info in pt_info]

        with autocast(enabled=self.amp):
            logits = self.model(inputs) if self.backbone_only else self.model(inputs, pad_mask=padding_mask, pt_info=pt_info)
            logits = logits.squeeze(-1) if not self.backbone_only else logits
            loss = self.val_loss(logits, labels.long() if self.backbone_only else labels.float())

        preds = F.softmax(logits, dim=-1) if self.backbone_only else F.sigmoid(logits)
        self.val_metrics.update(preds, labels.int())

        return loss.item()
    
    def train(
            self, 
            fold: int,
            batch_size: int,
            accum_steps: int,
            val_steps: int = 10
        ) -> None:

        '''
        Training loop.

        Args:
            train_ds (monai.data): Training dataset.
            metric (torch.nn): Metric to assess model performance while training/validating.
            min_epochs (int): Minimum number of epochs to train.
            val_every (int): Integer specifying the interval the model is validated.
            accum_steps (int): Number of accumulation steps to use before updating the model weights.
            early_stopping (bool): Whether to use early stopping.
            patience (int): Number of epochs to wait before aborting the training process.
            num_patches (int): Number of slice patches to use for training.
        '''
        start_time = time.time()
        step = 0
        accum_loss = 0.0
        running_train_loss = 0.0
        best_loss = 1.0
        best_metric = 0.0
        best_auprc = 0.0
        best_auroc = 0.0
        self.optim.zero_grad(set_to_none=True)

        for epoch in range(self.num_steps * accum_steps // len(self.dataloaders['train']) + 1):
            for idx, train_batch in enumerate(self.dataloaders['train']):

                step = epoch * len(self.dataloaders['train']) + idx
                update_step = step // accum_steps
                if self.gpu_id == 0 and step % (accum_steps * val_steps) == 0:
                    print('-' * 15)
                    print(f'Step {update_step}/{self.num_steps}')
                    print('-' * 15)

                accum_loss += self.training_step(train_batch, batch_size, accum_steps)

                if (step + 1) % accum_steps == 0:
                    self.accumulation_step(update_step, clip_grad=False)
                    if self.gpu_id == 0:
                        print(f"Step Loss: {accum_loss:.4f}")
                    running_train_loss += accum_loss
                    accum_loss = 0.0

                if (step + 1) % (accum_steps * val_steps) == 0:
                    running_val_loss = 0.0
                    for val_batch in self.dataloaders['val']:
                        running_val_loss += self.validation_step(val_batch)

                    train_loss = running_train_loss / val_steps
                    running_train_loss = 0.0
                    val_loss = torch.Tensor([running_val_loss / len(self.dataloaders['val'])])
                    dist.all_reduce(val_loss.to(self.gpu_id), op=dist.ReduceOp.AVG)
                    train_results = self.train_metrics.compute()
                    val_results = self.val_metrics.compute()
                    self.train_metrics.reset()
                    self.val_metrics.reset()
                    val_loss = 1.0 if val_loss.item() > 1.0 else val_loss.item()
                    val_metric = (val_results[self.val_metric_str[0]] + val_results[self.val_metric_str[1]]) / 2

                    if self.gpu_id == 0:
                        self.log_dict(phase='train', keys=['loss', 'auprc', 'auroc'], values=[train_loss, train_results[self.train_metric_str[0]].cpu().item(), train_results[self.train_metric_str[1]].cpu().item()])
                        self.log_dict(phase='val', keys=['loss', 'auprc', 'auroc'], values=[val_loss, val_results[self.val_metric_str[0]].cpu().item(), val_results[self.val_metric_str[1]].cpu().item()])
                        print(f"[GPU {self.gpu_id}] Step {update_step}/{self.num_steps}, Training Loss: {train_loss:.4f}, AUPRC: {train_results[self.train_metric_str[0]]:.4f}, and AUROC {train_results[self.train_metric_str[1]]:.4f}")
                        print(f"[GPU {self.gpu_id}] Step {update_step}/{self.num_steps}, Validation Loss: {val_loss:.4f}, AUPRC: {val_results[self.val_metric_str[0]]:.4f}, and AUROC {val_results[self.val_metric_str[1]]:.4f}")

                    if (val_metric * (1 - val_loss)) ** 0.5 > best_metric:
                        best_metric = (val_metric * (1 - val_loss)) ** 0.5
                        best_loss = val_loss
                        best_auprc = val_results[self.val_metric_str[0]]
                        best_auroc = val_results[self.val_metric_str[1]]
                        dist.barrier()
                        if self.gpu_id == 0:
                            print(f'[GPU {self.gpu_id}] New best Validation Loss: {best_loss:.4f} and Metric: {val_metric:.4f}. Saving model weights...')
                            self.save_output(self.model.module.state_dict(), 'weights', fold)
                        dist.barrier()

                if (step + 1) == self.num_steps * accum_steps:
                    break

        if self.gpu_id == 0:
            time_elapsed = time.time() - start_time
            print(f'[Fold {fold}] Training complete in {time_elapsed // 60:.0f}min {time_elapsed % 60:.0f}sec')
            print(f'[Fold {fold}] Loss: {best_loss:.4f}, AUPRC: {best_auprc:.4f}, and {best_auroc:.4f} of best model configuration.')
            self.save_output(self.results_dict, 'history', fold)
        dist.barrier()
    
    def visualize_training(
            self, 
            phases: List[str],
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
    parser.add_argument("--loss-fn", type=str, default='bce',
                        help="Loss function to use.")
    parser.add_argument("--gamma", type=int, default=2,
                        help="Gamma parameter to use for focal loss.")
    parser.add_argument("--pretrained", action='store_true',
                        help="Flag to use pretrained weights.")
    parser.add_argument("--use-v2", action='store_true',
                        help="Flag to use ConvNeXt version 2.")
    parser.add_argument("--kernel-size", type=int, default=3,
                        help="Kernel size of the convolutional layers.")
    parser.add_argument("--out-dim", type=int, default=4096,
                        help="Output dimension of pretrained projection head.")
    parser.add_argument("--distributed", action='store_true',
                        help="Flag to enable distributed training.")
    parser.add_argument("--amp", action='store_true',
                        help="Flag to enable automated mixed precision training.")
    parser.add_argument("--backbone-only", action='store_true',
                        help="Flag to train the CNN backbone.")

    parser.add_argument("--num-classes", default=2, type=int,
                        help="Number of output classes. Defaults to 2.")
    parser.add_argument("--label-smoothing", default=0.0, type=float, 
                        help="Label smoothing to use for training. Defaults to 0.")
    parser.add_argument("--num-steps", default=100, type=int, 
                        help="Number of steps to train for. Defaults to 100.")
    parser.add_argument("--warmup-steps", default=10, type=int, 
                        help="Number of warmup steps. Defaults to 10.")
    parser.add_argument("--val-interval", default=1, type=int,
                        help="Number of epochs to wait before running validation. Defaults to 2.")
    parser.add_argument("--k-folds", default=5, type=int, 
                        help="Number of folds to use in cross validation. Defaults to 5.")
    parser.add_argument("--batch-size", default=4, type=int, 
                        help="Batch size to use for training. Defaults to 4.")
    parser.add_argument("--effective-batch-size", default=32, type=int,
                        help="Total batch size: batch size x number of devices x number of accumulations steps.")
    parser.add_argument("--seq-length", default=4, type=int,
                        help="Flag to scale the raw model logits using temperature scalar.")
    parser.add_argument("--min-learning-rate", default=1e-6, type=float, 
                        help="Final learning rate. Defaults to 1e-6.")
    parser.add_argument("--weight-decay", default=0.05, type=float, 
                        help="Initial weight decay value. Defaults to 0.05.")
    parser.add_argument("--max-weight-decay", default=0.5, type=float, 
                        help="Final weight decay value. Defaults to 0.5.")
    parser.add_argument("--stochastic-depth", default=0.0, type=float, 
                        help="Stochastic depth rate to use for training. Defaults to 0.")
    parser.add_argument("--dropout", default=0.0, type=float, 
                        help="Dropout rate to use for training. Defaults to 0.")
    parser.add_argument("--epsilon", default=1e-5, type=float, 
                        help="Epsilon value to use for norm layers. Defaults to 1e-5.")
    parser.add_argument("--crop-size", default=72, type=int, 
                        help="Global crop size to use. Defaults to 72.")

    parser.add_argument("--mod-list", default=MODALITY_LIST, nargs='+', 
                        help="List of modalities to use for training")
    parser.add_argument("--seed", default=1234, type=int, 
                        help="Seed to use for reproducibility")
    parser.add_argument("--num-workers", default=8, type=int,
                        help="Number of workers to use for training. Defaults to 8.")
    parser.add_argument("--suffix", default='MedNet', type=str, 
                        help="File suffix for identification")
    parser.add_argument("--data-dir", default=DATA_DIR, type=str, 
                        help="Path to data directory")
    parser.add_argument("--weights-dir", default=WEIGHTS_DIR, type=str, 
                        help="Path to weights directory")
    parser.add_argument("--results-dir", default=RESULTS_DIR, type=str, 
                        help="Path to results directory")
    return parser.parse_args()

def load_train_data(
        args: argparse.Namespace,
        device: torch.device
    ) -> tuple:

    '''
    Load training objects.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        tuple: Training objects consisting of the datasets, dataloaders, the model, and the performance metric.
    '''
    folds = range(args.k_folds)
    phases = ['train', 'val']
    data_dict, label_df = DatasetPreprocessor(data_dir=args.data_dir).load_data(
        modalities=args.mod_list, 
        keys=['label','lirads','delta'], 
        file_name='labels.csv',
        verbose=False)
    lirads_dict, lirads_df = DatasetPreprocessor(data_dir=args.data_dir).load_data(
        modalities=args.mod_list, 
        keys=['label','lirads','delta'], 
        file_name='lirads.csv')
    dev, test = GroupStratifiedSplit(split_ratio=0.75).split_dataset(label_df)
    if args.backbone_only:
        lirads_test = lirads_df[lirads_df['patient_id'].isin(test['patient_id'])]
        lirads_dev = lirads_df[-lirads_df['patient_id'].isin(lirads_test['patient_id'])]
        dev, test, data_dict = lirads_dev, lirads_test, lirads_dict
    if args.k_folds > 1:
        cv_folds = StratifiedGroupKFold(n_splits=args.k_folds).split(dev, y=dev['label'], groups=dev['patient_id'])
        indices = [(dev.iloc[train_idx], dev.iloc[val_idx]) for train_idx, val_idx in list(cv_folds)]
        split_dict = [convert_to_dict([indices[k][0], indices[k][1]], data_dict=data_dict, split_names=phases) for k in folds]
    elif args.k_folds == 1:
        train, val = GroupStratifiedSplit(split_ratio=0.8).split_dataset(dev)
        split_dict = [convert_to_dict([train, val], data_dict=data_dict, split_names=phases, verbose=True) for k in folds]
    else:
        phases = ['train']
        folds = range(1)
        split_dict = [convert_to_dict([dev], data_dict=data_dict, split_names=phases, verbose=True) for k in folds]

    if args.backbone_only:
        counts = dev['lirads'].value_counts()
        pos_weight = counts.sum() / (counts * 2)
        class_dict = {x: [[patient['lirads'] for patient in split_dict[k][x]] for k in folds] for x in phases}
        split_dict = {x: [partition_dataset_classes(
            data=split_dict[k][x],
            classes=class_dict[x][k],
            num_partitions=dist.get_world_size(),
            shuffle=True,
            even_divisible=True
            )[dist.get_rank()] for k in folds] for x in phases}
        datasets = {x: [CacheDataset(
            data=split_dict[x][k], 
            transform=transforms(
                dataset=x,
                modalities=args.mod_list, 
                device=device,
                crop_size=ensure_tuple_rep(args.crop_size, 3)),
            num_workers=8,
            copy_cache=False
            ) for k in folds] for x in phases}
        dataloader = {x: [ThreadDataLoader(
            dataset=datasets[x][k], 
            batch_size=args.batch_size,
            shuffle=(True if x == 'train' else False),   
            drop_last=(True if x == 'train' else False),
            num_workers=0
            ) for k in folds] for x in phases}

    else:
        seq_split_dict = [convert_to_seqdict(split_dict[k], args.mod_list, phases) for k in folds]
        seq_class_dict = {x: [[max(patient['label']) for patient in seq_split_dict[k][x][0]] for k in folds] for x in phases}
        seq_split_dict = {x: [partition_dataset_classes(
            data=seq_split_dict[k][x][0],
            classes=seq_class_dict[x][k],
            num_partitions=dist.get_world_size(),
            shuffle=True,
            even_divisible=False
            )[dist.get_rank()] for k in folds] for x in phases}
        datasets = {x: [CacheSeqDataset(
            data=seq_split_dict[x][k],
            image_keys=args.mod_list,
            transform=transforms(
                dataset=x, 
                modalities=args.mod_list,
                device=device,
                crop_size=ensure_tuple_rep(args.crop_size, 3)),
            num_workers=args.num_workers,
            copy_cache=False
            ) for k in folds] for x in phases}
        dataloader = {x: [ThreadDataLoader(
            datasets[x][k], 
            batch_size=(args.batch_size if x == 'train' else 1), 
            shuffle=(True if x == 'train' else False),   
            drop_last=(True if x == 'train' else False),   
            num_workers=0,
            collate_fn=(SequenceBatchCollater(
                keys=['image','label','lirads','delta'], 
                seq_length=args.seq_length) if x == 'train' else list_data_collate)
            ) for k in folds] for x in phases}
        _, counts = np.unique(seq_class_dict['train'][0], return_counts=True)
        pos_weight = counts[1] / counts.sum()

    return dataloader, pos_weight

def load_train_objs(
        args: argparse.Namespace,
        model: nn.Module,
        learning_rate: float = 1e-4,
        pos_weight: float | List[float] | None = None
    ) -> tuple:

    if args.loss_fn == 'focal':
        train_fn = FocalLoss(gamma=args.gamma, alpha=pos_weight, label_smoothing=args.label_smoothing)
    else:
        train_fn = BinaryCELoss(weights=pos_weight, label_smoothing=args.label_smoothing)
    if args.backbone_only:
        train_fn = nn.CrossEntropyLoss(weight=torch.Tensor(pos_weight), label_smoothing=args.label_smoothing)
    loss_fn = [train_fn, BinaryCELoss() if not args.backbone_only else nn.CrossEntropyLoss()]
    params = get_params_groups(model)
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
    scheduler = [lr_schedule, wd_schedule]
    return loss_fn, optimizer, scheduler

def load_weights(
        args: argparse.Namespace,
        weights_path: str
    ) -> dict:

    weights = torch.load(weights_path, map_location='cpu')
    weights = {k.replace('backbone.', ''): v for k, v in weights.items()}
    weights['downsample_layers.0.0.weight'] = weights['downsample_layers.0.0.weight'].repeat(1, len(args.mod_list), 1, 1, 1)
    return weights

def transforms(
        dataset: str,
        modalities: list,
        device: torch.device,
        crop_size: tuple = (72, 72, 72),
        image_spacing: tuple = (1.5, 1.5, 1.5)
    ) -> transforms:

    '''
    Perform data transformations on image and image labels.

    Args:
        dataset (str): Dataset to apply transformations on. Can be 'train', 'val' or 'test'.

    Returns:
        transforms (monai.transforms): Data transformations to be applied.
    '''
    prep = [
        LoadImaged(keys=modalities, image_only=True),
        EnsureChannelFirstd(keys=modalities),
        Orientationd(keys=modalities, axcodes='PLI'),
        Spacingd(keys=modalities, pixdim=image_spacing, mode=3),
        ResampleToMatchd(keys=modalities, key_dst=modalities[0], mode=3),
        PercentileSpatialCropd(
            keys=modalities,
            roi_center=(0.5, 0.5, 0.5),
            roi_size=(0.85, 0.8, 0.99),
            min_size=(82, 82, 82)),
        CopyItemsd(keys=modalities[0], names='mask'),
        Lambdad(
            keys='mask', 
            func=lambda x: torch.where(x > torch.mean(x), 1, 0)),
        KeepLargestConnectedComponentd(keys='mask', connectivity=1),
        CropForegroundd(
            keys=modalities,
            source_key='mask',
            select_fn=lambda x: x > 0,
            k_divisible=1,
            allow_smaller=False),
        ConcatItemsd(keys=modalities, name='image'),
        DeleteItemsd(keys=modalities + ['mask']),
        YeoJohnsond(keys='image', lmbda=0.1, channel_wise=True),
        RobustNormalized(keys='image', channel_wise=True),
        SoftClipIntensityd(keys='image', min_value=-2.5, max_value=2.5, channel_wise=True),
        PercentileSpatialCropd(
            keys='image',
            roi_center=(0.45, 0.25, 0.3),
            roi_size=(0.6, 0.45, 0.4),
            min_size=(82, 82, 82)),
        CenterSpatialCropd(keys='image', roi_size=(96, 96, 96))
    ]

    train = [
        EnsureTyped(keys='image', track_meta=False, device=device, dtype=torch.float),
        RandSpatialCropd(keys='image', roi_size=crop_size, random_size=False),
        RandFlipd(keys='image', prob=0.5, spatial_axis=0),
        RandFlipd(keys='image', prob=0.5, spatial_axis=1),
        RandFlipd(keys='image', prob=0.5, spatial_axis=2),
        RandRotate90d(keys='image', prob=0.5),
        # RandScaleIntensityd(keys='image', prob=1.0, factors=0.1),
        # RandShiftIntensityd(keys='image', prob=1.0, offsets=0.1),
        # RandGibbsNoised(keys='image', prob=0.1, alpha=(0.1, 0.9)),
        # RandKSpaceSpikeNoised(keys='image', prob=0.1, channel_wise=True),
        # RandCoarseDropoutd(keys='image', prob=0.25, holes=1, spatial_size=(16, 16, 16), max_spatial_size=(48, 48, 48)),
        NormalizeIntensityd(
            keys='image', 
            subtrahend=(0.3736, 0.4208, 0.4409, 0.4367),
            divisor=(0.6728, 0.6765, 0.6912, 0.6864),
            channel_wise=True)
    ]

    test = [
        CenterSpatialCropd(keys='image', roi_size=(72, 72, 72)),
        NormalizeIntensityd(
            keys='image', 
            subtrahend=(0.3736, 0.4208, 0.4409, 0.4367),
            divisor=(0.6728, 0.6765, 0.6912, 0.6864),
            channel_wise=True),
        EnsureTyped(keys='image', track_meta=False, device=device, dtype=torch.float)
    ]

    if dataset == 'train':
        return Compose(prep + train)
    elif dataset in ['val', 'test']:
        return Compose(prep + test)
    else:
        raise ValueError ("Dataset must be 'train', 'val' or 'test'.")

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
    batch_size_gpu = args.batch_size * accum_steps
    learning_rate = scale_learning_rate(args.effective_batch_size)
    num_classes = args.num_classes if args.num_classes > 2 else 1

    cv_dataloader, pos_weight = load_train_data(args, device_id)
    set_track_meta(False)

    if args.backbone_only:
        steps = np.arange(2000, 32000 + 1, 2000)
        for step in steps:
            for k in range(args.k_folds): 
                dataloader = {x: cv_dataloader[x][k] for x in ['train','val']}
                model = convnext3d_tiny(
                    in_chans=len(args.mod_list), 
                    num_classes=3,
                    kernel_size=args.kernel_size,
                    drop_path_rate=args.stochastic_depth,
                    use_v2=args.use_v2, 
                    eps=args.epsilon)
                weights_path = os.path.join(args.results_dir, f'model_weights/weights_fold{step}_dmri_tiny_{args.out_dim}.pth')
                weights = load_weights(args, weights_path)
                model.load_state_dict(weights, strict=False)
                model = model.to(device_id)
                for name, param in model.named_parameters():
                    if not name.startswith(('head')):
                        param.requires_grad = False
                if args.distributed:
                    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id])
                loss_fn, optimizer, scheduler = load_train_objs(
                    args, 
                    model=model, 
                    pos_weight=pos_weight,
                    learning_rate=learning_rate)
                trainer = Trainer(
                    model=model, 
                    loss_fn=loss_fn,
                    dataloaders=dataloader, 
                    optimizer=optimizer,
                    scheduler=scheduler,
                    num_folds=(args.k_folds if args.k_folds > 0 else 1),
                    num_steps=args.num_steps,
                    amp=args.amp,
                    backbone_only=args.backbone_only,
                    suffix=args.suffix + f'_step{step}',
                    output_dir=args.results_dir)
                trainer.train(
                    fold=k,
                    batch_size=args.batch_size, 
                    accum_steps=accum_steps,
                    val_steps=args.val_interval)
            trainer.visualize_training(['train','val'], 'loss')
            trainer.visualize_training(['train','val'], 'auprc')
            trainer.visualize_training(['train','val'], 'auroc')
    
    else:
        for k in range(args.k_folds):
            dataloader = {x: cv_dataloader[x][k] for x in ['train','val']}
            backbone = convnext3d_tiny(
                in_chans=len(args.mod_list), 
                kernel_size=args.kernel_size,
                drop_path_rate=args.stochastic_depth,
                use_v2=args.use_v2, 
                eps=args.epsilon)
            model = MedNet(
                backbone, 
                num_classes=num_classes, 
                max_len=22,
                dropout=args.dropout, 
                eps=args.epsilon)
            if args.pretrained:
                weights = load_weights(os.path.join(args.results_dir, 'model_weights/weights_fold8000_dmri_tiny_v1.pth'))
                model.backbone.load_state_dict(weights, strict=False)
            model = model.to(device_id)
            for name, param in model.named_parameters():
                if name.startswith(('backbone')):
                    param.requires_grad = False
            if args.distributed:
                model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id])
            loss_fn, optimizer, scheduler = load_train_objs(
                args, 
                model=model, 
                pos_weight=pos_weight,
                learning_rate=learning_rate)
            trainer = Trainer(
                model=model, 
                loss_fn=loss_fn,
                dataloaders=dataloader, 
                optimizer=optimizer,
                scheduler=scheduler,
                num_folds=(args.k_folds if args.k_folds > 0 else 1),
                num_steps=args.num_steps,
                amp=args.amp,
                backbone_only=args.backbone_only,
                suffix=args.suffix,
                output_dir=args.results_dir)
            trainer.train(
                fold=k,
                batch_size=args.batch_size, 
                accum_steps=accum_steps,
                val_steps=args.val_interval)
        trainer.visualize_training(['train','val'], 'loss')
        trainer.visualize_training(['train','val'], 'auprc')
        trainer.visualize_training(['train','val'], 'auroc')

    if args.distributed:
        cleanup()
    print('Script finished')
    

RESULTS_DIR = '/Users/noltinho/thesis/results'
DATA_DIR = '/Users/noltinho/thesis/sensitive_data'
MODALITY_LIST = ['T1W_OOP','T1W_IP','T1W_DYN','T2W_TES','T2W_TEL','DWI_b150','DWI_b400','DWI_b800']
WEIGHTS_DIR = '/Users/noltinho/MedicalNet/pytorch_files/pretrain'

if __name__ == '__main__':
    args = parse_args()
    main(args)