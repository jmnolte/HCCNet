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
from torchmetrics.classification import BinaryAveragePrecision, BinaryAUROC
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
    Resized
)
from monai import transforms
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    set_track_meta,
    partition_dataset_classes,
    list_data_collate
)
from monai.utils import set_determinism
from data.transforms import PercentileSpatialCropd, YeoJohnsond, SoftClipIntensityd
from data.splits import GroupStratifiedSplit
from data.datasets import CacheSeqDataset
from data.utils import DatasetPreprocessor, convert_to_dict, convert_to_seqdict, SequenceBatchCollater
from models.mednet import MedNet
from models.convnext3d import convnext3d_atto, convnext3d_femto, convnext3d_pico, convnext3d_nano, convnext3d_tiny
from losses.focalloss import FocalLoss
from losses.binaryceloss import BinaryCELoss
from optimizer.adams import AdamS


class Trainer:
    
    def __init__(
            self, 
            model: nn.Module, 
            loss_fn: List[nn.Module],
            dataloaders: dict, 
            optimizer: optim,
            scheduler: lr_scheduler | None = None,
            num_folds: int = 1,
            num_steps: int = 1000,
            amp: bool = True,
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
        self.schedule = scheduler

        metrics = MetricCollection([BinaryAveragePrecision(), BinaryAUROC()])
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
        inputs, labels, pos_token, padding_mask = self.prep_batch(batch, batch_size=batch_size, device_id=self.gpu_id)
        weight = self.calc_weight(pos_token)

        with autocast(enabled=self.amp):
            logits = self.model(inputs, pos_token=pos_token, padding_mask=padding_mask)
            loss = self.train_loss(logits.squeeze(-1), labels.float())
            # loss = torch.mean(torch.mul(weight, loss)) / accum_steps
            loss /= accum_steps

        self.scaler.scale(loss).backward()
        preds = F.sigmoid(logits.squeeze(-1))
        self.train_metrics.update(preds, labels)

        return loss.item()

    def accumulation_step(
            self,
            clip_grad: bool = True
        ) -> None:

        if clip_grad:
            self.scaler.unscale_(self.optim)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2)

        self.scaler.step(self.optim)
        self.scaler.update()
        self.optim.zero_grad(set_to_none=True)
        if self.schedule is not None:
            self.schedule.step()

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
        inputs, labels, pos_token, padding_mask = self.prep_batch(batch, batch_size=batch_size, device_id=self.gpu_id)

        with autocast(enabled=self.amp):
            logits = self.model(inputs, pos_token=pos_token, padding_mask=None)
            loss = self.val_loss(logits.squeeze(-1), labels.float())

        preds = F.sigmoid(logits.squeeze(-1))
        self.val_metrics.update(preds, labels)

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
        best_metric = 0.0

        self.optim.zero_grad(set_to_none=True)
        for train_epoch in range(self.num_steps * accum_steps // len(self.dataloaders['train']) + 1):
            for train_step, train_batch in enumerate(self.dataloaders['train']):

                step = train_epoch * len(self.dataloaders['train']) + train_step
                if self.gpu_id == 0 and step % (accum_steps * val_steps) == 0:
                    print('-' * 15)
                    print(f'Step {step}/{self.num_steps}')
                    print('-' * 15)
                accum_loss += self.training_step(train_batch, batch_size, accum_steps)

                if (step + 1) % accum_steps == 0:
                    self.accumulation_step(clip_grad=False)
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
                    dist.reduce(val_loss.to(self.gpu_id), dst=0, op=dist.ReduceOp.AVG)
                    train_results = self.train_metrics.compute()
                    val_results = self.val_metrics.compute()
                    self.train_metrics.reset()
                    self.val_metrics.reset()
                    val_loss = 1.0 if val_loss.item() > 1.0 else val_loss.item()
                    val_metric = (val_results['val_BinaryAveragePrecision'] + val_results['val_BinaryAUROC']) / 2

                    if self.gpu_id == 0:
                        self.log_dict(phase='train', keys=['loss', 'auprc', 'auroc'], values=[train_loss, train_results['train_BinaryAveragePrecision'].cpu().item(), train_results['train_BinaryAUROC'].cpu().item()])
                        self.log_dict(phase='val', keys=['loss', 'auprc', 'auroc'], values=[val_loss, val_results['val_BinaryAveragePrecision'].cpu().item(), val_results['val_BinaryAUROC'].cpu().item()])
                        print(f"[GPU {self.gpu_id}] Step {step}/{self.num_steps}, Training Loss: {train_loss:.4f}, AUPRC: {train_results['train_BinaryAveragePrecision']:.4f}, and AUROC {train_results['train_BinaryAUROC']:.4f}")
                        print(f"[GPU {self.gpu_id}] Step {step}/{self.num_steps}, Validation Loss: {val_loss:.4f}, AUPRC: {val_results['val_BinaryAveragePrecision']:.4f}, and AUROC {val_results['val_BinaryAUROC']:.4f}")

                    if (val_metric * (1 - val_loss)) ** 0.5 > best_metric:
                        best_metric = (val_metric * (1 - val_loss)) ** 0.5
                        best_loss = val_loss
                        best_auprc = val_results['val_BinaryAveragePrecision']
                        best_auroc = val_results['val_BinaryAUROC']
                        dist_barrier()
                        if self.gpu_id == 0:
                            print(f'[GPU {self.gpu_id}] New best Validation Loss: {best_loss:.4f} and Metric: {val_metric:.4f}. Saving model weights...')
                            self.save_output(self.model.state_dict(), 'weights', fold)
                        dist.barrier()

                if step == self.num_steps * accum_steps:
                    break

        if self.gpu_id == 0:
            time_elapsed = time.time() - start_time
            print(f'[Fold {fold}] Training complete in {time_elapsed // 60:.0f}min {time_elapsed % 60:.0f}sec')
            print(f'[Fold {fold}] Loss: {best_loss:.4f}, AUPRC: {best_auprc:.4f}, and {best_auroc:.4f} of best model configuration.')
            self.save_output(self.results_dict, 'history', fold)
            
    @staticmethod
    def prep_batch(
            data: dict, 
            batch_size: int, 
            device_id: torch.device | None = None, 
            normalize: bool = True
        ) -> tuple:

        stats_dict = {'min': 18.0, 'max': 100.0}
        B, C, H, W, D = data['image'].shape
        for key in data:
            if key == 'image':
                data[key] = data[key].reshape(batch_size, B // batch_size, C, H, W, D)
            elif not isinstance(data[key], list):
                try:
                    data[key] = data[key].reshape(batch_size, B // batch_size)
                except:
                    pass
        data['label'] = torch.max(data['label'], dim=1).values
        data['age'] = torch.where(data['age'] != 0.0, (data['age'] - stats_dict['min']) / (stats_dict['max'] - stats_dict['min']), 0.0) if normalize else data['age']
        mask = torch.where(data['age'] == 0.0, 1, 0)

        if device_id is not None:
            return data['image'].to(device_id), data['label'].to(device_id, dtype=torch.int), data['age'].to(device_id, dtype=torch.float), mask.to(device_id, dtype=torch.float)
        else:
            return data['image'], data['label'].to(torch.int), data['age'].to(torch.float), mask.to(torch.float)

    @staticmethod
    def calc_weight(
            data: torch.Tensor, 
            base: int = 3, 
            dim: int = 1
        ) -> torch.Tensor:

        count = torch.count_nonzero(data, dim=dim)
        base = torch.Tensor([base])
        return torch.log(count + 1) / torch.log(base.to(count.device))
    
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


def get_wd_params(model: nn.Module):
    decay = list()
    no_decay = list()
    for name, param in model.named_parameters():
        if hasattr(param,'requires_grad') and not param.requires_grad:
            continue
        if 'weight' in name and 'norm' not in name and 'bn' not in name:
            decay.append(param)
        else:
            no_decay.append(param)
    return decay, no_decay


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
    parser.add_argument("--pooling-mode", default='cls', type=str,
                        help="Pooling mode. Can be cls, mean, max, or att. Defaults to cls.")
    parser.add_argument("--pretrained", action='store_true',
                        help="Flag to use pretrained weights.")
    parser.add_argument("--distributed", action='store_true',
                        help="Flag to enable distributed training.")
    parser.add_argument("--amp", action='store_true',
                        help="Flag to enable automated mixed precision training.")
    parser.add_argument("--scheduler", action='store_true',
                        help="Flag to use a LR scheduler.")
    parser.add_argument("--num-classes", default=2, type=int,
                        help="Number of output classes. Defaults to 2.")
    parser.add_argument("--num-steps", default=100, type=int, 
                        help="Number of epochs to train for. Defaults to 10.")
    parser.add_argument("--batch-size", default=4, type=int, 
                        help="Batch size to use for training. Defaults to 4.")
    parser.add_argument("--effective-batch-size", default=32, type=int,
                        help="Total batch size: batch size x number of devices x number of accumulations steps.")
    parser.add_argument("--seq-length", default=4, type=int,
                        help="Flag to scale the raw model logits using temperature scalar.")
    parser.add_argument("--k-folds", default=0, type=int, 
                        help="Number of folds to use in cross validation. Defaults to 0.8.")
    parser.add_argument("--label-smoothing", default=0.0, type=float, 
                        help="Label smoothing to use for training. Defaults to 0.")
    parser.add_argument("--weight-decay", default=0, type=float, 
                        help="Weight decay to use for training. Defaults to 0.")
    parser.add_argument("--dropout", default=0, type=float, 
                        help="Dropout rate to use for training. Defaults to 0.")
    parser.add_argument("--stochastic-depth", default=0, type=float, 
                        help="Stochastic depth rate to use for training. Defaults to 0.")
    parser.add_argument("--epsilon", default=1e-5, type=float, 
                        help="Epsilon value to use for norm layers. Defaults to 1e-5.")
    parser.add_argument("--val-interval", default=1, type=int,
                        help="Number of epochs to wait before running validation. Defaults to 2.")
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
    data_dict, label_df = DatasetPreprocessor(data_dir=args.data_dir).load_data(modalities=args.mod_list, keys=['label','age'], file_name='labels.csv')
    dev, test = GroupStratifiedSplit(split_ratio=0.75).split_dataset(label_df)
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
        transform=data_transforms(
            dataset=x, 
            modalities=args.mod_list,
            device=device), 
        num_workers=args.num_workers,
        copy_cache=False
        ) for k in folds] for x in phases}
    dataloader = {x: [ThreadDataLoader(
        datasets[x][k], 
        batch_size=(args.batch_size if x == 'train' else 1), 
        shuffle=(True if x == 'train' else False),   
        drop_last=(True if x == 'train' else False),   
        num_workers=0,
        collate_fn=(SequenceBatchCollater(keys=['image','label','age'], seq_length=args.seq_length) if x == 'train' else list_data_collate)
        ) for k in folds] for x in phases}
    _, counts = np.unique(seq_class_dict['train'][0], return_counts=True)
    pos_weight = counts[1] / counts.sum()

    return dataloader, pos_weight

def load_train_objs(
        args: argparse.Namespace,
        model: nn.Module,
        dataloader: dict,
        pos_weight: float | list | None = None,
        learning_rate: float = 1e-4,
        accum_steps: int = 1
    ) -> tuple:

    if args.loss_fn == 'focal':
        train_fn = FocalLoss(gamma=args.gamma, alpha=pos_weight, label_smoothing=args.label_smoothing)
    else:
        train_fn = BinaryCELoss(weights=pos_weight, label_smoothing=args.label_smoothing)
    loss_fn = [train_fn, BinaryCELoss()]
    decay, no_decay = get_wd_params(model)
    optimizer = optim.AdamW([{'params': no_decay, 'weight_decay': 0}, {'params': decay}], lr=learning_rate, weight_decay=args.weight_decay)
    # max_iter = len(dataloader['train']) / accum_steps * args.epochs
    scheduler = None
    return loss_fn, optimizer, scheduler

def scale_learning_rate(
        batch_size: int,
        num_devices: int
    ) -> float:

    alpha = {8: 0.0001, 16: 0.000141, 32: 0.0002, 64: 0.000282, 128: 0.0004, 256: 0.000565, 512: 0.0008}
    return alpha[batch_size] * np.sqrt(batch_size * num_devices) / np.sqrt(128)

def data_transforms(
        dataset: str,
        modalities: list,
        device: torch.device
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
        Spacingd(keys=modalities, pixdim=(1.5, 1.5, 1.5), mode=3),
        ResampleToMatchd(keys=modalities, key_dst=modalities[0], mode=3),
        PercentileSpatialCropd(
            keys=modalities,
            roi_center=(0.5, 0.5, 0.5),
            roi_size=(0.85, 0.8, 0.99),
            min_size=(96, 96, 96)),
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
        NormalizeIntensityd(keys='image', channel_wise=True),
        PercentileSpatialCropd(
            keys='image',
            roi_center=(0.45, 0.25, 0.3),
            roi_size=(0.6, 0.45, 0.4),
            min_size=(96, 96, 96)),
        SoftClipIntensityd(keys='image', min_value=-3.0, max_value=3.0, channel_wise=True)
    ]

    train = [
        EnsureTyped(keys='image', track_meta=False, device=device, dtype=torch.float),
        RandSpatialCropd(keys='image', roi_size=(72, 72, 72), random_size=False),
        RandFlipd(keys='image', prob=0.5, spatial_axis=0),
        RandFlipd(keys='image', prob=0.5, spatial_axis=1),
        RandFlipd(keys='image', prob=0.5, spatial_axis=2),
        RandRotate90d(keys='image', prob=0.5),
        RandScaleIntensityd(keys='image', prob=1.0, factors=0.1),
        RandShiftIntensityd(keys='image', prob=1.0, offsets=0.1),
        RandGibbsNoised(keys='image', prob=0.1, alpha=(0.1, 0.9)),
        RandKSpaceSpikeNoised(keys='image', prob=0.1, channel_wise=True),
        RandCoarseDropoutd(keys='image', prob=0.25, holes=1, spatial_size=(16, 16, 16), max_spatial_size=(48, 48, 48)),
        NormalizeIntensityd(
            keys='image', 
            subtrahend=(0.4467, 0.4416, 0.4409, 0.4212),
            divisor=(0.6522, 0.6429, 0.6301, 0.6041),
            channel_wise=True)
        # NormalizeIntensityd(keys='image', channel_wise=True)
    ]

    test = [
        CenterSpatialCropd(keys='image', roi_size=(72, 72, 72)),
        NormalizeIntensityd(
            keys='image', 
            subtrahend=(0.4467, 0.4416, 0.4409, 0.4212),
            divisor=(0.6522, 0.6429, 0.6301, 0.6041),
            channel_wise=True),
        # NormalizeIntensityd(keys='image', channel_wise=True),
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
    learning_rate = scale_learning_rate(batch_size_gpu, num_devices)
    num_classes = args.num_classes if args.num_classes > 2 else 1

    cv_dataloader, pos_weight = load_train_data(args, device_id)
    set_track_meta(False)

    for k in range(args.k_folds):
        dataloader = {x: cv_dataloader[x][k] for x in ['train','val']}
        backbone = convnext3d_femto(
            in_chans=len(args.mod_list), 
            use_grn=True, 
            drop_path_rate=args.stochastic_depth,
            eps=args.epsilon)
        model = MedNet(
            backbone, 
            num_classes=num_classes, 
            pooling_mode=args.pooling_mode, 
            dropout=args.dropout, 
            eps=args.epsilon)
        model = model.to(device_id)
        if args.distributed:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id])
        loss_fn, optimizer, scheduler = load_train_objs(
            args, 
            model=model, 
            dataloader=dataloader,
            pos_weight=pos_weight,
            learning_rate=learning_rate,
            accum_steps=accum_steps)
        trainer = Trainer(
            model=model, 
            loss_fn=loss_fn,
            dataloaders=dataloader, 
            optimizer=optimizer,
            scheduler=scheduler,
            num_folds=(args.k_folds if args.k_folds > 0 else 1),
            num_steps=args.num_steps,
            amp=args.amp,
            suffix=args.suffix,
            output_dir=args.results_dir)
        trainer.train(
            fold=k,
            batch_size=args.batch_size, 
            accum_steps=accum_steps,
            val_steps=args.val_interval)

    if args.distributed:
        cleanup()
    trainer.visualize_training('loss')
    trainer.visualize_training('auprc')
    trainer.visualize_training('auroc')
    print('Script finished')
    

RESULTS_DIR = '/Users/noltinho/thesis/results'
DATA_DIR = '/Users/noltinho/thesis/sensitive_data'
MODALITY_LIST = ['T1W_OOP','T1W_IP','T1W_DYN','T2W_TES','T2W_TEL','DWI_b150','DWI_b400','DWI_b800']
WEIGHTS_DIR = '/Users/noltinho/MedicalNet/pytorch_files/pretrain'

if __name__ == '__main__':
    args = parse_args()
    main(args)