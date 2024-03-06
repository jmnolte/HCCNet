from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
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
    RandGibbsNoised,
    RandRicianNoised,
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
from data.transforms import PercentileSpatialCropd, YeoJohnsond, SoftClipIntensityd, RobustNormalized
from data.splits import GroupStratifiedSplit
from data.datasets import CacheSeqDataset
from data.utils import DatasetPreprocessor, convert_to_dict, convert_to_seqdict, SequenceBatchCollater
from handlers.slidingwindowinference import SlidingWindowInferer
from models.mednet import MedNet
from models.convnext3d import convnext3d_atto, convnext3d_femto, convnext3d_pico, convnext3d_nano, convnext3d_tiny
from losses.focalloss import FocalLoss
from losses.binaryceloss import BinaryCELoss
from optimizer.adams import AdamS


class Trainer:
    
    def __init__(
            self, 
            model: nn.Module, 
            backbone: str, 
            loss_fn: str,
            amp: bool,
            dataloaders: dict, 
            learning_rate: float, 
            weight_decay: float,
            max_seq_length: int,
            accum_steps: int,
            pos_weight: float,
            output_dir: str,
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
        self.amp = amp
        self.backbone = backbone
        self.dataloaders = dataloaders
        self.output_dir = output_dir
        self.accum_steps = accum_steps

        self.scaler = GradScaler(enabled=amp)
        self.inferer = SlidingWindowInferer(self.model, self.gpu_id, max_length=max_seq_length)

        params = self.model.parameters()
        self.optim = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        self.schedule = optim.lr_scheduler.OneCycleLR(
            optimizer=self.optim,
            max_lr=learning_rate * 25,
            epochs=50,
            steps_per_epoch=int(len(dataloaders['train']) / accum_steps))

        if loss_fn == 'focal':
            self.train_bce = FocalLoss(gamma=2, alpha=pos_weight, label_smoothing=0.1).to(self.gpu_id)
            self.val_bce = FocalLoss(gamma=0).to(self.gpu_id)
        else:
            self.train_bce = BinaryCELoss(weights=pos_weight, label_smoothing=0.1).to(self.gpu_id)
            self.val_bce = BinaryCELoss().to(self.gpu_id)

        self.train_auprc = BinaryAveragePrecision().to(self.gpu_id)
        self.train_auroc = BinaryAUROC().to(self.gpu_id)
        self.val_auprc = BinaryAveragePrecision().to(self.gpu_id)
        self.val_auroc = BinaryAUROC().to(self.gpu_id)

    def save_output(
            self, 
            output_dict: dict, 
            output_type: str,
            fold: int | None = None,
            suffix: str | None = None
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
            folder_name = self.backbone + '_weights.pth'
        elif output_type == 'history':
            folder_name = self.backbone + f'_hist_fold{fold}' + suffix + '.npy'
        elif output_type == 'preds':
            folder_name = self.backbone + '_preds.npy'
        folder_path = os.path.join(self.output_dir, 'model_' + output_type, folder_name)
        folder_path_root = os.path.join(self.output_dir, 'model_' + output_type)

        if os.path.exists(folder_path):
            os.remove(folder_path)
        elif not os.path.exists(folder_path_root):
            os.makedirs(folder_path_root)

        if output_type == 'weights':
            torch.save(self.model.module.state_dict(output_dict), folder_path)
        else:
            np.save(folder_path, output_dict)

    def train(
            self,
            epoch: int,
            batch_size: int
            ) -> tuple:

        running_loss = 0.0
        self.model.train()
        self.optim.zero_grad(set_to_none=True)

        if self.gpu_id == 0:
            print('-' * 10)
            print(f'Epoch {epoch}')
            print('-' * 10)

        for step, batch_data in enumerate(self.dataloaders['train']):
            inputs, labels, pos_token, padding_mask = self.prep_batch(batch_data, batch_size=batch_size, device_id=self.gpu_id)
            weight = self.calc_weight(pos_token)

            with autocast(enabled=self.amp):
                logits = self.model(inputs, pos_token=pos_token, padding_mask=padding_mask)
                loss = self.train_bce(logits.squeeze(-1), labels.float())
                # loss = torch.mean(torch.mul(weight, loss)) / self.accum_steps
                loss /= self.accum_steps

            self.scaler.scale(loss).backward()
            running_loss += loss.item()
            preds = F.sigmoid(logits.squeeze(-1))
            self.train_auprc.update(preds, labels)
            self.train_auroc.update(preds, labels)

            if ((step + 1) % self.accum_steps == 0) or (step + 1 == len(self.dataloaders['train'])):
                self.scaler.unscale_(self.optim)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2)

                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad(set_to_none=True)
                # self.schedule.step()

                if self.gpu_id == 0:
                    print(f"{step + 1}/{len(self.dataloaders['train'])}, Batch Loss: {loss.item() * self.accum_steps:.4f}")

        epoch_loss = running_loss / (len(self.dataloaders['train']) // self.accum_steps)
        epoch_auprc = self.train_auprc.compute()
        epoch_auroc = self.train_auroc.compute()
        self.train_auprc.reset()
        self.train_auroc.reset()

        if self.gpu_id == 0:
            print(f"[GPU {self.gpu_id}] Epoch {epoch}, Training Loss: {epoch_loss:.4f}, AUPRC: {epoch_auprc:.4f}, and AUROC {epoch_auroc:.4f}")

        return epoch_loss, epoch_auprc, epoch_auroc

    def evaluate(
            self, 
            epoch: int,
            ) -> tuple:

        '''
        Validate the model.

        Args:
            epoch (int): Current epoch.
        
        Returns:
            tuple: Tuple containing the loss and F1-score.
        '''
        running_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for step, batch_data in enumerate(self.dataloaders['val']):
                inputs, labels, pos_token, padding_mask = self.prep_batch(batch_data, batch_size=1)
                labels = labels.to(self.gpu_id)

                with autocast(enabled=self.amp):
                    logits = self.inferer(inputs, pos_token=pos_token)
                    loss = self.val_bce(logits.squeeze(-1), labels.float())

                running_loss += loss.item()
                preds = F.sigmoid(logits.squeeze(-1))
                self.val_auprc.update(preds, labels)
                self.val_auroc.update(preds, labels)

        epoch_loss = torch.Tensor([running_loss / len(self.dataloaders['val'])])
        dist.reduce(epoch_loss.to(self.gpu_id), dst=0, op=dist.ReduceOp.AVG)
        epoch_auprc = self.val_auprc.compute()
        epoch_auroc = self.val_auroc.compute()
        self.val_auprc.reset()
        self.val_auroc.reset()

        if self.gpu_id == 0:
            print(f"[GPU {self.gpu_id}] Epoch {epoch}, Validation Loss: {epoch_loss.item():.4f}, AUPRC: {epoch_auprc:.4f}, and AUROC {epoch_auroc:.4f}")

        return epoch_loss.item(), epoch_auprc, epoch_auroc
    
    def training_loop(
            self, 
            fold: int,
            suffix: str,
            batch_size: int,
            max_epochs: int = 100,
            val_steps: int = 1,
            patience: int = 10
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
        since = time.time()
        best_loss = 10.0
        best_auprc = 0.0
        best_auroc = 0.0
        counter = 0
        log_book = {dataset: {log_type: [] for log_type in ['loss','metric']} for dataset in ['train','val']}
        stop_criterion = torch.zeros(1).to(self.gpu_id)

        for epoch in range(0, max_epochs):
            start_time = time.time()
            train_loss, train_auprc, train_auroc = self.train(epoch, batch_size)
            log_book['train']['loss'].append(train_loss)
            log_book['train']['metric'].append(train_auprc.cpu().item())

            if (epoch + 1) % val_steps == 0:
                val_loss, val_auprc, val_auroc = self.evaluate(epoch)
                log_book['val']['loss'].append(val_loss)
                log_book['val']['metric'].append(val_auprc.cpu().item())

                if self.gpu_id == 0:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_auprc = val_auprc
                        best_auroc = val_auroc
                        counter = 0
                        print(f'[GPU {self.gpu_id}] New best Validation Loss: {best_loss:.4f}. Saving model weights...')
                        best_weights = copy.deepcopy(self.model.state_dict())
                    else:
                        counter += 1
                        if counter >= patience and epoch >= max_epochs:
                            stop_criterion += 1

            train_time = time.time() - start_time
            if self.gpu_id == 0:
                print(f'Epoch {epoch} complete in {train_time // 60:.0f}min {train_time % 60:.0f}sec')

            dist.all_reduce(stop_criterion, op=dist.ReduceOp.SUM)
            if stop_criterion == 1:
                break

        if self.gpu_id == 0:
            time_elapsed = time.time() - since
            print(f'[Fold {fold}] Training complete in {time_elapsed // 60:.0f}min {time_elapsed % 60:.0f}sec')
            print(f'[Fold {fold}] Loss: {best_loss:.4f}, AUPRC: {best_auprc:.4f}, and {best_auroc:.4f} of best model configuration.')
            self.save_output(best_weights, 'weights')
            self.save_output(log_book, 'history', fold, suffix)
            
    @staticmethod
    def prep_batch(data: dict, batch_size: int, device_id: torch.device | None = None, normalize: bool = True) -> tuple:

        stats_dict = {'mean': 63.3394, 'std': 11.2231}
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
        data['age'] = torch.where(data['age'] != 0.0, (data['age'] - stats_dict['mean']) / stats_dict['std'], 0.0) if normalize else data['age']
        mask = torch.where(data['age'] == 0.0, 1, 0)

        if device_id is not None:
            return data['image'].to(device_id), data['label'].to(device_id, dtype=torch.int), data['age'].to(device_id, dtype=torch.float), mask.to(device_id, dtype=torch.float)
        else:
            return data['image'], data['label'].to(torch.int), data['age'].to(torch.float), mask.to(torch.float)

    @staticmethod
    def calc_weight(data: torch.Tensor, base: int = 3, dim: int = 1) -> torch.Tensor:

        count = torch.count_nonzero(data, dim=dim)
        base = torch.Tensor([base])
        return torch.log(count + 1) / torch.log(base.to(count.device))

    @staticmethod
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
    
    def visualize_training(
            self, 
            log_type: str,
            epochs: int = 50,
            k_folds: int = 10,
            suffix: str | None = None
            ) -> None:

        '''
        Visualize the training and validation history.

        Args:
            metric (str): String specifying the metric to be visualized. Can be 'loss' or 'f1score'.
        '''

        if log_type == 'loss':
            axis_label = 'Loss'
        elif log_type == 'metric':
            axis_label = 'AUPRC'
        plot_name = log_type + '_' + suffix + '.png' if suffix is not None else log_type + '.png'

        for dataset in ['train','val']:
            log_book = []
            for fold in range(k_folds):
                file_name = self.backbone + f'_hist_fold{fold}' + suffix + '.npy'
                fold_log = np.load(os.path.join(self.output_dir, 'model_history', file_name), allow_pickle='TRUE').item()
                log_book.append(fold_log[dataset][log_type])
                plt.plot(fold_log[dataset][log_type], color=('blue' if dataset == 'train' else 'orange'), alpha=0.2)
            log_df = pd.DataFrame(log_book)
            mean_log = log_df.mean(axis=0).tolist()
            plt.plot(mean_log, color=('blue' if dataset == 'train' else 'orange'), alpha=1.0)
            
        plt.ylabel(axis_label, fontsize=20, labelpad=10)
        plt.xlabel('Training Epoch', fontsize=20, labelpad=10)
        plt.legend(['Training', 'Validation'], loc='lower right')
        file_path = os.path.join(self.output_dir, 'model_history/diagnostics', plot_name)
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
    parser.add_argument("--backbone", type=str, default='convnext',
                        help="Model encoder to use. Defaults to ResNet50.")
    parser.add_argument("--loss-fn", type=str, default='bce',
                        help="Loss function to use.")
    parser.add_argument("--pooling-mode", default='cls', type=str,
                        help="Pooling mode. Can be cls, mean, max, or att. Defaults to cls.")
    parser.add_argument("--pretrained", action='store_true',
                        help="Flag to use pretrained weights.")
    parser.add_argument("--distributed", action='store_true',
                        help="Flag to enable distributed training.")
    parser.add_argument("--amp", action='store_true',
                        help="Flag to enable automated mixed precision training.")
    parser.add_argument("--num-classes", default=2, type=int,
                        help="Number of output classes. Defaults to 2.")
    parser.add_argument("--epochs", default=100, type=int, 
                        help="Number of epochs to train for. Defaults to 10.")
    parser.add_argument("--val-interval", default=1, type=int,
                        help="Number of epochs to wait before running validation. Defaults to 2.")
    parser.add_argument("--batch-size", default=4, type=int, 
                        help="Batch size to use for training. Defaults to 4.")
    parser.add_argument("--effective-batch-size", default=32, type=int,
                        help="Total batch size: batch size x number of devices x number of accumulations steps.")
    parser.add_argument("--seq-length", default=4, type=int,
                        help="Flag to scale the raw model logits using temperature scalar.")
    parser.add_argument("--num-workers", default=8, type=int,
                        help="Number of workers to use for training. Defaults to 8.")
    parser.add_argument("--k-folds", default=0, type=int, 
                        help="Number of folds to use in cross validation. Defaults to 0.8.")
    parser.add_argument("--learning-rate", default=1e-4, type=float, 
                        help="Learning rate to use for training. Defaults to 1e-4.")
    parser.add_argument("--weight-decay", default=0, type=float, 
                        help="Weight decay to use for training. Defaults to 0.")
    parser.add_argument("--dropout", default=0, type=float, 
                        help="Dropout rate to use for training. Defaults to 0.")
    parser.add_argument("--stochastic-depth", default=0, type=float, 
                        help="Stochastic depth rate to use for training. Defaults to 0.")
    parser.add_argument("--epsilon", default=1e-5, type=float, 
                        help="Epsilon value to use for norm layers. Defaults to 1e-5.")
    parser.add_argument("--patience", default=10, type=int, 
                        help="Patience to use for early stopping. Defaults to 10.")
    parser.add_argument("--mod-list", default=MODALITY_LIST, nargs='+', 
                        help="List of modalities to use for training")
    parser.add_argument("--seed", default=1234, type=int, 
                        help="Seed to use for reproducibility")
    parser.add_argument("--suffix", default='MedNet', type=str, 
                        help="File suffix for identification")
    parser.add_argument("--data-dir", default=DATA_DIR, type=str, 
                        help="Path to data directory")
    parser.add_argument("--weights-dir", default=WEIGHTS_DIR, type=str, 
                        help="Path to weights directory")
    parser.add_argument("--results-dir", default=RESULTS_DIR, type=str, 
                        help="Path to results directory")
    return parser.parse_args()

def load_train_objs(
        device_id,
        args: argparse.Namespace,
        phases: list = ['train', 'val']
        ) -> tuple:

    '''
    Load training objects.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        tuple: Training objects consisting of the datasets, dataloaders, the model, and the performance metric.
    '''
    folds = range(args.k_folds)
    data_dict, label_df = DatasetPreprocessor(data_dir=args.data_dir).load_data(modalities=args.mod_list)
    dev, test = GroupStratifiedSplit(split_ratio=0.8).split_dataset(label_df)
    if args.k_folds > 1:
        cv_folds = StratifiedGroupKFold(n_splits=args.k_folds).split(dev, y=dev['label'], groups=dev['patient_id'])
        indices = [(dev.iloc[train_idx], dev.iloc[val_idx]) for train_idx, val_idx in list(cv_folds)]
        split_dict = [convert_to_dict([indices[k][0], indices[k][1]], data_dict=data_dict, split_names=phases) for k in folds]
    elif args.k_folds == 1:
        train, val = GroupStratifiedSplit(split_ratio=0.85).split_dataset(dev)
        split_dict = [convert_to_dict([train, val], data_dict=data_dict, split_names=phases, verbose=True) for k in folds]
    else:
        phases = ['train']
        folds = range(1)
        split_dict = [convert_to_dict([dev], data_dict=data_dict, split_names=phases, verbose=True) for k in folds]

    if args.seq_length > 0:
        seq_split_dict = [convert_to_seqdict(split_dict[k], args.mod_list, phases) for k in folds]
        seq_class_dict = {x: [[max(patient['label']) for patient in seq_split_dict[k][x][0]] for k in folds] for x in phases}
        seq_split_dict = {x: [partition_dataset_classes(
            data=seq_split_dict[k][x][0],
            classes=seq_class_dict[x][k],
            num_partitions=dist.get_world_size(),
            shuffle=True,
            even_divisible=(True if x == 'train' else False)
            )[dist.get_rank()] for k in folds] for x in phases}
        datasets = {x: [CacheSeqDataset(
            data=seq_split_dict[x][k],
            image_keys=args.mod_list,
            transform=data_transforms(
                dataset=x, 
                modalities=args.mod_list,
                device=device_id), 
            num_workers=8,
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
    else:
        class_dict = {x: [[patient['label'] for patient in split_dict[k][x]] for k in folds] for x in phases}
        split_dict = {x: [partition_dataset_classes(
            data=split_dict[k][x],
            classes=class_dict[x][k],
            num_partitions=dist.get_world_size(),
            shuffle=True,
            even_divisible=(True if x == 'train' else False)
            )[dist.get_rank()] for k in folds] for x in phases}
        datasets = {x: [CacheDataset(
            data=split_dict[x][k], 
            transform=data_transforms(
                dataset=x, 
                modalities=args.mod_list, 
                device=device_id), 
            num_workers=8,
            copy_cache=False
            ) for k in folds] for x in phases}
        dataloader = {x: [ThreadDataLoader(
            dataset=datasets[x][k], 
            batch_size=(args.batch_size if x == 'train' else 1), 
            shuffle=(True if x == 'train' else False),   
            num_workers=0,
            drop_last=(True if x == 'train' else False)
            ) for k in folds] for x in phases}

    return dataloader, pos_weight

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
        YeoJohnsond(keys=modalities, lmbda=0.1),
        ConcatItemsd(keys=modalities, name='image'),
        DeleteItemsd(keys=modalities + ['mask']),
        RobustNormalized(keys='image', channel_wise=True),
        PercentileSpatialCropd(
            keys='image',
            roi_center=(0.45, 0.25, 0.3),
            roi_size=(0.6, 0.45, 0.4),
            min_size=(96, 96, 96)),
        SoftClipIntensityd(keys='image', min_value=-3.0, max_value=3.0, channel_wise=True),
        NormalizeIntensityd(
            keys='image',
            subtrahend=(0.2907, 0.3094, 0.3167, 0.3051),
            divisor=(0.7265, 0.7447, 0.7597, 0.7605),
            channel_wise=True)
    ]

    train = [
        EnsureTyped(keys='image', track_meta=False, device=device, dtype=torch.float),
        RandSpatialCropd(keys='image', roi_size=(72, 72, 72), random_size=False),
        RandFlipd(keys='image', spatial_axis=0, prob=0.5),
        RandFlipd(keys='image', spatial_axis=1, prob=0.5),
        RandFlipd(keys='image', spatial_axis=2, prob=0.5),
        RandRotate90d(keys='image', prob=0.5),
        # RandGibbsNoised(keys='image', prob=0.1, alpha=(0.1, 0.9)),
        # RandRicianNoised(keys='image', prob=0.1, mean=0, std=0.5, sample_std=True, channel_wise=True),
        RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
        RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0)
    ]

    test = [
        CenterSpatialCropd(keys='image', roi_size=(96, 96, 96)),
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
    accum_steps = args.effective_batch_size / args.batch_size / num_devices
    batch_size_gpu = args.batch_size * accum_steps
    alpha = 0.0001 if batch_size_gpu == 8 else 0.000141 if batch_size_gpu == 16 else 0.0002 if batch_size_gpu == 32 else 0.000282
    learning_rate = (alpha * np.sqrt(batch_size_gpu) / np.sqrt(128)) * np.sqrt(num_devices)
    num_classes = args.num_classes if args.num_classes > 2 else 1

    cv_dataloader, pos_weight = load_train_objs(device_id, args)
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
        if args.distributed:
            model = model.to(device_id)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id])
        else:
            model = model.to(device_id)
        trainer = Trainer(
            model=model, 
            backbone=args.backbone, 
            loss_fn=args.loss_fn,
            amp=args.amp, 
            dataloaders=dataloader, 
            learning_rate=learning_rate, 
            weight_decay=args.weight_decay, 
            max_seq_length=args.seq_length * args.batch_size, 
            accum_steps=accum_steps, 
            pos_weight=pos_weight, 
            output_dir=args.results_dir)
        trainer.training_loop(
            fold=k,
            suffix=args.suffix,
            max_epochs=args.epochs, 
            batch_size=args.batch_size, 
            val_steps=args.val_interval, 
            patience=args.patience)

    if args.distributed:
        cleanup()
    trainer.visualize_training('loss', epochs=args.epochs, k_folds=args.k_folds, suffix=args.suffix)
    trainer.visualize_training('metric', epochs=args.epochs, k_folds=args.k_folds, suffix=args.suffix)
    print('Script finished')
    

RESULTS_DIR = '/Users/noltinho/thesis/results'
DATA_DIR = '/Users/noltinho/thesis/sensitive_data'
MODALITY_LIST = ['T1W_OOP','T1W_IP','T1W_DYN','T2W_TES','T2W_TEL','DWI_b150','DWI_b400','DWI_b800']
WEIGHTS_DIR = '/Users/noltinho/MedicalNet/pytorch_files/pretrain'

if __name__ == '__main__':
    args = parse_args()
    main(args)


