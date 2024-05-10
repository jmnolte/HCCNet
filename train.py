from __future__ import annotations

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAveragePrecision, BinaryAUROC, MulticlassAveragePrecision, MulticlassAUROC
import argparse
import os
import time
import copy
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monai.data import set_track_meta
from monai.utils import set_determinism
from models.mednet import MedNet
from utils.transforms import transforms
from utils.preprocessing import load_backbone, load_data, load_objs
from utils.config import parse_args
from utils.utils import scale_learning_rate, prep_batch


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
            suffix: str | None = None,
            output_dir: str | None = None
        ) -> None:

        '''
        Args:
            model (nn.Module): Pytorch module object.
            loss_fn (List[nn.Module]): List containing the train and validation loss functions. Has to be of length 2.
            dataloaders (dict): Dataloader objects. Have to be provided as a dictionary, where the the entries are 'train' and 'val'. 
            optimizer (optim): Pytorch optimizer.
            scheduler (List[np.array]): List of learing rate and weight decay schedules. Has to be of length 2.
            num_folds (int): Number of cross-validation folds. Defaults to 1.
            num_steps (int): Number of training steps. Defaults to 1000.
            amp (bool): Boolean flag to enable automatic mixed precision training. Defaults to true.
            suffix (str | None): Unique string under which model results are stored.
            output_dir (str | None): Directory to store model outputs.
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
        self.lr_schedule, self.wd_schedule = scheduler[0], scheduler[1]
        self.params = self.model.parameters()
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
        Args:
            output_dict (dict): Dictionary containing the model outputs.
            output_type (str): Type of output. Can be 'weights', 'history', or 'preds'.
            fold (int): Current cross-valdiation fold.
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

    def training_step(
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
        inputs, labels, delta, padding_mask = prep_batch(batch, batch_size=batch_size, device=self.gpu_id)

        with autocast(enabled=self.amp):
            logits = self.model(inputs, pad_mask=padding_mask, pos=delta)
            loss = self.train_loss(logits.squeeze(-1), labels.float())
            loss /= accum_steps

        self.scaler.scale(loss).backward()
        preds = F.sigmoid(logits.squeeze(-1))
        self.train_metrics.update(preds, labels.int())
        return loss.item()

    def accumulation_step(
            self,
            step: int,
            clip_grad: bool = True
        ) -> None:

        '''
        Args:
            step (int): Current training step.
            clip_grad (bool): Boolean flag to clip parameter gradients.
        '''

        for i, param_group in enumerate(self.optim.param_groups):
            param_group['lr'] = self.lr_schedule[step]
            if i == 0:  # only the first group is regularized
                param_group['weight_decay'] = self.wd_schedule[step]

        if clip_grad:
            self.scaler.unscale_(self.optim)
            nn.utils.clip_grad_norm_(self.params, max_norm=3.0, norm_type=2)

        self.scaler.step(self.optim)
        self.scaler.update()
        self.optim.zero_grad(set_to_none=True)

    @torch.no_grad()
    def validation_step(
            self, 
            batch: dict,
            batch_size: int = 1
        ) -> float:

        '''
        Args:
            batch (dict): Batch obtained from a Pytorch dataloader.
            batch_size (int): Number of unique observations in the batch.
        '''

        self.model.eval()
        inputs, labels, delta, padding_mask = prep_batch(batch, batch_size=batch_size, device=self.gpu_id)

        with autocast(enabled=self.amp):
            logits = self.model(inputs, pad_mask=padding_mask, pos=delta)
            loss = self.val_loss(logits.squeeze(-1), labels.float())

        preds = F.sigmoid(logits.squeeze(-1))
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
        Args:
            fold (int): Current cross-validation fold.
            batch_size (int): Number of unique observations in the batch.
            accum_steps (int): Number of steps to accumulate before updating parameter gradients.
            val_steps (int): Number of steps to wait before evaluating the model on the validation set.
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
                    self.accumulation_step(update_step, clip_grad=True)
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
                        if self.gpu_id == 0:
                            print(f'[GPU {self.gpu_id}] New best Validation Loss: {best_loss:.4f} and Metric: {val_metric:.4f}. Saving model weights...')

                if (step + 1) == self.num_steps * accum_steps / 2:
                    best_weights = copy.deepcopy(self.model.module.state_dict())

                if (step + 1) == self.num_steps * accum_steps:
                    break

        if self.gpu_id == 0:
            time_elapsed = time.time() - start_time
            print(f'[Fold {fold}] Training complete in {time_elapsed // 60:.0f}min {time_elapsed % 60:.0f}sec')
            print(f'[Fold {fold}] Loss: {best_loss:.4f}, AUPRC: {best_auprc:.4f}, and {best_auroc:.4f} of best model configuration.')
            self.save_output(self.results_dict, 'history', fold)
            self.save_output(best_weights, 'weights', fold)
        dist.barrier()
    
    def visualize_training(
            self, 
            phases: List[str],
            log_type: str
        ) -> None:

        '''
        Args:
            phase (List[str]): List of strings. Should be 'train' and 'val'.
            log_type (str): String specifying the metric that should be visualized.
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
        plt.xlabel('Training Steps', fontsize=20, labelpad=10)
        plt.legend(loc='lower right')
        file_path = os.path.join(self.output_dir, 'model_diagnostics/learning_curves', plot_name)
        file_path_root, _ = os.path.split(file_path)
        if os.path.exists(file_path):
            os.remove(file_path)
        elif not os.path.exists(file_path_root):
            os.makedirs(file_path_root)
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()

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

    # Seed to enable deterministic training
    set_determinism(seed=args.seed)
    if args.distributed:
        setup()
    rank = dist.get_rank()
    num_devices = torch.cuda.device_count()
    device_id = rank % num_devices
    # Accumulation steps = effective batch size / batch size per GPU / number of GPUs
    accum_steps = args.effective_batch_size // args.batch_size // num_devices
    learning_rate = scale_learning_rate(args.effective_batch_size)
    num_classes = args.num_classes if args.num_classes > 2 else 1
    num_folds = args.k_folds if args.k_folds > 0 else 1
    seeds = np.random.randint(1000, 10000, args.num_seeds)

    cv_dataloader, pos_weight = load_data(args, device_id)
    # We disable tracking of metadata for optimized performance
    set_track_meta(False)

    for i, seed in enumerate(seeds):
        set_determinism(seed=seed)
        for k in range(num_folds):
            dataloader = {x: cv_dataloader[x][k] for x in ['train','val']}
            backbone = load_backbone(args)
            model = MedNet(
                backbone, 
                num_classes=num_classes, 
                classification=True,
                num_layers=4,
                max_len=12,
                dropout=args.dropout, 
                eps=args.epsilon)
            if args.pretrained:
                weights_path = os.path.join(args.results_dir, 'model_weights/weights_fold1000_dmri_te_tiny_fixed.pth')
                weights = torch.load(weights_path, map_location='cpu')
                model.load_state_dict(weights, strict=False)
            model = model.to(device_id)
            # for name, param in model.named_parameters():
            #     if not name.startswith(('pooler','head')):
            #         param.requires_grad = False
            if args.distributed:
                model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id])
            loss_fn, optimizer, scheduler = load_objs(
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
                num_folds=int(num_folds * args.num_seeds),
                num_steps=args.num_steps,
                amp=args.amp,
                suffix=args.suffix,
                output_dir=args.results_dir)
            trainer.train(
                fold=int(i * num_folds + k),
                batch_size=args.batch_size, 
                accum_steps=accum_steps,
                val_steps=args.log_every)

    if args.distributed:
        cleanup()
    trainer.visualize_training(['train','val'], 'loss')
    trainer.visualize_training(['train','val'], 'auprc')
    trainer.visualize_training(['train','val'], 'auroc')
    print('Script finished')

if __name__ == '__main__':
    args = parse_args()
    main(args)