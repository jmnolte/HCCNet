from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy, 
    BinaryRecall,
    BinaryPrecision,
    BinaryF1Score,
    BinaryAveragePrecision, 
    BinaryAUROC
)
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    partition_dataset_classes,
    set_track_meta,
    list_data_collate
)
from monai.visualize.class_activation_maps import GradCAMpp
from monai.visualize import OcclusionSensitivity
from monai.utils import set_determinism
from data.splits import GroupStratifiedSplit
from data.datasets import CacheSeqDataset
from data.utils import DatasetPreprocessor, convert_to_dict, convert_to_seqdict, SequenceBatchCollater
from models.mednet import MedNet
from utils.preprocessing import load_backbone, load_data
from utils.transforms import transforms
from utils.config import parse_args
from utils.utils import prep_batch
import argparse
from train import transforms

class Tester:
    
    def __init__(
            self, 
            model: nn.Module, 
            dataloaders: dict, 
            num_folds: int = 5,
            amp: bool = True,
            suffix: str | None = None,
            output_dir: str | None = None
            ) -> None:

        '''
        Args:
            model (nn.Module): Pytorch module object.
            dataloaders (dict): Dataloader objects. Have to be provided as a dictionary, where the the entries are 'train' and 'val'. 
            num_folds (int): Number of cross-validation folds. Defaults to 5.
            amp (bool): Boolean flag to enable automatic mixed precision training. Defaults to true.
            suffix (str | None): Unique string under which model results are stored.
            output_dir (str | None): Directory to store model outputs.
        '''
        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.model = model
        self.dataloaders = dataloaders
        self.num_folds = num_folds
        self.amp = amp
        self.suffix = suffix
        self.output_dir = output_dir

        self.metrics = MetricCollection([
            BinaryAccuracy(), BinaryRecall(), BinaryPrecision(), BinaryF1Score(),
            BinaryAveragePrecision(), BinaryAUROC()
        ]).to(self.gpu_id)

    @torch.no_grad()
    def test_step(
            self,
            batch: dict
        ) -> None:

        '''
        Args:
            batch (dict): Batch obtained from a Pytorch dataloader.
        '''

        self.model.eval()
        inputs, labels, delta, padding_mask = prep_batch(batch, batch_size=1, device=self.gpu_id)

        with autocast(enabled=self.amp):
            logits = self.model(inputs, pad_mask=padding_mask, pos=delta)

        probs = F.sigmoid(logits.squeeze(-1))
        self.metrics.update(probs, labels.int())
        return probs, labels

    def test(
            self,
            fold: int
        ) -> None:

        '''
        Args:
            fold (int): Current cross-validation fold.
        '''

        out_dict = {x: [] for x in ['probs','labels','uid']}
        for batch in self.dataloaders['test']:
            uid = batch['uid']
            uid = uid[0].split('_')[0] + '_' + uid[0].split('_')[1]
            probs, labels = self.test_step(batch)
            out_dict['probs'].append(probs.cpu())
            out_dict['labels'].append(labels.cpu())
            out_dict['uid'].append(uid)
        
        results = self.metrics.compute()
        if self.gpu_id == 0:
            print(results)
        self.metrics.reset()
        probs = torch.cat(out_dict['probs'])
        labels = torch.cat(out_dict['labels'])
        self.save_output(out_dict, 'preds', fold)

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
            fold (int): Current cross-validation fold.
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

    def visualize_results(
            self,
            metric_type: str,
            models: List[str],
            suffixes: List[str],
            modality: str
        ) -> None:

        store_path = os.path.join(self.output_dir, 'model_diagnostics/roc_pr_curves', modality + '_' + metric_type + '.png')
        preds_dict = {x: {y: [] for y in suffixes} for x in models}
        labels_dict = {x: {y: [] for y in suffixes} for x in models}
        for model, version in preds_dict.items():
            for suffix in version.keys():
                for fold in range(self.num_folds):
                    file_name = f'preds_fold{fold}_{modality}_{model}_{suffix}' + '.npy'
                    preds_path = os.path.join(self.output_dir, 'model_preds', file_name)
                    preds = np.load(preds_path, allow_pickle=True).item()
                    preds_dict[model][suffix].append(preds['probs'])
                    labels_dict[model][suffix].append(preds['labels'])

        metric_dict = {}
        for model, version in preds_dict.items():
            metric_dict[model] = {}
            for suffix, pred_probs in version.items():
                avg_pred_probs = np.mean(pred_probs, axis=0)
                labels = np.mean(labels_dict[model][suffix], axis=0)
                if metric_type == 'AUROC':
                    x_axis, y_axis, _ = metrics.roc_curve(labels, avg_pred_probs)
                    mean_metric = metrics.roc_auc_score(labels, avg_pred_probs)
                elif metric_type == 'AUPRC':
                    y_axis, x_axis, _ = metrics.precision_recall_curve(labels, avg_pred_probs)
                    mean_metric = metrics.average_precision_score(labels, avg_pred_probs)
                metric_dict[model][suffix] = (x_axis, y_axis, mean_metric)

        colors = ['blue', 'green', 'red', 'purple']
        linestyles = ['-', '--']

        for i, (model, model_data) in enumerate(metric_dict.items()):
            for j, suffix in enumerate(suffixes):
                x_axis, y_axis, mean_metric = model_data[suffix]
                version = 'pretrained' if suffix.startswith('pre') else ''
                plt.plot(x_axis, y_axis, color=colors[i], linestyle=linestyles[j],
                        label=f'HCCNet-{model.capitalize()} {version} (AUC = {mean_metric:.2f})')

        if metric_type == 'AUROC':
            plt.plot([0, 1], [0, 1], linestyle='--')
        else:
            plt.plot([0, 1], [0.138, 0.138], linestyle='--')
        plt.ylabel('True Positive Rate' if metric_type == 'AUROC' else 'Precision', fontsize=20, labelpad=10)
        plt.xlabel('False Positive Rate' if metric_type == 'AUROC' else 'Recall', fontsize=20, labelpad=10)
        plt.legend(loc=4)
        plt.savefig(store_path, dpi=300, bbox_inches="tight")
        plt.close()

def load_weights(
        weights_path: str
    ) -> dict:

    '''
    Args:
        weights_path (str): Path to weights directory.
    '''

    weights = torch.load(weights_path, map_location='cpu')
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
    num_classes = args.num_classes if args.num_classes > 2 else 1
    num_folds = args.k_folds if args.k_folds > 0 else 1

    dataloader, _ = load_data(args, device_id, phase='test')
    dataloader = {x: dataloader[x][0] for x in ['test']}
    set_track_meta(False)
    modality = args.suffix.split('_')[0]
    models = ['femto','pico','nano','tiny']
    suffixes = ['pre_800steps','1600steps']

    for arch in models:
        for suffix in suffixes:
            for k in range(num_folds):
                backbone = load_backbone(args, arch)
                model = MedNet(
                    backbone, 
                    num_classes=num_classes, 
                    pretrain=False,
                    max_len=12,
                    num_layers=4 if any(arch in x for x in ['femto', 'pico']) else 6,
                    dropout=args.dropout, 
                    eps=args.epsilon)
                file_name = f'weights_fold{k}_{modality}_{arch}_{suffix}.pth'
                weights = load_weights(os.path.join(args.weights_dir, file_name))
                model.load_state_dict(weights)
                model = model.to(device_id)
                if args.distributed:
                    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id])
                tester = Tester(
                    model=model, 
                    dataloaders=dataloader, 
                    num_folds=k,
                    amp=args.amp,
                    suffix=f'{modality}_{arch}_{suffix}',
                    output_dir=args.results_dir)
                tester.test(fold=k)
    tester.visualize_results(metric_type='AUROC', models=models, suffixes=suffixes, modality=modality)
    tester.visualize_results(metric_type='AUPRC', models=models, suffixes=suffixes, modality=modality)
    if args.distributed:
        cleanup()
    print('Script finished')

if __name__ == '__main__':
    args = parse_args()
    main(args)

