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

        mean_x_axis = np.linspace(0, 1, 100)
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
                metric_dict[model][suffix] = {}
                results_dict = {x: [] for x in ['acc','prec','rec','f1','auc','pr']}
                for fold in range(10):
                    probs = pred_probs[fold]
                    labels = labels_dict[model][suffix][fold]
                    if metric_type == 'AUROC':
                        x_axis, y_axis, _ = metrics.roc_curve(labels, probs)
                    elif metric_type == 'AUPRC':
                        y_axis, x_axis, _ = metrics.precision_recall_curve(labels, probs)
                    y_axes = np.interp(mean_x_axis, x_axis, y_axis)
                    y_axes[0] = 0.0

                    f1_scores = []
                    for thres in np.arange(0.0, 1.0, 0.001):
                        score = metrics.f1_score(labels, [1 if x >= thres else 0 for x in probs])
                        f1_scores.append((thres, score))
                    best_thres, _ = max(f1_scores, key=lambda x: x[1])

                    results_dict['acc'].append(metrics.accuracy_score(labels, np.where(probs >= best_thres, 1, 0)))
                    results_dict['prec'].append(metrics.precision_score(labels, np.where(probs >= best_thres, 1, 0)))
                    results_dict['rec'].append(metrics.recall_score(labels, np.where(probs >= best_thres, 1, 0)))
                    results_dict['f1'].append(metrics.f1_score(labels, np.where(probs >= best_thres, 1, 0)))
                    results_dict['auc'].append(metrics.roc_auc_score(labels, probs))
                    results_dict['pr'].append(metrics.average_precision_score(labels, probs))
                    metric_dict[model][suffix][fold] = (y_axes, results_dict['auc'] if metric_type == 'AUROC' else results_dict['pr'], labels, probs)
                mean_metrics = [np.mean(results_dict[x]) for x in ['acc','prec','rec','f1','auc','pr']]
                print(f'{model} {suffix} Accuracy: {mean_metrics[0]:.3f} STD: {np.std(results_dict["acc"]):.3f}')
                print(f'Precision: {mean_metrics[1]:.3f} STD: {np.std(results_dict["prec"]):.3f}')
                print(f'Recall: {mean_metrics[2]:.3f} STD: {np.std(results_dict["rec"]):.3f}')
                print(f'F1: {mean_metrics[3]:.3f} STD: {np.std(results_dict["f1"]):.3f}')
                print(f'AUC: {mean_metrics[4]:.3f} STD: {np.std(results_dict["auc"]):.3f}')
                print(f'AUPRC: {mean_metrics[5]:.3f} STD: {np.std(results_dict["pr"]):.3f}')

        colors = ['blue', 'green', 'red', 'purple']
        linestyles = ['-', '--']

        axis_dict = {x: {y: [] for y in suffixes} for x in models}
        results_dict = {x: {y: [] for y in suffixes} for x in models}
        target_dict = {x: {y: [] for y in suffixes} for x in models}
        probs_dict = {x: {y: [] for y in suffixes} for x in models}
        for i, (model, model_data) in enumerate(metric_dict.items()):
            for j, suffix in enumerate(suffixes):
                for fold in range(self.num_folds):
                    y_axes, fold_metric, labels, probs = model_data[suffix][fold]
                    axis_dict[model][suffix].append(y_axes)
                    results_dict[model][suffix].append(fold_metric)
                    target_dict[model][suffix].append(labels)
                    probs_dict[model][suffix].append(probs)
                mean_y_axis = np.mean(axis_dict[model][suffix], axis=0)
                mean_y_axis[-1] = 1.0
                if metric_type == 'AUPRC':
                    mean_y_axis, mean_x_axis, _ = metrics.precision_recall_curve(np.concatenate(target_dict[model][suffix]), np.concatenate(probs_dict[model][suffix]))
                mean_metric = metrics.auc(mean_x_axis, mean_y_axis) if metric_type == 'AUROC' else np.mean(results_dict[model][suffix])
                std_metric = np.std(results_dict[model][suffix])
                version = 'pretrained' if suffix.startswith('pre') else ''
                plt.plot(mean_x_axis, mean_y_axis, color=colors[i], linestyle=linestyles[j], linewidth=1,
                        label=f'HCCNet-{model[0].capitalize()} {version} (AUC = {mean_metric:.3f} $\pm$ {std_metric:.3f})')

        if metric_type == 'AUROC':
            plt.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=1)
        else:
            plt.plot([0, 1], [0.138, 0.138], linestyle='--', color='black', linewidth=1)
        plt.ylabel('True Positive Rate' if metric_type == 'AUROC' else 'Precision', fontsize=20, labelpad=10)
        plt.xlabel('False Positive Rate' if metric_type == 'AUROC' else 'Recall', fontsize=20, labelpad=10)
        plt.legend(fontsize=8, loc=4)
        plt.show()
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
    modality = args.suffix.split('_')[0] if args.suffix.split('_')[0] != 't1iop' else 't1iop_t2'
    models = ['femto','pico','nano','tiny']
    suffixes = ['pre_400steps','400steps']

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

