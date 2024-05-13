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

        out_dict = {x: [] for x in ['probs','labels']}
        for batch in self.dataloaders['test']:
            probs, labels = self.test_step(batch)
            out_dict['probs'].append(probs.cpu())
            out_dict['labels'].append(labels.cpu())
        
        results = self.metrics.compute()
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
        ) -> None:

        log_book = []
        for metric in ['AUROC','AUPRC']:
            store_path = os.path.join(self.output_dir, 'model_diagnostics/roc_pr_curves', self.suffix + '_' + metric + '.png')
            for fold in range(self.num_folds):
                file_name = os.path.join(self.output_dir, 'model_preds', f'preds_fold{fold}_' + self.suffix + '.npy')
                results = np.load(file_name, allow_pickle=True).item()
                log_book.append(results['probs'])
                if metric == 'AUROC':
                    x_axis, y_axis, _ = metrics.roc_curve(results['labels'], results['probs'])
                else:
                    y_axis, x_axis, _ = metrics.precision_recall_curve(results['labels'], results['probs'])
                plt.plot(x_axis, y_axis, color='blue', alpha=0.2)
            
            proba_array = np.mean(np.array(log_book), axis=0)
            if metric == 'AUROC':
                x_axis, y_axis, _ = metrics.roc_curve(results['labels'], proba_array)
                metric_val = metrics.roc_auc_score(results['labels'], proba_array)
                chance_val = 0.500
                chance = [0.0, 1.0]
            else:
                y_axis, x_axis, _ = metrics.precision_recall_curve(results['labels'], proba_array)
                metric_val = metrics.average_precision_score(results['labels'], proba_array)
                chance_val = 0.138
                chance = [chance_val, chance_val]
            plt.plot(x_axis, y_axis, color='blue', label='MedNet: ' + str(metric) + '=' + str(round(metric_val, 3)))
            plt.plot([0.0, 1.0], chance, color='darkgray', linestyle='--', label='Random Chance: ' + str(metric) + '=' + str(round(chance_val, 3)))
            plt.ylabel('True Positive Rate' if metric == 'AUROC' else 'Precision', fontsize=20, labelpad=10)
            plt.xlabel('False Positive Rate' if metric == 'AUROC' else 'Recall', fontsize=20, labelpad=10)
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

    for k in range(num_folds):
        backbone = load_backbone(args)
        model = MedNet(
            backbone, 
            num_classes=num_classes, 
            pretrain=False,
            max_len=12,
            num_layers=4,
            dropout=args.dropout, 
            eps=args.epsilon)
        weights = load_weights(os.path.join(args.weights_dir, f'weights_fold{k}_{args.suffix}.pth'))
        model.load_state_dict(weights)
        model = model.to(device_id)
        if args.distributed:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id])
        tester = Tester(
            model=model, 
            dataloaders=dataloader, 
            num_folds=k,
            amp=args.amp,
            suffix=args.suffix,
            output_dir=args.results_dir)
        tester.test(fold=k)
    tester.visualize_results()
    if args.distributed:
        cleanup()
    print('Script finished')

if __name__ == '__main__':
    args = parse_args()
    main(args)

