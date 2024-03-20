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
from models.convnext3d import convnext3d_atto, convnext3d_femto, convnext3d_pico, convnext3d_nano, convnext3d_tiny
from train import data_transforms
import argparse

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
        Initialize the training class.

        Args:
            model (torch.nn): Model to train.
            version (str): Model version. Can be 'resnet10', 'resnet18', 'resnet34', 'resnet50',
                'resnet101', 'resnet152', or 'resnet200'.
            device (torch.device): Device to use.
            dataloaders (dict): Dataloader objects.
            output_dir (str): Output directory.
        '''
        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.model = model
        self.dataloaders = dataloaders
        self.num_folds = num_folds
        self.amp = amp
        self.suffix = suffix
        self.output_dir = output_dir
        # self.grad_cam = GradCAMpp(self.model, target_layers=)
        # self.occ_sens = OcclusionSensitivity(backbone, n_batch=1)

        self.metrics = MetricCollection([
            BinaryAccuracy(), BinaryRecall(), BinaryPrecision(), BinaryF1Score(),
            BinaryAveragePrecision(), BinaryAUROC()
        ]).to(self.gpu_id)

    def test(
            self,
            fold: int
            ) -> None:

        '''
        Run inference on the test set.
        '''

        results_dict = {'preds': [], 'labels': []}
        self.model.eval()
        with torch.no_grad():
            for step, batch_data in enumerate(self.dataloaders['test']):
                inputs, labels, pos_token, mask = self.prep_batch(batch_data, batch_size=1, device_id=self.gpu_id)

                with autocast(enabled=self.amp):
                    logits = self.model(inputs, pos_token=pos_token, padding_mask=None)

                preds = F.sigmoid(logits.squeeze(-1))
                self.metrics.update(preds, labels)
                results_dict['preds'].append(preds.cpu())
                results_dict['labels'].append(labels.cpu())

        results = self.metrics.compute()
        self.metrics.reset()

        if self.gpu_id == 0:
            for metric in results:
                print(f'[Fold {fold}] {metric}: {results[metric].item()}')
        self.save_output(results_dict, 'preds', fold)
    
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
            return data['image'].to(device_id), data['label'].to(device_id, dtype=torch.int), data['age'].to(device_id, dtype=torch.float), mask.to(device_id)
        else:
            return data['image'], data['label'].to(torch.int), data['age'].to(torch.float), mask

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

    def visualize_results(
            self,
        ) -> None:

        log_book = []
        for metric in ['AUROC','AUPRC']:
            store_path = os.path.join(self.output_dir, 'model_diagnostics/roc_pr_curves', self.suffix + '_' + metric + '.png')
            for fold in range(self.num_folds):
                file_name = os.path.join(self.output_dir, 'model_preds', f'preds_fold{fold}_' + self.suffix + '.npy')
                results = np.load(file_name, allow_pickle=True).item()
                log_book.append(results['preds'])
                if metric == 'AUROC':
                    x_axis, y_axis, _ = metrics.roc_curve(results['labels'], results['preds'])
                else:
                    y_axis, x_axis, _ = metrics.precision_recall_curve(results['labels'], results['preds'])
                plt.plot(x_axis, y_axis, color='blue', alpha=0.2)
            
            proba_array = np.mean(np.array(log_book), axis=0)
            preds_array = [1 if proba >= 0.5 else 0 for proba in proba_array]
            if metric == 'AUROC':
                x_axis, y_axis, _ = metrics.roc_curve(results['labels'], proba_array)
                metric_val = metrics.roc_auc_score(results['labels'], proba_array)
                chance_val = 0.500
                chance = [0.0, 1.0]
            else:
                y_axis, x_axis, _ = metrics.precision_recall_curve(results['labels'], proba_array)
                metric_val = metrics.average_precision_score(results['labels'], proba_array)
                chance_val = 0.128
                chance = [chance_val, chance_val]
            plt.plot(x_axis, y_axis, color='blue', label='MedNet: ' + str(metric) + '=' + str(round(metric_val, 3)))
            plt.plot([0.0, 1.0], chance, color='darkgray', linestyle='--', label='Random Chance: ' + str(metric) + '=' + str(round(chance_val, 3)))
            plt.ylabel('True Positive Rate' if metric == 'AUROC' else 'Precision', fontsize=20, labelpad=10)
            plt.xlabel('False Positive Rate' if metric == 'AUROC' else 'Recall', fontsize=20, labelpad=10)
            plt.legend(loc=4)
            plt.savefig(store_path, dpi=300, bbox_inches="tight")
            plt.close()

        acc = metrics.accuracy_score(results['labels'], preds_array)
        prec = metrics.precision_score(results['labels'], preds_array)
        rec = metrics.recall_score(results['labels'], preds_array)
        fscore = metrics.f1_score(results['labels'], preds_array)
        print(f'Accuracy: {acc}, Precision: {prec}, Recall: {rec}, and F1-Score: {fscore}')

    def load_weights(
            self,
            fold: int,
            weights_dir: str
        ) -> None:

        '''
        Update the model dictionary with the weights from the best epoch.
        '''
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.gpu_id}
        weights_path = os.path.join(weights_dir, f'weights_fold{fold}_' + self.suffix + '.pth')
        weights = torch.load(weights_path, map_location=map_location)
        self.model.load_state_dict(weights)


def parse_args() -> argparse.Namespace:

    '''
    Parse command line arguments.

    Returns:
        argparse.Namespace: Arguments.
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-classes", default=2, type=int,
                        help="Number of output classes. Defaults to 2.")
    parser.add_argument("--pooling-mode", type=str, default='cls',
                        help="Pooling to use. Defaults to cls pooling")
    parser.add_argument("--distributed", action='store_true',
                        help="Flag to enable distributed training.")
    parser.add_argument("--amp", action='store_true',
                        help="Flag to enable automated mixed precision training.")
    parser.add_argument("--k-folds", default=4, type=int, 
                        help="Number of folds to evaluate over.")
    parser.add_argument("--batch-size", default=4, type=int, 
                        help="Batch size to use for training")
    parser.add_argument("--seq-length", default=4, type=int, 
                        help="Maxiaml training length of input sequence")
    parser.add_argument("--train-ratio", default=0.8, type=float, 
                        help="Ratio of training data to use for training")
    parser.add_argument("--epsilon", default=1e-5, type=float, 
                        help="Epsilon to use for normalization layers.")
    parser.add_argument("--mod-list", default=MODALITY_LIST, nargs='+', 
                        help="List of modalities to use for training")
    parser.add_argument("--suffix", type=str,
                        help="Suffix to use for identification")
    parser.add_argument("--seed", default=123, type=int, 
                        help="Seed to use for reproducibility")               
    parser.add_argument("--data-dir", default=DATA_DIR, type=str, 
                        help="Path to data directory")
    parser.add_argument("--results-dir", default=RESULTS_DIR, type=str, 
                        help="Path to results directory")
    parser.add_argument("--weights-dir", default=WEIGHTS_DIR, type=str, 
                        help="Path to pretrained weights")
    return parser.parse_args()

def load_test_objs(
        args: argparse.Namespace,
        device_id: torch.device
        ) -> tuple:

    '''
    Load training objects.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        tuple: Training objects consisting of the datasets, dataloaders, the model, and the performance metric.
    '''    
    data_dict, label_df = DatasetPreprocessor(data_dir=args.data_dir).load_data(modalities=args.mod_list)
    dev, test = GroupStratifiedSplit(split_ratio=0.75).split_dataset(label_df)
    train, val = GroupStratifiedSplit(split_ratio=0.8).split_dataset(dev)
    # test = test[(test['delta'] < -365.25) | (test['delta'].isnull())]
    split_dict = convert_to_dict([val], data_dict=data_dict, split_names=['test'], verbose=True)

    if args.seq_length > 0:
        seq_split_dict = convert_to_seqdict(split_dict, args.mod_list, ['test'])
        seq_class_dict = {x: [max(patient['label']) for patient in seq_split_dict[x][0]] for x in ['test']}
        seq_split_dict = {x: partition_dataset_classes(
            data=seq_split_dict[x][0],
            classes=seq_class_dict[x],
            num_partitions=dist.get_world_size(),
            shuffle=True,
            even_divisible=(True if x == 'train' else False))[dist.get_rank()] for x in ['test']}
        datasets = {x: CacheSeqDataset(
            data=seq_split_dict[x], 
            image_keys=args.mod_list,
            transform=data_transforms(
                dataset=x, 
                modalities=args.mod_list,
                device=device_id), 
            num_workers=8,
            copy_cache=False) for x in ['test']}
        dataloader = {x: ThreadDataLoader(
            dataset=datasets[x], 
            batch_size=(args.batch_size if x == 'train' else 1), 
            shuffle=(True if x == 'train' else False),   
            num_workers=0,
            drop_last=False,
            collate_fn=(SequenceBatchCollater(
                keys=['image','label','age'], 
                seq_length=args.seq_length) if x == 'train' else list_data_collate)) for x in ['test']}
    else:
        class_dict = {x: [patient['label'] for patient in split_dict[x]] for x in ['test']}
        split_dict = {x: partition_dataset_classes(
            data=split_dict[x],
            classes=class_dict[x],
            num_partitions=dist.get_world_size(),
            shuffle=True,
            even_divisible=(True if x == 'train' else False))[dist.get_rank()] for x in ['test']}
        datasets = {x: CacheDataset(
            data=split_dict[x], 
            transform=data_transforms(
                dataset=x, 
                modalities=args.mod_list, 
                device=device_id), 
            num_workers=8,
            copy_cache=False) for x in ['test']}
        dataloader = {x: ThreadDataLoader(
            dataset=datasets[x], 
            batch_size=(args.batch_size if x == 'train' else 1), 
            shuffle=(True if x == 'train' else False),   
            num_workers=0,
            drop_last=(True if x == 'train' else False)) for x in ['test']}

    return dataloader

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
    # Set a seed for reproducibility.
    set_determinism(seed=args.seed)
    # Setup distributed training.
    if args.distributed:
        setup()
    rank = dist.get_rank()
    num_devices = torch.cuda.device_count()
    device_id = rank % num_devices
    num_classes = args.num_classes if args.num_classes > 2 else 1

    dataloader = load_test_objs(args, device_id)
    set_track_meta(False)

    backbone = convnext3d_femto(
        in_chans=len(args.mod_list), 
        use_grn=True, 
        eps=args.epsilon)
    model = MedNet(
        backbone, 
        num_classes=num_classes, 
        pooling_mode=args.pooling_mode, 
        eps=args.epsilon)
    model = model.to(device_id)
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id])
    tester = Tester(
        model=model, 
        dataloaders=dataloader, 
        num_folds=args.k_folds,
        amp=args.amp,
        suffix=args.suffix,
        output_dir=args.results_dir)
    for k in range(args.k_folds):
        tester.load_weights(fold=k, weights_dir=args.weights_dir)
        tester.test(fold=k)
    tester.visualize_results()
    # Cleanup distributed training.
    if args.distributed:
        cleanup()
    print('Script finished')
    

RESULTS_DIR = '/Users/noltinho/thesis/results'
DATA_DIR = '/Users/noltinho/thesis_private/data'
MODALITY_LIST = ['T1W_OOP','T1W_IP','T1W_DYN','T2W_TES','T2W_TEL','DWI_b150','DWI_b400','DWI_b800']
WEIGHTS_DIR = '/Users/noltinho/MedicalNet/pytorch_files/pretrain'

if __name__ == '__main__':
    args = parse_args()
    main(args)

