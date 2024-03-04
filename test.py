import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
import numpy as np
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
from monai.utils import set_determinism
from data.splits import GroupStratifiedSplit
from data.datasets import CacheSeqDataset
from data.utils import DatasetPreprocessor, convert_to_dict, convert_to_seqdict, SequenceBatchCollater
from handlers.slidingwindowinference import SlidingWindowInferer
from models.sequencenet import SeqNet
from models.resnet3d import resnet10, resnet18, resnet34, resnet50
from models.swinvit import swinvit_tiny, swinvit_small, swinvit_base
from train import data_transforms
import argparse

class Tester():
    
    def __init__(
            self, 
            model: nn.Module, 
            backbone: str, 
            amp: bool,
            dataloaders: dict, 
            max_seq_length: int,
            output_dir: str
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
        self.amp = amp
        self.backbone = backbone
        self.dataloaders = dataloaders
        self.output_dir = output_dir

        self.inferer = SlidingWindowInferer(self.model, self.gpu_id, max_length=max_seq_length)
        self.metrics = MetricCollection([
            BinaryAccuracy(), BinaryRecall(), BinaryPrecision(), BinaryF1Score(),
            BinaryAveragePrecision(), BinaryAUROC()
        ]).to(self.gpu_id)

    def test(
            self,
            ) -> None:

        '''
        Run inference on the test set.
        '''

        results_dict = {'preds': [], 'labels': []}
        self.model.eval()
        with torch.no_grad():
            for step, batch_data in enumerate(self.dataloaders['test']):
                inputs, labels, pos_token, mask = self.prep_batch(batch_data, batch_size=1)
                labels = labels.to(self.gpu_id)

                with autocast(enabled=self.amp):
                    logits = self.inferer(inputs, pos_token=pos_token)

                preds = F.sigmoid(logits.squeeze(-1))
                self.metrics.update(preds, labels)
                results_dict['preds'].append(preds)
                results_dict['labels'].append(labels)

            results = self.metrics.compute()
            self.metrics.reset()

        if self.gpu_id == 0:
            for metric in results:
                print(f'{metric}: {results[metric].item()}')
        self.save_output(results_dict, 'preds')
    
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
            output_type: str
            ) -> None:

        '''
        Save the model's output.

        Args:
            output_dict (dict): Output dictionary.
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
            folder_name = self.backbone + '_hist.npy'
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

def visualize_results(
        self,
        backbone: str,
        results_dir: str
    ) -> None:

    results = np.load(os.path.join(results_dir, 'model_preds', backbone + '_preds.npy'), allow_pickle=True).item()
    for metric in ['AUROC','AUPRC']:
        if metric == 'AUROC':
            x_axis, y_axis, _ = metrics.roc_curve(results['labels'],  results['preds'])
            metric_val = metrics.roc_auc_score(results['labels'],  results['preds'])
            chance_val = 0.5
            chance = [0.0, 1.0]
        else:
            y_axis, x_axis, _ = metrics.precision_recall_curve(results['labels'],  results['preds'])
            metric_val = metrics.average_precision_score(results['labels'],  results['preds'])
            chance_val = len(results['labels'][results['labels'] == 1]) / len(results['labels'])
            chance = [chance_val, chance_val]
        store_path = os.path.join(results_dir, 'model_history/diagnostics', backbone + '_' + metric + '.png')
        plt.plot(x_axis, y_axis, label='SeqNet: ' + str(metric) + '=' + str(metric_val))
        plt.plot([0.0, 1.0], chance, linestyle='--', label='Random Chance: ' + str(metric) + '=' + str(chance_val))
        plt.legend(loc=4)
        plt.savefig(store_path, dpi=300, bbox_inches="tight")
        plt.close()

def load_model_weights(
        model: nn.Module,
        backbone: str,
        weights_path: str
        ) -> nn.Module:

    '''
    Update the model dictionary with the weights from the best epoch.

    Returns:
        dict: Updated model dictionary.
    '''
    model_dict = model.state_dict()
    weights_dict = torch.load(os.path.join(weights_path, backbone + '_weights.pth'))
    weights_dict = {k.replace('module.', ''): v for k, v in weights_dict.items()}
    model_dict.update(weights_dict)
    model.load_state_dict(model_dict)
    print('Model weights have bee sucessfully updated.')
    return model

def parse_args() -> argparse.Namespace:

    '''
    Parse command line arguments.

    Returns:
        argparse.Namespace: Arguments.
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-classes", default=2, type=int,
                        help="Number of output classes. Defaults to 2.")
    parser.add_argument("--backbone", type=str, 
                        help="Model encoder to use. Defaults to ResNet50.")
    parser.add_argument("--pretrained", action='store_true',
                        help="Flag to use pretrained weights")
    parser.add_argument("--distributed", action='store_true',
                        help="Flag to enable distributed training.")
    parser.add_argument("--amp", action='store_true',
                        help="Flag to enable automated mixed precision training.")
    parser.add_argument("--batch-size", default=4, type=int, 
                        help="Batch size to use for training")
    parser.add_argument("--seq-length", default=4, type=int, 
                        help="Maxiaml training length of input sequence")
    parser.add_argument("--train-ratio", default=0.85, type=float, 
                        help="Ratio of training data to use for training")
    parser.add_argument("--mod-list", default=MODALITY_LIST, nargs='+', 
                        help="List of modalities to use for training")
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
        device_id,
        args: argparse.Namespace
        ) -> tuple:

    '''
    Load training objects.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        tuple: Training objects consisting of the datasets, dataloaders, the model, and the performance metric.
    '''
    data_dict, label_df = DatasetPreprocessor(data_dir=args.data_dir).load_data(modalities=args.mod_list)
    train, val_test = GroupStratifiedSplit(split_ratio=args.train_ratio).split_dataset(label_df)
    test, val = GroupStratifiedSplit(split_ratio=0.6).split_dataset(val_test)
    split_dict = convert_to_dict(train, val, test, data_dict)

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

    dataloader = load_test_objs(device_id, args)
    set_track_meta(False)
    num_classes = args.num_classes if args.num_classes > 2 else 1
    if args.backbone == 'resnet10':
        backbone = resnet10(pretrained=args.pretrained, n_input_channels=len(args.mod_list), num_classes=num_classes, shortcut_type='B')
    elif args.backbone == 'resnet50':
        backbone = resnet50(pretrained=args.pretrained, n_input_channels=len(args.mod_list), num_classes=num_classes, shortcut_type='B')
    elif args.backbone == 'swinvit_t':
        backbone = swinvit_tiny(pretrained=False, in_chans=len(args.mod_list), num_classes=num_classes)
    elif args.backbone == 'swinvit_s':
        backbone = swinvit_small(pretrained=False, in_chans=len(args.mod_list), num_classes=num_classes)
    elif args.backbone == 'swinvit_b':
        backbone = swinvit_base(pretrained=True, in_chans=len(args.mod_list), num_classes=num_classes)
    model = SeqNet(backbone, num_classes=num_classes)
    model = load_model_weights(model, backbone=args.backbone, weights_path=args.weights_dir)
    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) if args.backbone.startswith('resnet') else model
        model = model.to(device_id)
        model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[device_id])
    else:
        model = model.to(device_id)
    # Train the model using the training data and validate the model on the validation data following each epoch.
    tester = Tester(
        model=model, 
        backbone=args.backbone, 
        amp=args.amp, 
        dataloaders=dataloader, 
        max_seq_length=args.seq_length * args.batch_size, 
        output_dir=args.results_dir)
    tester.test()
    visualize_results(args.backbone, args.results_dir)
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

