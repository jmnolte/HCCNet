import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
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
    partition_dataset_classes,
    set_track_meta,
    list_data_collate
)
from monai.visualize.class_activation_maps import GradCAMpp
from monai.visualize import OcclusionSensitivity
from monai.utils import set_determinism
from data.splits import GroupStratifiedSplit
from data.datasets import CacheSeqDataset
from data.transforms import PercentileSpatialCropd, YeoJohnsond, SoftClipIntensityd
from data.utils import DatasetPreprocessor, convert_to_dict, convert_to_seqdict, SequenceBatchCollater
from models.mednet import MedNet
from models.convnext3d import convnext3d_atto, convnext3d_femto, convnext3d_pico, convnext3d_nano, convnext3d_tiny
from utils.utils import prep_batch
import argparse

class Tester:
    
    def __init__(
            self, 
            model: nn.Module, 
            dataloaders: dict, 
            num_folds: int = 5,
            backbone_only: bool = False,
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
        self.knn = KNeighborsClassifier(n_neighbors=num_folds)
        self.backbone_only = backbone_only
        # self.grad_cam = GradCAMpp(self.model, target_layers=)
        # self.occ_sens = OcclusionSensitivity(backbone, n_batch=1)

        # self.metrics = MetricCollection([
        #     BinaryAccuracy(), BinaryRecall(), BinaryPrecision(), BinaryF1Score(),
        #     BinaryAveragePrecision(), BinaryAUROC()
        # ]).to(self.gpu_id)

    @torch.no_grad()
    def test_step_backbone(
            self,
            batch: dict
        ) -> torch.Tensor:

        self.model.eval()
        inputs, labels = batch['image'].to(self.gpu_id), batch['lirads'].to(self.gpu_id)

        with autocast(enabled=self.amp):
            feats = self.model(inputs)
            feats = F.normalize(feats, p=2, dim=-1)
        return feats, labels

    @torch.no_grad()
    def test_step_encoder(
            self,
            batch: dict
        ) -> None:

        '''
        Run inference on the test set.
        '''

        self.model.eval()
        inputs, labels, pt_info, padding_mask = prep_batch(batch, batch_size=1)
        inputs, labels, padding_mask = inputs.to(self.gpu_id), labels.to(self.gpu_id), padding_mask.to(self.gpu_id)
        pt_info = [info.to(self.gpu_id) for info in pt_info]

        with autocast(enabled=self.amp):
            logits = self.model(inputs, pad_mask=padding_mask, pt_info=pt_info)

        preds = F.sigmoid(logits.squeeze(-1))
        preds = torch.where(preds >= 0.5, 1, 0)
        return preds, labels

    def knn_classify(
            self
        ):

        out_dict = {x: [] for x in ['out','labels']}
        for batch in self.dataloaders['train']:
            if self.backbone_only:
                feats, labels = self.test_step_backbone(batch)
                out_dict['out'].append(feats.cpu())
        
        out = torch.cat(out_dict['out'])
        labels = torch.cat(out_dict['labels'])

        self.knn.fit(out, labels)
        preds = self.knn.predict(out)
        self.compute_metrics(labels, preds)

    def test(
            self
        ) -> None:

        if self.backbone_only:
            self.knn_classify()

        out_dict = {x: [] for x in ['preds','labels']}
        for batch in self.dataloaders['test']:
            if self.backbone_only:
                feats, labels = self.test_step_backbone(batch)
                out_dict['preds'].append(feats.cpu())
            else:
                preds, labels = self.test_step_encoder(batch)
                out_dict['preds'].append(preds.cpu())
            out_dict['labels'].append(labels.cpu())
        
        preds = torch.cat(out_dict['preds'])
        labels = torch.cat(out_dict['labels'])

        if self.backbone_only:
            preds = self.knn.predict(preds)
        self.compute_metrics(labels, preds)

    def compute_metrics(
            self,
            labels: torch.Tensor,
            preds: torch.Tensor
        ) -> None:
        
        acc = metrics.accuracy_score(labels, preds)
        prec = metrics.precision_score(labels, preds)
        rec = metrics.recall_score(labels, preds)
        fscore = metrics.f1_score(labels, preds)
        auroc = metrics.roc_auc_score(labels, preds)
        auprc = metrics.average_precision_score(labels, preds)
        print(f'Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1-Score: {fscore}, AUROC: {auroc}, and AUPRC: {auprc}')

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
            min_size=crop_size),
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
            min_size=crop_size),
        SoftClipIntensityd(keys='image', min_value=-3.0, max_value=3.0, channel_wise=True)
    ]

    test = [
        CenterSpatialCropd(keys='image', roi_size=(72, 72, 72)),
        SpatialPadd(keys='image', spatial_size=crop_size, method='symmetric'),
        NormalizeIntensityd(
            keys='image', 
            subtrahend=(0.4467, 0.4416, 0.4409, 0.4212),
            divisor=(0.6522, 0.6429, 0.6301, 0.6041),
            channel_wise=True),
        EnsureTyped(keys='image', track_meta=False, device=device, dtype=torch.float)
    ]
    return Compose(prep + test)

def load_weights(
        weights_path: str
    ) -> dict:

    weights = torch.load(weights_path, map_location='cpu')
    for key in list(weights.keys()):
        if 'backbone.' in key:
            weights[key.replace('backbone.', '')] = weights.pop(key)
        if 'head.' in key:
            weights.pop(key)
    weights['downsample_layers.0.0.weight'] = weights['downsample_layers.0.0.weight'].repeat(1, 4, 1, 1, 1)
    return weights

def parse_args() -> argparse.Namespace:

    '''
    Parse command line arguments.

    Returns:
        argparse.Namespace: Arguments.
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-classes", default=2, type=int,
                        help="Number of output classes. Defaults to 2.")
    parser.add_argument("--distributed", action='store_true',
                        help="Flag to enable distributed training.")
    parser.add_argument("--amp", action='store_true',
                        help="Flag to enable automated mixed precision training.")
    parser.add_argument("--backbone-only", action='store_true',
                        help="Flag to only test the backbone.")
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
    phases = ['train','test'] if args.backbone_only else ['test']
    data_dict, label_df = DatasetPreprocessor(
        data_dir=args.data_dir).load_data(modalities=args.mod_list, keys=['label','age'], file_name='labels.csv', verbose=False)
    lirads_dict, lirads_df = DatasetPreprocessor(
        data_dir=args.data_dir, no_labels=True).load_data(modalities=args.mod_list, keys=['lirads','age'], file_name='lirads.csv')
    dev, test = GroupStratifiedSplit(split_ratio=0.75).split_dataset(label_df)
    lirads_test = lirads_df[lirads_df['patient_id'].isin(test['patient_id'])]
    lirads_dev = lirads_df[-lirads_df['patient_id'].isin(lirads_test['patient_id'])]
    split_dict = convert_to_dict([lirads_dev, lirads_test], data_dict=lirads_dict, split_names=phases, verbose=False)

    if not args.backbone_only:
        seq_split_dict = convert_to_seqdict(split_dict, args.mod_list, phases)
        seq_class_dict = {x: [max(patient['label']) for patient in seq_split_dict[x][0]] for x in phases}
        seq_split_dict = {x: partition_dataset_classes(
            data=seq_split_dict[x][0],
            classes=seq_class_dict[x],
            num_partitions=dist.get_world_size(),
            shuffle=True,
            even_divisible=(True if x == 'train' else False))[dist.get_rank()] for x in phases}
        datasets = {x: CacheSeqDataset(
            data=seq_split_dict[x], 
            image_keys=args.mod_list,
            transform=transforms(
                dataset=x, 
                modalities=args.mod_list,
                device=device_id), 
            num_workers=8,
            copy_cache=False) for x in phases}
        dataloader = {x: ThreadDataLoader(
            dataset=datasets[x], 
            batch_size=(args.batch_size if x == 'train' else 1), 
            shuffle=(True if x == 'train' else False),   
            num_workers=0,
            drop_last=False,
            collate_fn=(SequenceBatchCollater(
                keys=['image','label','age'], 
                seq_length=args.seq_length) if x == 'train' else list_data_collate)) for x in phases}
    else:
        class_dict = {x: [patient['lirads'] for patient in split_dict[x]] for x in phases}
        split_dict = {x: partition_dataset_classes(
            data=split_dict[x],
            classes=class_dict[x],
            num_partitions=dist.get_world_size(),
            shuffle=True,
            even_divisible=(True if x == 'train' else False))[dist.get_rank()] for x in phases}
        datasets = {x: CacheDataset(
            data=split_dict[x], 
            transform=transforms(
                dataset=x, 
                modalities=args.mod_list, 
                device=device_id), 
            num_workers=8,
            copy_cache=False) for x in phases}
        dataloader = {x: ThreadDataLoader(
            dataset=datasets[x], 
            batch_size=(args.batch_size if x == 'train' else 1), 
            shuffle=(True if x == 'train' else False),   
            num_workers=0,
            drop_last=(True if x == 'train' else False)) for x in phases}

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

    model = convnext3d_tiny(
        in_chans=len(args.mod_list), 
        kernel_size=3, 
        drop_path_rate=0.1, 
        use_v2=False,
        eps=1e-6)
    weights = load_weights(os.path.join(args.results_dir, 'model_weights/weights_fold8000_dmri_tiny_v1.pth'))
    model.load_state_dict(weights, strict=False)
    model.head = nn.Identity()
    # model = MedNet(
    #     backbone, 
    #     num_classes=num_classes, 
    #     pooling_mode=args.pooling_mode, 
    #     eps=args.epsilon)
    model = model.to(device_id)
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id])
    for k in range(3, args.k_folds):
        tester = Tester(
            model=model, 
            dataloaders=dataloader, 
            num_folds=k,
            backbone_only=True,
            amp=args.amp,
            suffix=args.suffix,
            output_dir=args.results_dir)
        tester.test()
    # tester.visualize_results()
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

