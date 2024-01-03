from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
import torchmetrics
import argparse
import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ResizeWithPadOrCropd,
    Spacingd,
    CropForegroundd,
    ConcatItemsd,
    RandScaleIntensityd,
    ScaleIntensityd,
    NormalizeIntensityd,
    RandSpatialCropd,
    ResampleToMatchd,
    RandStdShiftIntensityd,
    RandFlipd,
    EnsureTyped,
    DeleteItemsd,
    ScaleIntensityRangePercentilesd,
    CenterSpatialCropd,
    Resized,
    Lambdad,
    KeepLargestConnectedComponentd,
    RandGridPatchd,
    GridPatchd,
    ForegroundMaskd,
    MaskIntensityd
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
from scipy.ndimage import binary_closing

from data.transforms import DivisiblePercentileCropd, ResizeToMatchd
from data.splits import GroupStratifiedSplit
from data.datasets import CacheSeqDataset
from data.utils import DatasetPreprocessor, convert_to_dict, convert_to_seqdict, collate_sequence_batch
from models.milnet import MILNet
from models.sequencenet import SequenceNet
from losses.focalloss import FocalLoss
from losses.recallloss import RecallLoss
from losses.ccloss import ContrastiveCenterBCELoss

class Trainer:
    
    def __init__(
            self, 
            model: nn.Module, 
            backbone: str, 
            amp: bool,
            dataloaders: dict, 
            learning_rate: float, 
            weight_decay: float,
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
        self.scaler = GradScaler(enabled=amp)
        self.val_scaler = GradScaler(enabled=amp)
        self.backbone = backbone
        self.dataloaders = dataloaders
        self.output_dir = output_dir

        model_params = self.model.parameters()
        self.optimizer = optim.AdamW(model_params, lr=learning_rate, weight_decay=weight_decay)

        self.bag_loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight])).to(self.gpu_id)
        self.inst_loss = nn.BCEWithLogitsLoss().to(self.gpu_id)

    def save_output(
            self, 
            output_dict: dict, 
            output_type: str
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

    def train(
            self, 
            metric: torchmetrics, 
            labels_dict: dict,
            epoch: int, 
            batch_size: int,
            accum_steps: int
            ) -> tuple:

        '''
        Train the model.

        Args:
            metric (torchmetrics): Metric to assess model performance while training.
            epoch (int): Current epoch.
            accum_steps (int): Number of accumulation steps to use before updating the model weights.

        Returns:
            tuple: Tuple containing the loss and F1-score.
        '''
        running_loss = 0.0
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        if self.gpu_id == 0:
            print('-' * 10)
            print(f'Epoch {epoch}')
            print('-' * 10)

        for step, batch_data in enumerate(self.dataloaders['train']):
            # inputs, labels, encodings = batch_data['image'].to(self.gpu_id), batch_data['label'].to(self.gpu_id), batch_data['age'].type(torch.float32).to(self.gpu_id)
            inputs, labels, identifiers = batch_data['image'].to(self.gpu_id), batch_data['label'].to(self.gpu_id), batch_data['uid']

            alpha = 0.8 if (0.95 - epoch * 0.01) < 0.8 else 0.95 - epoch * 0.01
            beta = 0.8 if (0.99 - (epoch - 10) * 0.01) < 0.8 else 0.99 if epoch < 10 else 0.99 - (epoch - 10) * 0.01
            labels = labels.unsqueeze(1).float()
            sh = inputs.shape

            if epoch == 0:
                soft_labels = labels.repeat(1, sh[1])
                labels_dict['id'].extend(identifiers)
                labels_dict['soft_label'].append(soft_labels)
            else:
                indices = [labels_dict['id'].index(element) for element in identifiers]
                soft_labels = labels_dict['soft_label'][indices]

            # input_mask = (labels == 99).bool()
            # encodings = (encodings - 64.214) / 11.826

            with autocast(enabled=self.amp):
                # logits = self.model(
                #     x=inputs, 
                #     x_mask=input_mask,
                #     encodings=encodings,
                #     batch_size=batch_size)
                inst_logits, bag_logits, upd_soft_labels = self.model(inputs, soft_labels.to(inputs.device), alpha=alpha, beta=beta)

                loss = 0.1 * self.bag_loss(bag_logits, labels) + self.inst_loss(inst_logits, upd_soft_labels)
                loss = loss / accum_steps

            if epoch > 0:
                labels_dict['soft_label'][indices] = upd_soft_labels.reshape(sh[0], -1)
            
            self.scaler.scale(loss).backward()
            running_loss += loss.item()
            batch_f1score = metric(bag_logits, labels)
    
            if ((step + 1) % accum_steps == 0) or (step + 1 == len(self.dataloaders['train'])):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                # self.scheduler.step()

                if self.gpu_id == 0:
                    print(f"{step + 1}/{len(self.dataloaders['train'])}, Batch Loss: {loss.item() * accum_steps:.4f}")

        if epoch == 0:
            labels_dict['soft_label'] = torch.cat(labels_dict['soft_label'], dim=0)

        epoch_loss = running_loss / (len(self.dataloaders['train']) // accum_steps)
        epoch_f1score = metric.compute()
        metric.reset()
        print(f"[GPU {self.gpu_id}] Epoch {epoch}, Training Loss: {epoch_loss:.4f}, and F1-Score: {epoch_f1score:.4f}")
        return epoch_loss, epoch_f1score, labels_dict

    def evaluate(
            self, 
            metric: torchmetrics, 
            labels_dict: dict,
            epoch: int,
            ) -> tuple:

        '''
        Validate the model.

        Args:
            metric (torchmetrics): Metric to assess model performance while validating.
            epoch (int): Current epoch.
            num_patches (int): Number of patches to split the input into.
        
        Returns:
            tuple: Tuple containing the loss and F1-score.
        '''
        logits_list = []
        labels_list = []
        running_loss = 0.0
        self.model.eval()

        with torch.no_grad():
            for step, batch_data in enumerate(self.dataloaders['val']):
                # inputs, labels, encodings = batch_data['image'].to(self.gpu_id), batch_data['label'].to(self.gpu_id), batch_data['age'].type(torch.float32)
                inputs, labels, identifiers = batch_data['image'].to(self.gpu_id), batch_data['label'].to(self.gpu_id), batch_data['uid']
                # encodings = (encodings - 64.214) / 11.826

                beta = 0.8 if (0.99 - (epoch - 10) * 0.01) < 0.8 else 0.99 if epoch < 10 else 0.99 - (epoch - 10) * 0.01
                labels = labels.unsqueeze(1).float()
                sh = inputs.shape

                if epoch == 0:
                    soft_labels = labels.repeat(1, sh[1])
                    labels_dict['id'].extend(identifiers)
                    labels_dict['soft_label'].append(soft_labels)
                else:
                    indices = [labels_dict['id'].index(element) for element in identifiers]
                    soft_labels = labels_dict['soft_label'][indices]

                with autocast(enabled=self.amp):
                    # logits = self.model(
                    #     x=inputs,
                    #     encodings=encodings,
                    #     batch_size=1)
                    inst_logits, bag_logits, upd_soft_labels = self.model(inputs, soft_labels.to(inputs.device), no_update=True, beta=beta)

                    loss = 0.1 * self.bag_loss(bag_logits, labels) + self.inst_loss(inst_logits, upd_soft_labels)

                if epoch > 0:
                    labels_dict['soft_label'][indices] = upd_soft_labels.reshape(sh[0], -1)

                running_loss += loss.item()
                metric.update(bag_logits, labels)

        if epoch == 0:
            labels_dict['soft_label'] = torch.cat(labels_dict['soft_label'], dim=0)

        epoch_loss = running_loss / len(self.dataloaders['val'])
        epoch_f1score = metric.compute()
        metric.reset()
        print(f"[GPU {self.gpu_id}] Epoch {epoch}, Validation Loss: {epoch_loss:.4f}, and F1-Score: {epoch_f1score:.4f}")

        return epoch_loss, epoch_f1score, labels_dict
    
    def training_loop(
            self, 
            metric: torchmetrics, 
            min_epochs: int, 
            val_every: int,
            batch_size: int,
            accum_steps: int, 
            patience: int
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
        max_epochs = min_epochs * 100
        best_loss = 10.0
        f1score = 0.0
        counter = 0
        stop_criterion = torch.zeros(1).to(self.gpu_id)
        history = {'train_loss': [], 'train_f1score': [], 'val_loss': [], 'val_f1score': []}
        soft_labels = {x: {'id': [], 'soft_label': []} for x in ['train','val']}

        for epoch in range(max_epochs):
            start_time = time.time()
            train_loss, train_f1score, soft_labels['train'] = self.train(metric, soft_labels['train'], epoch, batch_size, accum_steps)
            history['train_f1score'].append(train_f1score.cpu().item())
            history['train_loss'].append(train_loss)

            if (epoch + 1) % val_every == 0:
                val_loss, val_f1score, soft_labels['val'] = self.evaluate(metric, soft_labels['val'], epoch)
                history['val_loss'].append(val_loss)
                history['val_f1score'].append(val_f1score.cpu().item())

                if self.gpu_id == 0:
                    if val_f1score > f1score:
                        best_loss = val_loss
                        f1score = val_f1score
                        counter = 0
                        print(f'[GPU {self.gpu_id}] New best Validation Loss: {best_loss:.4f}. Saving model weights...')
                        best_weights = copy.deepcopy(self.model.state_dict())
                    else:
                        counter += 1
                        if counter >= patience and epoch >= min_epochs:
                            stop_criterion += 1
                
                dist.all_reduce(stop_criterion, op=dist.ReduceOp.SUM)
                if stop_criterion == 1:
                    break


            train_time = time.time() - start_time
            if self.gpu_id == 0:
                print(f'Epoch {epoch} complete in {train_time // 60:.0f}min {train_time % 60:.0f}sec')

        if self.gpu_id == 0:
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Loss {:.4f} and F1-Score {:.4f} of best model configuration:'.format(best_loss, f1score))
            self.save_output(best_weights, 'weights')
            self.save_output(history, 'history')


    # def extract_activations(self):

    #     activation = {}
    #     def get_activation(name):
    #         def hook(model, input, output):
    #             activation[name] = output.detach()
    #         return hook

    #     self.model.activations.register_forward_hook(get_activation('fc'))

    #     labels_list, activations_list = [], []

    #     for phase in ['train', 'val', 'test']:
    #         for batch_data in self.dataloaders[phase]:
    #             name = batch_data["image_meta_dict"]["filename_or_obj"]
    #             inputs = batch_data["image"].to(device)
    #             labels = batch_data["label"].to(device)
                
    #             _ = self.model(inputs)
                
    #             img_path.extend(name)
    #             labels_list.append(labels.detach().cpu().numpy())
    #             activations_list.append(activation['fc'].cpu().numpy())

    #     features = np.concatenate(activations_list)
    #     labels = np.concatenate(labels_list)

    #     id_df = pd.DataFrame(img_path, columns=['id'])
    #     labels_df = pd.DataFrame(labels, columns=['HCC'])
    #     feats_df = pd.DataFrame(features, 
    #                             columns = ["var%d" % (i + 1) 
    #                             for i in range(features.shape[1])])

    #     deep_feats = pd.concat([id_df.reset_index(drop=True), labels_df, feats_df], axis=1)

    #     # write dataframe to csv
    #     deep_feats.to_csv(os.path.join('/home/x3007104/thesis/results', 'DeepFeatures.csv'), index=False, header=True, sep=',')


    def visualize_training(
            self, 
            metric: str
            ) -> None:

        '''
        Visualize the training and validation history.

        Args:
            metric (str): String specifying the metric to be visualized. Can be 'loss' or 'f1score'.
        '''
        try: 
            assert any(metric == metric_item for metric_item in ['loss','f1score'])
        except AssertionError:
            print('Invalid input. Please choose from: loss or f1score.')
            exit(1)

        if metric == 'loss':
            metric_label = 'Loss'
        elif metric == 'f1score':
            metric_label = 'F1-Score'

        file_name = self.backbone + '_hist.npy'
        plot_name = self.backbone + '_' + metric + '.png'
        history = np.load(os.path.join(self.output_dir, 'model_history', file_name), allow_pickle='TRUE').item()
        plt.plot(history['train_' + metric], color='dimgray',)
        plt.plot(history['val_' + metric], color='darkgray',)
        plt.ylabel(metric_label, fontsize=20, labelpad=10)
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
    parser.add_argument("--mil-mode", default='att', type=str,
                        help="MIL pooling mode. Can be mean, max, att, att_trans, and att_trans_pyramid. Defaults to att.")
    parser.add_argument("--backbone", type=str, 
                        help="Model encoder to use. Defaults to ResNet50.")
    parser.add_argument("--pretrained", action='store_true',
                        help="Flag to use pretrained weights.")
    parser.add_argument("--temp-scaling", action='store_true',
                        help="Flag to scale the raw model logits using temperature scalar.")
    parser.add_argument("--distributed", action='store_true',
                        help="Flag to enable distributed training.")
    parser.add_argument("--amp", action='store_true',
                        help="Flag to enable automated mixed precision training.")
    parser.add_argument("--num-classes", default=2, type=int,
                        help="Number of output classes. Defaults to 2.")
    parser.add_argument("--min-epochs", default=10, type=int, 
                        help="Minimum number of epochs to train for. Defaults to 10.")
    parser.add_argument("--val-interval", default=1, type=int,
                        help="Number of epochs to wait before running validation. Defaults to 2.")
    parser.add_argument("--batch-size", default=4, type=int, 
                        help="Batch size to use for training. Defaults to 4.")
    parser.add_argument("--total-batch-size", default=32, type=int,
                        help="Total batch size: batch size x number of devices x number of accumulations steps.")
    parser.add_argument("--num-workers", default=8, type=int,
                        help="Number of workers to use for training. Defaults to 4.")
    parser.add_argument("--train-ratio", default=0.8, type=float, 
                        help="Ratio of data to use for training. Defaults to 0.8.")
    parser.add_argument("--learning-rate", default=1e-4, type=float, 
                        help="Learning rate to use for training. Defaults to 1e-4.")
    parser.add_argument("--weight-decay", default=0, type=float, 
                        help="Weight decay to use for training. Defaults to 0.")
    parser.add_argument("--patience", default=5, type=int, 
                        help="Patience to use for early stopping. Defaults to 5.")
    parser.add_argument("--mod-list", default=MODALITY_LIST, nargs='+', 
                        help="List of modalities to use for training")
    parser.add_argument("--seed", default=1234, type=int, 
                        help="Seed to use for reproducibility")
    parser.add_argument("--data-dir", default=DATA_DIR, type=str, 
                        help="Path to data directory")
    parser.add_argument("--weights-dir", default=WEIGHTS_DIR, type=str, 
                        help="Path to weights directory")
    parser.add_argument("--results-dir", default=RESULTS_DIR, type=str, 
                        help="Path to results directory")
    return parser.parse_args()

def load_train_objs(
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
    val, test = GroupStratifiedSplit(split_ratio=0.5).split_dataset(val_test)
    split_dict = convert_to_dict(train, val, test, data_dict)
    label_dict = {x: [patient['label'] for patient in split_dict[x]] for x in ['train', 'val']}

    metric = torchmetrics.F1Score(task=('multiclass' if args.num_classes > 2 else 'binary'), num_classes=2, ignore_index=99)
    values, counts = np.unique(label_dict['train'], return_counts=True)
    # weights = len(label_dict['train']) / (args.num_classes * torch.Tensor([class_count for class_count in counts]))
    pos_weight = counts[0] / counts[1]
    if args.distributed:
        split_dict = {x: partition_dataset_classes(
            data=split_dict[x],
            classes=label_dict[x],
            num_partitions=dist.get_world_size(),
            shuffle=True,
            even_divisible=(True if x == 'train' else False))[dist.get_rank()] for x in ['train','val']}

    # seq_split_dict = convert_to_seqdict(split_dict, ['train', 'val'])
    # seq_class_dict = {x: [max(patient['label']) for patient in seq_split_dict[x][0]] for x in ['train', 'val']}
    # seq_split_dict = {x: partition_dataset_classes(
    #     data=seq_split_dict[x][0],
    #     classes=seq_class_dict[x],
    #     num_partitions=dist.get_world_size(),
    #     shuffle=True,
    #     even_divisible=(True if x == 'train' else False))[dist.get_rank()] for x in ['train','val']}

    datasets = {x: CacheDataset(
        data=split_dict[x], 
        transform=data_transforms(
            dataset=x, 
            modalities=args.mod_list, 
            device=device_id), 
        num_workers=8,
        copy_cache=False) for x in ['train','val']}
    # datasets = {x: CacheSeqDataset(
    #     data=seq_split_dict[x], 
    #     transform=data_transforms(
    #         dataset=x, 
    #         modalities=args.mod_list,
    #         device=device_id), 
    #     num_workers=8,
    #     copy_cache=False) for x in ['train','val']}
    dataloader = {x: ThreadDataLoader(
        datasets[x], 
        batch_size=(args.batch_size if x == 'train' else 1), 
        shuffle=(True if x == 'train' else False),   
        num_workers=0,
        drop_last=False) for x in ['train','val']}

    return dataloader, metric, pos_weight

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
        Spacingd(keys=modalities, pixdim=(1.5, 1.5, 2.0), mode='bilinear'),
        ResampleToMatchd(keys=modalities, key_dst='T1W_OOP', mode='bilinear'),
        ConcatItemsd(keys='DWI_b0', name='image'),
        NormalizeIntensityd(keys='image', channel_wise=True),
        ScaleIntensityRangePercentilesd(keys='T1W_OOP', lower=33.0, upper=99.0, b_min=0.0, b_max=1.0, clip=True),
        ForegroundMaskd(keys='T1W_OOP', invert=True, new_key_prefix='MASK_', threshold=0.95),
        Lambdad(keys='MASK_T1W_OOP', func=lambda x: x[:, :, :, -(x.shape[3] // 10):]),
        KeepLargestConnectedComponentd(keys='MASK_T1W_OOP', connectivity=1),
        ResizeToMatchd(keys='MASK_T1W_OOP', dst_key='image', mode='nearest'),
        CropForegroundd(
            keys='image', 
            source_key='MASK_T1W_OOP', 
            select_fn=lambda x: x > 0,
            k_divisible=(144, 192, 96),
            allow_smaller=False),
        DeleteItemsd(keys=modalities + ['MASK_T1W_OOP']),
        ResizeWithPadOrCropd(keys='image', spatial_size=(144, 192, 96), mode='constant'),
        DivisiblePercentileCropd(keys='image', roi_center=(0.5, 0.4, 0.5), k_divisible=(144, 144, 96))
    ]

    train = [
        EnsureTyped(keys='image', track_meta=False, device=device),
        RandFlipd(keys='image', prob=0.1, spatial_axis=0),
        RandFlipd(keys='image', prob=0.1, spatial_axis=1),
        RandFlipd(keys='image', prob=0.1, spatial_axis=2),
        RandStdShiftIntensityd(keys='image', prob=0.5, factors=0.1),
        RandScaleIntensityd(keys='image', prob=0.5, factors=0.1),
        ScaleIntensityd(keys='image', minv=0.0, maxv=1.0, channel_wise=True),
        RandGridPatchd(
            keys='image',
            patch_size=(48, 48, 48),
            max_offset=(0, 0, 0)),
        RandSpatialCropd(keys='image', roi_size=(18, 32, 32, 32), random_size=False)
    ]

    val = [
        ScaleIntensityd(keys='image', minv=0.0, maxv=1.0, channel_wise=True),
        GridPatchd(
            keys='image', 
            patch_size=(32, 32, 32),
            offset=(8, 8, 0)),
        EnsureTyped(keys='image', track_meta=False, device=device)
    ]

    if dataset == 'train':
        return Compose(prep + train)
    elif dataset in ['val', 'test']:
        return Compose(prep + val)
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
    # Set a seed for reproducibility.
    set_determinism(seed=args.seed)
    # Setup distributed training.
    if args.distributed:
        setup()
        rank = dist.get_rank()
        num_devices = torch.cuda.device_count()
        device_id = rank % num_devices
        learning_rate = args.learning_rate * np.sqrt(num_devices)
        accum_steps = args.total_batch_size / args.batch_size / num_devices
    else:
        device_id = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        accum_steps = args.total_batch_size / args.batch_size

    dataloader, metric, pos_weight = load_train_objs(device_id, args)
    set_track_meta(False)
    # if args.backbone == 'resnet10':
    #     model = resnet10(pretrained=True, n_input_channels=len(args.mod_list), num_classes=args.num_classes, shortcut_type='B')
    # elif args.backbone == 'resnet50':
    #     model = resnet50(pretrained=True, n_input_channels=len(args.mod_list), num_classes=args.num_classes, shortcut_type='B')
    # elif args.backbone == 'swinvit':
    #     model = swinvit_base(pretrained=True, in_chans=len(args.mod_list), num_classes=args.num_classes)
    # else:
    model = MILNet(
        backbone=args.backbone,
        num_channels=1,
        num_spatial_dims=3,
        num_classes=args.num_classes,
        mil_mode=args.mil_mode,
        truncate_layer=None,
        pretrained=args.pretrained)
    # model = SequenceNet(model, num_classes=2, dropout=0.1)
    model.metric = metric
    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) if args.backbone != 'swinvit' else model
        model = model.to(device_id)
        model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[device_id])
    else:
        model = model.to(device_id)
    # Train the model using the training data and validate the model on the validation data following each epoch.
    trainer = Trainer(model, args.backbone, args.amp, dataloader, learning_rate, args.weight_decay, pos_weight, args.results_dir)
    trainer.training_loop(metric, args.min_epochs, args.val_interval, args.batch_size, accum_steps, args.patience)
    # Cleanup distributed training.
    if args.distributed:
        cleanup()
    # Plot the training and validation loss and F1 score.
    trainer.visualize_training('loss')
    trainer.visualize_training('f1score')
    print('Script finished')
    

RESULTS_DIR = '/Users/noltinho/thesis/results'
DATA_DIR = '/Users/noltinho/thesis/sensitive_data'
MODALITY_LIST = ['T1W_OOP','T1W_IP','T1W_DYN','T2W_TES','T2W_TEL','DWI_b150','DWI_b400','DWI_b800']
WEIGHTS_DIR = '/Users/noltinho/MedicalNet/pytorch_files/pretrain'

if __name__ == '__main__':
    args = parse_args()
    main(args)


