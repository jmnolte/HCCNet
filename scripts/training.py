import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import (
    DatasetPreprocessor, 
    GroupStratifiedSplit,
    transformations
)
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    SmartCacheDataset,
    DistributedSampler,
    partition_dataset,
    partition_dataset_classes,
    set_track_meta
)
from monai.utils import set_determinism
from monai.networks.nets import milmodel
from monai.optimizers import LearningRateFinder
import argparse
import torchmetrics
from milnet import MILNet

class Trainer:
    
    def __init__(
            self, 
            model: torch.nn.Module, 
            backbone: str, 
            amp: bool,
            dataloaders: dict, 
            learning_rate: float, 
            weight_decay: float,
            accum_steps: int,
            weights: torch.Tensor,
            smoothing_coef: float,
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
        self.backbone = backbone
        self.dataloaders = dataloaders
        self.output_dir = output_dir
        params = self.model.parameters()
        if args.mil_mode in ["att_trans", "att_trans_pyramid"]:
            params = [
                {"params": list(self.model.module.attention.parameters()) + list(self.model.module.myfc.parameters()) + list(self.model.module.net.parameters())},
                {"params": list(self.model.module.transformer.parameters()), "lr": 6e-6, "weight_decay": 0.1},
            ]
        self.optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CyclicLR(
            self.optimizer, 
            base_lr=learning_rate,
            max_lr=learning_rate*10,
            mode='triangular2', 
            step_size_up=int(len(dataloaders['train']) / accum_steps * 4),
            cycle_momentum=False)
        # self.scheduler = optim.lr_scheduler.ExponentialLR(
        #     self.optimizer, 
        #     gamma=0.5)
        self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.gpu_id), label_smoothing=smoothing_coef).to(self.gpu_id)

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
            epoch: int, 
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
        # self.dataloaders['train'].sampler.set_epoch(epoch)
        self.optimizer.zero_grad(set_to_none=True)

        if self.gpu_id == 0:
            print('-' * 10)
            print(f'Epoch {epoch}')
            print('-' * 10)

        for step, batch_data in enumerate(self.dataloaders['train']):
            inputs, labels = batch_data['image'].to(self.gpu_id), batch_data['label'].to(self.gpu_id)

            with autocast(enabled=self.amp):
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                loss /= accum_steps
                running_loss += loss.item()

            self.scaler.scale(loss).backward()
            preds = torch.argmax(logits, 1)
            batch_f1score = metric(preds, labels)
    
            if ((step + 1) % accum_steps == 0) or (step + 1 == len(self.dataloaders['train'])):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                if self.gpu_id == 0:
                    print(f"{step + 1}/{len(self.dataloaders['train'])}, Batch Loss: {loss.item() * accum_steps:.4f}, Batch F1-Score: {batch_f1score.item():.4f}")

        epoch_loss = running_loss / (len(self.dataloaders['train']) // accum_steps)
        epoch_f1score = metric.compute()
        metric.reset()
        if self.gpu_id == 0:
            print(f"[GPU {self.gpu_id}] Epoch {epoch}, Training Loss: {epoch_loss:.4f}, Training F1-Score: {epoch_f1score.item():.4f}")
        return epoch_loss, epoch_f1score

    def validate(
            self, 
            metric: torchmetrics, 
            epoch: int,
            num_patches: int
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
        running_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch_data in self.dataloaders['val']:
                inputs, labels = batch_data['image'].to(self.gpu_id), batch_data['label'].to(self.gpu_id)

                with autocast(enabled=self.amp):
                    if inputs.shape[1] > num_patches:
                        logits = []
                        logits2 = []

                        for i in range(int(np.ceil(inputs.shape[1] / float(num_patches)))):
                            data_slice = inputs[:, i * num_patches : (i + 1) * num_patches]
                            logits_slice = self.model(data_slice, no_head=True)
                            logits.append(logits_slice)

                            if self.model.module.mil_mode == 'att_trans_pyramid':
                                logits2.append([
                                        self.model.module.extra_outputs["layer1"],
                                        self.model.module.extra_outputs["layer2"],
                                        self.model.module.extra_outputs["layer3"],
                                        self.model.module.extra_outputs["layer4"]
                                        ])

                        logits = torch.cat(logits, dim=1)
                        if self.model.module.mil_mode == 'att_trans_pyramid':
                            self.model.module.extra_outputs["layer1"] = torch.cat([l[0] for l in logits2], dim=0)
                            self.model.module.extra_outputs["layer2"] = torch.cat([l[1] for l in logits2], dim=0)
                            self.model.module.extra_outputs["layer3"] = torch.cat([l[2] for l in logits2], dim=0)
                            self.model.module.extra_outputs["layer4"] = torch.cat([l[3] for l in logits2], dim=0)
                        logits = self.model.module.calc_head(logits)
                    else:
                        logits = self.model(inputs)

                    loss = self.criterion(logits, labels)

                running_loss += loss.item()
                preds = torch.argmax(logits, 1)
                metric.update(preds, labels)

            epoch_loss = running_loss / len(self.dataloaders['val'])
            epoch_f1score = metric.compute()
            metric.reset()
            if self.gpu_id == 0:
                print(f"[GPU {self.gpu_id}] Epoch {epoch}, Validation Loss: {epoch_loss:.4f}, Validation F1-Score: {epoch_f1score.item():.4f}")
        return epoch_loss, epoch_f1score
    
    def training_loop(
            self, 
            train_ds: monai.data,
            metric: torchmetrics, 
            min_epochs: int, 
            val_every: int,
            accum_steps: int, 
            early_stopping: bool, 
            patience: int,
            num_patches: int
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
        best_loss = 1000
        counter = 0
        early_stop_flag = torch.zeros(1).to(self.gpu_id)
        history = {'train_loss': [], 'train_f1score': [], 'val_loss': [], 'val_f1score': []}
        f1score = 0
        train_ds.start()

        for epoch in range(max_epochs):
            start_time = time.time()
            train_loss, train_f1score = self.train(metric, epoch, accum_steps)
            history['train_loss'].append(train_loss)
            history['train_f1score'].append(train_f1score.cpu().item())
            # self.scheduler.step()
            if (epoch + 1) % val_every == 0:
                val_loss, val_f1score = self.validate(metric, epoch, num_patches)
                history['val_loss'].append(val_loss)
                history['val_f1score'].append(val_f1score.cpu().item())
                if self.gpu_id == 0:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        f1score = val_f1score
                        counter = 0
                        print('New best validation loss. Saving model weights...')
                        best_weights = copy.deepcopy(self.model.state_dict())
                    elif val_loss >= best_loss:
                        counter += 1
                        if epoch >= min_epochs and counter >= patience:
                            early_stop_flag += 1
            train_ds.update_cache()
            train_time = time.time() - start_time
            if self.gpu_id == 0:
                print(f'Epoch {epoch} complete in {train_time // 60:.0f}min {train_time % 60:.0f}sec')
                
            torch.distributed.all_reduce(early_stop_flag, op=torch.distributed.ReduceOp.SUM)
            if early_stopping:
                if early_stop_flag == 1:
                    break

        train_ds.shutdown()
        if self.gpu_id == 0:
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Loss {:.4f} and F1-Score {:.4f} of best model configuration:'.format(best_loss, f1score.item()))
            self.save_output(best_weights, 'weights')
            self.save_output(history, 'history')

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

class FocalLoss(nn.Module):

    '''
    Focal loss implementation using cross entropy loss.
    '''

    def __init__(
            self, 
            gamma: float, 
            weight: torch.Tensor, 
            reduction: str = 'mean'
            ) -> None:
        super(FocalLoss, self).__init__()
        '''
        Initialize focal loss.

        Args:
            gamma (float): Focal loss gamma parameter.
            weight (torch.Tensor): Class weights.
            reduction (str): Loss reduction method.
        '''
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(
            self, 
            input: torch.Tensor, 
            target: torch.Tensor
            ) -> torch.Tensor:

        '''
        Forward pass.

        Args:
            input (torch.Tensor): Input tensor.
            target (torch.Tensor): Target tensor.
        '''
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce_loss)
        loss = (1 - p_t)**self.gamma * ce_loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss

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
    parser.add_argument("--distributed", action='store_true',
                        help="Flag to enable distributed training.")
    parser.add_argument("--amp", action='store_true',
                        help="Flag to enable automated mixed precision training.")
    parser.add_argument("--test-run", action='store_true',
                        help="Flag to perform test run on subset (n=100) of the data.")
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
    parser.add_argument("--num-patches", default=64, type=int,
                        help="Number of patches to use for training. Defaults to None.")
    parser.add_argument("--image-size", default=224, type=int,
                        help="Image size to use for training.")
    parser.add_argument("--augment-prob", default=0.5, type=float,
                        help="Probability with which random transform is to be applied.")
    parser.add_argument("--num-workers", default=8, type=int,
                        help="Number of workers to use for training. Defaults to 4.")
    parser.add_argument("--train-ratio", default=0.8, type=float, 
                        help="Ratio of data to use for training. Defaults to 0.8.")
    parser.add_argument("--learning-rate", default=1e-4, type=float, 
                        help="Learning rate to use for training. Defaults to 1e-4.")
    parser.add_argument("--weight-decay", default=0, type=float, 
                        help="Weight decay to use for training. Defaults to 0.")
    parser.add_argument("--smoothing-coef", default=0, type=float, 
                        help="Coefficient to smooth labels. Defaults to 0.")
    parser.add_argument("--tl-strategy", default='finetune', type=str, 
                        help="Transfer learning strategy to use. Can be finetuning (FT), layer-wise finetuning (LWFT), or TransFusion (TF). Defaults to FT.")
    parser.add_argument("--shrink-coef", default=1, type=int, 
                        help="Factor by which the network architecture should be reduced. Only applies if TransFusion is selected as transfer learning strategy.")                   
    parser.add_argument("--cutoff-point", default=-1, type=int, 
                        help="Cutoff point from where layers are to be updated when layer-wise finetuning is selected or from where layers are randomly initialized when transfusion is selected.")                 
    parser.add_argument("--early-stopping", action='store_true',
                        help="Flag to use early stopping.")
    parser.add_argument("--patience", default=10, type=int, 
                        help="Patience to use for early stopping. Defaults to 5.")
    parser.add_argument("--mod-list", default=MODALITY_LIST, nargs='+', 
                        help="List of modalities to use for training")
    parser.add_argument("--seed", default=1234, type=int, 
                        help="Seed to use for reproducibility")
    parser.add_argument("--data-dir", default=DATA_DIR, type=str, 
                        help="Path to data directory")
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
    multiclass = True if args.num_classes > 2 else False
    data_dict, label_df = DatasetPreprocessor(data_dir=args.data_dir, test_run=args.test_run).load_imaging_data(modalities=args.mod_list, multiclass=multiclass)
    train, val_test = GroupStratifiedSplit(split_ratio=args.train_ratio, multiclass=multiclass).split_dataset(label_df)
    val, test = GroupStratifiedSplit(split_ratio=0.5, multiclass=multiclass).split_dataset(val_test)
    split_dict = GroupStratifiedSplit(multiclass=multiclass).convert_to_dict(train, val, test, data_dict)
    label_dict = {x: [patient['label'] for patient in split_dict[x]] for x in ['train', 'val', 'test']}
    if args.distributed:
        if args.test_run:
            split_dict = {x: partition_dataset(
                data=split_dict[x],
                num_partitions=torch.distributed.get_world_size(),
                shuffle=True,
                even_divisible=True)[torch.distributed.get_rank()] for x in ['train','val']}
        else:
            split_dict = {x: partition_dataset_classes(
                data=split_dict[x],
                classes=label_dict[x],
                num_partitions=torch.distributed.get_world_size(),
                shuffle=True,
                even_divisible=True)[torch.distributed.get_rank()] for x in ['train','val']}
    datasets = {x: None for x in ['train','val']}
    datasets['train'] = SmartCacheDataset(
        data=split_dict['train'], 
        transform=transformations('train', args.mod_list, args.num_patches, args.image_size, args.augment_prob, device_id), 
        replace_rate=0.25,
        cache_num=(64 if args.test_run else int(512 / torch.cuda.device_count())),
        num_init_workers=args.num_workers,
        num_replace_workers=args.num_workers,
        copy_cache=False)
    datasets['val'] = CacheDataset(
        data=split_dict['val'], 
        transform=transformations('val', args.mod_list, args.num_patches, args.image_size, args.augment_prob, device_id), 
        num_workers=int(args.num_workers / 2),
        copy_cache=False)
    # if args.distributed:
    #     sampler = {x: DistributedSampler(
    #         dataset=datasets[x], 
    #         even_divisible=True, 
    #         shuffle=(True if x == 'train' else False)) for x in ['train','val']}
    # else:
    #     sampler = {x: None for x in ['train','val']}
    dataloader = {x: ThreadDataLoader(
        datasets[x], 
        batch_size=(args.batch_size if x == 'train' else int(args.batch_size / 2)), 
        shuffle=(True if x == 'train' else False), 
        num_workers=0) for x in ['train','val']}

    metric = torchmetrics.F1Score(task=('multiclass' if args.num_classes > 2 else 'binary'), num_classes=args.num_classes)
    values, counts = np.unique(label_dict['train'], return_counts=True)
    weights = len(label_dict['train']) / (args.num_classes * torch.Tensor([class_count for class_count in counts]))
    return datasets, dataloader, metric, weights

def setup() -> None:

    '''
    Setup distributed training.
    '''
    torch.distributed.init_process_group(backend="nccl")

def cleanup() -> None:

    '''
    Cleanup distributed training.
    '''
    torch.distributed.destroy_process_group()

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
        rank = torch.distributed.get_rank()
        num_devices = torch.cuda.device_count()
        device_id = rank % num_devices
        args.learning_rate = num_devices * args.learning_rate
        accum_steps = args.total_batch_size / args.batch_size / num_devices
    else:
        device_id = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        accum_steps = args.total_batch_size / args.batch_size
    datasets, dataloader, metric, weights = load_train_objs(device_id, args)
    model = MILNet(
        num_classes=args.num_classes, 
        mil_mode=args.mil_mode, 
        backbone=args.backbone, 
        pretrained=args.pretrained, 
        tl_strategy=args.tl_strategy, 
        shrink_coefficient=args.shrink_coef,
        load_up_to=args.cutoff_point)
    model.metric = metric
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(device_id)
        model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[device_id],
            find_unused_parameters=(True if args.backbone == 'densenet121' and args.mil_mode == 'att_trans_pyramid' else False))
    else:
        model = model.to(device_id)
    set_track_meta(False)
    # Train the model using the training data and validate the model on the validation data following each epoch.
    trainer = Trainer(model, args.backbone, args.amp, dataloader, args.learning_rate, args.weight_decay, accum_steps, weights, args.smoothing_coef, args.results_dir)
    trainer.training_loop(datasets['train'], metric, args.min_epochs, args.val_interval, accum_steps, args.early_stopping, args.patience, args.num_patches)
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
    
