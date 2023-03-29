import torch
import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from preprocessing import DataLoader
from models import ResNet
from utils import ReproducibilityUtils
from tqdm import tqdm
import argparse
import torchmetrics

class Trainer:
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        version: str, 
        dataloaders: dict, 
        optimizer: torch.optim,
        criterion: torch.nn,
        output_dir: str
    ) -> None:

        '''
        Initialize the training class.

        Args:
            model (torch.nn): Model to train.
            version (str): Model version. Can be 'resnet10', 'resnet18', 'resnet34', 'resnet50',
                'resnet101', 'resnet152', or 'resnet200'.
            dataloaders (dict): Dataloader objects.
            optimizer (torch.optim): Optimizer.
            criterion (torch.nn): Loss function.
            output_dir (str): Output directory.
        '''
        try: 
            assert any(version == version_item for version_item in ['resnet10','resnet18','resnet34','resnet50','resnet101','resnet152','resnet200'])
        except AssertionError:
            print('Invalid version. Please choose from: resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200')
            exit(1)

        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.model = model.to(self.gpu_id)
        self.version = version
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.criterion = criterion.to(self.gpu_id)
        self.output_dir = output_dir
        self.epochs_run = 0
        self.snapshot_path = os.path.join(output_dir, 'snapshots' + version + '_' + 'snapshot.pth')
        if self.epochs_run != 0 and os.path.exists(self.snapshot_path):
            print('Loading snapshot')
            self.load_snapshot(self.snapshot_path)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu_id])
    
    def load_snapshot(self, snapshot_path: str) -> dict:

        '''
        Load a snapshot of the model.
        '''
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot['MODEL_STATE'])
        self.epochs_run = snapshot['EPOCHS_RUN']
        print('Resuming training from epoch:'.format(self.epochs_run))

    def save_snapshot(self, epoch: int) -> None:
        
        '''
        Save a snapshot of the model.
        '''
        snapshot = {}
        snapshot['MODEL_STATE'] = self.model.module.state_dict()
        snapshot['EPOCHS_RUN'] = epoch
        folder_name = self.version + '_' + 'snapshot.pth'
        folder_path = os.path.join(self.output_dir, 'snapshots', folder_name)
        folder_path_root = os.path.join(self.output_dir, 'snapshots')

        if os.path.exists(folder_path):
            os.remove(folder_path)
        elif not os.path.exists(folder_path_root):
            os.makedirs(folder_path_root)
        torch.save(snapshot, folder_path)
        print('Epoch {}: Training snapshot saved at snapshot.pth'.format(epoch))

    def save_output(self, output_dict: dict, output_type: str) -> None:

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
            folder_name = self.version + '_weights.pth'
        elif output_type == 'history':
            folder_name = self.version + '_hist.npy'
        elif output_type == 'preds':
            folder_name = self.version + '_preds.npy'
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

    # def train_model(self, criterion: torch.nn, optimizer: torch.optim, early_stopping: bool, patience: int) -> None:

    #     '''
    #     Train the model.

    #     Args:
    #         min_epochs (int): Minimum number of epochs.
    #         criterion (torch.nn): Loss function.
    #         optimizer (torch.optim): Optimizer.
    #         early_stopping (bool): If True, early stopping is used.
    #         patience (int): Number of epochs to wait before early stopping.
    #     '''
    
    #     since = time.time()
    #     max_epochs = min_epochs * 100

    #     history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    #     best_loss = 10000.0
    #     counter = 0
    #     criterion.to(self.gpu_id)

    #     for epoch in range(self.epochs_run, max_epochs):

    #         self.dataloaders['train'].sampler.set_epoch(epoch)

    #         for phase in ['train', 'val']:
    #             if phase == 'train':
    #                 self.model.train() 
    #             else:
    #                 self.model.eval()

    #             running_loss = 0.0
    #             running_corrects = 0

    #             for batch_data in tqdm(self.dataloaders[phase]):
    #                 inputs, labels = batch_data['image'].to(self.gpu_id), batch_data['label'].to(self.gpu_id)
    #                 optimizer.zero_grad()

    #                 with torch.set_grad_enabled(phase == 'train'):
    #                     outputs = self.model(inputs)
    #                     loss = criterion(outputs, labels)

    #                     _, preds = torch.max(outputs, 1)

    #                     if phase == 'train':
    #                         loss.backward()
    #                         optimizer.step()
                    
    #                 running_loss += loss.item() * inputs.size(0)
    #                 running_corrects += torch.sum(preds == labels.data)

    #             epoch_loss = copy.deepcopy(running_loss / len(self.dataloaders[phase].dataset))
    #             epoch_acc = copy.deepcopy(running_corrects.double() / len(self.dataloaders[phase].dataset))
    #             epoch_kappa = (epoch_acc.item() - 0.075) / (1 - 0.075)

    #             print(f"[GPU{self.gpu_id}] Epoch {epoch}")
    #             print('-' * 10)
    #             print('{} Loss: {:.4f} Accuracy: {:.4f} Kappa: {:.4f}'.format(phase, epoch_loss, epoch_acc.item(), epoch_kappa))

    #             if phase == 'train':
    #                 history['train_loss'].append(epoch_loss)
    #                 history['train_acc'].append(epoch_acc.cpu().item())
    #             elif phase == 'val':
    #                 history['val_loss'].append(epoch_loss)
    #                 history['val_acc'].append(epoch_acc.cpu().item())

    #             if phase == 'val' and epoch_loss < best_loss:
    #                 best_loss = epoch_loss
    #                 acc = epoch_acc
    #                 kappa = epoch_kappa
    #                 counter = 0
    #                 if self.gpu_id == 0
    #                     best_weights = copy.deepcopy(self.model.state_dict())
    #             elif phase == 'val' and epoch_loss >= best_loss:
    #                 counter += 1

    #         if self.gpu_id == 0 and epoch % self.save_every == 0:
    #             self.save_snapshot(epoch)

    #         if early_stopping:
    #             if epoch > min_epochs - 1 and counter == patience:
    #                 break

    #     time_elapsed = time.time() - since
    #     self.save_output(best_weights, 'weights')
    #     self.save_output(history, 'history')

    #     print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #     print('Perfrormance Metrics on Validation set:')
    #     print('Accuracy:'.format(acc.item()))
    #     print('Kappa:'.format(kappa))

    def train(self, metric: torchmetrics, epoch: int) -> tuple:

        '''
        Train the model.

        Args:
            epoch (int): Current epoch.
        '''
        running_loss = 0.0
        epoch_loss_values = []
        self.model.train()
        self.dataloaders['train'].sampler.set_epoch(epoch)
        if self.gpu_id == 0:
            print(f'Epoch {epoch}')
            print('-' * 10)
        for step, batch_data in enumerate(self.dataloaders['train']):
            inputs, labels = batch_data['image'].to(self.gpu_id), batch_data['label'].to(self.gpu_id)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            batch_f1score = metric(preds, labels)
            epoch_len = len(self.dataloaders['train'].dataset) // self.dataloaders['train'].batch_size // 4
            if self.gpu_id == 0 and step % 5 == 0:
                print(f"{step}/{epoch_len}, Batch Loss: {loss.item():.4f}, Batch F1-Score: {batch_f1score.item():.4f}")
        epoch_loss = running_loss / (step + 1)
        epoch_loss_values.append(running_loss)
        epoch_f1score = metric.compute()
        metric.reset()
        if self.gpu_id == 0:
            print(f"[GPU {self.gpu_id}] Epoch {epoch}, Training Loss: {epoch_loss:.4f}, Training F1-Score: {epoch_f1score.item():.4f}")
        return epoch_loss, epoch_f1score

    def validate(self, metric: torchmetrics, epoch: int) -> tuple:

        '''
        Validate the model.

        Args:
            criterion (torch.nn): Loss function.
            epoch (int): Current epoch.
            args (argparse.Namespace): Arguments.
        '''
        running_loss = 0.0
        epoch_loss_values = []
        self.model.eval()
        with torch.no_grad():
            for step, batch_data in enumerate(self.dataloaders['val']):
                inputs, labels = batch_data['image'].to(self.gpu_id), batch_data['label'].to(self.gpu_id)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                metric.update(preds, labels)
            epoch_loss = running_loss / (step + 1)
            epoch_loss_values.append(running_loss)
            epoch_f1score = metric.compute()
            metric.reset()
            if self.gpu_id == 0:
                print(f"[GPU {self.gpu_id}] Epoch {epoch}, Validation Loss: {epoch_loss:.4f}, Validation F1-Score: {epoch_f1score.item():.4f}")
        return epoch_loss, epoch_f1score
    
    def training_loop(self, metric: torchmetrics, min_epochs: int, early_stopping: bool, patience: int) -> None:

        '''
        Training loop.

        Args:
            metric (torch.nn): Metric to optimize.
            min_epochs (int): Minimum number of epochs to train.
            patience (int): Number of epochs to wait before early stopping.
            early_stopping (bool): Whether to use early stopping.
        '''
        since = time.time()
        max_epochs = min_epochs * 100
        best_loss = np.inf
        counter = 0
        history = {'train_loss': [], 'train_f1score': [], 'val_loss': [], 'val_f1score': []}
        f1score = 0
        for epoch in range(max_epochs):
            start_time = time.time()
            train_loss, train_f1score = self.train(metric, epoch)
            val_loss, val_f1score = self.validate(metric, epoch)
            train_time = time.time() - start_time
            if self.gpu_id == 0:
                print(f'Epoch {epoch} complete in {train_time // 60}min {train_time % 60}sec')
            history['train_loss'].append(train_loss)
            history['train_f1score'].append(train_f1score)
            history['val_loss'].append(val_loss)
            history['val_f1score'].append(val_f1score)

            if val_loss < best_loss:
                best_loss = val_loss
                f1score = train_f1score
                counter = 0
                if self.gpu_id == 0:
                    best_weights = copy.deepcopy(self.model.state_dict())
            elif val_loss >= best_loss:
                counter += 1
                
            if early_stopping:
                if epoch > min_epochs and counter == patience:
                    time_elapsed = time.time() - since
                    self.save_output(best_weights, 'weights')
                    self.save_output(history, 'history')
                    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                    print('F1-Score of best model configuration:'.format(f1score.item()))
                    break

    def visualize_training(self, metric: str) -> None:

        '''
        Visualize the training and validation history.

        Args:
            metric (str): 'loss' or 'f1score'.
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

        file_name = self.version + '_hist.npy'
        plot_name = self.version + '_' + metric + '.png'
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
        argparse.Namespace: Arguments.
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", required=True, type=str, 
                        help="Model version to train")
    parser.add_argument("--pretrained", default=True, type=bool, 
                        help="Flag to use pretrained weights")
    parser.add_argument("--feature-extraction", default=True, type=bool, 
                        help="Flag to use feature extraction")
    parser.add_argument("--epochs", required=True, type=int, 
                        help="Number of epochs to train for")
    parser.add_argument("--batch-size", default=4, type=int, 
                        help="Batch size to use for training")
    parser.add_argument("--train-ratio", default=0.8, type=float, 
                        help="Ratio of training data to use for training")
    parser.add_argument("--learning-rate", default=1e-4, type=float, 
                        help="Learning rate to use for training")
    parser.add_argument("--weight-decay", default=1e-5, type=float, 
                        help="Weight decay to use for training")
    parser.add_argument("--early-stopping", default=True, type=bool, 
                        help="Flag to use early stopping")
    parser.add_argument("--patience", default=10, type=int, help="Patience to use for early stopping")
    parser.add_argument("--modality-list", default=MODALITY_LIST, nargs='+', 
                        help="List of modalities to use for training")
    parser.add_argument("--seed", default=123, type=int, 
                        help="Seed to use for reproducibility")
    parser.add_argument("--test-set", default=False, type=bool, 
                        help="Flag to load test or training and validation set")
    parser.add_argument("--quant-images", default=False, type=bool, 
                        help="Flag to run analysis on quant images")
    parser.add_argument("--data-dir", default=DATA_DIR, type=str, 
                        help="Path to data directory")
    parser.add_argument("--results-dir", default=RESULTS_DIR, type=str, 
                        help="Path to results directory")
    parser.add_argument("--weights-dir", default=WEIGHTS_DIR, type=str, 
                        help="Path to pretrained weights")
    return parser.parse_args()

def load_train_objs(args: argparse.Namespace) -> tuple:

    '''
    Load training objects.

    Returns:
        tuple: Training objects.
    '''
    dataloader = DataLoader(args.data_dir, args.modality_list)
    data_dict = dataloader.create_data_dict()
    dataloader_dict = dataloader.load_data(data_dict, args.train_ratio, args.batch_size, 2, args.test_set, args.quant_images)

    model = ResNet(args.version, 2, len(args.modality_list), args.pretrained, args.feature_extraction, args.weights_dir)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.quant_images:
        weights = 19 / torch.tensor([189, 19])
    else:
        weights = 60 / torch.tensor([739, 60])
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    metric = torchmetrics.F1Score(task='binary')
    model.metric = metric
    return dataloader_dict, model, optimizer, criterion, metric

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

def main(args: argparse.Namespace) -> None:

    '''
    Main function.

    Args:
        args (argparse.Namespace): Arguments.
    '''
    ReproducibilityUtils.seed_everything(args.seed)
    setup()
    dataloader_dict, model, optimizer, criterion, metric = load_train_objs(args)
    trainer = Trainer(model, args.version, dataloader_dict, optimizer, criterion, args.results_dir)
    trainer.training_loop(metric, args.epochs, args.early_stopping, args.patience)
    cleanup()
    trainer.visualize_training('loss')
    trainer.visualize_training('f1score')
    

RESULTS_DIR = '/Users/noltinho/thesis/results'
DATA_DIR = '/Users/noltinho/thesis/sensitive_data'
MODALITY_LIST = ['T1W_OOP','T1W_IP','T1W_DYN','T2W_TES','T2W_TEL','DWI_b150','DWI_b400','DWI_b800']
WEIGHTS_DIR = '/Users/noltinho/MedicalNet/pytorch_files/pretrain'

if __name__ == '__main__':
    args = parse_args()
    main(args)
    
