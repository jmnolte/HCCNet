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
from torch.nn.parallel import DistributedDataParallel as DDP

class Training:
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        version: str, 
        dataloaders: dict, 
        save_every: int,
        snapshot_path: str,
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
        try: 
            assert any(version == version_item for version_item in ['resnet10','resnet18','resnet34','resnet50','resnet101','resnet152','resnet200'])
        except AssertionError:
            print('Invalid version. Please choose from: resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200')
            exit(1)

        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.model = model.to(self.gpu_id)
        self.version = version
        self.dataloaders = dataloaders
        self.output_dir = output_dir
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = os.path.join(output_dir, 'snapshots' + version + '_' + 'snapshot.pth')
        if self.epochs_run != 0 and os.path.exists(self.snapshot_path):
            print('Loading snapshot')
            self.load_snapshot(self.snapshot_path)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
    
    def load_snapshot(self, snapshot_path: str) -> dict:

        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot['MODEL_STATE'])
        self.epochs_run = snapshot['EPOCHS_RUN']
        print('Resuming training from epoch:'.format(self.epochs_run))

    def save_snapshot(self, epoch: int) -> None:

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

    def train_model(self, min_epochs: int, criterion: torch.nn, optimizer: torch.optim, early_stopping: bool, patience: int) -> None:

        '''
        Train the model.

        Args:
            min_epochs (int): Minimum number of epochs.
            criterion (torch.nn): Loss function.
            optimizer (torch.optim): Optimizer.
            early_stopping (bool): If True, early stopping is used.
            patience (int): Number of epochs to wait before early stopping.
        '''
    
        since = time.time()
        max_epochs = min_epochs * 100

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_loss = 10000.0
        counter = 0
        criterion = criterion.to(self.gpu_id)

        for epoch in range(self.epochs_run, max_epochs):
            print(f"[GPU{self.gpu_id}] Epoch {epoch}")
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train() 
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                for batch_data in tqdm(self.dataloaders[phase]):
                    inputs, labels = batch_data['image'].to(self.gpu_id), batch_data['label'].to(self.gpu_id)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = copy.deepcopy(running_loss / len(self.dataloaders[phase].dataset))
                epoch_acc = copy.deepcopy(running_corrects.double() / len(self.dataloaders[phase].dataset))
                epoch_kappa = (epoch_acc.item() - 0.075) / (1 - 0.075)

                print('{} Loss: {:.4f} Accuracy: {:.4f} Kappa: {:.4f}'.format(phase, epoch_loss, epoch_acc.item(), epoch_kappa))

                if phase == 'train':
                    history['train_loss'].append(epoch_loss)
                    history['train_acc'].append(epoch_acc.cpu().item())
                elif phase == 'val':
                    history['val_loss'].append(epoch_loss)
                    history['val_acc'].append(epoch_acc.cpu().item())

                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    acc = epoch_acc
                    kappa = epoch_kappa
                    counter = 0
                    best_weights = copy.deepcopy(self.model.state_dict())
                elif phase == 'val' and epoch_loss >= best_loss:
                    counter += 1

            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self.save_snapshot(epoch)

            if early_stopping == True:
                if epoch > min_epochs - 1 and counter == patience:
                    break

        time_elapsed = time.time() - since
        self.save_output(best_weights, 'weights')
        self.save_output(history, 'history')

        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Perfrormance Metrics on Validation set:')
        print('Accuracy:'.format(acc.item()))
        print('Kappa:'.format(kappa))

    def visualize_training(self, metric: str) -> None:

        '''
        Visualize the training and validation history.

        Args:
            metric (str): 'loss' or 'acc'.
        '''
        try: 
            assert any(metric == metric_item for metric_item in ['loss','acc'])
        except AssertionError:
            print('Invalid input. Please choose from: loss or acc.')
            exit(1)

        if metric == 'loss':
            metric_label = 'Loss'
        elif metric == 'acc':
            metric_label = 'Accuracy'

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

def ddp_setup() -> None:

    torch.distributed.init_process_group(backend="nccl")

RESULTS_DIR = '/Users/noltinho/thesis/results'
DATA_DIR = '/Users/noltinho/thesis/sensitive_data'
MODALITY_LIST = ['T1W_OOP','T1W_IP','T1W_DYN','T2W_TES','T2W_TEL','DWI_b150','DWI_b400','DWI_b800']
WEIGHTS_DIR = '/Users/noltinho/MedicalNet/pytorch_files/pretrain'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version", required=True, type=str, help="Model version to train")
    parser.add_argument("-p", "--pretrained", default=True, type=bool, help="Flag to use pretrained weights")
    parser.add_argument("-fe", "--feature_extraction", default=True, type=bool, help="Flag to use feature extraction")
    parser.add_argument("-e", "--epochs", required=True, type=int, help="Number of epochs to train for")
    parser.add_argument("-b", "--batch_size", default=4, type=int, help="Batch size to use for training")
    parser.add_argument("-tr", "--train_ratio", default=0.8, type=float, help="Ratio of training data to use for training")
    parser.add_argument("-lr", "--learning_rate", default=1e-5, type=float, help="Learning rate to use for training")
    parser.add_argument("-w", "--weight_decay", default=1e-7, type=float, help="Weight decay to use for training")
    parser.add_argument("-es", "--early_stopping", default=True, type=bool, help="Flag to use early stopping")
    parser.add_argument("-pa", "--patience", default=10, type=int, help="Patience to use for early stopping")
    parser.add_argument("-ml", "--modality_list", default=MODALITY_LIST, nargs='+', help="List of modalities to use for training")
    parser.add_argument("-s", "--seed", default=123, type=int, help="Seed to use for reproducibility")
    parser.add_argument("-ts", "--test_set", default=False, type=bool, help="Flag to load test or training and validation set")
    parser.add_argument("-q", "--quant_images", default=False, type=bool, help="Flag to run analysis on quant images")
    parser.add_argument("-dd", "--data_dir", default=DATA_DIR, type=str, help="Path to data directory")
    parser.add_argument("-rd", "--results_dir", default=RESULTS_DIR, type=str, help="Path to results directory")
    parser.add_argument("-wd", "--weights_dir", default=WEIGHTS_DIR, type=str, help="Path to pretrained weights")
    args = vars(parser.parse_args())
    ReproducibilityUtils.seed_everything(args['seed'])
    ddp_setup()
    torch.multiprocessing.set_sharing_strategy('file_system')
    dataloader = DataLoader(args['data_dir'], args['modality_list'])
    data_dict = dataloader.create_data_dict()
    dataloader_dict = dataloader.load_data(data_dict, args['train_ratio'], args['batch_size'], 2, args['test_set'], args['quant_images'])
    model = ResNet(args['version'], 2, len(args['modality_list']), args['pretrained'], args['feature_extraction'], args['weights_dir'])
    if args['quant_images']:
        weight = torch.tensor([19, 189]) / 208
    else:
        weight = torch.tensor([60, 739]) / 799
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
    train = Training(model, args['version'], dataloader_dict, 10, os.path.join(args['results_dir'], 'snapshots'), args['results_dir'])
    train.train_model(args['epochs'], criterion, optimizer, args['early_stopping'], args['patience'])
    torch.distributed.destroy_process_group()
    train.visualize_training('loss')
    train.visualize_training('acc')
    
