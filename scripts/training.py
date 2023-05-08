import torch
from torch.autograd import Variable
import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import DataLoader
from models import ResNet
from models import EnsembleModel
from utils import ReproducibilityUtils
import argparse
import torchmetrics

class Trainer:
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        version: str, 
        dataloaders: dict, 
        learning_rate: float,
        weight_decay: float,
        output_dir: str,
    ) -> None:

        '''
        Initialize the training class.

        Args:
            model (torch.nn): Model to train.
            version (str): Model version. Can be 'resnet10', 'resnet18', 'resnet34', 'resnet50',
                'resnet101', 'resnet152', 'resnet200', or 'ensemble'.
            dataloaders (dict): Dataloader objects.
            learning_rate (float): Learning rate.
            weight_decay (float): Weight decay.
            output_dir (str): Output directory.
        '''
        try: 
            assert any(version == version_item for version_item in ['resnet10','resnet18','resnet34','resnet50','resnet101','resnet152','resnet200','ensemble'])
        except AssertionError:
            print('Invalid version. Please choose from: resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200, ensemble')
            exit(1)

        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.model = model.to(self.gpu_id)
        self.version = version
        self.dataloaders = dataloaders
        self.output_dir = output_dir
        self.epochs_run = 0
        self.snapshot_path = os.path.join(output_dir, 'snapshots' + version + '_' + 'snapshot.pth')
        if self.epochs_run != 0 and os.path.exists(self.snapshot_path):
            print('Loading snapshot')
            self.load_snapshot(self.snapshot_path)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu_id])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # weights = 759 / (2 * torch.Tensor([679, 80])).to(self.gpu_id)
        # self.criterion = FocalLoss(gamma=2, weight=weights).to(self.gpu_id)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.gpu_id)
    
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

    def train(self, metric: torchmetrics, epoch: int, accum_steps: int) -> tuple:

        '''
        Train the model.

        Args:
            epoch (int): Current epoch.

        Returns:
            tuple: Tuple containing the loss and f1 score.
        '''
        running_loss = 0.0
        epoch_loss_values = []
        self.model.train()
        self.dataloaders['train'].sampler.set_epoch(epoch)
        self.optimizer.zero_grad()

        if self.gpu_id == 0:
            print('-' * 10)
            print(f'Epoch {epoch}')
            print('-' * 10)

        for step, batch_data in enumerate(self.dataloaders['train']):
            inputs, labels = batch_data['image'].to(self.gpu_id), batch_data['label'].to(self.gpu_id)
            outputs = self.model(inputs.as_tensor())
            preds = torch.argmax(outputs, 1)
            loss = self.criterion(outputs, labels)
            loss /= accum_steps
            loss.backward()
            running_loss += loss.item()
            batch_f1score = metric(preds, labels)
    
            if ((step + 1) % accum_steps == 0) or (step + 1 == len(self.dataloaders['train'])):
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.gpu_id == 0:
                    print(f"{step + 1}/{len(self.dataloaders['train'])}, Batch Loss: {loss.item() * accum_steps:.4f}, Batch F1-Score: {batch_f1score.item():.4f}")

        epoch_loss = running_loss / (len(self.dataloaders['train']) // accum_steps + 1)
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
        
        Returns:
            tuple: Tuple containing the loss and f1 score.
        '''
        running_loss = 0.0
        epoch_loss_values = []
        self.model.eval()
        with torch.no_grad():
            for step, batch_data in enumerate(self.dataloaders['val']):
                inputs, labels = batch_data['image'].to(self.gpu_id), batch_data['label'].to(self.gpu_id)
                outputs = self.model(inputs.as_tensor())
                preds = torch.argmax(outputs, 1)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                metric.update(preds, labels)

            epoch_loss = running_loss / len(self.dataloaders['val'])
            epoch_loss_values.append(running_loss)
            epoch_f1score = metric.compute()
            metric.reset()
            if self.gpu_id == 0:
                print(f"[GPU {self.gpu_id}] Epoch {epoch}, Validation Loss: {epoch_loss:.4f}, Validation F1-Score: {epoch_f1score.item():.4f}")
        return epoch_loss, epoch_f1score
    
    def training_loop(self, metric: torchmetrics, min_epochs: int, accum_steps: int, early_stopping: bool, patience: int) -> None:

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
        best_loss = 1000
        counter = 0
        early_stop_flag = torch.zeros(1).to(self.gpu_id)
        history = {'train_loss': [], 'train_f1score': [], 'val_loss': [], 'val_f1score': []}
        f1score = 0
        for epoch in range(max_epochs):
            start_time = time.time()
            train_loss, train_f1score = self.train(metric, epoch, accum_steps)
            val_loss, val_f1score = self.validate(metric, epoch)
            train_time = time.time() - start_time
            if self.gpu_id == 0:
                print(f'Epoch {epoch} complete in {train_time // 60:.0f}min {train_time % 60:.0f}sec')
            history['train_loss'].append(train_loss)
            history['train_f1score'].append(train_f1score.cpu().item())
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
                
            torch.distributed.all_reduce(early_stop_flag, op=torch.distributed.ReduceOp.SUM)
            if early_stopping:
                if early_stop_flag == 1:
                    break

        if self.gpu_id == 0:
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Loss {:.4f} and F1-Score {:.4f} of best model configuration:'.format(best_loss, f1score.item()))
            self.save_output(best_weights, 'weights')
            self.save_output(history, 'history')

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

class FocalLoss(torch.nn.Module):

    '''
    Focal loss implementation using cross entropy loss.
    '''

    def __init__(self, gamma: float = 2.0, weight=None, reduction: str = 'mean') -> None:
        super(FocalLoss, self).__init__()

        '''
        Args:
            gamma (float): Focal loss gamma parameter.
            weight (torch.Tensor): Class weights.
            reduction (str): Loss reduction method.
        '''
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):

        '''
        Forward pass.

        Args:
            input (torch.Tensor): Input tensor.
            target (torch.Tensor): Target tensor.
        '''
        ce_loss = torch.nn.functional.cross_entropy(input, target, weight=self.weight, reduction="none")
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
        argparse.Namespace: Arguments.
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", required=True, type=str, 
                        help="Model version to train")
    parser.add_argument("--pretrained", action='store_true',
                        help="Flag to use pretrained weights")
    parser.add_argument("--feature-extraction", action='store_true',
                        help="Flag to use feature extraction")
    parser.add_argument("--epochs", required=True, type=int, 
                        help="Number of epochs to train for")
    parser.add_argument("--batch-size", default=4, type=int, 
                        help="Batch size to use for training")
    parser.add_argument("--accum-steps", default=1, type=int, 
                        help="Accumulation steps to take before computing the gradient")
    parser.add_argument("--train-ratio", default=0.8, type=float, 
                        help="Ratio of training data to use for training")
    parser.add_argument("--learning-rate", default=1e-4, type=float, 
                        help="Learning rate to use for training")
    parser.add_argument("--weight-decay", default=0.0, type=float, 
                        help="Weight decay to use for training")
    parser.add_argument("--early-stopping", default=True, type=bool, 
                        help="Flag to use early stopping")
    parser.add_argument("--patience", default=10, type=int, 
                        help="Patience to use for early stopping")
    parser.add_argument("--modality-list", default=MODALITY_LIST, nargs='+', 
                        help="List of modalities to use for training")
    parser.add_argument("--seed", default=123, type=int, 
                        help="Seed to use for reproducibility")
    parser.add_argument("--weighted-sampler", action='store_true',
                        help="Flag to use a weighted sampler")
    parser.add_argument("--quant-images", action='store_true',
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
        tuple: Training objects consisting of the data loader, the model, and the performance metric.
    '''
    dataloader = DataLoader(args.data_dir, args.modality_list)
    data_dict = dataloader.create_data_dict()
    dataloader_dict = dataloader.load_data(data_dict, args.train_ratio, args.batch_size, 2, args.weighted_sampler, args.quant_images)

    if args.version == 'ensemble':
        versions = ['resnet10','resnet18','resnet34','resnet50']
        resnet10 = ResNet('resnet10', 2, len(args.modality_list), args.pretrained, args.feature_extraction, args.weights_dir)
        resnet18 = ResNet('resnet18', 2, len(args.modality_list), args.pretrained, args.feature_extraction, args.weights_dir)
        resnet34 = ResNet('resnet34', 2, len(args.modality_list), args.pretrained, args.feature_extraction, args.weights_dir)
        resnet50 = ResNet('resnet50', 2, len(args.modality_list), args.pretrained, args.feature_extraction, args.weights_dir)

        for idx, model in enumerate([resnet10, resnet18, resnet34, resnet50]):
            model_dict = model.state_dict()
            if idx == 0:
                version = 'resnet10'
            elif idx == 1:
                version = 'resnet18'
            elif idx == 2:
                version = 'resnet34'
            elif idx == 3:
                version = 'resnet50'
            weights_dict = torch.load(os.path.join(args.results_dir, 'model_weights', version + '_weights.pth'))
            weights_dict = {k.replace('module.', ''): v for k, v in weights_dict.items()}
            model_dict.update(weights_dict)
            model.load_state_dict(model_dict)
        print('Model weights are updated.')
        model = EnsembleModel(resnet10, resnet18, resnet34, resnet50, versions, 2, args.results_dir)
    else:
        model = ResNet(args.version, 2, len(args.modality_list), args.pretrained, args.feature_extraction, args.weights_dir)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.weighted_sampler:
        print('Model is trained using weighted random sampling')
    else:
        print('Model is trained using focal loss')

    metric = torchmetrics.F1Score(task='binary')
    model.metric = metric
    return dataloader_dict, model, metric

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
    dataloader_dict, model, metric = load_train_objs(args)
    trainer = Trainer(model, args.version, dataloader_dict, args.learning_rate, args.weight_decay, args.results_dir)
    trainer.training_loop(metric, args.epochs, args.accum_steps, args.early_stopping, args.patience)
    cleanup()
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
    
