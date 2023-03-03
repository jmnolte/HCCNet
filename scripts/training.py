import torch
import monai
import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from preprocessing import Preprocessing
from models import ResNet
from utils import ReproducibilityUtils
import platform

class Training:
    
    def __init__(self, model, device: torch.device, dataloaders: dict, output_dir: str) -> None:

        '''
        Initialize the training class.

        Args:
            model (torch.nn): Model to train.
            version (str): Model version.
            device (torch.device): Device to use.
            dataloaders (dict): Dataloader objects.
            output_dir (str): Output directory.
        '''

        self.model = model
        self.model.to(device)
        self.version = str(model)
        self.device = device
        self.dataloaders = dataloaders
        self.output_dir = output_dir

    def save_best_model(self, best_model_weights: dict) -> None:

        '''
        Save the model's best set of weights.

        Args:
            best_model_weights (dict): Best model's weights.
        '''

        folder_name = 'resnet_' + str(self.version.strip('resnet')) + '_weights.pth'
        weights_path = os.path.join(self.output_dir, 'model_weights', folder_name)
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        torch.save(best_model_weights, weights_path)

    def save_training_history(self, training_hist: list) -> None:
            
        '''
        Save the training and validation history.
    
        Args:
            training_hist (list): Training and validation history.
        '''
    
        folder_name = 'resnet_' + str(self.version.strip('resnet')) + '_hist.npy'
        history_path = os.path.join(self.output_dir, 'model_history', folder_name)
        if not os.path.exists(history_path):
            os.makedirs(history_path)
        np.save(history_path, training_hist)

    def save_model_preds(self, preds: list) -> None:
            
        '''
        Save the model's predictions.
    
        Args:
            preds (list): The model's predicted labels.
        '''
    
        folder_name = 'resnet_' + str(self.version.strip('resnet')) + '_preds.npy'
        preds_path = os.path.join(self.output_dir, 'model_preds', folder_name)
        if not os.path.exists(preds_path):
            os.makedirs(preds_path)
        np.save(preds_path, preds)
            

    def train_model(self, min_epochs: int, criterion: torch.nn, optimizer: torch.optim, early_stopping: bool = True, patience: int = 10) -> None:

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
        max_epochs = min_epochs * 10

        history = ['train_loss','train_acc','val_loss','val_acc']
        best_loss = 10000.0
        counter = 0

        for epoch in range(max_epochs):
            print('Epoch {}'.format(epoch))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train() 
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                for batch_data in self.dataloaders[phase]:
                    inputs, labels = batch_data['image'].to(self.device), batch_data['label'].to(self.device)

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

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc.item()))

                if phase == 'train':
                    history['train_loss'].append(epoch_loss)
                    history['train_acc'].append(epoch_acc)
                elif phase == 'val':
                    history['val_loss'].append(epoch_loss)
                    history['val_acc'].append(epoch_acc)

                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    acc = epoch_acc
                    prec = metrics.precision_score(labels.cpu(), preds.cpu()) 
                    recall = metrics.recall_score(labels.cpu(), preds.cpu())
                    fscore = metrics.f1_score(labels.cpu(), preds.cpu())
                    counter = 0
                    best_model_weights = copy.deepcopy(self.model.state_dict())
                elif phase == 'val' and epoch_loss >= best_loss:
                    counter += 1

            if early_stopping == True:
                if epoch > min_epochs - 1 and counter == patience:
                    break

        time_elapsed = time.time() - since
        Training.save_best_model(best_model_weights)
        Training.save_training_history(history)

        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Performance Metrics on Validation set:')
        print('Acc: {:4f}'.format(acc.item())) 
        print('Precision: {:4f}'.format(prec.item()))
        print('Recall: {:4f}'.format(recall.item()))
        print('F1-Score: {:4f}'.format(fscore.item()))
    
    def run_inference(self) -> None:

        '''
        Run inference on the test set.
        '''

        self.model.eval()
        results = ['preds','proba']
        for batch_data in self.dataloaders['test']:
            inputs = batch_data['image'].to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                _, pred = torch.max(outputs, 1)
                proba = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
                results['preds'].append(pred.cpu().numpy())
                results['probabilities'].append(proba.cpu().numpy())
        results = np.concatenate(results)
        results['preds'] = results['preds'].astype(int)
        Training.save_model_preds(preds=results)

    def calculate_test_metrics(self) -> None:

        '''
        Calculate performance metrics on the test set.
        '''

        file_name = 'resnet_' + str(self.version.strip('resnet')) + '_preds.npy'
        path = os.path.join(self.output_dir, 'model_preds', file_name)
        results = np.load(path, allow_pickle='TRUE')
        preds, proba = results['preds'], results['proba']
        labels = self.dataloaders['test'].dataset.labels
        acc = metrics.accuracy_score(labels, preds)
        prec = metrics.precision_score(labels, preds)
        recall = metrics.recall_score(labels, preds)
        fscore = metrics.f1_score(labels, preds)
        auc_score = metrics.roc_auc_score(labels, proba)
        avg_prec_score = metrics.average_precision_score(labels, proba)
        print('Performance Metrics on Test set:')
        print('Acc: {:4f}'.format(acc))
        print('Precision: {:4f}'.format(prec))
        print('Recall: {:4f}'.format(recall))
        print('F1-Score: {:4f}'.format(fscore))
        print('AUC: {:4f}'.format(auc_score))
        print('AP: {:4f}'.format(avg_prec_score))

    def visualize_training(self, phase: str, metric: str) -> None:

        '''
        Visualize the training and validation history.

        Args:
            phase (str): 'train' or 'val'.
            metric (str): 'loss' or 'acc'.
        '''

        if metric == 'loss':
            metric_label = 'Loss'
        elif metric == 'acc':
            metric_label = 'Accuracy'

        file_name = 'resnet_' + str(self.version.strip('resnet')) + '_hist.npy'
        plot_name = 'resnet_' + str(self.version.strip('resnet')) + '_' + phase + '_' + metric + '.png'
        history = np.load(os.path.join(self.output_dir, 'model_history', file_name), allow_pickle='TRUE').item()
        plt.plot(history[phase + '_' + metric])
        plt.ylabel(metric_label, fontsize=20, labelpad=10)
        plt.xlabel('Training Epoch', fontsize=20, labelpad=10)
        plt.legend(['Training', 'Validation'], loc='lower right')
        plt.savefig(os.path.join(self.output_dir, 'model_history', plot_name), dpi=300, bbox_inches="tight")
        plt.gray()

    def plot_test_metrics(self, metric: str) -> None:

        '''
        Plot the ROC curve and calculate the AUC score.

        Args:
            metric (str): 'auc' or 'ap'.
        '''

        file_name = 'resnet_' + str(self.version.strip('resnet')) + '_preds.npy'
        path = os.path.join(self.output_dir, 'model_preds', file_name)
        results = np.load(path, allow_pickle='TRUE')
        _, proba = results['preds'], results['proba']
        labels = self.dataloaders['test'].dataset.labels

        if metric == 'auc':
            metric_x, metric_y, _ = metrics.roc_curve(labels, proba)
            score = metrics.auc(metric_x, metric_y)
            metric_label = 'AUC'
            metric_x_label = 'False Positive Rate'
            metric_y_label = 'True Positive Rate'
        elif metric == 'ap':
            metric_x, metric_y, _ = metrics.precision_recall_curve(labels, proba)
            score = metrics.average_precision_score(labels, proba)
            metric_label = 'AP'
            metric_x_label = 'Precision'
            metric_y_label = 'Recall'

        plt.plot(metric_x, metric_y, color='dimgray', lw=2, label=metric_label + '= %0.2f' % score)
        plt.plot([0, 1], [0, 1], color='darkgray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel(metric_x_label, fontsize=20, labelpad=10)
        plt.ylabel(metric_y_label, fontsize=20, labelpad=10)
        plt.legend(loc="lower right")
        file_name = 'resnet_' + str(self.version.strip('resnet')) + '_' + metric + '.png'
        plt.savefig(os.path.join(self.output_dir, 'model_history', file_name), dpi=300, bbox_inches="tight")
        plt.gray()

        
RESULTS_DIR = '/Users/noltinho/thesis/results'
PATH = '/Users/noltinho/thesis_private/data'
COMPLETE_IMAGE_LIST = ['T1W_OOP','T1W_IP','T1W_DYN','T2W_TES','T2W_TEL','DWI_b0','DWI_b150','DWI_b400','DWI_b800']
WEIGHTS_PATH = '/Users/noltinho/MedicalNet/pytorch_files/pretrain'

if __name__ == '__main__':
    print(platform.platform())
    ReproducibilityUtils.seed_everything(123)
    prep = Preprocessing(PATH)
    observation_list = prep.assert_observation_completeness(COMPLETE_IMAGE_LIST)
    observation_list = np.random.choice(observation_list, 10, replace=False)
    labels = np.random.randint(2, size=561)
    dwi_b0_paths = prep.split_observations_by_modality(observation_list, 'DWI_b0')
    dwi_b150_paths = prep.split_observations_by_modality(observation_list, 'DWI_b150')
    data_dict = [{'DWI_b0': dwi_b0, 'DWI_b150': dwi_b150, 'label': label} for dwi_b0, dwi_b150, label in zip(dwi_b0_paths, dwi_b150_paths, labels)]
    train_transforms = prep.apply_transformations(['DWI_b0', 'DWI_b150'], 'train')
    val_transforms = prep.apply_transformations(['DWI_b0','DWI_b150'], 'val')
    dataset = {'train': monai.data.CacheDataset(data_dict[:5], train_transforms), 
               'val': monai.data.CacheDataset(data_dict[5:], val_transforms)}
    dataloader_dict = {x: monai.data.DataLoader(dataset[x], batch_size=2, shuffle=True, num_workers=4) for x in ['train', 'val']}
    resnet50 = ResNet(version='resnet50', num_out_classes=2, num_in_channels=2, pretrained=True, feature_extraction=True, weights_path=WEIGHTS_PATH)
    # device = torch.device('cuda')
    # training = Training(resnet50, device, dataloader_dict, RESULTS_DIR)
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(resnet50.parameters(), lr=0.001, momentum=0.9)
    # training.train_model(5, criterion, optimizer)