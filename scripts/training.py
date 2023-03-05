import torch
import monai
import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from preprocessing import PreprocessingUtils, DataLoader
from models import ResNet
from utils import ReproducibilityUtils
from tqdm import tqdm

class Training:
    
    def __init__(self, model: torch.nn.Module, version: str, device: torch.device, dataloaders: dict, output_dir: str) -> None:

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

        self.model = model
        self.model.to(device)
        self.version = version
        self.device = device
        self.dataloaders = dataloaders
        self.output_dir = output_dir

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
            torch.save(self.model.state_dict(output_dict), folder_path)
        else:
            np.save(folder_path, output_dict)

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

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_loss = 10000.0
        counter = 0

        for epoch in range(0, max_epochs):
            print('Epoch {}'.format(epoch))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train() 
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                for batch_data in tqdm(self.dataloaders[phase]):
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
                    best_weights = copy.deepcopy(self.model.state_dict())
                elif phase == 'val' and epoch_loss >= best_loss:
                    counter += 1

            if early_stopping == True:
                if epoch > min_epochs - 1 and counter == patience:
                    break

        time_elapsed = time.time() - since
        self.save_output(best_weights, 'weights')
        self.save_output(history, 'history')

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

        weights_path = os.path.join(self.output_dir, 'model_weights', self.version + '_weights.pth')
        weights_dict = torch.load(weights_path)
        self.model.load_state_dict(weights_dict)
        self.model.eval()
        results = {'preds': [],'proba': [], 'labels': []}
        for batch_data in tqdm(self.dataloaders['test']):
            inputs, labels = batch_data['image'].to(self.device), batch_data['label'].to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                proba = torch.nn.functional.softmax(outputs, dim=1)[:,1]
                results['preds'].append(preds.cpu())
                results['proba'].append(proba.cpu())
                results['labels'].append(labels.cpu())
        results['preds'] = np.concatenate(results['preds'])
        results['proba'] = np.concatenate(results['proba'])
        results['labels'] = np.concatenate(results['labels'])
        self.save_output(results, 'preds')

    def calculate_test_metrics(self) -> None:

        '''
        Calculate performance metrics on the test set.
        '''

        path = os.path.join(self.output_dir, 'model_preds', self.version + '_preds.npy')
        results = np.load(path, allow_pickle='TRUE').item()
        preds, proba, labels = results['preds'], results['proba'][:,1], results['labels']
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
        try: 
            assert any(phase == phase_item for phase_item in ['train','val'])
            assert any(metric == metric_item for metric_item in ['loss','acc'])
        except AssertionError:
            print('Invalid input. Please choose phase from: train or val. Likewise, choose metric from: loss or acc.')
            exit(1)

        if metric == 'loss':
            metric_label = 'Loss'
        elif metric == 'acc':
            metric_label = 'Accuracy'

        file_name = self.version + '_hist.npy'
        plot_name = self.version + '_' + phase + '_' + metric + '.png'
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
        try:
            assert any(metric == metric_item for metric_item in ['auc','ap'])
        except AssertionError:
            print('Invalid input. Please choose metric from: auc or ap')
            exit(1)
        
        file_name = self.version + '_preds.npy'
        path = os.path.join(self.output_dir, 'model_preds', file_name)
        results = np.load(path, allow_pickle='TRUE').item()
        proba, labels = results['proba'][:,1], results['labels']

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
        file_name = self.version + '_' + metric + '.png'
        plt.savefig(os.path.join(self.output_dir, 'model_history', file_name), dpi=300, bbox_inches="tight")
        plt.gray()

    def get_random_test_image(self) -> torch.tensor:

        '''
        Get a random image from the test set.

        Returns:
            image (torch.tensor): The image.
            label (int): The label.
        '''
        test_data = next(iter(self.dataloaders['test']))
        return test_data['image'].to(self.device), test_data['label'].unsqueeze(0).to(self.device)
    
    def visualize_activations(self, batch_size: int) -> None:

        '''
        Get the occlusion sensitivity map for a given image.

        Args:
            test_image (torch.tensor): The image.
        '''
        image, label = self.get_random_test_image()
        weights_path = os.path.join(self.output_dir, 'model_weights', self.version + '_weights.pth')
        weights_dict = torch.load(weights_path)
        self.model.load_state_dict(weights_dict)
        self.model.eval()
        occ_sens = monai.visualize.OcclusionSensitivity(nn_module=self.model, n_batch=batch_size)
        z_slice = image.shape[-1] // 2
        occ_sens_b_box = [-1, -1, -1, -1, z_slice - 1, z_slice]

        occ_result, _ = occ_sens(image, b_box=occ_sens_b_box)
        occ_result = occ_result[0, label.argmax().item()][None]
        fig, axes = plt.subplots(1, 2, figsize=(25, 15), facecolor="white")

        for i, im in enumerate([image[:, :, z_slice, ...], occ_result]):
            cmap = "gray" if i == 0 else "jet"
            ax = axes[i]
            im_show = ax.imshow(np.squeeze(im[0][0].detach().cpu()), cmap=cmap)
            ax.axis("off")
            fig.colorbar(im_show, ax=ax)

        
RESULTS_DIR = '/Users/noltinho/thesis/results'
PATH = '/Users/noltinho/thesis_private/data'
COMPLETE_IMAGE_LIST = ['T1W_OOP','T1W_IP','T1W_DYN','T2W_TES','T2W_TEL','DWI_b0','DWI_b150','DWI_b400','DWI_b800']
WEIGHTS_PATH = '/Users/noltinho/MedicalNet/pytorch_files/pretrain'

if __name__ == '__main__':
    ReproducibilityUtils.seed_everything(123)
    prep = PreprocessingUtils(PATH)
    observation_list = prep.assert_observation_completeness(COMPLETE_IMAGE_LIST)
    observation_list = np.random.choice(observation_list, 20, replace=False)
    labels = np.random.randint(2, size=561)
    dwi_b0_paths = prep.split_observations_by_modality(observation_list, 'DWI_b0')
    dwi_b150_paths = prep.split_observations_by_modality(observation_list, 'DWI_b150')
    data_dict = [{'DWI_b0': dwi_b0, 'DWI_b150': dwi_b150, 'label': label} for dwi_b0, dwi_b150, label in zip(dwi_b0_paths, dwi_b150_paths, labels)]
    dataloader = DataLoader(image_list=['DWI_b0','DWI_b150'])
    dataloader_dict = dataloader.load_data(data_dict, split_ratio=[0.6, 0.2, 0.2], batch_size=1, num_workers=4)
    resnet50 = ResNet(version='resnet50', num_out_classes=2, num_in_channels=2, pretrained=True, feature_extraction=True, weights_path=WEIGHTS_PATH)
    device = torch.device('cpu')
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(resnet50.parameters(), lr=0.00001, momentum=0.9)
    train = Training(resnet50, 'resnet50', device, dataloader_dict, RESULTS_DIR)
    # train.train_model(1, criterion, optimizer)
    # train.run_inference()
    train.visualize_activations(1)

