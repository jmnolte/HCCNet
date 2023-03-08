import torch
import monai
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from preprocessing import PreprocessingUtils, DataLoader
from models import ResNet
from utils import ReproducibilityUtils
from training import Training
from tqdm import tqdm
import argparse

class Inference(Training):
    
    def __init__(self, model: torch.nn.Module, version: str, device: torch.device, dataloaders: dict, output_dir: str) -> None:
        super().__init__(model, version, device, dataloaders, output_dir)

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
        super().save_output(results, 'preds')

    def calculate_test_metrics(self) -> None:

        '''
        Calculate performance metrics on the test set.
        '''

        path = os.path.join(self.output_dir, 'model_preds', self.version + '_preds.npy')
        results = np.load(path, allow_pickle='TRUE').item()
        preds, proba, labels = results['preds'], results['proba'], results['labels']
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
    
    def plot_test_metrics(self, metric: str) -> None:

        '''
        Plot the ROC or PR curve on the test set.

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
        proba, labels = results['proba'], results['labels']

        if metric == 'auc':
            metric_x, metric_y, _ = metrics.roc_curve(labels, proba)
            score = metrics.roc_auc_score(labels, proba)
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
        file_path = os.path.join(self.output_dir, 'model_history/diagnostics', file_name)
        file_path_root, _ = os.path.split(file_path)
        if os.path.exists(file_path):
            os.remove(file_path)
        elif not os.path.exists(file_path_root):
            os.makedirs(file_path_root)
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()

    def get_random_image(self, phase: str) -> torch.tensor:

        '''
        Get a random image from the test set.

        Args:
            phase (str): 'train' or 'val'.

        Returns:
            image (torch.tensor): The image.
            label (int): The label.
        '''
        test_data = next(iter(self.dataloaders[phase]))
        return test_data['image'].to(self.device), test_data['label'].unsqueeze(0).to(self.device)
    
    def visualize_activations(self) -> None:

        '''
        Get the occlusion sensitivity map for a given image.
        '''
        image, label = self.get_random_image('test')
        weights_path = os.path.join(self.output_dir, 'model_weights', self.version + '_weights.pth')
        weights_dict = torch.load(weights_path)
        self.model.load_state_dict(weights_dict)
        self.model.eval()
        occ_sens = monai.visualize.OcclusionSensitivity(nn_module=self.model, n_batch=1)
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
DATA_DIR = '/Users/noltinho/thesis_private/data'
MODALITY_LIST = ['T1W_OOP','T1W_IP','T1W_DYN','T2W_TES','T2W_TEL','DWI_b0','DWI_b150','DWI_b400','DWI_b800']
WEIGHTS_DIR = '/Users/noltinho/MedicalNet/pytorch_files/pretrain'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version", required=True, type=str, help="Model version to train")
    parser.add_argument("-p", "--pretrained", default=True, type=bool, help="Flag to use pretrained weights")
    parser.add_argument("-fe", "--feature_extraction", default=True, type=bool, help="Flag to use feature extraction")
    parser.add_argument("-b", "--batch_size", default=4, type=int, help="Batch size to use for training")
    parser.add_argument("-ml", "--modality_list", default=MODALITY_LIST, nargs='+', help="List of modalities to use for training")
    parser.add_argument("-s", "--seed", default=123, type=int, help="Seed to use for reproducibility")
    parser.add_argument("-d", "--device", default='cuda', type=str, help="Device to use for training")
    parser.add_argument("-dd", "--data_dir", default=DATA_DIR, type=str, help="Path to data directory")
    parser.add_argument("-rd", "--results_dir", default=RESULTS_DIR, type=str, help="Path to results directory")
    parser.add_argument("-wd", "--weights_dir", default=WEIGHTS_DIR, type=str, help="Path to pretrained weights")
    args = vars(parser.parse_args())
    ReproducibilityUtils.seed_everything(args['seed'])
    prep = PreprocessingUtils(args['data_dir'])
    observation_list = prep.assert_observation_completeness(args['modality_list'])
    observation_list = np.random.choice(observation_list, 20, replace=False)
    path_dict = {modality: prep.split_observations_by_modality(observation_list, modality) for modality in args['modality_list']}
    path_dict['label'] = list(np.random.randint(2, size=20))
    data_dict = [dict(zip(path_dict.keys(), vals)) for vals in zip(*(path_dict[k] for k in path_dict.keys()))]
    dataloader = DataLoader(image_list=args['modality_list'])
    dataloader_dict = dataloader.load_data(data_dict, [0.6, 0.2, 0.2], args['batch_size'], 4)
    model = ResNet(args['version'], 2, len(args['modality_list']), args['pretrained'], args['feature_extraction'], args['weights_dir'])
    inference = Inference(model, args['version'], args['device'], dataloader_dict, args['results_dir'])
    inference.run_inference()
    if args['batch_size'] == 1:
        inference.visualize_activations()
    else:
        inference.calculate_test_metrics()
        inference.plot_test_metrics('auc')
        inference.plot_test_metrics('ap')

