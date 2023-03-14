import torch
import os
import numpy as np
from sklearn import metrics
from preprocessing import DataLoader
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
        self.model.load_state_dict(torch.load(weights_path))
        print('Model weights are updated.')
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
    

RESULTS_DIR = '/Users/noltinho/thesis/results'
DATA_DIR = '/Users/noltinho/thesis_private/data'
MODALITY_LIST = ['T1W_OOP','T1W_IP','T1W_DYN','T2W_TES','T2W_TEL','DWI_b0','DWI_b150','DWI_b400','DWI_b800']
WEIGHTS_DIR = '/Users/noltinho/MedicalNet/pytorch_files/pretrain'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version", required=True, type=str, help="Model version to train")
    parser.add_argument("-p", "--pretrained", default=True, type=bool, help="Flag to use pretrained weights")
    parser.add_argument("-fe", "--feature_extraction", default=True, type=bool, help="Flag to use feature extraction")
    parser.add_argument("-b", "--batch_size", default=4, type=int, help="Batch size to use for inference")
    parser.add_argument("-ml", "--modality_list", default=MODALITY_LIST, nargs='+', help="List of modalities to use for inference")
    parser.add_argument("-s", "--seed", default=123, type=int, help="Seed to use for reproducibility")
    parser.add_argument("-d", "--device", default='cuda', type=str, help="Device to use for inference")
    parser.add_argument("-ts", "--test_set", default=True, type=bool, help="Flag to load test or training and validation set")
    parser.add_argument("-dd", "--data_dir", default=DATA_DIR, type=str, help="Path to data directory")
    parser.add_argument("-rd", "--results_dir", default=RESULTS_DIR, type=str, help="Path to results directory")
    parser.add_argument("-wd", "--weights_dir", default=WEIGHTS_DIR, type=str, help="Path to pretrained weights")
    args = vars(parser.parse_args())
    torch.multiprocessing.set_sharing_strategy('file_system')
    ReproducibilityUtils.seed_everything(args['seed'])
    dataloader = DataLoader(args['data_dir'], args['modality_list'])
    labels = np.random.randint(2, size=20)
    data_dict = dataloader.create_data_dict(labels)
    dataloader_dict = dataloader.load_data(data_dict, [0.6, 0.2, 0.2], args['batch_size'], 2, args['test_set'])
    model = ResNet(args['version'], 2, len(args['modality_list']), args['pretrained'], args['feature_extraction'], args['weights_dir'])
    inference = Inference(model, args['version'], args['device'], dataloader_dict, args['results_dir'])
    inference.run_inference()
    inference.calculate_test_metrics()

