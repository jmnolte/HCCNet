import torch
import os
import numpy as np
from sklearn import metrics
from preprocessing import DataLoader
from models import ResNet, EnsembleModel
from utils import ReproducibilityUtils
from tqdm import tqdm
import argparse

class Inference():
    
    def __init__(
            self, 
            model: torch.nn.Module, 
            version: str, 
            dataloaders: dict, 
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
            assert any(version == version_item for version_item in ['resnet10','resnet18','resnet34','resnet50','resnet101','resnet152','resnet200','ensemble'])
        except AssertionError:
            print('Invalid version. Please choose from: resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200','ensemble')
            exit(1)

        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.model = model
        self.version = version
        self.dataloaders = dataloaders
        self.output_dir = output_dir
        model_dict = self.update_model_dict()
        self.model.load_state_dict(model_dict)
        print('Model weights are updated.')
        self.model = self.model.to(self.gpu_id)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu_id])

    def update_model_dict(
            self
            ) -> dict:

        '''
        Update the model dictionary with the weights from the best epoch.

        Returns:
            dict: Updated model dictionary.
        '''

        model_dict = self.model.state_dict()
        weights_dict = torch.load(os.path.join(self.output_dir, 'model_weights', self.version + '_weights.pth'))
        weights_dict = {k.replace('module.', ''): v for k, v in weights_dict.items()}
        model_dict.update(weights_dict)
        return model_dict

    def run_inference(
            self
            ) -> None:

        '''
        Run inference on the test set.
        '''

        self.model.eval()
        results = {'preds': [],'proba': [], 'labels': []}
        for batch_data in tqdm(self.dataloaders['test']):
            inputs, labels = batch_data['image'].to(self.gpu_id), batch_data['label'].to(self.gpu_id)
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

    def save_output(
            self, 
            output_dict: dict, 
            output_type: str
            ) -> None:

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

    def calculate_test_metrics(
            self
            ) -> None:

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
    parser.add_argument("--epochs", type=int, 
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
    Main function. The function loads the test data, loads the updated model weights, and runs inference 
    on the test set. It saves the models' predicted labels and performance metrics to the results directory.

    Args:
        args (argparse.Namespace): Arguments.
    '''
    # Set a seed for reproducibility.
    ReproducibilityUtils.seed_everything(args.seed)
    # Setup distributed processing.
    setup()
    # Load the test data.
    dataloader = DataLoader(args.data_dir, args.modality_list)
    data_dict = dataloader.create_data_dict()
    dataloader_dict = dataloader.load_data(data_dict, args.train_ratio, args.batch_size, 2, args.weighted_sampler, args.quant_images)
    # Load the model. If the model is an ensemble, load the individual models and create an ensemble model.
    if args.version == 'ensemble':
        versions = ['resnet10','resnet18','resnet34','resnet50']
        resnet10 = ResNet('resnet10', 2, len(args.modality_list), args.pretrained, args.feature_extraction, args.weights_dir)
        resnet18 = ResNet('resnet18', 2, len(args.modality_list), args.pretrained, args.feature_extraction, args.weights_dir)
        resnet34 = ResNet('resnet34', 2, len(args.modality_list), args.pretrained, args.feature_extraction, args.weights_dir)
        resnet50 = ResNet('resnet50', 2, len(args.modality_list), args.pretrained, args.feature_extraction, args.weights_dir)
        model = EnsembleModel(resnet10, resnet18, resnet34, resnet50, versions, 2, args.results_dir)
    else:
        model = ResNet(args.version, 2, len(args.modality_list), args.pretrained, args.feature_extraction, args.weights_dir)
    # Run inference on the test set and calculate performance metrics.
    inference = Inference(model, args.version, dataloader_dict, args.learning_rate, args.weight_decay, args.results_dir)
    inference.run_inference()
    inference.calculate_test_metrics()
    # Cleanup distributed processing.
    cleanup()
    print('Script finished')
    

RESULTS_DIR = '/Users/noltinho/thesis/results'
DATA_DIR = '/Users/noltinho/thesis_private/data'
MODALITY_LIST = ['T1W_OOP','T1W_IP','T1W_DYN','T2W_TES','T2W_TEL','DWI_b150','DWI_b400','DWI_b800']
WEIGHTS_DIR = '/Users/noltinho/MedicalNet/pytorch_files/pretrain'

if __name__ == '__main__':
    args = parse_args()
    main(args)

