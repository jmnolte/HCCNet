import torch
import monai
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from preprocessing import DataLoader
from models import ResNet, EnsembleModel
from utils import ReproducibilityUtils
import argparse
import seaborn as sns

class Diagnostics():
    
    def __init__(
            self, 
            model: torch.nn.Module, 
            version: str, 
            dataloaders: dict, 
            output_dir: str
            ) -> None:

        '''
        Initialize the diagnostics class.

        Args:
            model (torch.nn): Model to train.
            version (str): Model version. Can be 'resnet10', 'resnet18', 'resnet34', 'resnet50',
                'resnet101', 'resnet152', or 'resnet200'.
            dataloaders (dict): Dataloader objects.
            output_dir (str): Output directory.
        '''
        try: 
            assert any(version == version_item for version_item in ['resnet10','resnet18','resnet34','resnet50','resnet101','resnet152','resnet200','ensemble'])
        except AssertionError:
            print('Invalid version. Please choose from: resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200, ensemble')
            exit(1)
        
        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.model = model
        self.version = version
        self.dataloaders = dataloaders
        self.output_dir = output_dir
        model_dict = self.model.state_dict()
        weights_dict = torch.load(os.path.join(self.output_dir, 'model_weights', self.version + '_weights.pth'))
        weights_dict = {k.replace('module.', ''): v for k, v in weights_dict.items()}
        model_dict.update(weights_dict)
        self.model.load_state_dict(model_dict)
        print('Model weights are updated.')
        self.model = self.model.to(self.gpu_id)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu_id])
    
    def plot_test_metrics(
            self, 
            metric: str
            ) -> None:

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
        
        for file_name in sorted(os.listdir(os.path.join(self.output_dir, 'model_preds'))):
            model = file_name.split('_')[0]
            path = os.path.join(self.output_dir, 'model_preds', file_name)
            results = np.load(path, allow_pickle='TRUE').item()
            proba, labels = results['proba'], results['labels']

            if metric == 'auc':
                metric_x, metric_y, _ = metrics.roc_curve(labels, proba)
                score = metrics.roc_auc_score(labels, proba)
                metric_label = 'AUROC'
                metric_x_label = 'False Positive Rate'
                metric_y_label = 'True Positive Rate'
            elif metric == 'ap':
                metric_x, metric_y, _ = metrics.precision_recall_curve(labels, proba)
                score = metrics.average_precision_score(labels, proba)
                metric_label = 'mAP'
                metric_x_label = 'Precision'
                metric_y_label = 'Recall'

            with sns.color_palette("husl", n_colors=len(os.listdir(os.path.join(self.output_dir, 'model_preds')))):
                plt.plot(metric_x, metric_y, lw=2, label=model + ', ' + metric_label + '= %0.2f' % score)
        plt.xlabel(metric_x_label, fontsize=20, labelpad=10)
        plt.ylabel(metric_y_label, fontsize=20, labelpad=10)
        plt.legend(loc="lower right")
        file_path = os.path.join(self.output_dir, 'model_history/diagnostics', metric)
        file_path_root, _ = os.path.split(file_path)
        if os.path.exists(file_path):
            os.remove(file_path)
        elif not os.path.exists(file_path_root):
            os.makedirs(file_path_root)
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()

    def get_random_image(
            self, 
            phase: str, 
            positive: bool
            ) -> torch.tensor:

        '''
        Get a random image from the test set.

        Args:
            phase (str): 'train' or 'val'.
            positive (bool): Whether to get a positive or negative image.

        Returns:
            image (torch.tensor): The image.
            label (int): The label.
        '''
        for batch_data in self.dataloaders[phase]:
            inputs, labels = batch_data['image'], batch_data['label']
            if positive:
                if labels.unsqueeze(0) == 1:
                    break
                else:
                    continue
            else:
                if labels.unsqueeze(0) == 0:
                    break
                else:
                    continue
        return inputs.to(self.gpu_id), labels.unsqueeze(0).to(self.gpu_id)
    
    def visualize_activations(
            self, 
            positive: bool
            ) -> None:

        '''
        Get the occlusion sensitivity map for a given image.

        Args:
            positive (bool): Whether to get a positive or negative image.
        '''
        if positive:
            print('Getting positive image...')
            file_suffix = '_hcc'
        else:
            print('Getting negative image...')
            file_suffix = '_nohcc'
        image, label = self.get_random_image('test', positive)
        print('Image loaded')
        self.model.eval()

        occ_sens = monai.visualize.OcclusionSensitivity(nn_module=self.model, n_batch=1)
        depth_slice = image.shape[2] // 2
        occ_sens_b_box = [depth_slice - 1, depth_slice, -1, -1, -1, -1]
        occ_result, _ = occ_sens(x=image, b_box=occ_sens_b_box)
        occ_result = occ_result[0, label.argmax().item()][None]

        for idx, im in enumerate([image[:, :, depth_slice, ...], occ_result]):
            plt.imshow(np.squeeze(im[0][0].detach().cpu()), cmap='jet')
            plt.axis('off')
        
        file_path = os.path.join(self.output_dir, 'model_history/diagnostics', self.version + '_occ_sens' + file_suffix)
        file_path_root, _ = os.path.split(file_path)
        if os.path.exists(file_path):
            os.remove(file_path)
        elif not os.path.exists(file_path_root):
            os.makedirs(file_path_root)
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()

        image = np.squeeze(image[0][0].detach().cpu())
        plt.imshow(image[:, :, depth_slice], cmap='gray')
        plt.axis('off')
        file_path = os.path.join(self.output_dir, 'model_history/diagnostics', self.version + '_orig_img' + file_suffix)
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
    parser.add_argument("--pretrained", action='store_true',
                        help="Flag to use pretrained weights")
    parser.add_argument("--feature-extraction", action='store_true',
                        help="Flag to use feature extraction")
    parser.add_argument("--occ-sens", action='store_true',
                        help="Flag to use occlusion sensitivity")
    parser.add_argument("--positive", action='store_true',
                        help="Flag to use positive images")
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
    Main function. The function saves receiver operating characteristic (ROC) and precision-recall (PR) 
    curves for the models to the results directorcy. Additionally, the function saves occlusion 
    sensitivity maps of the respective model architectures for a positive and a negative image to the
    results directory.

    Args:
        args (argparse.Namespace): Arguments.
    '''
    # Set a seed for reproducibility.
    ReproducibilityUtils.seed_everything(args.seed)
    # Setup distributed processing.
    setup()
    if args.occ_sens:
        batch_size = 1
    else:
        batch_size = args.batch_size
    # Load the test data and create a data dictionary.
    dataloader = DataLoader(args.data_dir, args.modality_list)
    data_dict = dataloader.create_data_dict()
    dataloader_dict = dataloader.load_data(data_dict, args.train_ratio, batch_size, 2, args.weighted_sampler, args.quant_images)
    # Load the model. If the model is an ensemble model, load the individual models and create an ensemble model.
    if args.version == 'ensemble':
        versions = ['resnet10','resnet18','resnet34','resnet50']
        resnet10 = ResNet('resnet10', 2, len(args.modality_list), args.pretrained, args.feature_extraction, args.weights_dir)
        resnet18 = ResNet('resnet18', 2, len(args.modality_list), args.pretrained, args.feature_extraction, args.weights_dir)
        resnet34 = ResNet('resnet34', 2, len(args.modality_list), args.pretrained, args.feature_extraction, args.weights_dir)
        resnet50 = ResNet('resnet50', 2, len(args.modality_list), args.pretrained, args.feature_extraction, args.weights_dir)
        model = EnsembleModel(resnet10, resnet18, resnet34, resnet50, versions, 2, args.results_dir)
    else:
        model = ResNet(args.version, 2, len(args.modality_list), args.pretrained, args.feature_extraction, args.weights_dir)
    # Return occlusion sensitivity maps and save ROC and PR curves.
    inference = Diagnostics(model, args.version, dataloader_dict, args.results_dir)
    inference.visualize_activations(True)
    inference.visualize_activations(False)
    inference.plot_test_metrics('auc')
    inference.plot_test_metrics('ap')
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