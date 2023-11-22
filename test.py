import torch
import torch.nn as nn
import os
import numpy as np
from sklearn import metrics
from torch.cuda.amp import GradScaler, autocast
from preprocessing import (
    DatasetPreprocessor, 
    GroupStratifiedSplit,
    transformations,
    mil_transformations
)
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    SmartCacheDataset,
    DistributedSampler,
    DistributedWeightedRandomSampler,
    partition_dataset,
    partition_dataset_classes,
    set_track_meta,
    decollate_batch
)
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, SaveImage
from monai.networks.nets import SwinUNETR, ViT
from tqdm import tqdm
from models.milnet import MILNet
import argparse

class Inference():
    
    def __init__(
            self, 
            model: nn.Module, 
            task: str,
            version: str, 
            amp: bool,
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
        self.task = task
        self.model = model
        self.amp = amp
        self.version = version
        self.dataloaders = dataloaders
        self.output_dir = output_dir

    def run_inference(
            self,
            num_patches: int
            ) -> None:

        '''
        Run inference on the test set.
        '''

        results = {'preds': [],'proba': [], 'labels': []}
        post_pred = AsDiscrete(argmax=True, to_onehot=2)
        save_image = SaveImage(output_dir="./output", output_ext=".nii.gz", output_postfix="seg")
        self.model.eval()
        with torch.no_grad():
            for batch_data in tqdm(self.dataloaders['test']):

                if self.task == 'classification':
                    inputs, labels = batch_data['image'].to(self.gpu_id), batch_data['label'].to(self.gpu_id)
                    logits = self.model(inputs)
                    preds = torch.argmax(logits, 1)
                    proba = torch.nn.functional.softmax(logits, dim=1)[:,1]
                    results['preds'].append(preds.cpu())
                    results['proba'].append(proba.cpu())
                    results['labels'].append(labels.cpu())

                elif self.task == 'segmentation':
                    inputs = batch_data['image'].to(self.gpu_id)
                    logits = sliding_window_inference(inputs, (96, 96, 96), 4, self.model)
                    logits_list = decollate_batch(logits)
                    logits_convert = [post_pred(tensor) for tensor in logits_list]
                    for logit in logits_convert:
                        logit.meta['filename_or_obj'], _ = os.path.split(str(logit.meta['filename_or_obj']))
                        save_image(logit)

            if self.task == 'classification':
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
        prec = metrics.precision_score(labels, preds, average='macro')
        recall = metrics.recall_score(labels, preds, average='macro')
        fscore = metrics.f1_score(labels, preds, average='macro')
        # fscore_ind = metrics.f1_score(labels, preds, average=None)
        auc_score = metrics.roc_auc_score(labels, proba)
        avg_prec_score = metrics.average_precision_score(labels, proba)
        print('Performance Metrics on Test set:')
        print('Acc: {:4f}'.format(acc))
        print('Precision: {:4f}'.format(prec))
        print('Recall: {:4f}'.format(recall))
        print('F1-Score: {:4f}'.format(fscore))
        # print(f'F1-Score of Group 0: {fscore_ind[0]:4f}, F1-Score of Group 1: {fscore_ind[1]:4f}, and F1-Score of Group 2: {fscore_ind[2]:4f}')
        print('AUC: {:4f}'.format(auc_score))
        print('AP: {:4f}'.format(avg_prec_score))

def load_model_weights(
        model: nn.Module,
        backbone: str,
        weights_path: str
        ) -> nn.Module:

    '''
    Update the model dictionary with the weights from the best epoch.

    Returns:
        dict: Updated model dictionary.
    '''

    model_dict = model.state_dict()
    weights_dict = torch.load(os.path.join(weights_path, backbone + '_weights.pth'))
    weights_dict = {k.replace('module.', ''): v for k, v in weights_dict.items()}
    model_dict.update(weights_dict)
    model.load_state_dict(model_dict)
    print('Model weights have bee sucessfully updated.')
    return model

def parse_args() -> argparse.Namespace:

    '''
    Parse command line arguments.

    Returns:
        argparse.Namespace: Arguments.
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-classes", default=2, type=int,
                        help="Number of output classes. Defaults to 2.")
    parser.add_argument("--mil-mode", default='att', type=str,
                        help="MIL pooling mode. Can be mean, max, att, att_trans, and att_trans_pyramid. Defaults to att.")
    parser.add_argument("--backbone", type=str, 
                        help="Model encoder to use. Defaults to ResNet50.")
    parser.add_argument("--pretrained", action='store_true',
                        help="Flag to use pretrained weights")
    parser.add_argument("--distributed", action='store_true',
                        help="Flag to enable distributed training.")
    parser.add_argument("--amp", action='store_true',
                        help="Flag to enable automated mixed precision training.")
    parser.add_argument("--epochs", type=int, 
                        help="Number of epochs to train for")
    parser.add_argument("--batch-size", default=4, type=int, 
                        help="Batch size to use for training")
    parser.add_argument("--total-batch-size", default=32, type=int,
                        help="Total batch size: batch size x number of devices x number of accumulations steps.")
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
    parser.add_argument("--augment-prob", default=0.5, type=float,
                        help="Probability with which random transform is to be applied.")
    parser.add_argument("--mod-list", default=MODALITY_LIST, nargs='+', 
                        help="List of modalities to use for training")
    parser.add_argument("--num-patches", default=64, type=int,
                        help="Number of patches to use for training. Defaults to None.")
    parser.add_argument("--image-size", default=224, type=int,
                        help="Image size to use for training.")
    parser.add_argument("--seed", default=123, type=int, 
                        help="Seed to use for reproducibility")
    parser.add_argument("--weighted-sampler", action='store_true',
                        help="Flag to use a weighted sampler")
    parser.add_argument("--tl-strategy", default='finetune', type=str, 
                        help="Transfer learning strategy to use. Can be finetuning (FT), layer-wise finetuning (LWFT), or TransFusion (TF). Defaults to FT.")
    parser.add_argument("--shrink-coef", default=1, type=int, 
                        help="Factor by which the network architecture should be reduced. Only applies if TransFusion is selected as transfer learning strategy.")                   
    parser.add_argument("--cutoff-point", default=-1, type=int, 
                        help="Cutoff point from where layers are to be updated when layer-wise finetuning is selected or from where layers are randomly initialized when transfusion is selected.")
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
    set_determinism(seed=args.seed)
    # Setup distributed processing.
    if args.distributed:
        setup()
        rank = torch.distributed.get_rank()
        num_devices = torch.cuda.device_count()
        device_id = rank % num_devices
        learning_rate = args.learning_rate * np.sqrt(num_devices)
        accum_steps = args.total_batch_size / args.batch_size / num_devices
    # Load the test data.
    task = 'classification'
    multiclass = True if args.num_classes > 2 else False
    MOD_LIST = args.mod_list[:-1] if len(args.mod_list[:-1]) == 3 else 3 * args.mod_list[:-1]
    voxel_int = {x: [] for x in ['mean', 'std']}
    for entry in MOD_LIST:
        voxel_int['mean'].append(612.983 if 'T1WI' == entry else 0.3209 if 'T1A' == entry else 0.4042 if 'T1V' == entry else 0.4479 if 'T1D' == entry else 429.407 if 'T1W_OOP' == entry else 595.868 if 'T1W_IP' == entry else 0.3743 if 'T2W_TES' == entry else 441.226 if 'T2W_TEL' == entry else 0.374 if 'DWI_b0' == entry else 0.2066 if 'DWI_b150' == entry else 135.552 if 'DWI_b400' == entry else 0.132)
        voxel_int['std'].append(308.69 if 'T1WI' == entry else 0.2275 if 'T1A' == entry else 0.2732 if 'T1V' == entry else 0.2675 if 'T1D' == entry else 328.031 if 'T1W_OOP' == entry else 373.299 if 'T1W_IP' == entry else 0.2658 if 'T2W_TES' == entry else 424.472 if 'T2W_TEL' == entry else 0.328 if 'DWI_b0' == entry else 0.2114 if 'DWI_b150' == entry else 230.139 if 'DWI_b400' == entry else 0.177)
    split_dict, label_df = DatasetPreprocessor(data_dir=args.data_dir, test_run=False).load_imaging_data(modalities=args.mod_list, task=task, multiclass=multiclass)
    if task == 'classification':
        train, val_test = GroupStratifiedSplit(split_ratio=args.train_ratio, multiclass=multiclass).split_dataset(label_df)
        test, val = GroupStratifiedSplit(split_ratio=0.6, multiclass=multiclass).split_dataset(val_test)
        split_dict = GroupStratifiedSplit(multiclass=multiclass).convert_to_dict(train, val, test, split_dict)
    datasets = {x: CacheDataset(
        data=split_dict[x], 
        transform=mil_transformations(
            dataset=x, 
            modalities=args.mod_list, 
            image_size=args.image_size, 
            num_patches=args.num_patches,
            voxel_int=voxel_int,
            device=device_id), 
        num_workers=args.batch_size,
        copy_cache=False) for x in ['test']}
    dataloader = {x: ThreadDataLoader(
        datasets[x], 
        batch_size=1, 
        shuffle=False, 
        num_workers=0) for x in ['test']}
    # Load the model. If the model is an ensemble, load the individual models and create an ensemble model.
    model = MILNet(
        num_classes=args.num_classes, 
        mil_mode=args.mil_mode, 
        backbone=args.backbone, 
        pretrained=args.pretrained, 
        tl_strategy=args.tl_strategy, 
        shrink_coefficient=args.shrink_coef,
        load_up_to=args.cutoff_point)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = SwinUNETR(
    #     img_size=(96, 96, 96),
    #     in_channels=1,
    #     out_channels=2,
    #     feature_size=48,
    #     drop_rate=0.0,
    #     attn_drop_rate=0.0,
    #     dropout_path_rate=0.0)
    model = load_model_weights(model, args.backbone, '/home/x3007104/thesis/results/model_weights')
    model = model.to(device_id)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id])
    # Run inference on the test set and calculate performance metrics.
    inference = Inference(model, task, args.backbone, False, dataloader, args.results_dir)
    inference.run_inference(num_patches=64)
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

