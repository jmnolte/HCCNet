from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    CropForegroundd,
    Resized,
    ConcatItemsd,
    ScaleIntensityRangePercentilesd,
    NormalizeIntensityd,
    GridPatchd,
    SqueezeDimd,
    CenterSpatialCropd,
    RandSpatialCropd,
    RandGridPatchd,
    RandFlipd,
    RandRotated,
    RandZoomd,
    RandGibbsNoised,
    EnsureTyped,
    Lambdad,
    RandKSpaceSpikeNoised
)
from monai import transforms
from monai.data import (
    DataLoader,
    CacheDataset,
    DistributedSampler
)
import torch.distributed as dist
import numpy as np
import glob
import os
from tqdm import tqdm
from natsort import natsorted
from glob import glob
import dicom2nifti
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from utils import ReproducibilityUtils
import argparse
import matplotlib.pyplot as plt


class Dicom2NiftiConverter:

    def __init__(
            self, 
            data_dir: str
            ) -> None:
        '''
        Initialize the data utils class.

        Args:
            path (str): Path to the data.
        '''

        self.data_dir = data_dir
        self.dicom_dir = os.path.join(data_dir, 'dicom')
        self.nifti_dir = os.path.join(data_dir, 'nifti')
        self.dicom_patients = glob(os.path.join(self.dicom_dir, '*'), recursive = True)
        self.nifti_patients = glob(os.path.join(self.nifti_dir, '*'), recursive = True)

    def get_dicom_observations(
            self
            ) -> list:
        '''
        Get list of existing observations.

        Returns:
            observation_list (list): List of observations.
        '''

        observation_list = []
        for patient_path in tqdm(natsorted(self.dicom_patients)):
            _, patient_id = os.path.split(patient_path)
            observations = natsorted(glob(os.path.join(patient_path, 'DICOM/*'), recursive = True))
            for entry in list(zip([patient_id]*len(observations), [observation.split('/')[-1] for observation in observations])):
                observation_list.append(entry)
        return observation_list
    
    def convert_dicom_to_nifti(
            self, 
            labels_dict: dict,
            observation_list: list
            ) -> None:
        '''
        Convert dicom files into nifti file format.

        Args:
            labels_dict (dict): Dictionary of labels.
            observation_list (list): List of observations.
        '''

        for entry in tqdm(observation_list):
            if entry in natsorted(list(zip(labels_dict['id'], labels_dict['date']))):
                index = natsorted(list(zip(labels_dict['id'], labels_dict['date']))).index(entry)
                patient_id, date, observation_id = natsorted(list(zip(labels_dict['id'], labels_dict['date'], labels_dict['observation'])))[index]
                if observation_id < 10:
                    observation_id = '00' + str(observation_id)
                else:
                    observation_id = '0' + str(observation_id)
                for series in natsorted(glob(os.path.join(self.dicom_dir, patient_id, 'DICOM', date, '*/*'))):
                    folder_name = patient_id + '_' + observation_id
                    file_name = os.path.split(series)[-1] + '.nii.gz'
                    if not os.path.exists(os.path.join(self.nifti_dir, folder_name)):
                        os.makedirs(os.path.join(self.nifti_dir, folder_name))
                    if not os.path.exists(os.path.join(self.nifti_dir, folder_name, file_name)):
                        try:
                            dicom2nifti.dicom_series_to_nifti(series, os.path.join(self.nifti_dir, folder_name, file_name))
                        except:
                            print('Error converting DICOM series {} to nifti {}.'.format(series, os.path.join(self.nifti_dir, folder_name, file_name)))
            else:
                print('File {} not found.'.format(entry))

    def __call__(
            self,
            ) -> None:
        '''
        Call the convert dicom to nifti function.
        '''

        labels_df = os.path.join(self.data_dir, 'labels/labels.csv')
        try:
            assert os.path.exists(labels_df)
        except AssertionError:
            print('Labels file not found.')
            exit(1)
        labels_dict = pd.read_csv(labels_df).to_dict('list')
        labels_dict['date'] = [str(date) for date in labels_dict['date']]
        observation_list = self.get_dicom_observations()
        self.convert_dicom_to_nifti(labels_dict, observation_list)


class DirectoryCleaner:

    def __init__(
            self, 
            data_dir: str,
            modality_list: list
            ) -> None:

        '''
        Initialize the directory cleaner class.

        Args:
            path (str): Path to the data.
            modality_list (list): List of image modalities to keep.
        '''
        self.nifti_dir = os.path.join(data_dir, 'nifti')
        self.dicom_dir = os.path.join(data_dir, 'dicom')
        self.nifti_patients = glob(os.path.join(self.nifti_dir, '*'), recursive = True)
        self.dicom_patients = glob(os.path.join(self.dicom_dir, '*'), recursive = True)
        self.modality_list = modality_list

    def extract_image_names(
            self
            ) -> list:

        '''
        Extract image names from the nifti files.

        Returns:
            image_list (list): List of images.
        '''
        image_list = []
        for observation in tqdm(natsorted(self.nifti_patients)):
            images = glob(os.path.join(observation, '*.nii.gz'))
            for image in images:
                _, image_head = os.path.split(image)
                image_list.append(image_head)
        return image_list

    def clean_directory(
            self, 
            ) -> None:

        '''
        Remove non-convertable images from the directory.
        '''
        for observation in tqdm(natsorted(self.nifti_patients)):
            images = glob(os.path.join(observation, '*.nii.gz'))
            for image in images:
                image_head = os.path.split(image)[-1]
                modality = image_head.split('.')[0]
                if modality not in self.modality_list:
                    os.remove(image)
                    print('Deleting {}.'.format(image))



    # def create_label_list(
    #         self, 
    #         df_path: str
    #         ) -> list:

    #     '''
    #     Create a list of labels.

    #     Args:
    #         df_path (str): Path to the dataframe of labels.

    #     Returns:
    #         label_list (list): List of labels.
    #     '''
    #     path_root, _ = os.path.split(df_path)
    #     labels_df = pd.read_csv(df_path, sep=';')
    #     labels_df = labels_df.loc[:,~labels_df.columns.str.contains('other', case=False)]
    #     labels_df = labels_df.loc[:,~labels_df.columns.str.contains('CT', case=False)]
    #     labels_df = labels_df.loc[:,~labels_df.columns.str.contains('comments', case=False)]
    #     labels_df = labels_df.loc[:,~labels_df.columns.str.contains('Unnamed', case=False)]
    #     labels_df.columns = labels_df.columns.str.strip().str.lower()
    #     labels_df.columns = ['_'.join(x.split('_')[::-1]) for x in labels_df.columns]
    #     labels_df.columns = labels_df.columns.str.replace('mri','')
    #     labels_df = labels_df.rename({'nr_study': 'id'}, axis=1)
    #     labels_df = pd.wide_to_long(labels_df, stubnames=['date', 'outcome'], 
    #                 i='id', j='observation', suffix='.*', sep='_').reset_index()
    #     labels_df['outcome'] = labels_df['outcome'].replace(['1+3','2+3'], '3')
    #     labels_df = labels_df._convert(numeric=True)
    #     labels_df['outcome'] = np.where(labels_df['outcome'] != 3, 0, 1)
    #     labels_df['date'] = pd.to_datetime(labels_df['date'], errors='coerce').dt.strftime('%Y%m%d')
    #     labels_df = labels_df.dropna(subset=['date'])
    #     return labels_df.to_csv(os.path.join(path_root, 'labels.csv'), encoding='utf-8', index=False)


class ModalityCheck:

    def __init__(
            self, 
            data_dir: str
            ) -> None:

        '''
        Initialize the modality check class.

        Args:
            data_dir (str): Path to the data.
            modalities (list): List of modalities to check.
        '''
        self.nifti_dir = os.path.join(data_dir, 'nifti')
        self.nifti_patients = glob(os.path.join(self.nifti_dir, '*'), recursive = True)
        self.label_dir = os.path.join(data_dir, 'labels')

    def assert_observation_completeness(
            self,
            modalities: list,
            verbose: bool = True
            ) -> list:

        '''
        Assert that all observations include all required images.

        Args:
            verbose (bool): Print the number of observations that include all required images.

        Returns:
            observation_list (list): List of observations that include all required images.
        '''
        image_names_list = [image_name + '.nii.gz' for image_name in modalities]
        observation_list = []
        for observation in natsorted(self.nifti_patients):
            images = [image for image in glob(os.path.join(observation, '*.nii.gz'))]
            common_prefix = os.path.commonprefix(images)
            image_names = [image_name[len(common_prefix):] for image_name in images]
            if all(names in image_names for names in image_names_list):
                observation_list.append(observation)
        if verbose:
            print('{} out of {} observations include all required images'.format(len(observation_list), len(self.nifti_patients)))
        return observation_list
    

class GroupStratifiedSplit(GroupShuffleSplit):

    def __init__(
            self,
            split_ratio: float = 0.8,
            n_splits: int = 100,
            multiclass: bool = False
        ) -> None:
        super().__init__(
            train_size=split_ratio, 
            n_splits=n_splits
        )
        
        '''
        Initialize the group stratified split class.

        Args:
            test_ratio (float): Ratio of the data to be used for testing.
            n_splits (int): Number of splits to perform.
        '''
        self.test_ratio = 1 - split_ratio
        self.multiclass = multiclass

    def select_best_split(
            self,
            dataframe: pd.DataFrame,
            splits: list,
            ) -> pd.DataFrame:

        '''
        Split the data into training, validation and test sets.

        Args:
            dataframe (pd.DataFrame): Dataframe to split.
            splits (list): List of splits.

        Returns:
            dataframes (pd.DataFrame): Dataframes split according to the test ratio.
        '''
        label_ratio = 1
        for split in splits:
            set1, set2 = dataframe.iloc[split[0]], dataframe.iloc[split[1]]
            if abs(set1['label'].mean() - set2['label'].mean()) < label_ratio:
                if len(set1) / (1 - self.test_ratio) > len(set2) / self.test_ratio * 0.95 and len(set1) / (1 - self.test_ratio) < len(set1) / self.test_ratio * 1.05:
                    label_ratio = abs(set1['label'].mean() - set2['label'].mean())
                    best_split = split
        return best_split
    
    def split_dataset(
            self,
            dataframe: pd.DataFrame,
            ) -> pd.DataFrame:
        
        '''
        Call the group stratified split class.

        Args:
            dataframe (pd.DataFrame): Dataframe to split.
            group_id (str): Column name of the group id.

        Returns:
            dataframes (pd.DataFrame): Dataframes split according to the test ratio.
        '''
        if self.multiclass:
            dataframe['prevlabel'] = np.where(dataframe['label'] == 2, 1, 0)
            dataframe['label'] = np.where(dataframe['label'] == 2, 0, dataframe['label'])
        splits = super().split(dataframe, groups=dataframe['patient_id'])
        split1, split2 = self.select_best_split(dataframe, splits)
        if self.multiclass:
            dataframe['label'] = np.where(dataframe['prevlabel'] == 1, 2, dataframe['label'])
            dataframe = dataframe.drop(columns=['prevlabel'])
        return dataframe.iloc[split1], dataframe.iloc[split2]
    
    def convert_to_dict(
            self,
            train: pd.DataFrame,
            val: pd.DataFrame,
            test: pd.DataFrame,
            data_dict: dict
            ) -> dict:
        
        '''
        Convert the dataframes to dictionaries.

        Args:
            train (pd.DataFrame): Training dataframe.
            val (pd.DataFrame): Validation dataframe.
            test (pd.DataFrame): Test dataframe.
            data_dict (dict): Dictionary of data.
        
        Returns:
            split_dict (dict): Dictionary of split data.
        '''
        for idx, df in enumerate([train, val, test]):
            name = 'training' if idx == 0 else 'validation' if idx == 1 else 'test'
            if not self.multiclass:
                print('{} total observations in {} set with {} positive cases ({} %).'.format(len(df), name, df['label'].sum(), round(df['label'].mean(), ndigits=3)))
            else:
                print('{} total observations in {} set with {} diagnosed HCC ({} %) and {} developing HCC cases ({} %).'.format(len(df), name, df['label'].value_counts()[1], round(df['label'].value_counts()[1] / len(df), ndigits=3), df['label'].value_counts()[2], round(df['label'].value_counts()[2] / len(df), ndigits=3)))
        split_dict = {'train': train[['uid']].values, 'val': val[['uid']].values, 'test': test[['uid']].values}
        split_dict = {x: [patient for patient in data_dict if patient['uid'] in split_dict[x]] for x in ['train', 'val', 'test']} 

        for phase in split_dict:
            for patient in split_dict[phase]:
                del patient['uid']
        return split_dict
        

class DatasetPreprocessor(ModalityCheck):

    def __init__(
            self,
            data_dir: str,
            test_run: bool = False
        ) -> dict:
        super().__init__(
            data_dir=data_dir
        )
        
        '''
        Initialize the dataset preprocessor class.

        Args:
            data_dir (str): Path to the data.
            test_run (bool): Run the code on a small subset of the data.

        Returns:
            data_dict (dict): Dictionary containing the paths to the images and corresponding labels.
        '''
        self.nifti_dir = os.path.join(data_dir, 'nifti')
        self.nifti_patients = glob(os.path.join(self.nifti_dir, '*'), recursive = True)
        if test_run:
            self.nifti_patients = self.nifti_patients[100:250]
        self.label_dir = os.path.join(data_dir, 'labels')

    @staticmethod
    def split_observations_by_modality(
            observation_list: list, 
            modality: str
            ) -> list:

        '''
        Create a dictionary of paths to images.

        Args:
            observation_list (list): List of observations to split.
            modality (str): Modality to split by.

        Returns:
            image_paths (list): List of paths to images.
        '''
        return [glob(os.path.join(observation, modality + '.nii.gz')) for observation in observation_list]
    
    @staticmethod
    def create_label_dict(
            observation_list: list, 
            label_path: str
            ) -> dict:

        '''
        Extract the labels from the dataframe.

        Args:
            observation_list (list): List of observations to create labels for.
            label_path (str): Path to the dataframe containing the labels.

        Returns:
            label_dict (dict): Dictionary containing the labels.
        '''
        label_dict = {'uid': [], 'label': []}
        labels_df = pd.read_csv(label_path)
        for idx, row in labels_df.iterrows():
            if labels_df.loc[idx, 'observation'] < 10:
                labels_df.loc[idx, 'uid'] = row['id'] + '_00' + str(row['observation'])
            else:
                labels_df.loc[idx, 'uid'] = row['id'] + '_0' + str(row['observation'])
        for observation in observation_list:
            observation_id = os.path.basename(observation)
            if observation_id not in labels_df['uid'].values:
                continue
            else:
                label = labels_df.loc[labels_df['uid'] == observation_id, 'outcome'].values.item()
            label_dict['label'].append(label)
            label_dict['uid'].append(observation_id)
        return [dict(zip(label_dict.keys(), vals)) for vals in zip(*(label_dict[k] for k in label_dict.keys()))]
    
    @staticmethod
    def create_data_dict(
            observation_list: list,
            input_dict: dict
            ) -> dict:
        
        '''
        Transpose the dictionary.

        Args:
            observation_list (list): List of observations to create labels for.
            input_dict (dict): Dictionary containing the paths to the images.
        
        Returns:
            data_dict (dict): Dictionary containing the paths to the images.
        '''
        input_dict['uid'] = [os.path.basename(observation) for observation in observation_list]
        return [dict(zip(input_dict.keys(), vals)) for vals in zip(*(input_dict[k] for k in input_dict.keys()))]
    
    def load_imaging_data(
            self,
            modalities: list,
            label_column: str = 'label',
            multiclass: bool = False,
            verbose: bool = True
            ) -> list:
        
        '''
        Call the dataset preprocessor class.

        Args:
            modality_list (list): List of modalities to load.
            label_column (str): Column name of the label.
            verbose (bool): Print statement indicating the number of observations that include all required images.

        Returns:
            data_dict (dict): Dictionary containing the paths to the images and corresponding labels.
        '''
        observation_list = super().assert_observation_completeness(modalities, verbose)
        modality_dict = {modality: self.split_observations_by_modality(observation_list, modality) for modality in modalities}
        if not multiclass:
            label_dict = self.create_label_dict(observation_list, os.path.join(self.label_dir, 'labels.csv'))
        else:
            label_dict = self.create_label_dict(observation_list, os.path.join(self.label_dir, 'labels_mc.csv'))
        label_df = pd.DataFrame.from_dict(label_dict)
        label_df['patient_id'] = label_df['uid'].str.split('_').str[1]
        data_dict = self.create_data_dict(observation_list, modality_dict)
        for idx, patient in enumerate(data_dict):
            if patient['uid'] in [label['uid'] for label in label_dict]:
                data_dict[idx][label_column] = label_dict[[label['uid'] for label in label_dict].index(patient['uid'])][label_column]
            else:
                data_dict[idx][label_column] = None
        data_dict = [patient for patient in data_dict if not (patient[label_column] == None)]
        return data_dict, label_df

class NoneTransform(object):
    """ Does nothing to the image, to be used instead of None
    
    Args:
        image in, image out, nothing is done
    """
    def __call__(self, image):       
        return image


def transformations(
        dataset: str,
        modalities: list,
        num_patches: int,
        image_size: int,
        random_prob: float,
        device
        ) -> transforms:
    '''
    Perform data transformations on image and image labels.

    Args:
        dataset (str): Dataset to apply transformations on. Can be 'train', 'val' or 'test'.

    Returns:
        transforms (monai.transforms): Data transformations to be applied.
    '''
    resized_image = int(image_size * 1.25)
    # min_offset = None if random_prob == 0 else (0, 0, 0)
    # max_offset = None if random_prob == 0 else (16, 16, 0)
    preprocessing = Compose([
        LoadImaged(keys=modalities, image_only=False),
        EnsureChannelFirstd(keys=modalities),
        Orientationd(keys=modalities, axcodes='PLI'),
        Resized(keys=modalities, spatial_size=(resized_image, resized_image, 64)),          
        Lambdad(
            keys=modalities, 
            func=lambda x: x.repeat(3, 1, 1, 1)
            ) if len(modalities) == 1 else NoneTransform(),
        ConcatItemsd(keys=modalities, name='image', dim=0),
        ScaleIntensityRangePercentilesd(
            keys='image', 
            lower=1, 
            upper=99, 
            b_min=0, 
            b_max=1, 
            clip=True, 
            channel_wise=True)
    ])
        
    train_specific = Compose([
        EnsureTyped(keys=['image', 'label'], device=device, track_meta=False),
        RandGridPatchd(
            keys='image', 
            patch_size=(resized_image, resized_image, 1), 
            sort_fn='random', 
            num_patches=num_patches, 
            pad_mode='constant', 
            constant_values=0),
        SqueezeDimd(keys='image', dim=-1),
        RandRotated(keys='image', prob=random_prob, range_x=np.pi/8),
        RandFlipd(keys='image', prob=random_prob, spatial_axis=1),
        RandFlipd(keys='image', prob=random_prob, spatial_axis=2),
        RandZoomd(
            keys='image', 
            prob=random_prob, 
            min_zoom=(1, 0.9, 0.9), 
            max_zoom=(1, 1.1, 1.1),
            padding_mode='constant', 
            constant_values=0),
        RandSpatialCropd(keys='image', roi_size=(num_patches, image_size, image_size), random_size=False),
        NormalizeIntensityd(keys='image', channel_wise=True),
        RandGibbsNoised(keys='image', prob=random_prob, alpha=(0.1, 0.8)),
        RandKSpaceSpikeNoised("image", prob=random_prob, intensity_range=(1, 11))
    ])

    val_test_specific = Compose([
        GridPatchd(
            keys='image', 
            patch_size=(resized_image, resized_image, 1), 
            pad_mode='constant', 
            constant_values=0),
        SqueezeDimd(keys='image', dim=-1),
        CenterSpatialCropd(keys='image', roi_size=(-1, image_size, image_size)),
        NormalizeIntensityd(keys='image', channel_wise=True),
        EnsureTyped(keys=['image', 'label'], device=device, track_meta=False)
    ])

    if dataset == 'train':
        return Compose([preprocessing, train_specific])
    elif dataset in ['val', 'test']:
        return Compose([preprocessing, val_test_specific])
    else:
        raise ValueError ("Dataset must be 'train', 'val' or 'test'.")
    
def parse_args() -> argparse.Namespace:

    '''
    Parse command line arguments.

    Returns:
        argparse.Namespace: Arguments.
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--summ-stats", action='store_true',
                        help="Flag to return summary statistics")
    parser.add_argument("--dicom2nifti", action='store_true',
                        help="Flag to convert dicom images to nifti")
    parser.add_argument("--dataset", type=str, 
                        help="Dataset to return summary statistics for")
    parser.add_argument("--seed", type=int, default=123, 
                        help="Seed for reproducibility")
    parser.add_argument("--modality-list", type=str, default=MODALITY_LIST,
                        help="Path to the data directory")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help="Path to the data directory")
    return parser.parse_args()

def main(
        args: argparse.Namespace
        ) -> None:

    '''
    Main function. The function can be used to return summary statistics or to convert dicom images 
    to nifti file format. To return the former set the --summ-stats flag to True and specify the
    dataset to return summary statistics for using the --dataset flag. To convert dicom images to
    nifti file format set the --dicom2nifti flag to True.

    Args:
        args (argparse.Namespace): Arguments.
    '''
    # Set a seed for reproducibility.
    ReproducibilityUtils.seed_everything(args.seed)
    # If args.dict2nifti is set to True convert dicom images to nifti file format.
    if args.dicom2nifti:
        # Disable validation of slice increment.
        dicom2nifti.settings.disable_validate_slice_increment()
        # Convert dicom images to nifti file format.
        DicomToNifti(args.data_dir).convert_dicom_to_nifti(os.path.join(args.data_dir, 'labels/labels.csv'))
        # Only retain specified imaging modalities.
        prep = PreprocessingUtils(args.data_dir)
        prep.clean_directory(args.modality_list)
    # If args.summ_stats is set to True return summary statistics.
    if args.summ_stats:
        # Load the data and create a dictionary containing the data.
        dataloader = DataLoader(args.data_dir, args.modality_list)
        data_dict = dataloader.create_data_dict()
        split_dict = dataloader.split_dataset(0.8, False)
        data_split_dict = {x: [patient for patient in data_dict if patient['uid'] in split_dict[x]] for x in ['train', 'val', 'test']} 
        data_split_dict = {x: [patient['uid'][:-4] for patient in data_split_dict[x]] for x in ['train', 'val', 'test']}
        # Load data on preconditions and general patient information.
        preconditions_df = pd.read_csv(os.path.join(args.data_dir, 'preconditions.csv'), sep=';')
        preconditions_df = preconditions_df[['Study_nr','NASH','HBV','HCV','Alcoholic_Cirrosis','Haemochromatosis','PBC','PSC','Cryptogenic','AIH','Other_Etiology']]
        pt_info_df = pd.read_csv(os.path.join(args.data_dir, 'patient_info.csv'), sep=';')
        pt_info_df = pt_info_df[['Study_nr','Age','Sex']]
        summ_stats_df = preconditions_df.merge(pt_info_df, how='inner', on='Study_nr')
        # Specify which dataset to return summary statistics for.
        if args.dataset == 'train':
            summ_stats_df = summ_stats_df[summ_stats_df['Study_nr'].isin(data_split_dict['train'])]
        elif args.dataset == 'val':
            summ_stats_df = summ_stats_df[summ_stats_df['Study_nr'].isin(data_split_dict['val'])]
        elif args.dataset == 'test':
            summ_stats_df = summ_stats_df[summ_stats_df['Study_nr'].isin(data_split_dict['test'])]
        # Print summary statistics.
        print('Summary Statistics for Dichotumous Variables (Abs. and Rel. Frequency):')
        abs_freq = round(summ_stats_df.iloc[:,1:-2].mean() * summ_stats_df.iloc[:,1:-2].count())
        rel_freq = round(summ_stats_df.iloc[:,1:-2].mean() * 100, 1)
        print(pd.concat([abs_freq, rel_freq], axis=1))
        print('Summary Statistics for Continuous Variables (Mean and Std.):')
        print('Age',round(summ_stats_df.iloc[:,11].mean(), 1), round(summ_stats_df.iloc[:,11].std(), 1))

DATA_DIR = '/Users/noltinho/thesis/sensitive_data'
IMAGES_TO_KEEP = ['T1W_OOP','T1W_IP','T1W_DYN','T1W_QNT','T2W_TES','T2W_TEL','DWI_b0','DWI_b150','DWI_b400','DWI_b800']
MODALITY_LIST = ['T1W_OOP','T1W_IP','T1W_DYN','T2W_TES','T2W_TEL','DWI_b150','DWI_b400','DWI_b800']
MOD_LIST = ['T1W_OOP']

if __name__ == '__main__':
    # args = parse_args()
    # main(args)
    ReproducibilityUtils.seed_everything(124)
    data_dict, label_df = DatasetPreprocessor(data_dir=DATA_DIR, test_run=True).load_imaging_data(MOD_LIST)
    train, val_test = GroupStratifiedSplit(split_ratio=0.6).split_dataset(label_df)
    val, test = GroupStratifiedSplit(split_ratio=0.5).split_dataset(val_test)
    split_dict = GroupStratifiedSplit().convert_to_dict(train, val, test, data_dict)
    distributed = False
    batch_size = 4
    num_workers = 4
    datasets = {x: CacheDataset(data=split_dict[x], transform=transformations(x, MOD_LIST, 32, 224)) for x in ['train','val','test']}
    if distributed:
        sampler = {x: DistributedSampler(dataset=datasets[x], even_divisible=True, shuffle=(True if x == 'train' else False)) for x in ['train','val','test']}
    else:
        sampler = {x: None for x in ['train','val','test']}
    dataloader = {x: DataLoader(datasets[x], batch_size=(batch_size if x == 'train' else 1), shuffle=(True if x == 'train' and sampler[x] is None else False), num_workers=num_workers, sampler=sampler[x], pin_memory=True) for x in ['train','val','test']}
    for i, batch in enumerate(dataloader['val']):
        image, label = batch['image'], batch['label']
        print(image.shape)
        plt.imshow(image[0,16,0,...], cmap="gray")
        plt.savefig('/Users/noltinho/thesis/miscellaneous/images/new3.png', dpi=300, bbox_inches="tight")
        plt.close()
        if i == 0:
            break