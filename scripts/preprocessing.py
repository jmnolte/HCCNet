import monai
import numpy as np
import glob
import os
import tempfile
import torch
from tqdm import tqdm
from natsort import natsorted
from glob import glob
import dicom2nifti
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from utils import ReproducibilityUtils
import argparse


class DicomToNifti:

    def __init__(
            self, 
            path: str
            ) -> None:

        '''
        Initialize the data utils class.

        Args:
            path (str): Path to the data.
        '''

        self.dicom_dir = os.path.join(path, 'dicom')
        self.nifti_dir = os.path.join(path, 'nifti')
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
            df_path: str
            ) -> None:

        '''
        Convert dicom files into nifti file format.

        Args:
            df_path (str): Path to the dataframe.
        '''

        labels_dict = pd.read_csv(df_path).to_dict('list')
        labels_dict['date'] = [str(date) for date in labels_dict['date']]
        observation_list = self.get_dicom_observations()
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

class PreprocessingUtils:

    def __init__(
            self, 
            path: str
            ) -> None:

        '''
        Initialize the preprocessing utils class.

        Args:
            path (str): Path to the data.
        '''

        self.nifti_dir = os.path.join(path, 'nifti')
        self.dicom_dir = os.path.join(path, 'dicom')
        self.nifti_patients = glob(os.path.join(self.nifti_dir, '*'), recursive = True)
        self.dicom_patients = glob(os.path.join(self.dicom_dir, '*'), recursive = True)

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
            image_list: list
            ) -> None:

        '''
        Clean the directory.

        Args:
            image_list (list): List of images to keep.
        '''
        for observation in tqdm(natsorted(self.nifti_patients)):
            images = glob(os.path.join(observation, '*.nii.gz'))
            for image in images:
                image_head = os.path.split(image)[-1]
                image_name = image_head.split('.')[0]
                if image_name not in image_list:
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

class DataLoader:

    def __init__(
            self, 
            path: str, 
            modality_list
            ) -> None:

        '''
        Initialize the data loader class.

        Args:
            path (str): Path to the data.
            modality_list (list): List of modalities to load.
        '''
        self.modality_list = modality_list
        self.nifti_dir = os.path.join(path, 'nifti')
        self.nifti_patients = glob(os.path.join(self.nifti_dir, '*'), recursive = True)
        self.label_dir = os.path.join(path, 'labels')

    def apply_transformations(
            self, 
            dataset_indicator: str
            ) -> monai.transforms:

        '''
        Perform data transformations on image and image labels.

        Args:
            dataset_indicator (str): Dataset to apply transformations on. Can be 'train', 'val' or 'test'.

        Returns:
            transforms (monai.transforms): Data transformations to be applied.
        '''

        preprocessing = monai.transforms.Compose([
            monai.transforms.LoadImaged(keys=self.modality_list),
            monai.transforms.EnsureChannelFirstd(keys=self.modality_list),
            monai.transforms.Orientationd(keys=self.modality_list, axcodes='PLI'),
            monai.transforms.Resized(keys=self.modality_list, spatial_size=(96, 96, 96)),
            monai.transforms.ConcatItemsd(keys=self.modality_list, name='image', dim=0),
            monai.transforms.ScaleIntensityRangePercentilesd(keys='image', lower=5, upper=95, b_min=0, b_max=1, clip=True, channel_wise=True),
            monai.transforms.NormalizeIntensityd(keys='image', channel_wise=True)
        ])
        
        augmentation = monai.transforms.Compose([
            monai.transforms.RandRotated(keys='image', prob=0.2, range_x=np.pi/8),
            monai.transforms.RandFlipd(keys='image', prob=0.2, spatial_axis=2),
            monai.transforms.RandZoomd(keys='image', prob=0.2, min_zoom=1.1, max_zoom=1.3),
            monai.transforms.RandGibbsNoised(keys='image', prob=0.1, alpha=(0.6, 0.8)),
            monai.transforms.RandAdjustContrastd(keys='image', prob=0.1, gamma=(0.5, 2.0))
        ])
        
        postprocessing = monai.transforms.Compose([
            monai.transforms.ToTensord(keys=['image', 'label'])
        ])

        if dataset_indicator == 'train':
            return monai.transforms.Compose([preprocessing, augmentation, postprocessing])
        else:
            return monai.transforms.Compose([preprocessing, postprocessing])
        
    def assert_observation_completeness(
            self, 
            modality_list: list, 
            print_statement: bool
            ) -> list:

        '''
        Assert that all observations include all required images.

        Args:
            modality_list (list): List of modalities to load.
            print_statement (bool): Print statement indicating the number of observations that include all required images.

        Returns:
            observation_list (list): List of observations that include all required images.
        '''

        image_names_list = [image_name + '.nii.gz' for image_name in modality_list]
        observation_list = []
        for observation in natsorted(self.nifti_patients):
            images = [image for image in glob(os.path.join(observation, '*.nii.gz'))]
            common_prefix = os.path.commonprefix(images)
            image_names = [image_name[len(common_prefix):] for image_name in images]
            if all(names in image_names for names in image_names_list):
                observation_list.append(observation)
        if print_statement:
            print('{} out of {} observations include all required images'.format(len(observation_list), len(self.nifti_patients)))
        return observation_list
    
    @staticmethod
    def split_observations_by_modality(
            observation_list: list, 
            modality: str
            ) -> list:

        '''
        Split the observation paths by modality.

        Args:
            observation_list (list): List of observations to split.
            modality (str): Modality to split by.
        
        Returns:
            observation_list (list): List of observations split by modality.
        '''
        return [glob(os.path.join(observation, modality + '.nii.gz')) for observation in observation_list]
    
    @staticmethod
    def create_label_dict(
            observation_list: list, 
            df_path: str
            ) -> dict:

        '''
        Create a list of labels.

        Args:
            observation_list (list): List of observations to create labels for.
            df_path (str): Path to the dataframe containing the labels.

        Returns:
            label_dict (dict): Dictionary containing the labels.
        '''
        label_dict = {'uid': [], 'label': []}
        labels_df = pd.read_csv(df_path)
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
        return label_dict
            
    def create_data_dict(
            self
            ) -> dict:

        '''
        Create a dictionary containing the data.

        Returns:
            data_dict (dict): Dictionary containing the data.
        '''
        observation_list = self.assert_observation_completeness(self.modality_list, True)
        path_dict = {modality: self.split_observations_by_modality(observation_list, modality) for modality in self.modality_list}
        label_dict = self.create_label_dict(observation_list, os.path.join(self.label_dir, 'labels.csv'))
        path_dict['uid'] = [os.path.basename(observation) for observation in observation_list]
        path_dict = [dict(zip(path_dict.keys(), vals)) for vals in zip(*(path_dict[k] for k in path_dict.keys()))]
        label_dict = [dict(zip(label_dict.keys(), vals)) for vals in zip(*(label_dict[k] for k in label_dict.keys()))]
        for idx, patient in enumerate(path_dict):
            if patient['uid'] in [label['uid'] for label in label_dict]:
                path_dict[idx]['label'] = label_dict[[label['uid'] for label in label_dict].index(patient['uid'])]['label']
            else:
                path_dict[idx]['label'] = None
        return [patient for patient in path_dict if not (patient['label'] == None)]
    
    @staticmethod
    def stratified_data_split(
            df: pd.DataFrame, 
            test_ratio: float, 
            n_splits: int = 100
            ) -> pd.DataFrame:

        '''
        Split the data into training, validation and test sets.

        Args:
            df (pd.DataFrame): Dataframe containing the data.
            test_ratio (float): Ratio of the data to be used for testing.
            n_splits (int): Number of splits to perform.

        Returns:
            dataframes (pd.DataFrame): Dataframes split according to the test ratio.
        '''
        label_ratio = 1
        splits = GroupShuffleSplit(test_size=test_ratio, n_splits=n_splits).split(df, groups=df['patient_id'])
        for split in splits:
            set1, set2 = df.iloc[split[0]], df.iloc[split[1]]
            if abs(set1['label'].mean() - set2['label'].mean()) < label_ratio:
                if len(set1) / (1 - test_ratio) > len(set2) / test_ratio * 0.95 and len(set1) / (1 - test_ratio) < len(set1) / test_ratio * 1.05:
                    label_ratio = abs(set1['label'].mean() - set2['label'].mean())
                    best_split = split
        return best_split

    def split_dataset(
            self, 
            train_ratio: float, 
            quant: bool
            ) -> dict:

        '''
        Split the dataset into training, validation and test sets.

        Args:
            train_ratio (float): Ratio of the data to be used for training.
            quant (bool): Whether to only use observations including quant images.

        Returns:
            dataframes (dict): Dictionary containing the dataframes split according to the train ratio.
        '''
        quant_list = self.assert_observation_completeness(['T1W_QNT'], False)
        observation_list = self.assert_observation_completeness(self.modality_list, False)
        quant_dict = {'uid': [os.path.split(observation)[-1] for observation in quant_list], 'quant': [1 for _ in range(len(quant_list))]}
        label_dict = self.create_label_dict(observation_list, os.path.join(self.label_dir, 'labels.csv'))
        quant_df = pd.DataFrame.from_dict(quant_dict)
        label_df = pd.DataFrame.from_dict(label_dict)
        df = pd.merge(label_df, quant_df, on='uid', how='left').fillna(0)
        df['patient_id'] = df['uid'].str.split('_').str[1]
        train, val_test = train_test_split(df, test_size=1-train_ratio, stratify=df['label'])
        val, test = train_test_split(val_test, test_size=0.5, stratify=val_test['label'])
        if quant:
            for idx, df in enumerate([train, val, test]):
                df = df.drop(df[df['quant'] == 0].index)
                train, val, test = df if idx == 0 else train, df if idx == 1 else val, df if idx == 2 else test
                name = 'training' if idx == 0 else 'validation' if idx == 1 else 'test'
                print('{} total observations in {} set with {} positive cases ({} %).'.format(len(df), name, df['label'].sum(), round(df['label'].mean(), ndigits=3)))
            return {'train': train[['uid']].values, 'val': val[['uid']].values, 'test': test[['uid']].values}
        else:
            for idx, df in enumerate([train, val, test]):
                name = 'training' if idx == 0 else 'validation' if idx == 1 else 'test'
                print('{} total observations in {} set with {} positive cases ({} %).'.format(len(df), name, df['label'].sum(), round(df['label'].mean(), ndigits=3)))
            return {'train': train[['uid']].values, 'val': val[['uid']].values, 'test': test[['uid']].values}
        
    @staticmethod
    def get_class_weights(
            data_split_dict: dict
            ) -> dict:

        '''
        Calculate the sample class weights per dataset.

        Args:
            data_split_dict (dict): Dictionary containing the dataframes split according to the train ratio.
        
        Returns:
            class_weights (dict): Dictionary containing the class weights for each set.
        '''
        labels = {x: [patient['label'] for patient in data_split_dict[x]] for x in ['train', 'val', 'test']}
        class_sample_count = {x: np.array([len(np.where(labels[x] == t)[0]) for t in np.unique(labels[x])]) for x in ['train', 'val', 'test']}
        weights = {x: 1. / class_sample_count[x] for x in ['train', 'val', 'test']}
        sample_weights = {x: np.array([weights[x][t] for t in labels[x]]) for x in ['train', 'val', 'test']}
        sample_weights = {x: torch.from_numpy(sample_weights[x]) for x in ['train','val','test']}
        return {x: sample_weights[x].double() for x in ['train','val','test']}  
        
    def load_data(
            self, 
            data_dict: dict, 
            train_ratio: float, 
            batch_size: int, 
            num_workers: int, 
            weighted_sampler: bool, 
            quant_images: bool
            ) -> dict:

        '''
        Load the data.

        Args:
            data_dict (dict): Dictionary containing the data.
            train_ratio (float): Ratio of the data to be used for training.
            batch_size (int): Batch size to use.
            num_workers (int): Number of workers to use.
            weighted_sampler (bool): Whether to use a weighted sampler.
            quant_images (bool): Whether to only use observations including quant images.

        Returns:
            data_loaders (dict): Dictionary containing the data loaders.
        '''
        split_dict = self.split_dataset(train_ratio, quant_images)
        data_split_dict = {x: [patient for patient in data_dict if patient['uid'] in split_dict[x]] for x in ['train', 'val', 'test']} 
        sample_weights = self.get_class_weights(data_split_dict)     
        for phase in data_split_dict:
            for patient in data_split_dict[phase]:
                del patient['uid']
        persistent_cache = os.path.join(tempfile.mkdtemp(), 'persistent_cache')
        datasets = {x: monai.data.PersistentDataset(data=data_split_dict[x], transform=self.apply_transformations(x), cache_dir=persistent_cache) for x in ['train','val','test']}
        sampler = {x: [] for x in ['train','val','test']}
        if weighted_sampler:
            sampler['train'] = monai.data.DistributedWeightedRandomSampler(dataset=datasets['train'], weights=sample_weights['train'], even_divisible=True, shuffle=True)
            sampler['val'] = monai.data.DistributedSampler(dataset=datasets['val'], even_divisible=True, shuffle=True)
            sampler['test'] = monai.data.DistributedSampler(dataset=datasets['test'], even_divisible=True, shuffle=True)

        else:
            sampler = {x: monai.data.DistributedSampler(dataset=datasets[x], even_divisible=True, shuffle=True) for x in ['train','val','test']}
        return {x: monai.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=(sampler[x] is None), num_workers=num_workers, sampler=sampler[x], pin_memory=True) for x in ['train','val','test']}
        
    
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

if __name__ == '__main__':
    args = parse_args()
    main(args)
