import monai
import numpy as np
import glob
import os
import tempfile
from tqdm import tqdm
from natsort import natsorted
from glob import glob
import dicom2nifti
from utils import ReproducibilityUtils
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

class DicomToNifti:

    def __init__(self, path: str) -> None:

        '''
        Initialize the data utils class.

        Args:
            path (str): Path to the data.
        '''

        self.dicom_dir = os.path.join(path, 'dicom')
        self.nifti_dir = os.path.join(path, 'nifti')
        self.dicom_patients = glob(os.path.join(self.dicom_dir, '*'), recursive = True)
        self.nifti_patients = glob(os.path.join(self.nifti_dir, '*'), recursive = True)

    def get_dicom_observations(self) -> list:

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
    
    def convert_dicom_to_nifti(self, df_path: str) -> None:

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

    def __init__(self, path: str) -> None:

        '''
        Initialize the preprocessing utils class.

        Args:
            path (str): Path to the data.
        '''

        self.nifti_dir = os.path.join(path, 'nifti')
        self.dicom_dir = os.path.join(path, 'dicom')
        self.nifti_patients = glob(os.path.join(self.nifti_dir, '*'), recursive = True)
        self.dicom_patients = glob(os.path.join(self.dicom_dir, '*'), recursive = True)

    def extract_image_names(self) -> list:

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

    def clean_directory(self, image_list: list) -> None:

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

    # def create_label_list(self, df_path: str) -> list:

    #     '''
    #     Create a list of labels.

    #     Args:
    #         label_path (str): Path to the label file.
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

    def __init__(self, path: str, modality_list) -> None:

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

    def apply_transformations(self, dataset_indicator: str) -> monai.transforms:

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
            monai.transforms.Resized(keys=self.modality_list, spatial_size=(128, 128, 128)),
            monai.transforms.NormalizeIntensityd(keys=self.modality_list, channel_wise=True),
            monai.transforms.ConcatItemsd(keys=self.modality_list, name='image', dim=0)
        ])
        
        augmentation = monai.transforms.Compose([
            monai.transforms.RandRotated(keys='image', prob=0.1,
                                         range_x=np.pi/8, range_y=np.pi/8, range_z=np.pi/8),
            monai.transforms.RandGibbsNoised(keys='image', prob=0.1, alpha=(0.6, 0.8)),
            monai.transforms.RandFlipd(keys='image', prob=0.1, spatial_axis=2),
            monai.transforms.RandAdjustContrastd(keys='image', prob=0.1, gamma=(0.5, 2.0))
        ])
        
        postprocessing = monai.transforms.Compose([
            monai.transforms.IntensityStatsd(keys='image', key_prefix='orig', ops=['mean', 'std'], channel_wise=True),
            monai.transforms.ToTensord(keys=['image', 'label'])
        ])

        if dataset_indicator == 'train':
            return monai.transforms.Compose([preprocessing, augmentation, postprocessing])
        else:
            return monai.transforms.Compose([preprocessing, postprocessing])
        
    def assert_observation_completeness(self, modality_list: list, print_statement: bool) -> list:

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
    def split_observations_by_modality(observation_list: list, modality: str) -> list:

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
    def create_label_dict(observation_list: list, df_path: str) -> dict:

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
            label = labels_df.loc[labels_df['uid'] == observation_id, 'outcome'].item()
            label_dict['label'].append(label)
            label_dict['uid'].append(observation_id)
        return label_dict
            
    def create_data_dict(self) -> dict:

        '''
        Create a dictionary containing the data.

        Returns:
            data_dict (dict): Dictionary containing the data.
        '''
        observation_list = self.assert_observation_completeness(self.modality_list, True)
        path_dict = {modality: self.split_observations_by_modality(observation_list, modality) for modality in self.modality_list}
        path_dict['label'] = self.create_label_dict(observation_list, os.path.join(self.label_dir, 'labels.csv'))['label']
        path_dict['uid'] = self.create_label_dict(observation_list, os.path.join(self.label_dir, 'labels.csv'))['uid']
        return [dict(zip(path_dict.keys(), vals)) for vals in zip(*(path_dict[k] for k in path_dict.keys()))]
    
    @staticmethod
    def stratified_data_split(df: pd.DataFrame, test_ratio: float, n_splits: int = 100) -> pd.DataFrame:

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
                label_ratio = abs(set1['label'].mean() - set2['label'].mean())
                best_split = split
        return df.iloc[best_split[0]], df.iloc[best_split[1]]

    def split_dataset(self, train_ratio: float, quant: bool) -> dict:

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
        train, val_test = self.stratified_data_split(df, 1 - train_ratio)
        val, test = self.stratified_data_split(val_test, 0.5)
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
        
    def load_data(self, data_dict: dict, train_ratio: float, batch_size: int, num_workers: int, test_set: bool, quant_images: bool) -> dict:

        '''
        Load the data.

        Args:
            data_dict (dict): Dictionary containing the data.
            train_ratio (float): Ratio of the data to be used for training.
            batch_size (int): Batch size to use.
            num_workers (int): Number of workers to use.
            test_set (bool): Whether to load the test set.

        Returns:
            data_loaders (dict): Dictionary containing the data loaders.
        '''
        split_dict = self.split_dataset(train_ratio, quant_images)
        data_split_dict = {x: [patient for patient in data_dict if patient['uid'] in split_dict[x]] for x in ['train', 'val', 'test']}
        for phase in data_split_dict:
            for patient in data_split_dict[phase]:
                del patient['uid']
        persistent_cache = os.path.join(tempfile.mkdtemp(), 'persistent_cache')
        if test_set:
            test_dataset = monai.data.PersistentDataset(data=data_split_dict['test'], transform=self.apply_transformations('test'), cache_dir=persistent_cache)
            return {'test': monai.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)}
        else:
            datasets = {x: monai.data.PersistentDataset(data=data_split_dict[x], transform=self.apply_transformations(x), cache_dir=persistent_cache) for x in ['train', 'val']}
            return {x: monai.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}
            

if __name__ == '__main__':
    DATA_DIR = '/Users/noltinho/thesis/sensitive_data'
    QUANT_LIST = ['T1W_QNT']
    MODALITY_LIST = ['T1W_OOP','T1W_IP','T1W_DYN','T2W_TES','T2W_TEL','DWI_b150','DWI_b400','DWI_b800']
    IMAGES_TO_KEEP = ['T1W_OOP','T1W_IP','T1W_DYN','T1W_QNT','T2W_TES','T2W_TEL','DWI_b0','DWI_b150','DWI_b400','DWI_b800']
    # dicom2nifti.settings.disable_validate_slice_increment()
    # DicomToNifti(DATA_DIR).convert_dicom_to_nifti(os.path.join(DATA_DIR, 'labels/labels.csv'))
    # prep = PreprocessingUtils(DATA_DIR)
    # prep.clean_directory(IMAGES_TO_KEEP)
    ReproducibilityUtils.seed_everything(123)
    dataloader = DataLoader(DATA_DIR, MODALITY_LIST)
    data_dict = dataloader.create_data_dict()
    dataloader_dict = dataloader.load_data(data_dict, 0.8, 16, 4, False, True)