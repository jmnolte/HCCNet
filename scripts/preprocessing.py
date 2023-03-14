import monai
import numpy as np
import glob
import os
import tempfile
from tqdm import tqdm
from natsort import natsorted
from glob import glob
import dicom2nifti
from utils import MetadataUtils, ReproducibilityUtils
import pandas as pd

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

    def convert_dicom_to_nifti(self) -> None:

        '''
        Convert dicom files into nifti file format.
        '''

        for patient_path in tqdm(natsorted(self.dicom_patients)):
            _, patient_id = os.path.split(patient_path)
            observations = natsorted(glob(os.path.join(patient_path, 'DICOM/*'), recursive = True))
            for idx, observation_path in enumerate(observations):
                if idx < 9:
                    observation_id = os.path.join('00' + str(idx+1))
                elif idx >= 9:
                    observation_id = os.path.join('0' + str(idx+1))
                for series_path in glob(os.path.join(observation_path, '*/*')):
                    _, series_id = os.path.split(series_path)
                    folder_name = os.path.join(patient_id + '_' + observation_id)
                    file_name = os.path.join(series_id + '.nii.gz')
                    if not os.path.exists(os.path.join(self.nifti_dir, folder_name)):
                        os.makedirs(os.path.join(self.nifti_dir, folder_name))
                    if not os.path.exists(os.path.join(self.nifti_dir, folder_name, file_name)):
                        try:
                            dicom2nifti.dicom_series_to_nifti(series_path, os.path.join(self.nifti_dir, folder_name, file_name))
                        except:
                            print('Error converting DICOM series {} to nifti {}.'.format(series_path, folder_name, file_name))


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

    def clean_directory(self, image_list) -> None:

        '''
        Clean the directory.

        Args:
            image_list (list): List of images to delete.
        '''

        for observation in tqdm(natsorted(self.nifti_patients)):
            images = glob(os.path.join(observation, '*.nii.gz'))
            for image in images:
                _, image_head = os.path.split(image)
                if image_head.startswith('S') or any(str(image + '.nii.gz') == image_head for image in image_list):
                    os.remove(image)
                    print('Deleting {}.'.format(image))

    def create_label_list(self, label_path) -> list:

        '''
        Create a list of labels.

        Args:
            label_path (str): Path to the label file.
        '''
        observation_dict = {'Study_nr': [], 'date': [], 'observation': []}
        for folder_path in natsorted(self.dicom_patients):
            _, patient_id = os.path.split(folder_path)
            for idx, observation in enumerate(natsorted(glob(os.path.join(folder_path, 'DICOM/*')))):
                _, date = os.path.split(observation)
                observation_dict['Study_nr'].append(patient_id)
                observation_dict['observation'].append(idx+1)
                observation_dict['date'].append(date)
        observation_df = pd.DataFrame.from_dict(observation_dict)
        labels_df = pd.read_csv(label_path, sep=';')
        labels_df = labels_df.loc[:, ['Study_nr', 'HCC', 'HCC_date_of_diagnosis']]
        labels_df['Study_nr'] = labels_df['Study_nr'].astype('string')
        labels_df['HCC_date_of_diagnosis'] = pd.to_datetime(labels_df['HCC_date_of_diagnosis'], errors='coerce')
        labels_df['HCC_date_of_diagnosis'] = labels_df['HCC_date_of_diagnosis'].dt.strftime('%Y%m%d')
        labels_df = labels_df.merge(observation_df, on=['Study_nr'], how='left')
        labels_df = labels_df.dropna(subset=['observation'])
        labels_df['observation'] = labels_df['observation'].astype('int64')
        labels_df['label'] = np.where(labels_df['HCC_date_of_diagnosis'] < labels_df['date'], 1, 0)
        labels_df = labels_df.drop(columns=['HCC', 'HCC_date_of_diagnosis', 'date'])
        return labels_df

class DataLoader:

    def __init__(self, path: str, modality_list: list) -> None:

        '''
        Initialize the data loader class.

        Args:
            path (str): Path to the data.
            modality_list (list): List of modalities to load.
        '''

        self.modality_list = modality_list
        self.nifti_patients = glob(os.path.join(path, '*'), recursive = True)

    def apply_transformations(self, dataset_indicator: str) -> monai.transforms:

        '''
        Perform data transformations on image and image labels.

        Args:
            dataset_indicator (str): Dataset to apply transformations on. Can be 'train', 'val' or 'test'.
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
        
    def assert_observation_completeness(self) -> list:

        '''
        Assert that all observations include all required images.

        Args:
            image_list (list): List of images to check.
            include_quant (bool): Whether to include the quant series images.
        '''
        image_names_list = [image_name + '.nii.gz' for image_name in self.modality_list]
        observation_list = []
        for observation in tqdm(natsorted(self.nifti_patients)):
            images = [image for image in glob(os.path.join(observation, '*.nii.gz'))]
            common_prefix = os.path.commonprefix(images)
            image_names = [image_name[len(common_prefix):] for image_name in images]
            if all(names in image_names for names in image_names_list):
                observation_list.append(observation)

        print('{} out of {} observations include all required images'.format(len(observation_list), len(self.nifti_patients)))
        return observation_list
    
    @staticmethod
    def split_observations_by_modality(observation_list: list, modality: str) -> list:

        '''
        Split the observation paths by modality.

        Args:
            observation_list (list): List of observations to split.
            modality (str): Modality to split by.
        '''
        return [glob(os.path.join(observation, modality + '.nii.gz')) for observation in observation_list]
            
    def create_data_dict(self, label_list: list) -> dict:

        '''
        Create a dictionary containing the data.

        Args:
            label_list (list): List of labels to add.
        '''
        observation_list = self.assert_observation_completeness()
        observation_list = np.random.choice(observation_list, 20, replace=False)
        path_dict = {modality: self.split_observations_by_modality(observation_list, modality) for modality in self.modality_list}
        path_dict['label'] = list(label_list)
        return [dict(zip(path_dict.keys(), vals)) for vals in zip(*(path_dict[k] for k in path_dict.keys()))]
        
    def load_data(self, data_dict, split_ratio: list, batch_size: int, num_workers: int, test_set: bool) -> dict:

        '''
        Load the data.

        Args:
            data_dict (dict): Dictionary containing the data.
            split_ratio (list): List of ratios to split the data into. First value corresponds to 
                the training set, second to the validation set and third to the test set.
            batch_size (int): Batch size to use.
            num_workers (int): Number of workers to use.
        '''
        data_split = monai.data.utils.partition_dataset(data_dict, shuffle=True, ratios=split_ratio)
        data_split_dict = {x: [] for x in ['train', 'val', 'test']}
        data_split_dict['train'], data_split_dict['val'], data_split_dict['test'] = data_split[0], data_split[1], data_split[2]
        persistent_cache = os.path.join(tempfile.mkdtemp(), "persistent_cache")
        if test_set:
            test_dataset = monai.data.PersistentDataset(data=data_split_dict['test'], transform=self.apply_transformations('test'), cache_dir=persistent_cache)
            return {'test': monai.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)}
        else:
            datasets = {x: monai.data.PersistentDataset(data=data_split_dict[x], transform=self.apply_transformations(x), cache_dir=persistent_cache) for x in ['train', 'val']}
            return {x: monai.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}
        

if __name__ == '__main__':
    # ReproducibilityUtils.seed_everything(123)
    DATA_DIR = '/Users/noltinho/thesis_private/data'
    LABEL_DIR = '/Users/noltinho/thesis'
    MODALITY_LIST = ['T1W_OOP','T1W_IP','T1W_DYN','T2W_TES','T2W_TEL','DWI_b0','DWI_b150','DWI_b400','DWI_b800']
    # dicom2nifti.settings.disable_validate_slice_increment()
    # DicomToNifti(DATA_DIR).convert_dicom_to_nifti()
    prep = PreprocessingUtils(DATA_DIR)
    # image_list = prep.extract_image_names()
    # metadata = MetadataUtils(DATA_DIR)
    # metadata.count_series_freq(image_list)
    # IMAGES_TO_DELETE = ['T1_dyn','T2_short_ET','T1_in_out','T2W_ETS']
    # prep.clean_directory(IMAGES_TO_DELETE)
    prep.create_label_list(os.path.join(LABEL_DIR, 'hcc_labels.csv'))
