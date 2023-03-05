import monai
import numpy as np
import glob
import os
import nibabel as nib
from tqdm import tqdm
from natsort import natsorted
from glob import glob
import dicom2nifti
from utils import MetadataUtils

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
        self.nifti_patients = glob(os.path.join(self.nifti_dir, '*'), recursive = True)

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
    
    def assert_observation_completeness(self, image_list: list) -> list:

        '''
        Assert that all observations include all required images.

        Args:
            image_list (list): List of images to check.
            include_quant (bool): Whether to include the quant series images.
        '''
        image_names_list = [image_name + '.nii.gz' for image_name in image_list]
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

class DataLoader:

    def __init__(self, image_list: list) -> None:

        '''
        Initialize the data loader class.

        Args:
            image_list (list): List of images to load.
        '''

        self.image_list = image_list

    def apply_transformations(self, dataset_indicator: str) -> monai.transforms:

        '''
        Perform data transformations on image and image labels.

        Args:
            dataset_indicator (str): Dataset to apply transformations on. Can be 'train', 'val' or 'test'.
        '''

        preprocessing = monai.transforms.Compose([
            monai.transforms.LoadImaged(keys=self.image_list),
            monai.transforms.EnsureChannelFirstd(keys=self.image_list),
            monai.transforms.Orientationd(keys=self.image_list, axcodes='ASL'),
            monai.transforms.Resized(keys=self.image_list, spatial_size=(224, 224, 224)),
            monai.transforms.NormalizeIntensityd(keys=self.image_list, channel_wise=True)
            ])
        
        augmentation = monai.transforms.Compose([
            monai.transforms.RandRotated(keys=self.image_list, prob=0.1,
                                         range_x=np.pi/8, range_y=np.pi/8, range_z=np.pi/8),
            monai.transforms.RandGaussianNoised(keys=self.image_list, prob=0.1, mean=0, std=0.1),
            monai.transforms.RandAxisFlipd(keys=self.image_list, prob=0.1),
            monai.transforms.RandAdjustContrastd(keys=self.image_list, prob=0.1, gamma=(0.5, 2.0)),
        ])
        
        postprocessing = monai.transforms.Compose([
            monai.transforms.ConcatItemsd(keys=self.image_list, name='image', dim=0),
            monai.transforms.IntensityStatsd(keys='image', key_prefix='orig', ops=['mean', 'std'], channel_wise=True),
            monai.transforms.ToTensord(keys=['image', 'label'])
        ])

        if dataset_indicator == 'train':
            return monai.transforms.Compose([preprocessing, augmentation, postprocessing])
        else:
            return monai.transforms.Compose([preprocessing, postprocessing])
        
    def load_data(self, data_dict, split_ratio: list, batch_size: int, num_workers: int) -> dict:

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
        datasets = {x: monai.data.CacheDataset(data=data_split_dict[x], transform=self.apply_transformations(x)) for x in ['train', 'val', 'test']}
        return {x: monai.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val', 'test']}
        

if __name__ == '__main__':
    PATH = '/Users/noltinho/thesis_private/data'
    prep = DicomToNifti(PATH)
    # dicom2nifti.settings.disable_validate_slice_increment()
    # prep.convert_dicom_to_nifti()
    # image_list = prep.extract_image_names()
    # metadata = MetadataUtils(PATH)
    # metadata.count_series_freq(image_list)
    COMPLETE_IMAGE_LIST = ['T1W_OOP','T1W_IP','T1W_DYN','T2W_TES','T2W_TEL','DWI_b0','DWI_b150','DWI_b400','DWI_b800']
    observation_list = prep.assert_observation_completeness(COMPLETE_IMAGE_LIST)
    observation_list = np.random.choice(observation_list, 5, replace=False)
    # IMAGES_TO_DELETE = ['T1_dyn','T2_short_ET','T1_in_out','T2W_ETS']
    # prep.clean_directory(IMAGES_TO_DELETE)