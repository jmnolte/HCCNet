import torch
import os
import numpy as np
import pandas as pd
from glob import glob
from natsort import natsorted
from collections import defaultdict
from monai.data.utils import collate_meta_tensor


def pad_or_trunc_seq(data: list, seq_length: int) -> list:

    sequences = {x: [seq[x] for seq in data] for x in ['image','label','age']}
    zero_image = sequences['image'][0]
    zero_label = torch.IntTensor([99])
    zero_age = torch.FloatTensor([0])
    if len(sequences['image']) < seq_length:
        while len(sequences['image']) < seq_length:
            sequences['image'].append(zero_image)
            sequences['label'].append(zero_label)
            sequences['age'].append(zero_age)
    elif len(sequences['image']) > seq_length:
        while len(sequences['image']) > seq_length:
            idx_to_remove = np.random.choice(len(sequences['image']))
            sequences['image'].pop(idx_to_remove)
            sequences['label'].pop(idx_to_remove)
            sequences['age'].pop(idx_to_remove)

    data = []
    for i in range(seq_length):
        data.append({'image': sequences['image'][i], 'label': sequences['label'][i], 'age': sequences['age'][i]})

    return data


def collate_sequence_batch(batch):
    data = []
    for patient in range(len(batch)):
        elem = [i for k in [batch[patient]] for i in k]
        data.append(pad_or_trunc_seq(elem, 4))
    data = [i for k in data for i in k]

    return collate_meta_tensor(data)


def collate_shuffled_batch(batch):

    data = [i for k in batch for i in k]
    np.random.shuffle(data)

    return collate_meta_tensor(data)


def convert_to_dict(
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        data_dict: dict
    ) -> dict:
        
    for idx, df in enumerate([train, val, test]):
        name = 'training' if idx == 0 else 'validation' if idx == 1 else 'test'
        print(f'{len(df)} total observations in {name} set with {df["label"].sum()} positive cases ({round(df["label"].mean(), ndigits=3)} %)')
    split_dict = {'train': train[['uid']].values, 'val': val[['uid']].values, 'test': test[['uid']].values}
    split_dict = {x: [patient for patient in data_dict if patient['uid'] in split_dict[x]] for x in ['train', 'val', 'test']} 

    return split_dict


def convert_to_seqdict(
        split_dict: dict,
        splits: list
    ) -> dict:

    seq_dict = {x: [] for x in splits}
    for phase in splits:
        desired_split_dict = defaultdict(lambda: defaultdict(list))
        for entry in split_dict[phase]:
            uid = entry['uid'].split('_')[1]  # Extract the common part of 'uid'
            for key, value in entry.items():
                if key == 'uid':
                    continue
                elif key == 'label' or key == 'age':
                    value = [value]
                desired_split_dict[uid][key].extend(value)

        desired_split_list = [{'uid': uid, **entry} for uid, entry in desired_split_dict.items()]
        seq_dict[phase].append(desired_split_list)
    return seq_dict


class DatasetPreprocessor:

    def __init__(
            self,
            data_dir: str,
        ) -> None:
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
            if len(images) == 1:
                tail, _ = os.path.split(images[0])
                common_prefix = tail + '/'
            else:
                common_prefix = os.path.commonprefix(images)
            image_names = [image_name[len(common_prefix):] for image_name in images]
            if all(names in image_names for names in image_names_list):
                observation_list.append(observation)
        if verbose:
            print('{} out of {} observations include all required images'.format(len(observation_list), len(self.nifti_patients)))
        return observation_list

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
                label = labels_df.loc[labels_df['uid'] == observation_id, 'label'].values.item()
                # age = labels_df.loc[labels_df['uid'] == observation_id, 'age'].values.item()
            label_dict['label'].append(label)
            label_dict['uid'].append(observation_id)
            # label_dict['age'].append(age)
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
    
    def load_data(
            self,
            modalities: list,
            label_column: str = 'label',
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
        observation_list = self.assert_observation_completeness(modalities, verbose)
        modality_dict = {modality: self.split_observations_by_modality(observation_list, modality) for modality in modalities}
        label_dict = self.create_label_dict(observation_list, os.path.join(self.label_dir, 'labels.csv'))
        label_df = pd.DataFrame.from_dict(label_dict)
        label_df['patient_id'] = label_df['uid'].str.split('_').str[1]
        data_dict = self.create_data_dict(observation_list, modality_dict)
        for idx, patient in enumerate(data_dict):
            if patient['uid'] in [label['uid'] for label in label_dict]:
                data_dict[idx][label_column] = label_dict[[label['uid'] for label in label_dict].index(patient['uid'])][label_column]
                # data_dict[idx]['age'] = label_dict[[label['uid'] for label in label_dict].index(patient['uid'])]['age']
            else:
                data_dict[idx][label_column] = None
                # data_dict[idx]['age'] = None
        data_dict = [patient for patient in data_dict if not (patient[label_column] == None)]

        return data_dict, label_df