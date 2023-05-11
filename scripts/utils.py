import os
from glob import glob
import pydicom as pm
import shutil
from tqdm import tqdm
from natsort import natsorted
from collections import Counter, OrderedDict
from operator import itemgetter
import random
import numpy as np
import torch

class ReproducibilityUtils:

    def seed_everything(seed):

        '''
        Seed everything for reproducibility.

        Args:
            seed (int): Seed value.
        '''
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class EnvironmentUtils:

    def __init__(
            self, 
            path: str
            ) -> None:

        '''
        Initialize the environment utils class.

        Args:
            path (str): Path to the data.
        '''
        self.path = path
        self.folders = glob(os.path.join(self.path, '*/DICOM/*'), recursive = True)

    def create_subfolders(
            self, 
            subfolders: list
            ) -> None:

        '''
        Create subfolders in the data directory.

        Args:
            subfolders (list): List of subfolders to create.
        '''
        for folder_path in tqdm(self.folders):
            for subfolder_path in subfolders:
                new_subfolder = os.path.join(folder_path, subfolder_path)
                if not os.path.exists(new_subfolder):
                    os.makedirs(new_subfolder)
                    print('Creating {}.'.format(new_subfolder))

    def delete_subfolders_by_path(
            self, 
            subfolder_paths: list
            ) -> None:

        '''
        Delete subfolders in the data directory.

        Args:
            subfolders (list): List of subfolder paths to delete.
        '''
        for subfolder_path in tqdm(subfolder_paths):
            if os.path.exists(subfolder_path):
                shutil.rmtree(subfolder_path)
                print('Deleting {}.'.format(subfolder_path))

    def delete_subfolders_by_name(
            self, 
            subfolder_list: list, 
            new_folder_structure: bool
            ) -> None:

        '''
        Delete subfolders in the data directory.

        Args:
            subfolders (list): List of subfolder names to delete.
            new_folder_structure (bool): If True, the adjusted folder structure is used.
        '''
        for folder_path in tqdm(self.folders):
            if new_folder_structure:
                new_folder_path = glob(os.path.join(folder_path, '*'), recursive = True)
                for new_folder in new_folder_path:
                    folder_path = new_folder
                    for subfolder in subfolder_list:
                        subfolder_path = os.path.join(folder_path, subfolder)
                        if os.path.exists(subfolder_path):
                            shutil.rmtree(subfolder_path)
                            print('Deleting {}.'.format(subfolder_path))
            else:
                for subfolder in subfolder_list:
                    subfolder_path = os.path.join(folder_path, subfolder)
                    if os.path.exists(subfolder_path):
                        shutil.rmtree(subfolder_path)
                        print('Deleting {}.'.format(subfolder_path))

    def move_subfolders(
            self
            ) -> None:

        '''
        Move subfolders in the data directory.
        '''
        for folder_path in tqdm(self.folders):
            t1_path = os.path.join(folder_path, 'T1')
            t2_path = os.path.join(folder_path, 'T2')
            diff_path = os.path.join(folder_path, 'DIFFUSION')
            unknown_path = os.path.join(folder_path, 'UNKNOWN')
            series = glob(os.path.join(folder_path, '*'), recursive = True)
            for series_path in series:
                _, path_ext = os.path.split(series_path)
                if path_ext.startswith('S') and len(os.listdir(series_path)) > 1:
                    sequences = natsorted(glob(os.path.join(series_path, '*'), recursive = True))
                    metadata = pm.dcmread(sequences[0], stop_before_pixels=True)
                    try:
                        contrast_id = metadata.AcquisitionContrast
                        # contrast_id = metadata[0x2005140f][0][0x00089209].value
                        series_id = metadata.SeriesDescription
                        if contrast_id == 'T1' or series_id[:2] == 'T1':
                            shutil.move(series_path, t1_path)
                        elif contrast_id == 'T2' or series_id[:2] == 'T2':
                            shutil.move(series_path, t2_path)
                        elif contrast_id == 'DIFFUSION':
                            shutil.move(series_path, diff_path)
                        else:
                            shutil.move(series_path, unknown_path)
                    except:
                        print('Extracting metadata for file {} failed.'.format(series_path))

    def rename_subfolders(
            self
            ) -> None:

        '''
        Rename subfolders in the data directory.
        '''
        for folder_path in tqdm(self.folders):
            path_root, _ = os.path.split(folder_path)
            old_path = folder_path
            dicoms = glob(os.path.join(folder_path, '*/*/I10'), recursive = True)
            for dicom in dicoms:
                new_path = dicom
                metadata = pm.dcmread(new_path, stop_before_pixels=True)
                new_subfolder = metadata.StudyDate
            new_path = os.path.join(path_root, str(new_subfolder))
            if old_path != new_path:
                print('Folder {} does not match dicom metadata and is renamed to {}.'.format(old_path, new_path))
                os.rename(old_path, new_path)
            else:
                print('Folder {} matches dicom metadata.'.format(old_path))

class MetadataUtils:
    def __init__(
            self, 
            path: str
            ) -> None:

        '''
        Initialize the sequence utils class.

        Args:
            path (str): Path to the data directory.
        '''
        self.path = path
        self.folders = glob(os.path.join(self.path, '*/DICOM/*'), recursive = True)

    def extract_metadata(
            self, 
            contrast_type: str, 
            metadata_list: list, 
            all_sequences: bool
            ) -> list:

        '''
        Get the metadata of the sequences.

        Args:
            contrast_type (str): Contrast type of the image.
            metadata_list (list): List of metadata to extract.
            all_sequences (bool): Whether to extract metadata from all or just the first sequence.
        
        Returns:
            metadata_list (list): List of metadata.
        '''    
        metadata_dict = {key: [] for key in metadata_list}
        for folder_path in tqdm(self.folders):
            series = glob(os.path.join(folder_path, contrast_type, '*'), recursive = True)
            for series_path in series:
                sequences = natsorted(glob(os.path.join(series_path, '*'), recursive = True))
                if all_sequences:
                    for sequence_path in sequences:
                        metadata = pm.dcmread(sequence_path, stop_before_pixels=True)
                        if 'description' in metadata_list:
                            metadata_dict['description'].append(metadata.SeriesDescription)
                        if 'orientation' in metadata_list:
                            metadata_dict['orientation'].append(metadata[0x2001100b].value)
                        if 'scanning' in metadata_list:
                            metadata_dict['scanning'].append(metadata.ScanningSequence)
                        if 'variation' in metadata_list:
                            metadata_dict['variation'].append(metadata.SequenceVariant)
                        if 'option' in metadata_list:
                            metadata_dict['option'].append(metadata.ScanOptions)
                        if 'dynamic' in metadata_list:
                            metadata_dict['dynamic'].append(metadata[0x20011012].value)
                        if 'b_factor' in metadata_list:
                            metadata_dict['b_factor'].append(metadata[0x20011003].value)
                else:
                    metadata = pm.dcmread(sequences[0], stop_before_pixels=True)
                    if 'description' in metadata_list:
                        metadata_dict['description'].append(metadata.SeriesDescription)
                    if 'orientation' in metadata_list:
                        metadata_dict['orientation'].append(metadata[0x2001100b].value)
                    if 'scanning' in metadata_list:
                        metadata_dict['scanning'].append(metadata.ScanningSequence)
                    if 'variation' in metadata_list:
                        metadata_dict['variation'].append(metadata.SequenceVariant)
                    if 'option' in metadata_list:
                        metadata_dict['option'].append(metadata.ScanOptions)
                    if 'dynamic' in metadata_list:
                        metadata_dict['dynamic'].append(metadata[0x20011012].value)
                    if 'b_factor' in metadata_list:
                        metadata_dict['b_factor'].append(metadata[0x20011003].value)
        
        return zip(*list(metadata_dict.values()))
                    
    def count_series_freq(
            self, 
            sequence_list: list
            ) -> None:

        '''
        Count the frequency of each sequence in a list

        Args:
            sequence_list (list): List of sequences to count.
        '''
        sequence_set = Counter(sequence_list).keys()
        sequence_counts = Counter(sequence_list).values()
        sequence_dict = dict(zip(sequence_set, sequence_counts))
        sequence_ordered = OrderedDict(sorted(sequence_dict.items(), key=itemgetter(1), reverse=True))
        print("{:<8} {:<10}".format('Sequence','Frequency'))
        for keys in sequence_ordered.items():
            sequence, freq = keys
            print(sequence, freq)

    def extract_list_of_series(
            self, 
            series_feature: str, 
            feature_label: list
            ) -> list:

        '''
        Extract list of series.

        Args:
            series_feature (str): Feature of the series to extract.
            feature_label (str): Label of the feature to extract.
        
        Returns:
            series_list (list): List of series.
        '''
        series_list = []
        for folder_path in tqdm(self.folders):
            series = glob(os.path.join(folder_path, '*/*'), recursive = True)
            for series_path in series:
                sequences = natsorted(glob(os.path.join(series_path, '*'), recursive = True))
                try:
                    metadata = pm.dcmread(sequences[0], stop_before_pixels=True)
                    if series_feature == 'orientation':
                        if any(label == metadata[0x2001100b].value.lower() for label in feature_label):
                            dicom_series, _ = os.path.split(sequences[0])
                            series_list.append(dicom_series)
                    elif series_feature == 'description':
                        if any(label in metadata.SeriesDescription.lower() for label in feature_label):
                            dicom_series, _ = os.path.split(sequences[0])
                            series_list.append(dicom_series)
                except:
                    print('Error reading metadata from: ', series_path)
        return series_list
    
    def rename_series(
            self, 
            old_name: list, 
            new_name: str, 
            dynamic_scan: str
            ) -> None:
            
        '''
        Rename series in the data directory.
    
        Args:
            old_name (str): Old name of the series.
            new_name (str): New name of the series.
            dynamic_scan (str): Dynamic scan of the series.
        '''
        for folder_path in tqdm(self.folders):
            series = glob(os.path.join(folder_path, '*/*'), recursive = True)
            for series_path in series:
                path_root, _ = os.path.split(series_path)
                sequences = natsorted(glob(os.path.join(series_path, '*'), recursive = True))
                if new_name == 'T1W_IOP' or new_name == 'T1W_DYN':
                    try:
                        metadata = pm.dcmread(sequences[0], stop_before_pixels=True)
                        if any(name == metadata.SeriesDescription for name in old_name) and metadata[0x20011012].value == dynamic_scan:
                            new_series_path = os.path.join(path_root, new_name)
                            os.rename(series_path, new_series_path)
                    except:
                        print('Folder {} could not be renamed.'.format(series_path))
                else:
                    try:
                        metadata = pm.dcmread(sequences[0], stop_before_pixels=True)
                        if any(name == metadata.SeriesDescription for name in old_name):
                            new_series_path = os.path.join(path_root, new_name)
                            os.rename(series_path, new_series_path)
                    except:
                        print('Folder {} could not be renamed.'.format(series_path))
        
    def separate_dicom_series(
            self, 
            series_type: str
            ) -> None:

        '''
        Separate T1-weighted images in the data directory by echo time and DWI by b-value.

        Args:
            series_type (str): Type of series to be separated.
        '''
        for folder_path in tqdm(self.folders):
            if series_type == 'T1W':
                sequences = natsorted(glob(os.path.join(folder_path, 'T1/T1W_IOP/*'), recursive = True))
            elif series_type == 'DWI':
                sequences = natsorted(glob(os.path.join(folder_path, 'DIFFUSION/DWI/*'), recursive = True))
            for sequence_path in sequences:
                path_root, _ = os.path.split(sequence_path)
                parent_folder, _ = os.path.split(path_root)
                metadata = pm.dcmread(sequence_path, stop_before_pixels=True)
                if series_type == 'T1W':
                    if metadata[0x20051011].value == 'OP':
                        new_folder = os.path.join(parent_folder, 'T1W_OOP')
                        if not os.path.exists(new_folder):
                            os.makedirs(new_folder)
                        shutil.move(sequence_path, new_folder)
                    elif metadata[0x20051011].value == 'IP':
                        new_folder = os.path.join(parent_folder, 'T1W_IP')
                        if not os.path.exists(new_folder):
                            os.makedirs(new_folder)
                        shutil.move(sequence_path, new_folder)
                elif series_type == 'DWI':
                    if round(metadata[0x20011003].value) <= 50:
                        new_folder = os.path.join(parent_folder, 'DWI_b0')
                        if not os.path.exists(new_folder):
                            os.makedirs(new_folder)
                        shutil.move(sequence_path, new_folder)
                    elif round(metadata[0x20011003].value) == 150:
                        new_folder = os.path.join(parent_folder, 'DWI_b150')
                        if not os.path.exists(new_folder):
                            os.makedirs(new_folder)
                        shutil.move(sequence_path, new_folder)
                    elif round(metadata[0x20011003].value) == 400:
                        new_folder = os.path.join(parent_folder, 'DWI_b400')
                        if not os.path.exists(new_folder):
                            os.makedirs(new_folder)
                        shutil.move(sequence_path, new_folder)
                    elif round(metadata[0x20011003].value) == 800:
                        new_folder = os.path.join(parent_folder, 'DWI_b800')
                        if not os.path.exists(new_folder):
                            os.makedirs(new_folder)
                        shutil.move(sequence_path, new_folder)

if __name__ == '__main__':
    PATH = '/Users/noltinho/thesis/sensitive_data/dicom'
    T1_IOP = ['mDIXON BH','DIXON','mDIXON','DIXON 4 reconstructies','mDIXON obese','4 reconstructies DIXON','mDIXON W 4s','mDIXON BH b-buik','mDIXON BH laat']
    T1_DYN = ['mDIXON W DYN BH','DIXON 4 FASEN','DIXON','mDIXON_dyn','DIXON fast 4 FASEN']
    T1_QUANT = ['mDIXON-Quant_BH','QUANT','QUANT T0','QUANT T1','QUANT T2','mDIXON-Quant_BH obese']
    T2_SHORT_TE = ['T2 TSE NAV','T2W_TSE_nav_FB 5mm','T2W_TSE_nav','T2W_TSE','T2  TSE','T2  TSE NAV','T2 TSE NAV S1.5','T2 TSE','T2 TSE single shot']
    T2_LONG_TE = ['T2 LANGE TE NAV','T2 LANGE TE BH']
    DWI = ['DWI 3B','DWI 3B NAV','DWI_11b_free breath tijdelijk indiengoed geen 3Bwaarden maken','DWI 3bw getriggert','DWI 3B OUD','DWI_11b_free breath','DWI 3B standaard']
    env = EnvironmentUtils(PATH)
    env.create_subfolders(['T1', 'T2', 'DIFFUSION', 'UNKNOWN'])
    env.move_subfolders()
    env.rename_subfolders()
    env.delete_subfolders_by_name(['S00','S10'], True)
    metadata = MetadataUtils(PATH)
    series_paths = metadata.extract_list_of_series('orientation',['coronal','sagittal'])
    env.delete_subfolders_by_path(series_paths)
    series_paths = metadata.extract_list_of_series('description',['reg','thrive','gado','dual','spair','t2 tse bh','t2w bh','mrcp','doorademen','dadc','dwip','ddwi','ediff'])
    env.delete_subfolders_by_path(series_paths)
    metadata.rename_series(T1_IOP, 'T1W_IOP', 'N')
    metadata.rename_series(T1_DYN, 'T1W_DYN', 'Y')
    metadata.rename_series(T1_QUANT, 'T1W_QNT', 'not relevant')
    metadata.rename_series(T2_SHORT_TE, 'T2W_TES', 'not relevant')
    metadata.rename_series(T2_LONG_TE, 'T2W_TEL', 'not relevant')
    metadata.rename_series(DWI, 'DWI', 'not relevant')
    metadata.separate_dicom_series('T1W')
    metadata.separate_dicom_series('DWI')
    env.delete_subfolders_by_name(['UNKNOWN'], False)
    env.delete_subfolders_by_name(['T1W_IOP','DWI'], True)


