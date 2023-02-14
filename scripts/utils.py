import os
from glob import glob
import pydicom as pm
import shutil
from tqdm import tqdm
from natsort import natsorted

class EnvironmentUtils:
    def __init__(self, path: str) -> None:

        '''
        Initialize the environment utils class.

        Args:
            path (str): Path to the data.
        '''

        self.path = path
        self.sequences = glob(os.path.join(self.path, '*/DICOM/*/*'), recursive = True)
        self.folders = glob(os.path.join(self.path, '*/DICOM/*'), recursive = True)

        # elif self.subfolders == 'names':
        #     self.folders = glob(os.path.join(self.path, '*'), recursive = True)

    def create_subfolders(self, subfolders: list) -> None:

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

    def delete_subfolders_by_path(self, subfolder_paths: list) -> None:

        '''
        Delete subfolders in the data directory.

        Args:
            subfolders (list): List of subfolder paths to delete.
        '''
        for subfolder_path in tqdm(subfolder_paths):
            if os.path.exists(subfolder_path):
                shutil.rmtree(subfolder_path)
                print('Deleting {}.'.format(subfolder_path))

    def delete_subfolders_by_name(self, subfolder_list: list) -> None:

        '''
        Delete subfolders in the data directory.

        Args:
            subfolders (list): List of subfolder names to delete.
        '''
        for folder_path in tqdm(self.folders):
            for subfolder in subfolder_list:
                subfolder_path = os.path.join(folder_path, subfolder)
                if os.path.exists(subfolder_path):
                    shutil.rmtree(subfolder_path)
                    print('Deleting {}.'.format(subfolder_path))

    def move_subfolders(self) -> None:

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
                    contrast_id = metadata.AcquisitionContrast
                    series_id = metadata.SeriesDescription

                    if contrast_id == 'T1' or series_id[:2] == 'T1':
                        shutil.move(series_path, t1_path)
                    elif contrast_id == 'T2' or series_id[:2] == 'T2':
                        shutil.move(series_path, t2_path)
                    elif contrast_id == 'DIFFUSION':
                        shutil.move(series_path, diff_path)
                    else:
                        shutil.move(series_path, unknown_path)

    def rename_subfolders(self, folder_id: str) -> None:

        '''
        Rename subfolders in the data directory.

        Args:
            folder_id (str): ID to rename the subfolders.
        '''

        for folder in self.folders:
    
            path_root, _ = os.path.split(folder)
            old_path = folder

            if folder_id == 'date_id':
                dicoms = glob(os.path.join(folder, '*/*/I10'), recursive = True)
            elif folder_id == 'name_id':
                dicoms = glob(os.path.join(folder, 'DICOM/*/*/I10'), recursive = True)

            for dicom in dicoms:
                new_path = dicom
                metadata = pm.dcmread(new_path, stop_before_pixels=True)

                if folder_id == 'date_id':
                    new_subfolder = metadata.StudyDate
                elif folder_id == 'name_id':
                    new_subfolder = metadata.PatientName

            new_path = os.path.join(path_root, str(new_subfolder))
            if old_path != new_path:
                print('Folder {} does not match dicom metadata and is renamed to {}.'.format(old_path, new_path))
                os.rename(old_path, new_path)
            else:
                print('Folder {} matches dicom metadata.'.format(old_path))

PATH = '/Users/noltinho/thesis_private/data'
SUBFOLDERS = ['S00']

# env = EnvironmentUtils(PATH)
# env.delete_subfolders_by_name(SUBFOLDERS)

