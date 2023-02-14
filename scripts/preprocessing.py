import monai
import numpy as np
import glob
import os
import nibabel as nib

class Transforms(monai.transforms):
    def __init__(self, image_path: str, label_path: str) -> None:
        super().__init__()
        self.image_path = image_path
        self.label_path = label_path
        self.image_files = glob(os.path.join(self.image_path, '*.nii.gz'))
        self.label_files = glob(os.path.join(self.label_path, '*.nii.gz'))

    def concatenate_volumes(self, patient_id: str) -> np.ndarray:
        image_files = glob(os.path.join(self.image_path, patient_id, '*.nii.gz'))
        return np.concatenate([np.expand_dims(np.array(nib.load(image_file).get_fdata()), axis=0) for image_file in image_files], axis=0)

    def data_transformation(self, mean_intensity: list, std_intensity: list) -> monai.transforms:
        transforms = monai.transforms.Compose([
            monai.transforms.LoadImaged(keys=['image', 'label']),
            monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
            monai.transforms.Orientationd(keys=['image', 'label'], axcodes='PLI'),
            monai.tarnsforms.Resized(keys=['image', 'label'], spatial_size=(224, 224, 224)),
            monai.transforms.Spacingd(keys=['image', 'label'], 
                                    pixdim=(1.5, 1.5, 2.0)),
            monai.transforms.NormalizeIntensityd(keys=['image'], 
                                                 subtrahend=mean_intensity, 
                                                 divisor=std_intensity,
                                                 channel_wise=True),
            monai.transforms.ToTensord(keys=['image', 'label'])
            ])
        return transforms
    
    def data_augmentation(self, transform: str, mean_intensity: list, std_intensity: list) -> monai.transforms:
        transforms = monai.transforms.Compose([
            monai.transforms.LoadImaged(keys=['image', 'label']),
            monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
            monai.transforms.Orientationd(keys=['image', 'label'], axcodes='PLI'),
            monai.tarnsforms.Resized(keys=['image', 'label'], spatial_size=(224, 224, 224)),
            monai.transforms.Spacingd(keys=['image', 'label'], 
                                    pixdim=(1.5, 1.5, 2.0)),
            monai.transforms.NormalizeIntensityd(keys=['image'], 
                                                 subtrahend=mean_intensity, 
                                                 divisor=std_intensity,
                                                 channel_wise=True)            
            ])
        rand_rotation = monai.transforms.RandRotated(keys=['image', 'label'], prob=1,
                                                    range_x=np.pi/8, range_y=np.pi/8, range_z=np.pi/8)
        rand_gaussian_noise = monai.transforms.RandGaussianNoised(keys=['image'], prob=1, mean=0, std=0.1)
        combined = monai.transforms.Compose([rand_rotation, rand_gaussian_noise])
        to_tensor = monai.transforms.ToTensord(keys=['image', 'label'])

        if transform == 'rand_rotation':
            return monai.transforms.Compose([transforms, rand_rotation, to_tensor])
        elif transform == 'gaussian_noise':
            return monai.transforms.Compose([transforms, rand_gaussian_noise, to_tensor])
        elif transform == 'all':
            return monai.transforms.Compose([transforms, combined, to_tensor])