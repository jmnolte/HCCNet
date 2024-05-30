import torch
from monai import transforms
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    CropForegroundd,
    ConcatItemsd,
    CenterSpatialCropd,
    ResampleToMatchd,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
    DeleteItemsd,
    RandSpatialCropd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    CopyItemsd,
    KeepLargestConnectedComponentd,
    Lambdad,
    NormalizeIntensityd,
    SpatialPadd
)
from data.transforms import (
    PercentileSpatialCropd, 
    YeoJohnsond, 
    SoftClipOutliersd, 
    ResampleToMatchFirstd, 
    RandSelectChanneld
)

def transforms(
        dataset: str,
        modalities: list,
        device: torch.device,
        crop_size: tuple = (72, 72, 72),
        image_spacing: tuple = (1.5, 1.5, 1.5)
    ) -> transforms:

    '''
    Args:
        dataset (str): Dataset to apply transformations on. Can be 'train', 'val' or 'test'.
        modalities (list): List of image modalities to perform transformations on.
        device (torch.device): Pytorch device.
        crop_size (tuple): Tuple of integers specifying the image size.
        image_spacing (tuple): Tuple of floats specifying the spacing between MRI slides.
    '''
    if any('DWI' in mod for mod in modalities):
        mean = (0.5396, 0.5280, 0.5601, 0.5737)
        std = (0.8415, 0.7934, 0.7670, 0.7032) 
    elif any('T1WI' in mod for mod in modalities):
        mean = (0.6991, 0.6642, 0.7852, 0.8884)
        std = (0.7520, 0.7403, 0.7812, 0.8444)
    elif any('T1W_IP' in mod for mod in modalities):
        mean = (-0.2312, -0.3776)
        std = (0.6274, 0.7104)
    elif any('T2W' in mod for mod in modalities):
        mean = (-0.1640, -0.0783)
        std = (0.5422, 0.5877)

    prep = [
        LoadImaged(keys=modalities, image_only=True),
        EnsureChannelFirstd(keys=modalities),
        Orientationd(keys=modalities, axcodes='PLI'),
        YeoJohnsond(keys=modalities, lmbda=0.5),
        ResampleToMatchd(keys=modalities, key_dst=modalities[0], mode=3),
        ConcatItemsd(keys=modalities, name='image'),
        CopyItemsd(keys=modalities[0], names='mask'),
        PercentileSpatialCropd(
            keys=['image','mask'],
            roi_center=(0.5, 0.5, 0.5),
            roi_size=(0.85, 0.8, 0.99),
            min_size=(82, 82, 82)),
        Lambdad(
            keys='mask', 
            func=lambda x: torch.where(x > torch.mean(x), 1, 0)),
        KeepLargestConnectedComponentd(keys='mask', connectivity=1),
        CropForegroundd(
            keys='image',
            source_key='mask',
            select_fn=lambda x: x > 0,
            k_divisible=1,
            allow_smaller=False),
        DeleteItemsd(keys=modalities + ['mask']),
        SoftClipOutliersd(keys='image', scale_factor=3.5, channel_wise=True),
        NormalizeIntensityd(keys='image', channel_wise=True),
        Spacingd(keys='image', pixdim=(1.5, 1.5, 1.5), mode=3),
        PercentileSpatialCropd(
            keys='image',
            roi_center=(0.5, 0.3, 0.3),
            roi_size=(0.6, 0.5, 0.4),
            min_size=(82, 82, 82)),
        SpatialPadd(keys='image', spatial_size=(72, 72, 72)),
        CenterSpatialCropd(keys='image', roi_size=(96, 96, 96))
    ]

    train = [
        EnsureTyped(keys='image', track_meta=False, device=device, dtype=torch.float),
        RandSpatialCropd(keys='image', roi_size=(72, 72, 72), random_size=False),
        RandFlipd(keys='image', spatial_axis=0, prob=0.5),
        RandFlipd(keys='image', spatial_axis=1, prob=0.5),
        RandFlipd(keys='image', spatial_axis=2, prob=0.5),
        RandRotate90d(keys='image', prob=0.5),
        RandScaleIntensityd(keys='image', prob=1.0, factors=0.1, channel_wise=True),
        RandShiftIntensityd(keys='image', prob=1.0, offsets=0.1, channel_wise=True),
        NormalizeIntensityd(keys='image', subtrahend=mean, divisor=std, channel_wise=True)
    ]

    test = [
        CenterSpatialCropd(keys='image', roi_size=(72, 72, 72)),
        NormalizeIntensityd(keys='image', subtrahend=mean, divisor=std, channel_wise=True),
        EnsureTyped(keys='image', track_meta=False, device=device, dtype=torch.float)
    ]

    if dataset == 'train':
        return Compose(prep + train)
    elif dataset in ['val', 'test']:
        return Compose(prep + test)
    else:
        raise ValueError ("Dataset must be 'train', 'val' or 'test'.")

def dino_transforms(
        modalities: list,
        device: torch.device,
        global_crop_size: tuple = (72, 72, 72),
        local_crop_size: tuple = (48, 48, 48),
        image_spacing: tuple = (1.5, 1.5, 1.5)
    ) -> transforms:
    '''
    Args:
        modalities (list): List of image modalities to perform transformations on.
        device (torch.device): Pytorch device.
        global_crop_size (tuple): Tuple of integers specifying the size of the global views.
        global_crop_size (tuple): Tuple of integers specifying the size of the local views.
        image_spacing (tuple): Tuple of floats specifying the spacing between MRI slides.
    '''
    if any('DWI' in mod for mod in modalities):
        mean = 0.545
        std = 0.778
    elif any('T1WI' in mod for mod in modalities):
        mean = 0.748
        std = 0.784
    elif any('T1W_IP' in mod for mod in modalities):
        mean = -0.208
        std = 0.619

    prep = [
        LoadImaged(keys=modalities, image_only=True, allow_missing_keys=True),
        EnsureChannelFirstd(keys=modalities, allow_missing_keys=True),
        Orientationd(keys=modalities, axcodes='PLI', allow_missing_keys=True),
        YeoJohnsond(keys=modalities, lmbda=0.5, allow_missing_keys=True),
        ResampleToMatchFirstd(keys=modalities, mode=3, allow_missing_keys=True),
        ConcatItemsd(keys=modalities, name='image', allow_missing_keys=True),
        CopyItemsd(keys='image', names='mask'),
        Lambdad(keys='mask', func=lambda x: x[:1]),
        PercentileSpatialCropd(
            keys=['image','mask'],
            roi_center=(0.5, 0.5, 0.5),
            roi_size=(0.85, 0.8, 0.99),
            min_size=(82, 82, 82)),
        Lambdad(keys='mask', func=lambda x: torch.where(x > torch.mean(x), 1, 0)),
        KeepLargestConnectedComponentd(keys='mask', connectivity=1),
        CropForegroundd(
            keys='image',
            source_key='mask',
            select_fn=lambda x: x > 0,
            k_divisible=1,
            allow_smaller=False),
        SoftClipOutliersd(keys='image', scale_factor=3.5, channel_wise=True),
        NormalizeIntensityd(keys='image', channel_wise=True),
        Spacingd(keys='image', pixdim=(1.5, 1.5, 1.5), mode=3),
        PercentileSpatialCropd(
            keys='image',
            roi_center=(0.5, 0.3, 0.3),
            roi_size=(0.6, 0.5, 0.4),
            min_size=(82, 82, 82)),
        SpatialPadd(keys='image', spatial_size=(72, 72, 72)),
        CenterSpatialCropd(keys='image', roi_size=(96, 96, 96)),
        CopyItemsd(keys='image', times=4, names=['gv1','gv2','lv1','lv2']),
        DeleteItemsd(keys=modalities + ['image','mask']),
        EnsureTyped(keys=['gv1','gv2','lv1','lv2'], track_meta=False, device=device, dtype=torch.float),
    ]

    global_crop = [
        RandSpatialCropd(keys=['gv1','gv2'], roi_size=global_crop_size, random_size=False)
    ]

    local_crop = [
        RandSpatialCropd(keys=['lv1','lv2'], roi_size=local_crop_size, random_size=False)
    ]

    post = [
        RandFlipd(keys=['gv1','gv2','lv1','lv2'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['gv1','gv2','lv1','lv2'], prob=0.5, spatial_axis=1),
        RandFlipd(keys=['gv1','gv2','lv1','lv2'], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=['gv1','gv2','lv1','lv2'], prob=0.5),
        RandScaleIntensityd(keys=['gv1','gv2','lv1','lv2'], prob=1.0, factors=0.1, channel_wise=True),
        RandShiftIntensityd(keys=['gv1','gv2','lv1','lv2'], prob=1.0, offsets=0.1, channel_wise=True),
        RandSelectChanneld(keys=['gv1','gv2','lv1','lv2'], num_channels=1),
        NormalizeIntensityd(keys=['gv1','gv2','lv1','lv2'], subtrahend=mean, divisor=std)
    ]
    return Compose(prep + global_crop + local_crop + post)