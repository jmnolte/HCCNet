import torch
from typing import Sequence
from monai.transforms import Transform, Randomizable, ResampleToMatchd
from monai.transforms.spatial.functional import resize
from monai.config import DtypeLike, KeysCollection, SequenceStr
from collections.abc import Hashable, Mapping, Sequence
from monai.utils import GridSampleMode
import numpy as np

class PercentileSpatialCropd(Transform):

    '''
    Generic transform that crops images based on provided percentiles. For a given image of shape 80x80x80
    and a roi_size of (0.75, 0.75, 0.75), the resulting image is of shape 60x60x60.
    '''

    def __init__(
            self,
            keys: str | list,
            roi_center: Sequence[float],
            roi_size: Sequence[float],
            min_size: Sequence[int]
        ) -> None:

        '''
        Args:
            keys (str | list): String or list of strings to perform transform on.
            roi_center (Sequence[float]): Percentile of the center voxels per axis.
            roi_size (Sequence[float]): Size of the region of interest in percent.
            min_size (Sequence[float]): Minimum size of the cropped image in absolute values.
        '''

        self.keys = [keys] if isinstance(keys, str) else keys
        self.roi_center = roi_center
        self.roi_size = roi_size
        self.min_size = min_size
        
    def __call__(self, image: torch.Tensor):

        for key in self.keys:
            cropped_image = []
            for channel in image[key]:
                shape = channel.shape            
                roi_center = [int(self.roi_center[i] * shape[i]) for i in range(len(shape))]
                roi_size = [int(self.roi_size[i] * shape[i]) for i in range(len(shape))]
                roi_start = [int(roi_center[i] - roi_size[i] / 2) for i in range(len(shape))]
                roi_end = [int(roi_center[i] + roi_size[i] / 2) for i in range(len(shape))]
                roi_end = [self.min_size[i] + roi_start[i] if roi_end[i] - roi_start[i] < self.min_size[i] else roi_end[i] for i in range(len(shape))]
                cropped_image.append(channel[roi_start[0]:roi_end[0], roi_start[1]:roi_end[1], roi_start[2]:roi_end[2]])
            image[key] = torch.stack(cropped_image)
        return image

class YeoJohnsond(Transform):

    '''
    Transform to adjust the intensity values of the provided images using the Yeo-Johnson transformation.
    '''

    def __init__(
            self,
            keys: str | list,
            lmbda: float | Sequence[float],
            channel_wise: bool = False,
            allow_missing_keys: bool = False
        ) -> None:

        '''
        Args:
            keys (str | list): String or list of strings to perform transform on.
            lmbda (float | Sequence[float]): Strength of the transformation. Smaller values represents a stronger intensity adjustment.
            channel_wise (bool): Whether to perform the transform on all channels independently.
            allow_missing_keys (bool): Whether to raise an exception when encountering missing keys. 
        '''

        self.keys = [keys] if isinstance(keys, str) else keys
        self.lmbda = lmbda
        self.channel_wise = channel_wise
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, image: torch.Tensor):

        for key in self.keys:
            if self.allow_missing_keys and key not in image:
                continue
            if self.channel_wise:
                for channel in range(image[key].shape[0]):
                    lmbda = self.lmbda[channel] if isinstance(self.lmbda, Sequence) else self.lmbda
                    positives = image[key][channel] >= 0
                    negatives = image[key][channel] < 0
                    if lmbda == 0 or lmbda == 2:
                        image[key][channel][positives] = torch.log1p(image[key][channel][positives])
                        image[key][channel][negatives] = -torch.log1p(-image[key][channel][negatives])
                    else:
                        image[key][channel][positives] = ((image[key][channel][positives] + 1) ** lmbda - 1) / lmbda
                        image[key][channel][negatives] = -((-image[key][channel][negatives] + 1) ** (2 - lmbda) - 1) / (2 - lmbda)
            else:
                positives = image[key] >= 0
                negatives = image[key] < 0
                if self.lmbda == 0 or self.lmbda == 2:
                    image[key][positives] = torch.log1p(image[key][positives])
                    image[key][negatives] = -torch.log1p(-image[key][negatives])
                else:
                    image[key][positives] = ((image[key][positives] + 1) ** self.lmbda - 1) / self.lmbda
                    image[key][negatives] = -((-image[key][negatives] + 1) ** (2 - self.lmbda) - 1) / (2 - self.lmbda)
        return image

class RandSelectChanneld(Transform, Randomizable):

    '''
    Transform that randomly selects a channel from a multi-channel image.
    '''

    def __init__(
            self,
            keys: str | list,
            num_channels: int
        ) -> None:

        '''
        Args:
            keys (str | list): String or list of strings to perform transform on.
            num_channels (int): Number of channels to select randomly.
        '''

        self.keys = [keys] if isinstance(keys, str) else keys
        self.num_channels = num_channels

    def __call__(self, image: torch.Tensor):

        for key in self.keys:
            shape = image[key].shape
            image[key] = image[key][torch.randperm(shape[0])[:self.num_channels]]
        return image

class ResampleToMatchFirstd(ResampleToMatchd):

    '''
    Adaptation of ResampleToMatchd transform that automatically selects the first channel to resample the remaining 
    channels to. 
    '''

    def __init__(
            self,
            keys: KeysCollection,
            mode: SequenceStr = GridSampleMode.BILINEAR,
            align_corners: Sequence[bool] | bool = False,
            dtype: Sequence[DtypeLike] | DtypeLike = np.float64,
            allow_missing_keys: bool = False,
            lazy: bool = False
        ) -> None:

        '''
        Args:
            keys (str | list): String or list of strings to perform transform on.
            mode (SequenceStr): Resampling mode.
            align_corners (Sequence[bool] | bool): Whether to align corners.
            dtype (Sequence[DtypeLike] | DtypeLike): Output data type.
            allow_missing_keys (bool): Whether to raise an exception when encountering missing keys. 
            lazy (bool): Whether to allow lazy transformation.
        '''

        super().__init__(
            keys=keys,
            key_dst='dst',
            mode=mode,
            align_corners=align_corners,
            dtype=dtype,
            allow_missing_keys=allow_missing_keys,
            lazy=lazy)
        self.key_dst = None
        
    def __call__(
            self, 
            data: Mapping[Hashable, torch.Tensor], 
            lazy: bool | None = None
        ) -> dict[Hashable, torch.Tensor]:

        lazy_ = self.lazy if lazy is None else lazy
        key_dst = self.key_dst
        d = dict(data)
        for key, mode, padding_mode, align_corners, dtype in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners, self.dtype
        ):
            key_dst = key if key_dst is None else key_dst
            d[key] = self.resampler(
                img=d[key],
                img_dst=d[key_dst],
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
                dtype=dtype,
                lazy=lazy_,
            )
        return d

class SoftClipOutliersd(Transform):

    '''
    Transform that clips the intensity values of a provided image based on the median absolute deviation.
    '''

    def __init__(
            self,
            keys: str | list,
            scale_factor: float = 1.5,
            channel_wise: bool = False
        ) -> None:

        '''
        Args:
            keys (str | list): String or list of strings to perform transform on.
            scale_factor (float): Maximum median absolute deviation.
            Whether to perform the transform on all channels independently.
        '''

        self.keys = [keys] if isinstance(keys, str) else keys
        self.scale_factor = scale_factor
        self.channel_wise = channel_wise

    @staticmethod
    def softplus(x: torch.Tensor):

        other = torch.tensor([0])
        return torch.log(1 + torch.exp(-torch.abs(x))) + torch.maximum(x, other)
    
    def softminus(self, x: torch.Tensor):

        return -self.softplus(-x)
    
    def softclip(self, x: torch.Tensor, lower: float | None, upper: float | None):

        tanh = 1 - torch.tanh(torch.tensor([1]))
        const = torch.log(torch.tensor([2])) / tanh

        if lower is not None and upper is not None:
            const /= (upper - lower) / 2

        v = x
        if lower is not None:
            v = v - self.softminus(const * (x - lower)) / const
        if upper is not None:
            v = v - self.softplus(const * (x - upper)) / const
        return v
    
    def __call__(self, image: torch.Tensor):

        for key in self.keys:
            if self.channel_wise:
                for channel in range(image[key].shape[0]):
                    median = torch.median(image[key][channel])
                    mad = torch.median(torch.abs(image[key][channel] - median)) * 1.4826
                    min_value = median - mad * self.scale_factor
                    max_value = median + mad * self.scale_factor
                    image[key][channel] = self.softclip(image[key][channel], min_value, max_value)
            else:
                median = torch.median(image[key])
                mad = torch.median(torch.abs(image[key] - median)) * 1.4826
                min_value = median - mad * self.scale_factor
                max_value = median + mad * self.scale_factor
                image[key] = self.softclip(image[key], min_value, max_value)
        return image