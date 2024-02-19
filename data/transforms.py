import torch
from typing import Sequence
from monai.transforms import Transform
from monai.transforms.spatial.functional import resize
import SimpleITK as sitk
import numpy as np
from scipy.interpolate import interp1d


class SpatialCropPercentiled(Transform):

    def __init__(
            self,
            keys: str | list,
            roi_center: Sequence[float],
            roi_size: Sequence[float],
            min_size: Sequence[int]
        ) -> None:

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
        

class DivisiblePercentileCropd(Transform):

    def __init__(
            self,
            keys: str | list,
            roi_center: Sequence[float],
            k_divisible: Sequence[int],
        ) -> None:

        self.keys = [keys] if isinstance(keys, str) else keys
        self.roi_center = roi_center
        self.k_divisible = k_divisible
        
    def __call__(self, image: torch.Tensor):

        for key in self.keys:
            cropped_image = []
            for channel in image[key]:
                shape = channel.shape
                roi_center = [int(self.roi_center[i] * shape[i]) for i in range(len(shape))]
                roi_size = [round(roi_center[i] / self.k_divisible[i] * 2) / 2 for i in range(len(shape))]
                roi_size = [roi_size[i] - 0.5 if roi_center[i] / roi_size[i] < self.k_divisible[i] else roi_size[i] for i in range(len(shape))]
                roi_start = [int(roi_center[i] - roi_size[i] * self.k_divisible[i]) for i in range(len(shape))]
                roi_end = [int(roi_center[i] + roi_size[i] * self.k_divisible[i]) for i in range(len(shape))]
                cropped_image.append(channel[roi_start[0]:roi_end[0], roi_start[1]:roi_end[1], roi_start[2]:roi_end[2]])
            image[key] = torch.stack(cropped_image)
        return image


class ResizeToMatchd(Transform):

    def __init__(
            self,
            keys: str | list,
            dst_key: str,
            mode: str = 'nearest'
        ) -> None:

        self.keys = [keys] if isinstance(keys, str) else keys
        self.dst_key = [dst_key] if isinstance(dst_key, str) else dst_key
        self.mode = mode

    def __call__(self, image: torch.Tensor):

        image_list = []
        shape = [list(image[key].size()[1:]) for key in self.dst_key][0]
        for key in self.keys:
            resized_image = resize(
                image[key], tuple(shape), mode=self.mode, align_corners=None, dtype=torch.float32,
                input_ndim=3, anti_aliasing=False, anti_aliasing_sigma=None, lazy=False, transform_info=None)
            image_list.append(resized_image.squeeze(0))
            image[key] = torch.stack(image_list)
        return image


class CorrectBiasFieldd(Transform):

    def __init__(
            self,
            keys: str | list,
            shrink_factor: int = 2,
            per_slice: bool = False,
            fit_levels: int = 4,
            max_iter: int = 50
        ) -> None:

        self.keys = [keys] if isinstance(keys, str) else keys
        self.shrink_factor = shrink_factor
        self.per_slice = per_slice
        self.fit_levels = fit_levels
        self.max_iter = max_iter

    def __call__(self, image: torch.Tensor):

        for key in self.keys:
            for channel in range(image[key].shape[0]):
                if self.per_slice:
                    for slice in range(image[key].shape[-1]):
                        image_sitk = sitk.GetImageFromArray(image[key][channel, :, :, slice])
                        image_shrunk = sitk.Shrink(image_sitk, [self.shrink_factor] * image_sitk.GetDimension())
                        corrector = sitk.N4BiasFieldCorrectionImageFilter()
                        corrector.Execute(image_shrunk)
                        log_bias = corrector.GetLogBiasFieldAsImage(image_sitk)
                        image[key][channel, :, :, slice] = torch.from_numpy(sitk.GetArrayFromImage(image_sitk / sitk.Exp(log_bias)))
                else:
                    image_sitk = sitk.GetImageFromArray(image[key][channel])
                    image_shrunk = sitk.Shrink(image_sitk, [self.shrink_factor] * image_sitk.GetDimension())
                    corrector = sitk.N4BiasFieldCorrectionImageFilter()
                    corrector.SetMaximumNumberOfIterations([self.max_iter] * self.fit_levels)
                    corrector.Execute(image_shrunk)
                    log_bias = corrector.GetLogBiasFieldAsImage(image_sitk)
                    image[key][channel] = torch.from_numpy(sitk.GetArrayFromImage(image_sitk / sitk.Exp(log_bias)))
        return image
    

class NyulNormalized(Transform):

    def __init__(
            self,
            keys: str | list,
            standard_hist: dict,
            min_perc: float = 0.01,
            max_perc: float = 0.99,
            step_size: int = 1,
            clip: bool = False,
        ) -> None:

        self.keys = [keys] if isinstance(keys, str) else keys
        self.standard_hist = standard_hist
        self.min_perc = min_perc * 100
        self.max_perc = max_perc * 100
        self.step_size = step_size
        self.clip = clip

    def set_standard_hist(self, standard_hist: dict):

        if self.step_size == 1:
            perc_idx = np.arange(self.min_perc - 1, self.max_perc, self.step_size)
        else:
            perc_idx = np.concatenate(([self.min_perc - 1], np.arange(self.step_size, 100, self.step_size), [self.max_perc - 1]))
        perc_idx = [int(x) for x in perc_idx]
        standard_hist = standard_hist[perc_idx]
        interp = interp1d([standard_hist[0], standard_hist[-1]], [0, 1])
        return interp(standard_hist)

    def get_image_landmarks(self, image: torch.Tensor):

        if self.step_size == 1:
            percs = np.arange(self.min_perc, self.max_perc + 1, self.step_size)
        else:
            percs = np.concatenate(([self.min_perc], np.arange(self.step_size, 100, self.step_size), [self.max_perc]))
        return np.percentile(image, percs)

    def __call__(self, image: torch.Tensor):

        for key in self.keys:
            mask = (image[key] > torch.mean(image[key]) - 0.25 * torch.std(image[key])) & (image[key] < torch.mean(image[key]) + 2.75 * torch.std(image[key]))
            masked_image = image[key][mask > 0]
            standard_landmarks = self.set_standard_hist(self.standard_hist[key])
            image_landmarks = self.get_image_landmarks(masked_image)
            if self.clip:
                interp = interp1d(image_landmarks, standard_landmarks, kind='linear', fill_value=(0, 1), bounds_error=False)
            else:
                interp = interp1d(image_landmarks, standard_landmarks, kind='linear', fill_value='extrapolate')
            image[key] = torch.from_numpy(interp(image[key]))
        return image


class SoftClipIntensityd(Transform):

    def __init__(
            self,
            keys: str | list,
            min_value: float | None = None,
            max_value: float | None = None,
            channel_wise: bool = False
        ) -> None:

        self.keys = [keys] if isinstance(keys, str) else keys
        self.min_value = min_value
        self.max_value = max_value
        self.channel_wise = channel_wise

    @staticmethod
    def softplus(x: torch.Tensor):

        return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x, 0)
    
    def softminus(self, x: torch.Tensor):

        return -self.softplus(-x)
    
    def softclip(self, x: torch.Tensor):

        tanh = 1 - np.tanh(1)
        const = np.log(2) / tanh

        if self.min_value is not None and self.max_value is not None:
            const /= (self.max_value - self.min_value) / 2

        v = x
        if self.min_value is not None:
            v = v - self.softminus(const * (x - self.min_value)) / const
        if self.max_value is not None:
            v = v - self.softplus(const * (x - self.max_value)) / const
        return v

    
    def __call__(self, image: torch.Tensor):

        for key in self.keys:
            if self.channel_wise:
                for channel in range(image[key].shape[0]):
                    image[key][channel] = torch.from_numpy(self.softclip(image[key][channel]))
            else:
                image[key] = torch.from_numpy(self.softclip(image[key]))
        return image


class RobustNormalized(Transform):

    def __init__(
            self,
            keys: str | list,
            channel_wise: bool = False
        ) -> None:

        self.keys = [keys] if isinstance(keys, str) else keys
        self.channel_wise = channel_wise

    def __call__(self, image: torch.Tensor):

        for key in self.keys:
            if self.channel_wise:
                for channel in range(image[key].shape[0]):
                    median = torch.median(image[key][channel])
                    mad = torch.median(torch.abs(image[key][channel] - median))
                    image[key][channel] = (image[key][channel] - median) / mad
            else:
                median = torch.median(image[key])
                mad = torch.median(torch.abs(image[key] - median))
                image[key] = (image[key] - median) / mad
        return image