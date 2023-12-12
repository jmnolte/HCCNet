import torch
from typing import Sequence
from monai.transforms import Transform
from monai.transforms.spatial.functional import resize


class SpatialCropPercentiled(Transform):

    def __init__(
            self,
            keys: str | list,
            roi_center: Sequence[float],
            roi_size: Sequence[int],
        ) -> None:

        self.keys = [keys] if isinstance(keys, str) else keys
        self.roi_center = roi_center
        self.roi_size = roi_size
        
    def __call__(self, image: torch.Tensor):

        for key in self.keys:
            cropped_image = []
            for channel in image[key]:
                shape = channel.shape
                roi_center = [int(self.roi_center[i] * shape[i]) for i in range(len(shape))]
                roi_size = [self.roi_size[i] if self.roi_size[i] != -1 else shape[i] for i in range(len(shape))]
                roi_start = [int(roi_center[i] - roi_size[i] / 2) for i in range(len(shape))]
                roi_end = [int(roi_center[i] + roi_size[i] / 2) for i in range(len(shape))]
                cropped_image.append(channel[roi_start[0]:roi_end[0], roi_start[1]:roi_end[1], roi_start[2]:roi_end[2]])
            image[key] = torch.stack(cropped_image)
        return image


class DivisiblePercentileCropd(Transform):

    def __init__(
            self,
            keys: str | list,
            roi_center: Sequence[float],
            k_divisible: int = 1,
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
                roi_size = [round(roi_center[i] / self.k_divisible * 2) / 2 for i in range(len(shape))]
                roi_size = [roi_size[i] - 0.5 if roi_center[i] / roi_size[i] < self.k_divisible else roi_size[i] for i in range(len(shape))]
                roi_start = [int(roi_center[i] - roi_size[i] * self.k_divisible) for i in range(len(shape))]
                roi_end = [int(roi_center[i] + roi_size[i] * self.k_divisible) for i in range(len(shape))]
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