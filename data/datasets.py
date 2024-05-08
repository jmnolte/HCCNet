from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from monai.transforms import (
    Compose,
    RandomizableTrait,
    Transform,
    apply_transform,
    convert_to_contiguous
)
from monai.data import CacheDataset
from copy import deepcopy
from monai.data.utils import pickle_hashing
from multiprocessing.managers import ListProxy
from multiprocessing.pool import ThreadPool
from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset
from torch.multiprocessing import Manager
import torch
import sys
import warnings
from tqdm import tqdm

has_tqdm = True

class SeqDataset(_TorchDataset):
    """
    A generic dataset for longitudinal imaging data that has a length property and an 
    optional callable data transform when fetching a data sample.

    For example, typical input data can be a list of dictionaries::

        [{                                                                      {
             'img': ['image1_1.nii.gz', 'image1_2.nii.gz', 'image1_3.nii.gz'],      'img': ['image2_1.nii.gz', 'image2_2.nii.gz', 'image2_3.nii.gz']
             'seg': ['seg1_1.nii.gz', 'seg1_2.nii.gz', 'seg1_3.nii.gz'],            'seg': ['seg2_1.nii.gz', 'seg2_2.nii.gz', 'seg2_3.nii.gz']
             'extra': [123, 456, 789]                                               'extra': [123, 456, 789]
         },                                                                     }]
    """

    def __init__(self, data: Sequence, image_keys: str | list, transform: Callable | None = None) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            image_keys: keys to create imaging data from (e.g., 'T1' or 'T2').
            transform: a callable data transform on input data.

        """
        self.data = data
        self.image_keys = [image_keys] if isinstance(image_keys, str) else image_keys
        self.transform: Any = transform

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, idx: int):
        """
        Fetch single data item from `self.data`.
        """
    
        data_keys = list(self.data[idx].keys())
        data_keys = [key for key in data_keys if key not in self.image_keys]
        seq_len = len(self.data[idx][self.image_keys[0]])
        item_seq = []
        for i in range(seq_len):
            image_dict = {key: [self.data[idx][key][i]] for key in self.image_keys}
            data_dict = {key: self.data[idx][key][i] for key in data_keys}
            image_dict.update(data_dict) 
            data = apply_transform(self.transform, image_dict)
            item_seq.append(data)

        return item_seq

    def __getitem__(self, index: int | slice | Sequence[int]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices=index)
        data = self._transform(index)
        return data
    

class CacheSeqDataset(SeqDataset, CacheDataset):
    """
    Dataset for longitudinal imaging data with cache mechanism that can load data and cache deterministic transforms' result during training.

    By caching the results of non-random preprocessing transforms, it accelerates the training data pipeline.
    If the requested data is not in the cache, all transforms will run normally
    (see also :py:class:`monai.data.dataset.Dataset`).

    Users can set the cache rate or number of items to cache.
    It is recommended to experiment with different `cache_num` or `cache_rate` to identify the best training speed.

    The transforms which are supposed to be cached must implement the `monai.transforms.Transform`
    interface and should not be `Randomizable`. This dataset will cache the outcomes before the first
    `Randomizable` `Transform` within a `Compose` instance.
    So to improve the caching efficiency, please always put as many as possible non-random transforms
    before the randomized ones when composing the chain of transforms.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, if the transform is a `Compose` of::

        transforms = Compose([
            LoadImaged(),
            EnsureChannelFirstd(),
            Spacingd(),
            Orientationd(),
            ScaleIntensityRanged(),
            RandCropByPosNegLabeld(),
            ToTensord()
        ])

    when `transforms` is used in a multi-epoch training pipeline, before the first training epoch,
    this dataset will cache the results up to ``ScaleIntensityRanged``, as
    all non-random transforms `LoadImaged`, `EnsureChannelFirstd`, `Spacingd`, `Orientationd`, `ScaleIntensityRanged`
    can be cached. During training, the dataset will load the cached results and run
    ``RandCropByPosNegLabeld`` and ``ToTensord``, as ``RandCropByPosNegLabeld`` is a randomized transform
    and the outcome not cached.

    During training call `set_data()` to update input data and recompute cache content, note that it requires
    `persistent_workers=False` in the PyTorch DataLoader.

    Note:
        `CacheDataset` executes non-random transforms and prepares cache content in the main process before
        the first epoch, then all the subprocesses of DataLoader will read the same cache content in the main process
        during training. it may take a long time to prepare cache content according to the size of expected cache data.
        So to debug or verify the program before real training, users can set `cache_rate=0.0` or `cache_num=0` to
        temporarily skip caching.

    Lazy Resampling:
        If you make use of the lazy resampling feature of `monai.transforms.Compose`, please refer to
        its documentation to familiarize yourself with the interaction between `CacheDataset` and
        lazy resampling.

    """

    def __init__(
        self,
        data: Sequence,
        image_keys: str | list,
        transform: Sequence[Callable] | Callable | None = None,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: int | None = 1,
        progress: bool = True,
        copy_cache: bool = True,
        as_contiguous: bool = True,
        hash_as_key: bool = False,
        hash_func: Callable[..., bytes] = pickle_hashing,
        runtime_cache: bool | str | list | ListProxy = False,
    ) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            image_keys: keys to create imaging data from (e.g., 'T1' or 'T2').
            transform: transforms to execute operations on input data.
            cache_num: number of items to be cached. Default is `sys.maxsize`.
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
            cache_rate: percentage of cached data in total, default is 1.0 (cache all).
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
            num_workers: the number of worker threads if computing cache in the initialization.
                If num_workers is None then the number returned by os.cpu_count() is used.
                If a value less than 1 is specified, 1 will be used instead.
            progress: whether to display a progress bar.
            copy_cache: whether to `deepcopy` the cache content before applying the random transforms,
                default to `True`. if the random transforms don't modify the cached content
                (for example, randomly crop from the cached image and deepcopy the crop region)
                or if every cache item is only used once in a `multi-processing` environment,
                may set `copy=False` for better performance.
            as_contiguous: whether to convert the cached NumPy array or PyTorch tensor to be contiguous.
                it may help improve the performance of following logic.
            hash_as_key: whether to compute hash value of input data as the key to save cache,
                if key exists, avoid saving duplicated content. it can help save memory when
                the dataset has duplicated items or augmented dataset.
            hash_func: if `hash_as_key`, a callable to compute hash from data items to be cached.
                defaults to `monai.data.utils.pickle_hashing`.
            runtime_cache: mode of cache at the runtime. Default to `False` to prepare
                the cache content for the entire ``data`` during initialization, this potentially largely increase the
                time required between the constructor called and first mini-batch generated.
                Three options are provided to compute the cache on the fly after the dataset initialization:

                1. ``"threads"`` or ``True``: use a regular ``list`` to store the cache items.
                2. ``"processes"``: use a ListProxy to store the cache items, it can be shared among processes.
                3. A list-like object: a users-provided container to be used to store the cache items.

                For `thread-based` caching (typically for caching cuda tensors), option 1 is recommended.
                For single process workflows with multiprocessing data loading, option 2 is recommended.
                For multiprocessing workflows (typically for distributed training),
                where this class is initialized in subprocesses, option 3 is recommended,
                and the list-like object should be prepared in the main process and passed to all subprocesses.
                Not following these recommendations may lead to runtime errors or duplicated cache across processes.

        """
        if not isinstance(transform, Compose):
            transform = Compose(transform)
        self.image_keys = [image_keys] if isinstance(image_keys, str) else image_keys
        SeqDataset.__init__(
            self, 
            data=data, 
            image_keys=self.image_keys, 
            transform=transform)
        CacheDataset.__init__(
            self,
            data=data,
            transform=transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
            progress=progress,
            copy_cache=copy_cache,
            as_contiguous=as_contiguous,
            hash_as_key=hash_as_key,
            hash_func=hash_func,
            runtime_cache=runtime_cache)

    def _load_cache_item(self, idx: int):
        """
        Args:
            idx: the index of the input data sequence.
        """
        first_random = self.transform.get_index_of_first(
            lambda t: isinstance(t, RandomizableTrait) or not isinstance(t, Transform)
        )

        data_keys = list(self.data[idx].keys())
        data_keys = [key for key in data_keys if key not in self.image_keys]
        seq_len = len(self.data[idx][self.image_keys[0]])
        item_seq = []
        for i in range(seq_len):
            image_dict = {key: [self.data[idx][key][i]] for key in self.image_keys}
            data_dict = {key: self.data[idx][key][i] for key in data_keys}
            image_dict.update(data_dict) 
            data = self.transform(image_dict, end=first_random, threading=True)
            item_seq.append(data)

        if self.as_contiguous:
            item_seq = convert_to_contiguous(item_seq, memory_format=torch.contiguous_format)
        return item_seq

    def _transform(self, index: int):
        cache_index = None
        if self.hash_as_key:
            key = self.hash_func(self.data[index])
            if key in self._hash_keys:
                # if existing in cache, try to get the index in cache
                cache_index = self._hash_keys.index(key)
        elif index % len(self) < self.cache_num:  # support negative index
            cache_index = index

        if cache_index is None:
            # no cache for this index, execute all the transforms directly
            return SeqDataset._transform(index)

        if self._cache is None:
            raise RuntimeError("cache buffer is not initialized, please call `set_data()` first.")
        data = self._cache[cache_index]
        # runtime cache computation
        if data is None:
            data = self._cache[cache_index] = self._load_cache_item(cache_index)

        # load data from cache and execute from the first random transform
        if not isinstance(self.transform, Compose):
            raise ValueError("transform must be an instance of monai.transforms.Compose.")

        first_random = self.transform.get_index_of_first(
            lambda t: isinstance(t, RandomizableTrait) or not isinstance(t, Transform)
        )
        if first_random is not None:
            data = deepcopy(data) if self.copy_cache is True else data
            data = self.transform(data, start=first_random)

        return data

