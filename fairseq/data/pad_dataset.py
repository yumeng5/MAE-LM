# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.data import data_utils
from functools import lru_cache
import numpy as np
import torch

from . import BaseWrapperDataset


class PadDataset(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad

    def collater(self, samples):
        return data_utils.collate_tokens(samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8)


class LeftPadDataset(PadDataset):
    def __init__(self, dataset, pad_idx):
        super().__init__(dataset, pad_idx, left_pad=True)


class RightPadDataset(PadDataset):
    def __init__(self, dataset, pad_idx):
        super().__init__(dataset, pad_idx, left_pad=False)


class SplitRightPadDataset(RightPadDataset):
    def __init__(self, dataset, pad_idx):
        super().__init__(dataset, pad_idx)

    def collater(self, samples):
        size = max(v.size(-1) for v in samples)
        pad_to_multiple = 8
        if pad_to_multiple != 1 and size % pad_to_multiple != 0:
            size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
        res = samples[0].new(len(samples), samples[0].size(0), size).fill_(self.pad_idx)
        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)
        for i, v in enumerate(samples):
            copy_tensor(v, res[i][:, : v.size(-1)])
        return res


class AttentionRightPadDataset(RightPadDataset):
    def __init__(self, dataset, pad_idx):
        super().__init__(dataset, pad_idx)

    def collater(self, samples):
        size = max(v.size(-1) for v in samples)
        pad_to_multiple = 8
        if pad_to_multiple != 1 and size % pad_to_multiple != 0:
            size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
        res = samples[0].new(len(samples), size, size).fill_(False)
        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)
        for i, v in enumerate(samples):
            copy_tensor(v, res[i][:v.size(0), :v.size(1)])
        return res


class RandomPadDataset(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, max_pad, seed=1, left_pad=True, return_num_pad=False):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.epoch = 0
        self.max_pad = max_pad
        self.seed = seed
        self.left_pad = left_pad
        self.return_num_pad = return_num_pad

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.seed, self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, seed: int, epoch: int, index: int):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            if self.max_pad > 0:
                num_pad = np.random.randint(0, self.max_pad)
            else:
                num_pad = 0
            if self.return_num_pad:
                return torch.tensor(num_pad)
            pads = np.full(num_pad, self.pad_idx)
            if self.left_pad:
                new_item = np.concatenate([pads, item])
            else:
                new_item = np.concatenate([item, pads])
            return new_item

    def collater(self, samples):
        return torch.tensor(samples)
        
    