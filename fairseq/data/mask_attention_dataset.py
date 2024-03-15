# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import numpy as np
import torch
from fairseq.data import Dictionary, data_utils

from . import BaseWrapperDataset, LRUCacheDataset


class MaskAttentionDataset(BaseWrapperDataset):
    """
    A wrapper Dataset for masked language modeling.

    Input items are masked according to the specified masking probability.

    Args:
        dataset: Dataset to wrap.
        sizes: Sentence lengths
        vocab: Dictionary with the vocabulary and special tokens.
        pad_idx: Id of pad token in vocab
        mask_idx: Id of mask token in vocab
        return_masked_tokens: controls whether to return the non-masked tokens
            (the default) or to return a tensor with the original masked token
            IDs (and *pad_idx* elsewhere). The latter is useful as targets for
            masked LM training.
        seed: Seed for random number generator for reproducibility.
        mask_prob: probability of replacing a token with *mask_idx*.
        leave_unmasked_prob: probability that a masked token is unmasked.
        random_token_prob: probability of replacing a masked token with a
            random token from the vocabulary.
        freq_weighted_replacement: sample random replacement words based on
            word frequencies in the vocab.
        mask_whole_words: only mask whole words. This should be a byte mask
            over vocab indices, indicating whether it is the beginning of a
            word. We will extend any mask to encompass the whole word.
        bpe: BPE to use for whole-word masking.
        mask_multiple_length : repeat each mask index multiple times. Default
            value is 1.
        mask_stdev : standard deviation of masks distribution in case of
            multiple masking. Default value is 0.
    """

    @classmethod
    def apply_mask(cls, dataset: torch.utils.data.Dataset, *args, **kwargs):
        """Return the source and target datasets for masked LM training."""
        dataset = LRUCacheDataset(dataset)
        return LRUCacheDataset(cls(dataset, *args, **kwargs))

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        vocab: Dictionary,
        pad_idx: int,
        mask_idx: int,
        seed: int = 1,
        mask_prob: float = 0.15,
        layers: int = 12,
    ):
        assert 0.0 < mask_prob < 1.0

        self.dataset = dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.seed = seed
        self.mask_prob = mask_prob
        self.layers = layers

        self.epoch = 0

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True  # only the noise changes, not item sizes

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.seed, self.epoch, index)

    def collater(self, samples):
        size = max(v.size(1) for v in samples)
        size = int(((size - 0.1) // 8 + 1) * 8)
        res = samples[0].new(len(samples), samples[0].size(0), size, size).fill_(True)
        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)
        for i, v in enumerate(samples):
            copy_tensor(v, res[i][:, : v.size(1), : v.size(2)])
        return res

    @lru_cache(maxsize=8)
    def __getitem_cached__(self, seed: int, epoch: int, index: int):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            sz = len(item)

            assert (
                self.mask_idx not in item
            ), "Dataset contains mask_idx (={}), this is not expected!".format(
                self.mask_idx,
            )

            all_masks = []
            for i in range(2*self.layers):
                layer_masks = []
                for j in range(sz):
                    # decide elements to mask
                    mask = np.full(sz, True)
                    num_mask = int(
                        # add a random number for probabilistic rounding
                        self.mask_prob * sz
                        + np.random.rand()
                    )
                    # multiple masking as described in the vq-wav2vec paper (https://arxiv.org/abs/1910.05453)
                    mask_idc = np.random.choice(sz, num_mask, replace=False)

                    mask_idc = mask_idc[mask_idc < len(mask)]
                    try:
                        mask[mask_idc] = False
                    except:  # something wrong
                        print(
                            "Assigning mask indexes {} to mask {} failed!".format(
                                mask_idc, mask
                            )
                        )
                        raise
                    layer_masks.append(mask)
                all_masks.append(layer_masks)
            all_masks = np.array(all_masks)
            return torch.from_numpy(all_masks)
