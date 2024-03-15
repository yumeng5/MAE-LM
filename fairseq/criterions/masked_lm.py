# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("masked_lm")
class MaskedLmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, task, tpu=False):
        super().__init__(task)
        self.tpu = tpu
        self.mask_idx = self.task.mask_idx
        self.mask_attn = task.args.mask_attn
        self.mask_attn_prob = task.args.mask_attn_prob
        self.no_self_attn = task.args.no_self_attn
        self.task = task
        self.no_return_mask = task.args.no_return_mask
        self.debug = task.args.debug
        self.train_on_all = task.args.train_on_all
        self.no_pos_align = task.args.no_pos_align
        self.no_corruption = task.args.no_corruption
        self.detect_mask = task.args.detect_mask
        self.mask_subsets = task.args.mask_subsets
        if self.detect_mask:
            assert self.no_return_mask and self.no_pos_align

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if self.no_return_mask:
            padding_tokens = sample["target"].eq(self.padding_idx)
            masked_tokens = sample["target"].ne(self.mask_idx) & ~padding_tokens
        else:
            masked_tokens = sample["target"].ne(self.padding_idx)
            padding_tokens = sample["net_input"]["src_tokens"].eq(self.padding_idx)
            if self.train_on_all:
                sample["target"][~masked_tokens] = sample["net_input"]["src_tokens"][~masked_tokens]
                masked_tokens = ~padding_tokens
            if self.no_corruption:
                sample["net_input"]["src_tokens"][masked_tokens] = sample["target"][masked_tokens]
        sample_size = masked_tokens.int().sum() * self.mask_subsets
        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        
        if self.tpu:
            masked_tokens = None  # always project all tokens on TPU
        elif masked_tokens.device == torch.device("cpu"):
            if not masked_tokens.any():
                masked_tokens = None
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )
        logits, extra = model(**sample["net_input"], masked_tokens=masked_tokens, position_index=sample["pos"] if (self.no_return_mask and not self.no_pos_align) else None, 
                        padding_mask=padding_tokens)
        
        targets = model.get_targets(sample, [logits])
        if masked_tokens is not None:
            targets = targets.unsqueeze(1).repeat(1, self.mask_subsets, 1)
            masked_tokens = masked_tokens.unsqueeze(1).repeat(1, self.mask_subsets, 1)
            targets = targets[masked_tokens]

        loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )
        
        logging_output = {
            "loss": loss if self.tpu else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar('sample_size', sample_size, 1, round=4)
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
