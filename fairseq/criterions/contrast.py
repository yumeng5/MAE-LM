# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import copy
import pickle

import torch
import torch.nn.functional as F
import torch.distributed as dist

from fairseq import metrics, modules, utils, distributed_utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('contrast')
class ContrastLoss(FairseqCriterion):

    def __init__(self, task, tpu=False):
        super().__init__(task)
        self.tpu = tpu
        self.mask_idx = self.task.dictionary.index('<mask>')
        self.padding_idx = self.task.dictionary.pad()
        self.temperature = self.args.temperature
        self.train_on_all = not self.args.train_on_mask
        self.teacher_negative_only = self.args.teacher_negative_only
        self.student_negative_only = self.args.student_negative_only
        self.student_orig_input = self.args.student_orig_input
        self.target_mask_input = self.args.target_mask_input
        self.negative_samples = self.args.negative_samples
        self.average_top_k_layers = self.args.average_top_k_layers
        self.global_pool = self.args.global_pool
        self.with_lm_head = self.args.with_lm_head
        self.exclude_negative = self.args.exclude_negative
        self.task = task
        if self.global_pool:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def get_seq_label(self, sim_matrix):
        bsz = sim_matrix.size(0)
        if self.seq_label is None or bsz > self.seq_label.size(0):
            self.seq_label = torch.arange(0, bsz, device=sim_matrix.device).view(-1, 2)
            self.seq_label[:, 0] += 1
            self.seq_label[:, 1] += -1
            # label is [1, 0, 3, 2, 5, 4, ...]
            self.seq_label = self.seq_label.view(-1)
            return self.seq_label
        else:
            return self.seq_label[:bsz]

    def seqcontrast_sem(self, out_1, out_2, temperature, global_negative_pool):
        batch_size = out_1.size(0)
        pad_bsz = int(self.args.batch_size) if global_negative_pool else batch_size
        if batch_size < pad_bsz:
            pad_len = 2 * (pad_bsz - batch_size)
            # fill padding with -inf, to block gradient
            pad = torch.full((pad_len, out_1.size(1)), float('-inf'), device=out_1.device, dtype=out_1.dtype)
            out = torch.cat([out_1, out_2], dim=-1).view(2 * batch_size, -1)
            out = torch.cat([out, pad], dim=0)
        else:
            # [2*B, D], orig and span interleavely
            out = torch.cat([out_1, out_2], dim=-1).view(2 * batch_size, -1)

        def pool_by_all_gather(out):
            out_to_send = out.detach()
            out_list = [
                out_to_send if i == self.rank else torch.empty_like(out_to_send) for i in range(self.world_size)
            ]
            dist.all_gather(out_list, out_to_send)
            # backward for the output in current rank
            out_list[self.rank] = out
            global_out = torch.cat(out_list, dim=0)
            return global_out

        def pool_by_all_reduce(out):
            out_to_send = out.detach()
            out_size = out.size(0)
            global_out = torch.zeros((out_size * self.world_size, out.size(1)), device=out.device, dtype=out.dtype)
            start = self.rank * out_size
            global_out[start : start + out_size] = out_to_send
            dist.all_reduce(global_out)
            global_out[start : start + out_size] = out
            return global_out
        
        # don't use global pool in eval, faster
        global_out = pool_by_all_gather(out) if global_negative_pool else out
        # [2*GB, 2*GB]
        sim_matrix = torch.mm(global_out, global_out.t()) / temperature
        global_batch_size = sim_matrix.size(0)
        sim_matrix.masked_fill_(torch.eye(global_batch_size, device=sim_matrix.device, dtype=torch.bool), float('-inf'))
        truth = self.get_seq_label(sim_matrix)
        contrast_loss = 0.5 * F.nll_loss(
            F.log_softmax(sim_matrix, dim=-1, dtype=torch.float32),
            truth,
            reduction='sum',
        )
        acc = (sim_matrix.argmax(dim=-1) == truth).sum().item() / global_batch_size
        return contrast_loss, acc

    def contrast_loss(self, masked_rep, target_rep, mask_tokens, negative_mask=None):
        if self.temperature > 0:
            temperature = self.temperature
        else:
            temperature = 1
        if self.teacher_negative_only:
            sim_matrix = torch.mm(masked_rep, target_rep.t()) / temperature
            labels = torch.arange(0, sim_matrix.size(1), device=sim_matrix.device)
        elif self.student_negative_only:
            sim_matrix = torch.mm(masked_rep, masked_rep.t()) / temperature
            if negative_mask is not None:
                sim_matrix.masked_fill_(negative_mask, float('-inf'))
            else:
                sim_matrix.masked_fill_(torch.eye(sim_matrix.size(0), sim_matrix.size(1), device=sim_matrix.device, dtype=torch.bool), float('-inf'))
            if self.negative_samples > 0 and self.negative_samples < sim_matrix.size(1):
                # sim_matrix, _ = sim_matrix.topk(k=self.negative_samples, dim=-1, sorted=False)
                sample_probs = F.softmax(sim_matrix, dim=-1, dtype=torch.float32)
                neg_samples = torch.multinomial(sample_probs, self.negative_samples)
                sim_matrix = sim_matrix.gather(dim=-1, index=neg_samples)
            target_sim = (masked_rep*target_rep).sum(dim=-1,keepdim=True) / temperature
            sim_matrix = torch.cat((target_sim, sim_matrix), dim=-1)
            labels = torch.zeros(sim_matrix.size(0), device=sim_matrix.device, dtype=torch.int64)
        else:
            all_rep = torch.cat((masked_rep, target_rep), dim=0)
            # print(temperature)
            sim_matrix = torch.mm(masked_rep, all_rep.t()) / temperature
            # print(sim_matrix.shape)
            sim_matrix.masked_fill_(torch.eye(sim_matrix.size(0), sim_matrix.size(1), device=sim_matrix.device, dtype=torch.bool), float('-inf'))
            labels = torch.arange(sim_matrix.size(0), sim_matrix.size(1), device=sim_matrix.device)
        if self.train_on_all:
            train_matrix = sim_matrix
            train_labels = labels
        else:
            train_matrix = sim_matrix[mask_tokens]
            train_labels = labels[mask_tokens]
        # target_val = train_matrix.gather(-1, train_labels.unsqueeze(-1))
        # print(target_val)
        loss = modules.cross_entropy(
            train_matrix,
            train_labels,
            reduction='sum',
        )
        # print(f'loss: {loss}')
        preds = sim_matrix.argmax(dim=-1)
        masked_correct = (preds[mask_tokens] == labels[mask_tokens]).sum().item()
        unmasked_correct = (preds[~mask_tokens] == labels[~mask_tokens]).sum().item()
        ret_dict = {
            "loss": loss,
            "masked_correct": masked_correct,
            "unmasked_correct": unmasked_correct,
            "correct": masked_correct + unmasked_correct,
        }
        return ret_dict

    def pool_rep(self, rep):
        pad_bsz = int(self.args.max_tokens)
        if rep.size(0) < pad_bsz:
            pad_len = pad_bsz - rep.size(0)
            # fill padding with -inf, to block gradient
            pad = torch.full((pad_len, rep.size(1)), float('-inf'), device=rep.device, dtype=rep.dtype)
            out_rep = torch.cat([rep, pad], dim=0)
        else:
            out_rep = rep
        out_to_send = out_rep.detach()
        out_list = [
            out_to_send if i == self.rank else torch.empty_like(out_to_send) for i in range(self.world_size)
        ]
        dist.all_gather(out_list, out_to_send)
        # backward for the output in current rank
        out_list[self.rank] = out_rep
        out_rep = torch.cat(out_list, dim=0)
        return out_rep
    
    def pooled_contrast_loss(self, masked_rep, target_rep, mask_tokens, offset):
        if self.temperature > 0:
            temperature = self.temperature
        else:
            temperature = 1
        all_rep = torch.cat((masked_rep, target_rep), dim=0)
        # print(temperature)
        sim_matrix = torch.mm(masked_rep, all_rep.t()) / temperature
        # print(sim_matrix.shape)
        sim_matrix.masked_fill_(torch.eye(sim_matrix.size(0), sim_matrix.size(1), device=sim_matrix.device, dtype=torch.bool), float('-inf'))
        pad_bsz = int(self.args.max_tokens)
        labels = self.rank * pad_bsz + torch.arange(sim_matrix.size(0), 2*sim_matrix.size(0), device=sim_matrix.device)
        labels = labels - offset
        if self.train_on_all:
            train_matrix = sim_matrix
            train_labels = labels
            # print(train_matrix.size())
            # print(train_labels)
        else:
            train_matrix = sim_matrix[mask_tokens]
            train_labels = labels[mask_tokens]
        # print(train_matrix.shape)
        target_val = train_matrix.gather(-1, train_labels.unsqueeze(-1))
        # print(torch.any(torch.isinf(target_val)))
        # exit()
        # print(target_val)
        loss = modules.cross_entropy(
            train_matrix,
            train_labels,
            reduction='sum',
        )
        # print(f'loss: {loss}')
        preds = sim_matrix.argmax(dim=-1)
        masked_correct = (preds[mask_tokens] == labels[mask_tokens]).sum().item()
        unmasked_correct = (preds[~mask_tokens] == labels[~mask_tokens]).sum().item()
        ret_dict = {
            "loss": loss,
            "masked_correct": masked_correct,
            "unmasked_correct": unmasked_correct,
            "correct": masked_correct + unmasked_correct,
        }
        return ret_dict

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # masked_tokens = sample['net_input']['src_tokens'].eq(self.mask_idx)
        masked_tokens = sample['target'].ne(self.padding_idx)
        padding_tokens = sample['net_input']['src_tokens'].eq(self.padding_idx)


        sample_size = masked_tokens.int().sum()

        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        if self.tpu:
            masked_tokens = None  # always project all tokens on TPU
        elif masked_tokens.device == torch.device('cpu'):
            if not masked_tokens.any():
                masked_tokens = None
        # else:
        #     masked_tokens = torch.where(
        #         masked_tokens.any(),
        #         masked_tokens,
        #         masked_tokens.new([True]),
        #     )
        
        # model.eval()
        target_input = sample['net_input']['src_tokens'] if self.target_mask_input else sample['net_input']['orig_tokens']
        with torch.no_grad():
            target_rep, extra = model(
                src_tokens=target_input, mode='teacher'
            )
            if self.average_top_k_layers > 1:
                target_rep = target_rep[-self.average_top_k_layers :]
                target_rep = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in target_rep]
                target_rep = sum(target_rep) / len(target_rep)
                target_rep = target_rep.transpose(0, 1)
            decay = extra['decay'] if 'decay' in extra else 0
        # model.train()
        
        student_input = sample['net_input']['orig_tokens'] if self.student_orig_input else sample['net_input']['src_tokens']
        masked_rep, extra = model(
            src_tokens=student_input, masked_tokens=masked_tokens
        )

        if self.exclude_negative:
            orig_tokens = sample['net_input']['orig_tokens'][~padding_tokens]
            negative_mask = orig_tokens.unsqueeze(0) == orig_tokens.unsqueeze(1)
        else:
            negative_mask = None

        target_rep = target_rep[~padding_tokens]
        masked_rep = masked_rep[~padding_tokens]
        # print(f"rank {self.rank} batch {sample['net_input']['src_tokens'].shape} shape {masked_rep.shape}")
        masked_tokens = masked_tokens[~padding_tokens]
        if self.train_on_all:
            sample_size = masked_rep.size(0)

        if self.temperature > 0:
            masked_rep = F.normalize(masked_rep.float(), dim=-1) #.type_as(masked_rep)
            target_rep = F.normalize(target_rep.float(), dim=-1) #.type_as(target_rep)
        
        # with torch.no_grad():
        #     sim_matrix = torch.mm(target_rep, target_rep.t())
        #     # print(sim_matrix.shape)
        #     sim_matrix.masked_fill_(torch.eye(sim_matrix.size(0), sim_matrix.size(1), device=sim_matrix.device, dtype=torch.bool), float('-inf'))
        #     logits_matrix = sim_matrix / 0.1
        #     pos_sim, pos_idx = logits_matrix.topk(k=10, dim=-1, sorted=False)
        #     sample_probs = F.softmax(pos_sim, dim=-1, dtype=torch.float32)
        #     pos_samples = torch.multinomial(sample_probs, 1)
        #     pos_tokens = pos_idx.gather(dim=-1, index=pos_samples)
        #     bottom_threshold = logits_matrix.size(1) - 50
        #     neg_sim, neg_idx = logits_matrix.topk(k=bottom_threshold, dim=-1, largest=False, sorted=False)
        #     sample_probs = F.softmax(neg_sim, dim=-1, dtype=torch.float32)
        #     neg_samples = torch.multinomial(sample_probs, 1000)
        #     neg_tokens = neg_idx.gather(dim=-1, index=neg_samples)

        #     # token_map = self.task.dictionary
        #     # all_tokens = sample['target'][~padding_tokens]
        #     # assert len(all_tokens) == len(sim_matrix)
        #     # for i in range(0,30):
        #     #     top_sims = pos_sim[i]
        #     #     top_indices = pos_idx[i]
        #     #     top_token_indices = [all_tokens[j.item()] for j in top_indices]
        #     #     sample_probs = sample_prob[i]
        #     #     neg_sims = neg_sim[i]
        #     #     neg_indices = neg_idx[i]
        #     #     neg_token_indices = [all_tokens[j.item()] for j in neg_indices]
        #     #     # top_sims = top_sims[40:]
        #     #     # top_indices = top_indices[40:]
        #     #     # top_token_indices = top_token_indices[40:]
        #     #     res = " ".join([f"{idx.item()} {token_map[token]} ({sim}, {prob})" for idx, token, sim, prob in zip(top_indices, top_token_indices, top_sims, sample_probs)])
        #     #     print(f"{token_map[all_tokens[i]]}: {res}")
        #     #     res = " ".join([f"{idx.item()} {token_map[token]} ({sim})" for idx, token, sim in zip(neg_indices, neg_token_indices, neg_sims)])
        #     #     print(f"{token_map[all_tokens[i]]}: {res}\n")
        # exit()

        if not self.global_pool:
            outputs = self.contrast_loss(masked_rep, target_rep, masked_tokens, negative_mask)
        else:
            target_rep = self.pool_rep(target_rep)
            pad_row = torch.any(torch.isinf(target_rep), dim=-1)
            target_rep = target_rep[~pad_row]
            offset = torch.cumsum(pad_row.int(), dim=0)
            pad_bsz = int(self.args.max_tokens)
            offset = offset[self.rank*pad_bsz: self.rank*pad_bsz+masked_rep.size(0)]
            outputs = self.pooled_contrast_loss(masked_rep, target_rep, masked_tokens, offset)
        loss = outputs['loss']
        contrast_loss = loss.item()
        if self.with_lm_head:
            # masked_tokens = sample['net_input']['src_tokens'].eq(self.mask_idx)
            masked_tokens = sample['target'].ne(self.padding_idx)
            lm_logits = extra['lm_outputs']
            targets = sample['target']
            # lm_logits = lm_logits[masked_tokens]
            targets = targets[masked_tokens]
            lm_loss = modules.cross_entropy(
                lm_logits.view(-1, lm_logits.size(-1)),
                targets.view(-1),
                reduction="sum",
                ignore_index=self.padding_idx,
            )
            loss = loss * self.args.contrast_loss_weight + sample_size * lm_loss * self.args.lm_loss_weight / masked_tokens.int().sum()
        masked_correct = outputs['masked_correct']
        unmasked_correct = outputs['unmasked_correct']
        correct = outputs['correct']
        
        # print(f"sample_size {sample_size} ntokens {sample['ntokens']}")
        # print(f"{self.rank}: {masked_tokens.int().sum()}")
        with torch.no_grad():
            logging_output = {
                'loss': loss if self.tpu else loss.data,
                'contrast_loss': contrast_loss,
                'ntokens': sample['ntokens'],
                'nsentences': sample['nsentences'],
                'sample_size': sample_size,
                'mask_size': masked_tokens.int().sum(),
                'unmask_size': (~masked_tokens).int().sum(),
                'masked_correct': masked_correct,
                'unmasked_correct': unmasked_correct,
                'correct': correct,
                'decay': decay,
                'device_count': 1,
            }
            if self.with_lm_head:
                logging_output['lm_loss'] = lm_loss if self.tpu else lm_loss.data
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        contrast_loss_sum = sum(log.get('contrast_loss', 0) for log in logging_outputs)
        lm_loss_sum = sum(log.get('lm_loss', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        mask_size = sum(log.get('mask_size', 0) for log in logging_outputs)
        unmask_size = sum(log.get('unmask_size', 0) for log in logging_outputs)
        masked_correct = sum(log.get('masked_correct', 0) for log in logging_outputs)
        unmasked_correct = sum(log.get('unmasked_correct', 0) for log in logging_outputs)
        correct = sum(log.get('correct', 0) for log in logging_outputs)
        # assert mask_size + unmask_size == sample_size
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        metrics.log_scalar('masked_acc', masked_correct / mask_size, mask_size, round=4)
        metrics.log_scalar('unmasked_acc', unmasked_correct / unmask_size, unmask_size, round=4)
        metrics.log_scalar('acc', correct / ntokens, ntokens, round=4)
        
        
        # assert ntokens == sample_size
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=4)
        metrics.log_scalar('contrast_loss', contrast_loss_sum / sample_size / math.log(2), sample_size, round=4)
        metrics.log_scalar('lm_loss', lm_loss_sum / mask_size / math.log(2), mask_size, round=4)

        metrics.log_scalar('sample_size', sample_size, 1, round=4)
        metrics.log_scalar('mask_size', mask_size, 1, round=4)

        decay = sum(log.get('decay', 0) for log in logging_outputs)
        divide = sum(log.get('device_count', 0) for log in logging_outputs)
        metrics.log_scalar('decay', decay/divide, 1, round=4)
        
        # metrics.log_scalar('nll_loss', gen_loss_sum / sample_size / math.log(2), sample_size, round=4)
        # metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
