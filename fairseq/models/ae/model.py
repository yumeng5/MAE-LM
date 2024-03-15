# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
from unicodedata import decimal

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models.squad import SQuADHead
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, TransformerEncoder
from fairseq.modules import LayerNorm, TransformerSentenceEncoder, TransformerSentenceDecoder
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.modules.transformer_sentence_encoder import init_bert_params


logger = logging.getLogger(__name__)


@register_model("ae")
class AutoEncodingModel(FairseqEncoderModel):

    def __init__(self, args, encoder, gen_decoder, disc_decoder, encoder_to_decoder, mask_token_emb, task):
        super().__init__(encoder)
        self.args = args
        self.gen_decoder = gen_decoder
        self.disc_decoder = disc_decoder
        self.encoder_to_decoder = encoder_to_decoder
        self.mask_token_emb = mask_token_emb
        # We follow BERT's random weight initialization
        self.padding_idx = task.source_dictionary.pad()
        self.apply(init_bert_params)
        if args.criterion == 'masked_lm' or args.criterion == 'ae':
            self.mask_idx = task.mask_idx
            self.mask_subsets = args.mask_subsets
            self.decoder_mask_prob = args.decoder_mask_prob
        self.criterion = args.criterion
        self.classification_heads = nn.ModuleDict()
        

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument('--decoder-sample-mode', choices=['train', 'eval', 'zero-dropout'],
                            help='which mode the decoder is in when sampling from its MLM output')
        parser.add_argument('--decoder-attn-mode', choices=['train', 'eval', 'zero-dropout'],
                            help='which mode the decoder is in when self attention')
        parser.add_argument('--gen-decoder-layers', type=int)
        parser.add_argument('--disc-decoder-layers', type=int)
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--bottleneck-embed-dim",
            type=int,
            help="decoder bottleneck embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-dim-apply",
            help="apply decoder dim to which decoder",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--load-teacher-weights",
            help="load teacher model directory",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--with-lm-head",
            action="store_true",
            help="add language modeling head",
        )
        parser.add_argument(
            "--share-pos-emb",
            action="store_true",
            help="share position embeddings between encoder and decoder",
        )
        parser.add_argument(
            "--restore-decoder-input",
            action="store_true",
            help="restore input embedding to decoder",
        )
        parser.add_argument(
            "--restore-encoder-mask",
            action="store_true",
            help="restore mask embedding for encoder",
        )
        parser.add_argument(
            "--no-fill-mask",
            action="store_true",
            help="do not restore mask",
        )
        parser.add_argument(
            "--fill-real-token",
            action="store_true",
            help="fill in real token embedding for decoder",
        )
        parser.add_argument(
            "--share-decoder-pos-emb",
            action="store_true",
            help="share position embeddings between decoders",
        )
        parser.add_argument(
            "--with-contrast-head",
            action="store_true",
            help="add contrastive learning head",
        )
        parser.add_argument(
            "--mix-token-emb",
            action="store_true",
            help="mix token embeddings for encoder",
        )
        parser.add_argument(
            "--full-context-alignment",
            action="store_false",
            help="decoder bidirectional attention",
        )
        parser.add_argument(
            "--decoder-cross-attn",
            action="store_true",
            help="decoder with cross attention to encoder",
        )
        parser.add_argument(
            "--no-decoder-self-attn",
            action="store_true",
            help="no decoder self attention",
        )
        parser.add_argument(
            "--self-teacher",
            action="store_true",
            help="use self as teacher",
        )
        parser.add_argument(
            "--mlm-detach",
            action="store_true",
            help="detach mlm loss from training transformer representation",
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--load-checkpoint-heads",
            action="store_true",
            help="(re-)register and load heads when loading checkpoints",
        )
        parser.add_argument(
            "--lm-linear-head",
            action="store_true",
            help="linear layer for mlm head",
        )
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument(
            "--encoder-layerdrop",
            type=float,
            metavar="D",
            default=0,
            help="LayerDrop probability for encoder",
        )
        parser.add_argument(
            "--encoder-layers-to-keep",
            default=None,
            help="which layers to *keep* when pruning as a comma-separated list",
        )
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument(
            "--quant-noise-pq",
            type=float,
            metavar="D",
            default=0,
            help="iterative PQ quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-pq-block-size",
            type=int,
            metavar="D",
            default=8,
            help="block size of quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-scalar",
            type=float,
            metavar="D",
            default=0,
            help="scalar quantization noise and scalar quantization at training time",
        )
        # args for "Better Fine-Tuning by Reducing Representational Collapse" (Aghajanyan et al. 2020)
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            default=False,
            help="Apply spectral normalization on the classification head",
        )
        parser.add_argument(
            "--decoder-apply-mask",
            action="store_true",
            help="Use masks for decoder",
        )
        parser.add_argument(
            "--encoder-out-with-pos",
            action="store_true",
            help="encoder output with position",
        )
        parser.add_argument(
            "--keep-encoder-dim",
            action="store_true",
            help="keep encoder dim",
        )
        parser.add_argument(
            "--no-mask-emb",
            action="store_true",
            help="no mask embedding",
        )
        parser.add_argument(
            "--first-cross-attn",
            action="store_true",
            help="first cross attn then self attn",
        )
        parser.add_argument(
            "--cross-self-attention",
            action="store_true",
            help="both cross attn and self attn",
        )
        # args for Fully Sharded Data Parallel (FSDP) training
        parser.add_argument(
            "--min-params-to-wrap",
            type=int,
            metavar="D",
            default=DEFAULT_MIN_PARAMS_TO_WRAP,
            help=(
                "minimum number of params for a layer to be wrapped with FSDP() when "
                "training with --ddp-backend=fully_sharded. Smaller values will "
                "improve memory efficiency, but may make torch.distributed "
                "communication less efficient due to smaller input sizes. This option "
                "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
                "--offload-activations are passed."
            )
        )
        parser.add_argument('--rel-pos', 
                            type=int, 
                            help='whether to use relative position or not; 0 = not use; 1 = use')
        parser.add_argument('--decoder-rel-pos', 
                            type=int, 
                            help='whether to use relative position or not; 0 = not use; 1 = use')
        parser.add_argument('--rel-pos-bins', 
                            type=int, 
                            help='number of relative position buckets')
        parser.add_argument('--max-rel-pos', 
                            type=int, 
                            help='max relative positions')
        parser.add_argument('--teacher-layer-idx', 
                            type=int, 
                            help='layer index to get teacher encoder representation')
        parser.add_argument('--mix-topk', 
                            type=int, 
                            help='token embedding mix top k')
        parser.add_argument('--mix-num', 
                            type=int, 
                            help='number of tokens to mix')
        parser.add_argument('--checkpoint-activations', action='store_true',
                            help='checkpoint activations at each layer, which saves GPU '
                                 'memory usage at the cost of some additional compute')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = AEEncoder(args, 
                            task.source_dictionary,)
        mask_token_emb = None
        if args.criterion == 'masked_lm':
            if args.decoder_embed_dim != args.encoder_embed_dim:
                if args.keep_encoder_dim:
                    mask_token_emb = nn.Parameter(torch.zeros(1, 1, args.decoder_embed_dim))
                    encoder_to_decoder = None
                else:
                    encoder_to_decoder = nn.Linear(args.encoder_embed_dim, args.decoder_embed_dim, bias=True)
                pos_emb = None
            elif args.bottleneck_embed_dim > 0:
                encoder_to_decoder = nn.Sequential(nn.Linear(args.encoder_embed_dim, args.bottleneck_embed_dim, bias=True),
                                                   nn.Linear(args.bottleneck_embed_dim, args.decoder_embed_dim, bias=True),)
                pos_emb = encoder.sentence_encoder.embed_positions if args.share_pos_emb else None
            else:
                encoder_to_decoder = None
                pos_emb = encoder.sentence_encoder.embed_positions if args.share_pos_emb else None
            gen_decoder = AEDecoder(args, task.source_dictionary, encoder, pos_emb=pos_emb, decoder_type='gen')
            disc_decoder = None
        else:
            gen_decoder = None
            disc_decoder = None
            encoder_to_decoder = None
        return cls(args, encoder, gen_decoder, disc_decoder, encoder_to_decoder, mask_token_emb, task)

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        masked_tokens=None,
        targets=None,
        padding_mask=None,
        **kwargs
    ):
        if classification_head_name is not None:
            features_only = True

        x, extra = self.encoder(src_tokens, features_only, return_all_hiddens, masked_tokens, padding_mask=src_tokens.eq(self.padding_idx), **kwargs)
        
        if self.gen_decoder is not None:
            if self.args.no_return_mask:
                enc_outputs = torch.zeros((x.size(0), masked_tokens.size(1), x.size(2)), device=x.device, dtype=x.dtype)
                enc_outputs[~masked_tokens & ~padding_mask] = x[src_tokens.ne(self.padding_idx)]
            else:
                if self.mask_subsets > 1:
                    if self.args.fill_real_token:
                        real_token_emb = self.encoder.sentence_encoder.embed_tokens(src_tokens)
                        all_mask = masked_tokens.any(dim=1)
                        x[all_mask] = real_token_emb[all_mask]
                    enc_outputs = x.unsqueeze(1).repeat(1, self.mask_subsets, 1, 1)
                else:
                    enc_outputs = x
            token_embs = enc_outputs
            if not self.args.no_fill_mask:
                token_embs[masked_tokens] = self.encoder.sentence_encoder.embed_tokens.weight[self.mask_idx]
            decoder_pos = None
            if self.args.decoder_only_mask:
                decoder_pos = kwargs.get("decoder_pos", None)
                decoder_tokens = kwargs.get("decoder_tokens", None)
                if self.args.no_mask_emb:
                    new_embs = 0
                elif self.mask_token_emb is not None:
                    new_embs = self.mask_token_emb.repeat(decoder_tokens.shape[0], decoder_tokens.shape[1], 1)
                else:
                    new_embs = self.gen_decoder.sentence_encoder.embed_tokens(decoder_tokens)
                decoder_masked_tokens = decoder_tokens.eq(self.mask_idx)
                decoder_padding_mask = decoder_tokens.eq(self.padding_idx)
            else:
                new_embs = token_embs
                decoder_masked_tokens = masked_tokens
                decoder_padding_mask = padding_mask
            
            if self.encoder_to_decoder is not None:
                if not self.args.no_mask_emb:
                    new_embs = self.encoder_to_decoder(new_embs)
                # x = self.encoder_to_decoder(x)
            
            gen_x, extra_gen = self.gen_decoder(new_embs, 
                                                return_all_hiddens, 
                                                masked_tokens=decoder_masked_tokens, 
                                                padding_mask=decoder_padding_mask, 
                                                position_index=decoder_pos
                                                )
            extra.update(extra_gen)
        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        if self.criterion == 'masked_lm':
            return gen_x, extra
        else:
            return x, extra

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = AEClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            q_noise=self.args.quant_noise_pq,
            qn_block_size=self.args.quant_noise_pq_block_size,
            do_spectral_norm=self.args.spectral_norm_classification_head,
        )

    def register_question_answering_head(self, name, num_classes=None):
        self.classification_heads[name] = SQuADHead(
            self.args.encoder_embed_dim,
        )

    @property
    def supported_targets(self):
        return {"self"}

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""

        # rename decoder -> encoder before upgrading children modules
        # for k in list(state_dict.keys()):
        #     if k.startswith(prefix + "decoder"):
        #         new_k = prefix + "encoder" + k[len(prefix + "decoder") :]
        #         state_dict[new_k] = state_dict[k]
        #         del state_dict[k]

        # rename emb_layer_norm -> layernorm_embedding
        # for k in list(state_dict.keys()):
        #     if ".emb_layer_norm." in k:
        #         new_k = k.replace(".emb_layer_norm.", ".layernorm_embedding.")
        #         state_dict[new_k] = state_dict[k]
        #         del state_dict[k]

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
            ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v


class AELMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        else:
            if weight.size(-1) != embed_dim:
                self.embed_linear = nn.Linear(embed_dim, weight.size(-1), bias=False)
            else:
                self.embed_linear = None
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        if self.embed_linear is not None:
            x = self.embed_linear(x)
        x = F.linear(x, self.weight) + self.bias
        return x


class AELMLinearHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        else:
            if weight.size(-1) != embed_dim:
                self.embed_linear = nn.Linear(embed_dim, weight.size(-1), bias=False)
            else:
                self.embed_linear = None
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = features
        # project back to size of vocabulary with bias
        if self.embed_linear is not None:
            x = self.embed_linear(x)
        x = F.linear(x, self.weight) + self.bias
        return x


class AEClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        q_noise=0,
        qn_block_size=8,
        do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )
        if do_spectral_norm:
            if q_noise != 0:
                raise NotImplementedError(
                    "Attempting to use Spectral Normalization with Quant Noise. This is not officially supported"
                )
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class AEDiscHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, activation_fn):
        super().__init__()
        self.embed_dim = embed_dim
        # Todo: check projection is needed or not
        # self.dense = nn.Linear(embed_dim, embed_dim)
        # self.activation_fn = utils.get_activation_fn(activation_fn)
        # self.layer_norm = LayerNorm(embed_dim)

        self.out_proj = nn.Linear(embed_dim, 1, bias=True)
        # self.out_proj.bias.data.zero_()

    def forward(self, x, masked_tokens=None, **kwargs):
        # Only project the unmasked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            x = x[masked_tokens, :]

        # x = self.dense(x)
        # x = self.activation_fn(x)
        # x = self.layer_norm(x)
        return self.out_proj(x)


class AEDecoder(FairseqEncoder):

    def __init__(self, args, dictionary, main_encoder, pos_emb=None, decoder_type='gen'):
        super().__init__(dictionary)
        self.args = args
        self.restore_decoder_input = args.restore_decoder_input
        self.decoder_cross_attn = args.decoder_cross_attn
        self.full_context_alignment = args.full_context_alignment
        if args.decoder_dim_apply == 'both' or args.decoder_dim_apply == decoder_type:
            embed_dim = int(args.decoder_embed_dim)
            ffn_embed_dim = int(args.decoder_ffn_embed_dim)
            attention_heads = int(args.decoder_attention_heads)
        else:
            embed_dim = int(args.encoder_embed_dim)
            ffn_embed_dim = int(args.encoder_ffn_embed_dim)
            attention_heads = int(args.encoder_attention_heads)
        num_layers = args.gen_decoder_layers if decoder_type == 'gen' else args.disc_decoder_layers
        if self.decoder_cross_attn:
            encoder_out_dim = args.encoder_embed_dim if args.keep_encoder_dim else embed_dim
            self.sentence_encoder = TransformerSentenceDecoder(
                padding_idx=dictionary.pad(),
                vocab_size=len(dictionary),
                num_encoder_layers=num_layers,
                embedding_dim=embed_dim,
                encoder_embedding_dim=encoder_out_dim, #args.encoder_embed_dim,
                ffn_embedding_dim=ffn_embed_dim,
                num_attention_heads=attention_heads,
                dropout=args.dropout if args.decoder_sample_mode != "zero-dropout" else 0,
                attention_dropout=args.attention_dropout if args.decoder_attn_mode != "zero-dropout" else 0,
                activation_dropout=args.activation_dropout if args.decoder_sample_mode != "zero-dropout" else 0,
                layerdrop=args.encoder_layerdrop,
                max_seq_len=args.max_positions,
                num_segments=0,
                apply_bert_init=True,
                activation_fn=args.activation_fn,
                q_noise=args.quant_noise_pq,
                qn_block_size=args.quant_noise_pq_block_size,
                rel_pos=args.decoder_rel_pos,
                share_embed_tokens=main_encoder.sentence_encoder.embed_tokens,
                share_embed_positions=pos_emb,
                shared_embedding_dim=embed_dim,
                rel_pos_bins=args.rel_pos_bins,
                max_rel_pos=args.max_rel_pos,
                no_self_attn=args.no_decoder_self_attn,
                first_cross_attn=args.first_cross_attn,
                cross_self_attention=args.cross_self_attention,
            )
        else:
            self.sentence_encoder = TransformerSentenceEncoder(
                padding_idx=dictionary.pad(),
                vocab_size=len(dictionary),
                num_encoder_layers=num_layers,
                embedding_dim=embed_dim,
                ffn_embedding_dim=ffn_embed_dim,
                num_attention_heads=attention_heads,
                dropout=args.dropout if args.decoder_sample_mode != "zero-dropout" else 0,
                attention_dropout=args.attention_dropout if args.decoder_sample_mode != "zero-dropout" else 0,
                activation_dropout=args.activation_dropout if args.decoder_sample_mode != "zero-dropout" else 0,
                layerdrop=args.encoder_layerdrop,
                max_seq_len=args.max_positions,
                num_segments=0,
                apply_bert_init=True,
                activation_fn=args.activation_fn,
                q_noise=args.quant_noise_pq,
                qn_block_size=args.quant_noise_pq_block_size,
                rel_pos=args.decoder_rel_pos,
                share_embed_tokens=main_encoder.sentence_encoder.embed_tokens,
                share_embed_positions=pos_emb,
                shared_embedding_dim=embed_dim,
                rel_pos_bins=args.rel_pos_bins,
                max_rel_pos=args.max_rel_pos,
            )
        self.decoder_type = decoder_type
        if self.decoder_type == 'gen':
            if args.lm_linear_head:
                self.lm_head = AELMLinearHead(
                    embed_dim=embed_dim,
                    output_dim=len(dictionary),
                    activation_fn=args.activation_fn,
                    weight=main_encoder.sentence_encoder.embed_tokens.weight,
                )
            else:
                self.lm_head = AELMHead(
                    embed_dim=embed_dim,
                    output_dim=len(dictionary),
                    activation_fn=args.activation_fn,
                    weight=main_encoder.sentence_encoder.embed_tokens.weight,
                )
        else:
            self.lm_head = AEDiscHead(
                embed_dim=embed_dim,
                activation_fn=args.activation_fn,
            )

    def forward(self, token_embs, return_all_hiddens=False, masked_tokens=None, position_index=None, **unused):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states.
        """
        x, extra = self.extract_features(token_embs,
                                         return_all_hiddens,
                                         encoder_out=unused.get("encoder_out", None),
                                         padding_mask=unused.get("padding_mask", None),
                                         encoder_padding_mask=unused.get("encoder_padding_mask", None),
                                         masked_tokens=masked_tokens,
                                         position_index=position_index)
        x = self.output_layer(x, masked_tokens=masked_tokens, padding_mask=unused.get("padding_mask", None))
        return x, extra

    def extract_features(self, token_embs, return_all_hiddens=False, position_index=None, **unused):
        # if self.restore_decoder_input:
        #     masked_tokens = unused.get("masked_tokens", None)
        #     restore_pos = ~masked_tokens
        # else:
        #     restore_pos = None
        inner_states, _ = self.sentence_encoder(
            token_embeddings=token_embs,
            encoder_out=unused.get("encoder_out", None),
            last_state_only=not return_all_hiddens,
            padding_mask=unused.get("padding_mask", None),
            encoder_padding_mask=unused.get("encoder_padding_mask", None),
            use_ext_padding_mask=True,
            position_index=position_index,
            full_context_alignment=self.full_context_alignment,
        )
        features = inner_states[-1]
        return features, {'inner_states': inner_states if return_all_hiddens else None}

    def output_layer(self, features, masked_tokens=None, padding_mask=None, **unused):
        return self.lm_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions


class AEEncoder(FairseqEncoder):

    def __init__(self, args, dictionary, with_lm_head=False, with_contrast_head=False, layer_idx=None):
        super().__init__(dictionary)

        # set any missing default values
        base_architecture(args)
        self.args = args
        self.with_lm_head = with_lm_head
        self.with_contrast_head = with_contrast_head
        self.layer_idx = layer_idx
        self.mlm_detach = args.mlm_detach
        self.binary_classification = False
        self.restore_encoder_mask = args.restore_encoder_mask
        self.mix_token_emb = args.mix_token_emb
        self.mix_topk = args.mix_topk
        self.mix_num = args.mix_num
        self.detect_mask = False
        if args.criterion == 'masked_lm' or args.criterion == 'ae':
            self.detect_mask = args.detect_mask

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        # embed_tokens = self.build_embedding(
        #     len(dictionary), args.encoder_embed_dim, dictionary.pad()
        # )
        # self.sentence_encoder = self.build_encoder(args, dictionary, embed_tokens)
        self.sentence_encoder = TransformerSentenceEncoder(
            padding_idx=dictionary.pad(),
            vocab_size=len(dictionary),
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            layerdrop=args.encoder_layerdrop,
            max_seq_len=args.max_positions,
            num_segments=0,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
            q_noise=args.quant_noise_pq,
            qn_block_size=args.quant_noise_pq_block_size,
            rel_pos=args.rel_pos,
            rel_pos_bins=args.rel_pos_bins,
            max_rel_pos=args.max_rel_pos,
        )
        
        self.position_indices = None
        if self.detect_mask:
            self.binary_head = AEDiscHead(embed_dim=args.encoder_embed_dim,
                                          activation_fn=args.activation_fn,)
        else:
            self.binary_head = None

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = TransformerEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

    def build_lm_head(self, embed_dim, output_dim, activation_fn, weight):
        return AELMHead(embed_dim, output_dim, activation_fn, weight)
    
    def build_contrast_head(self, embed_dim, output_dim, activation_fn):
        return AELMHead(embed_dim, output_dim, activation_fn)

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        position_index=None,
        **unused
    ):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        padding_mask = unused.get("padding_mask", None)
        x, extra = self.extract_features(
            src_tokens, 
            return_all_hiddens=return_all_hiddens,
            position_index=position_index,
            attn_mask=unused.get("attn_mask", None),
            masked_tokens=masked_tokens,
            padding_mask=padding_mask,
        )
        # if not features_only:
        #     x = self.output_layer(x, masked_tokens=masked_tokens)
        
        # if self.binary_classification:
        #     num_seqs = x.size(0) // 2
        #     input1 = x[:num_seqs]
        #     input2 = x[num_seqs:]
        #     padding_mask = unused.get("padding_mask")[:num_seqs]
        #     x = self.binary_head(input1, input2, padding_mask)
        
        return x, extra

    def extract_features(self, src_tokens, position_index=None, return_all_hiddens=False, **kwargs):
        # encoder_out = self.sentence_encoder(
        #     src_tokens,
        #     return_all_hiddens=return_all_hiddens,
        #     token_embeddings=kwargs.get("token_embeddings", None),
        # )
        # features = encoder_out["encoder_out"][0]
        # inner_states = encoder_out["encoder_states"] if return_all_hiddens else None
        # return features, {"inner_states": inner_states}
        if self.restore_encoder_mask:
            restore_pos = kwargs.get("masked_tokens", None)
        else:
            restore_pos = None
        inner_states, _ = self.sentence_encoder(
            tokens=src_tokens,
            last_state_only=not return_all_hiddens,
            position_index=position_index,
            attn_mask=kwargs.get("attn_mask", None),
            restore_pos=restore_pos,
            padding_mask=kwargs.get("padding_mask", None),
            use_ext_padding_mask=True,
        )
        features = inner_states[-1]
        return features, {'inner_states': inner_states if return_all_hiddens else None}

    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions


@register_model_architecture("ae", "ae")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.bottleneck_embed_dim = getattr(args, "bottleneck_embed_dim", 0)
    args.decoder_dim_apply = getattr(args, "decoder_dim_apply", 'both')
    
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)

    args.max_source_positions = getattr(args, "max_positions", 512)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )

    # BERT has a few structural differences compared to the original Transformer
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")

    args.share_pos_emb = getattr(args, "share_pos_emb", False)
    args.share_decoder_pos_emb = getattr(args, "share_decoder_pos_emb", False)
    args.restore_decoder_input = getattr(args, "restore_decoder_input", False)
    args.restore_encoder_mask = getattr(args, "restore_encoder_mask", False)
    args.no_fill_mask = getattr(args, "no_fill_mask", False)
    args.fill_real_token = getattr(args, "fill_real_token", False)
    args.decoder_apply_mask = getattr(args, "decoder_apply_mask", False)
    args.encoder_out_with_pos = getattr(args, "encoder_out_with_pos", False)
    args.lm_linear_head = getattr(args, "lm_linear_head", False)
    args.keep_encoder_dim = getattr(args, "keep_encoder_dim", False)
    args.no_mask_emb = getattr(args, "no_mask_emb", False)

    # Adaptive input config
    args.adaptive_input = getattr(args, "adaptive_input", False)

    # LayerDrop config
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)

    # Quantization noise config
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

    # R4F config
    args.spectral_norm_classification_head = getattr(
        args, "spectral_norm_classification_head", False
    )
    args.mlm_detach = getattr(args, "mlm_detach", False)
    args.mix_token_emb = getattr(args, "mix_token_emb", False)
    args.decoder_cross_attn = getattr(args, "decoder_cross_attn", False)
    args.no_decoder_self_attn = getattr(args, "no_decoder_self_attn", False)
    args.full_context_alignment = getattr(args, "full_context_alignment", True)
    args.first_cross_attn = getattr(args, "first_cross_attn", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.mix_topk = getattr(args, 'mix_topk', 100)
    args.mix_num = getattr(args, 'mix_num', 1)
    args.load_teacher_weights = getattr(args, "load_teacher_weights", None)

    args.rel_pos = getattr(args, 'rel_pos', 0)
    args.decoder_rel_pos = getattr(args, 'decoder_rel_pos', 0)
    args.rel_pos_bins = getattr(args, 'rel_pos_bins', 32)
    args.max_rel_pos = getattr(args, 'max_rel_pos', 128)
    args.gen_decoder_layers = getattr(args, 'gen_decoder_layers', 4)
    args.disc_decoder_layers = getattr(args, 'disc_decoder_layers', 4)
    args.decoder_sample_mode = getattr(args, 'decoder_sample_mode', 'zero-dropout')
    args.decoder_attn_mode = getattr(args, 'decoder_attn_mode', 'zero-dropout')
    

# @register_model_architecture("contrast", "contrast_prenorm")
# def contrast_prenorm_architecture(args):
#     args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
#     args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
#     base_architecture(args)


@register_model_architecture("ae", "ae_base")
def ae_base_architecture(args):
    base_architecture(args)


@register_model_architecture("ae", "ae_large")
def ae_large_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.rel_pos_bins = getattr(args, 'rel_pos_bins', 128)
    args.max_rel_pos = getattr(args, 'max_rel_pos', 256)
    base_architecture(args)

