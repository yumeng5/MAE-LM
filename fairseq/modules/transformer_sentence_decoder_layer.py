# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise


class TransformerSentenceDecoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        encoder_embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        init_fn: Callable = None,
        no_self_attn: bool = False,
        first_cross_attn: bool = False,
        cross_self_attention: bool = False,
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
        self.first_cross_attn = first_cross_attn
        self.cross_self_attention = cross_self_attention

        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.activation_dropout_module = FairseqDropout(
            activation_dropout, module_name=self.__class__.__name__
        )

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        if no_self_attn:
            self.self_attn = None
            self.self_attn_layer_norm = None
        else:
            self.self_attn = self.build_self_attention(
                self.embedding_dim,
                num_attention_heads,
                dropout=attention_dropout,
                self_attention=True,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
            )

            # layer norm associated with the self attention layer
            self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.encoder_attn = self.build_encoder_attention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            encoder_embed_dim=encoder_embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embedding_dim,
            ffn_embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.fc2 = self.build_fc2(
            ffn_embedding_dim,
            self.embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        self_attention,
        q_noise,
        qn_block_size,
    ):
        return MultiheadAttention(
            embed_dim,
            num_attention_heads,
            dropout=dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def build_encoder_attention(
        self, 
        embed_dim, 
        num_attention_heads, 
        encoder_embed_dim, 
        dropout, 
        q_noise, 
        qn_block_size,
    ):
        return MultiheadAttention(
            embed_dim,
            num_attention_heads,
            kdim=encoder_embed_dim,
            vdim=encoder_embed_dim,
            dropout=dropout,
            encoder_decoder_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        
        if self.self_attn is not None and not self.first_cross_attn:
            residual = x
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=True,
                attn_mask=self_attn_mask,
            )
            x = self.dropout_module(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.cross_self_attention:
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
            cross_attn_mask = self_attn_mask
            cross_padding_mask = self_attn_padding_mask
        else:
            y = encoder_out
            cross_attn_mask = None
            cross_padding_mask = encoder_padding_mask

        x, attn = self.encoder_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=cross_padding_mask,
            static_kv=True,
            need_weights=False,
            attn_mask=cross_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        x = self.encoder_attn_layer_norm(x)

        if self.self_attn is not None and self.first_cross_attn:
            residual = x
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=True,
                attn_mask=self_attn_mask,
            )
            x = self.dropout_module(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn
