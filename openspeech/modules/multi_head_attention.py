# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional, Tuple

import torch.nn as nn
import numpy as np
from torch import Tensor

from openspeech.modules.dot_product_attention import DotProductAttention
from openspeech.modules.wrapper import Linear


class MultiHeadAttention(nn.Module):
    r"""
    Multi-Head Attention proposed in "Attention Is All You Need"
    Instead of performing a single attention function with d_model-dimensional keys, values, and queries,
    project the queries, keys and values h times with different, learned linear projections to d_head dimensions.
    These are concatenated and once again projected, resulting in the final values.
    Multi-head attention allows the model to jointly attend to information from different representation
    subspaces at different positions.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
        where head_i = Attention(Q · W_q, K · W_k, V · W_v)

    Args:
        dim (int): The dimension of model (default: 512)
        num_heads (int): The number of attention heads. (default: 8)

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoders.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoders.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features.
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoders outputs.
    """

    def __init__(self, dim: int = 512, num_heads: int = 8) -> None:
        super(MultiHeadAttention, self).__init__()

        assert dim % num_heads == 0, "hidden_dim % num_heads should be zero."

        self.d_head = int(dim / num_heads)
        self.num_heads = num_heads
        self.query_proj = Linear(dim, dim)
        self.key_proj = Linear(dim, dim)
        self.value_proj = Linear(dim, dim)
        self.output_proj = Linear(dim, dim)
        self.scaled_dot_attn = DotProductAttention(dim, scale=True)


    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
        d_head, n_head = self.d_head, self.num_heads
        batch_size, len_q, len_k, len_v = query.size(0), query.size(1), key.size(1), value.size(1)

        q = self.query_proj(query).view(batch_size, len_q, n_head, d_head)
        k = self.key_proj(key).view(batch_size, len_k, n_head, d_head)
        v = self.value_proj(value).view(batch_size, len_v, n_head, d_head)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_head)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_head)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_head)
        
        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)

        output = self.scaled_dot_attn(q, k, v, mask)
        output = output.view(n_head, batch_size, len_q, d_head)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, len_q, -1)

        return self.output_proj(output)
