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

import torch
from torch import Tensor


def get_transformer_non_pad_mask(inputs: Tensor, input_lengths: Tensor = None, 
                                 input_length: int = None) -> Tensor:
        """Padding position is set to 0, either use input_lengths or pad_id"""
        batch_size = inputs.size(0)

        if len(inputs.size()) == 2:
            non_pad_mask = inputs.new_ones(inputs.size())  # B x T
        elif len(inputs.size()) == 3:
            non_pad_mask = inputs.new_ones(inputs.size()[:-1])  # B x T
        else:
            raise ValueError(f"Unsupported input shape {inputs.size()}")

        if input_lengths is not None:
            for i in range(batch_size):
                non_pad_mask[i, input_lengths[i] :] = 0
        elif input_length is not None:
            non_pad_mask[:, input_length :] = 0
        else:
            raise ValueError("Either input_lengths or input_length should be provided")

        return non_pad_mask.lt(1)

def expand_mask(mask, expand_length):
    """mask position is set to 1"""
    return mask.unsqueeze_(1).expand(-1, expand_length, -1)

def get_attn_pad_mask(inputs, input_lengths, expand_length, input_length=None):
    """mask position is set to 1"""
    pad_mask = get_transformer_non_pad_mask(inputs, input_lengths, input_length)
    return expand_mask(pad_mask, expand_length)


def get_attn_subsequent_mask(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((sz_b, len_s, len_s), device=seq.device, dtype=torch.bool), diagonal=1)
    #subsequent_mask = subsequent_mask.unsqueeze_(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask
