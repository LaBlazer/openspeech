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

import random
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from openspeech.decoders import OpenspeechDecoder
from openspeech.modules import (
    Linear,
    View,
)


class LSTMDecoder(OpenspeechDecoder):
    r"""
    Converts higher level features (from encoders) into output utterances
    by specifying a probability distribution over sequences of characters.

    Args:
        num_classes (int): number of classification
        hidden_state_dim (int): the number of features in the decoders hidden state `h`
        num_layers (int, optional): number of recurrent layers (default: 2)
        rnn_type (str, optional): type of RNN cell (default: lstm)
        pad_id (int, optional): index of the pad symbol (default: 0)
        sos_id (int, optional): index of the start of sentence symbol (default: 1)
        eos_id (int, optional): index of the end of sentence symbol (default: 2)
        dropout_p (float, optional): dropout probability of decoders (default: 0.2)

    Inputs: inputs, encoder_outputs, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_state_dim): tensor with containing the outputs of the encoders.
          Used for attention mechanism (default is `None`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Returns: logits
        * logits (torch.FloatTensor) : log probabilities of model's prediction
    """
    supported_rnns = {
        "lstm": nn.LSTM,
        "gru": nn.GRU,
        "rnn": nn.RNN,
    }

    def __init__(
        self,
        num_classes: int,
        max_length: int = 150,
        hidden_state_dim: int = 1024,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        dropout_p: float = 0.3,
    ) -> None:
        super(LSTMDecoder, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.max_length = max_length
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.pad_id = pad_id
        self.embedding = nn.Embedding(num_classes, hidden_state_dim)
        self.input_dropout = nn.Dropout(dropout_p)
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(
            input_size=hidden_state_dim,
            hidden_size=hidden_state_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=False,
        )

        self.fc = nn.Sequential(
            Linear(hidden_state_dim << 1, hidden_state_dim),
            nn.Tanh(),
            View(shape=(-1, self.hidden_state_dim), contiguous=True),
            Linear(hidden_state_dim, num_classes),
        )

    def forward_step(
        self,
        input_var: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, output_lengths = input_var.size(0), input_var.size(1)

        embedded = self.embedding(input_var)
        print(embedded.size())
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()

        outputs, hidden_states = self.rnn(embedded, hidden_states)


        step_outputs = self.fc(outputs.view(-1, self.hidden_state_dim << 1)).log_softmax(dim=-1)
        step_outputs = step_outputs.view(batch_size, output_lengths, -1).squeeze(1)

        return step_outputs, hidden_states

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        encoder_output_lengths: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward propagate a `encoder_outputs` for training.

        Args:
            targets (torch.LongTensr): A target sequence passed to decoders. `IntTensor` of size ``(batch, seq_length)``
            encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            encoder_output_lengths: The length of encoders outputs. ``(batch)``
            teacher_forcing_ratio (float): ratio of teacher forcing

        Returns:
            * logits (torch.FloatTensor): Log probability of model predictions.
        """
        hidden_states = encoder_outputs

        targets, batch_size, max_length = self.validate_args(targets, encoder_outputs, teacher_forcing_ratio)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        outputs = torch.zeros(batch_size, max_length, self.num_classes, device=encoder_outputs.device)

        if use_teacher_forcing:
            targets = targets[targets != self.eos_id].view(batch_size, -1)

            for di in range(1, max_length):
                step_outputs, hidden_states = self.forward_step(
                    input_var=targets[:, di - 1].unsqueeze(1),
                    hidden_states=hidden_states,
                )
                outputs[:, di, :] = step_outputs

        else:
            input_var = targets[:, 0].unsqueeze(1)
            print(hidden_states.shape)
            print(targets.shape)
            print(targets[:, 0].unsqueeze(1).shape)

            for di in range(1, max_length):
                step_outputs, hidden_states = self.forward_step(
                    input_var=input_var,
                    hidden_states=hidden_states,
                )
                outputs[:, di, :] = step_outputs
                input_var = step_outputs.argmax(dim=-1, keepdim=True)

        return outputs

    def validate_args(
        self,
        targets: Optional[Any] = None,
        encoder_outputs: torch.Tensor = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> Tuple[torch.Tensor, int, int]:
        assert encoder_outputs is not None
        batch_size = encoder_outputs.size(0)

        if targets is None:  # inference
            max_length = self.max_length
            targets = torch.full((batch_size, 1, 1), self.sos_id, dtype=torch.long, device=encoder_outputs.device)

            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no targets is provided.")

        else:
            max_length = targets.size(1) - 1  # minus the start of sequence symbol

        return targets, batch_size, max_length
