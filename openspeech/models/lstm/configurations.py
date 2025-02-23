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

from dataclasses import dataclass, field

from openspeech.dataclass.configurations import OpenspeechDataclass


@dataclass
class LstmCTCConfigs(OpenspeechDataclass):
    r"""
    This is the configuration class to store the configuration of
    a :class:`~openspeech.models.LstmCTC`.

    It is used to initiated an `LstmCTC` model.

    Configuration objects inherit from :class: `~openspeech.dataclass.configs.OpenspeechDataclass`.

    Args:
        model_name (str): Model name (default: lstm)
        num_encoder_layers (int): The number of encoder layers. (default: 3)
        num_decoder_layers (int): The number of decoder layers. (default: 2)
        hidden_state_dim (int): The hidden state dimension of encoder. (default: 512)
        encoder_dropout_p (float): The dropout probability of encoder. (default: 0.3)
        encoder_bidirectional (bool): If True, becomes a bidirectional encoders (default: True)
        rnn_type (str): Type of rnn cell (rnn, lstm, gru) (default: lstm)
        joint_ctc_attention (bool): Flag indication joint ctc attention or not (default: False)
        max_length (int): Max decoding length. (default: 128)
        num_attention_heads (int): The number of attention heads. (default: 1)
        decoder_dropout_p (float): The dropout probability of decoder. (default: 0.2)
        decoder_attn_mechanism (str): The attention mechanism for decoder. (default: dot)
        teacher_forcing_ratio (float): The ratio of teacher forcing. (default: 1.0)
        optimizer (str): Optimizer for training. (default: adam)
    """
    model_name: str = field(default="lstm_ctc", metadata={"help": "Model name"})
    num_encoder_layers: int = field(default=3, metadata={"help": "The number of encoder layers."})
    hidden_state_dim: int = field(default=192, metadata={"help": "The hidden state dimension of encoder."})
    encoder_dropout_p: float = field(default=0.3, metadata={"help": "The dropout probability of encoder."})
    encoder_bidirectional: bool = field(default=True, metadata={"help": "If True, becomes a bidirectional encoders"})
    rnn_type: str = field(default="lstm", metadata={"help": "Type of rnn cell (rnn, lstm, gru)"})
    extractor: str = field(default="ds2", metadata={"help": "The CNN feature extractor."})
    activation: str = field(default="hardtanh", metadata={"help": "Type of activation function"})
    joint_ctc_attention: bool = field(default=False, metadata={"help": "Flag indication joint ctc attention or not"})
    max_length: int = field(default=256, metadata={"help": "Max decoding length."})
    decoder_dropout_p: float = field(default=0.2, metadata={"help": "The dropout probability of decoder."})
    teacher_forcing_ratio: float = field(default=1.0, metadata={"help": "The ratio of teacher forcing. "})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})

@dataclass
class LstmCEConfigs(OpenspeechDataclass):
    r"""
    This is the configuration class to store the configuration of
    a :class:`~openspeech.models.LstmCE`.

    It is used to initiated an `LstmCE` model.

    Configuration objects inherit from :class: `~openspeech.dataclass.configs.OpenspeechDataclass`.

    Args:
        model_name (str): Model name (default: lstm)
        num_encoder_layers (int): The number of encoder layers. (default: 3)
        num_decoder_layers (int): The number of decoder layers. (default: 2)
        hidden_state_dim (int): The hidden state dimension of encoder. (default: 512)
        encoder_dropout_p (float): The dropout probability of encoder. (default: 0.3)
        encoder_bidirectional (bool): If True, becomes a bidirectional encoders (default: True)
        rnn_type (str): Type of rnn cell (rnn, lstm, gru) (default: lstm)
        joint_ctc_attention (bool): Flag indication joint ctc attention or not (default: False)
        max_length (int): Max decoding length. (default: 128)
        num_attention_heads (int): The number of attention heads. (default: 1)
        decoder_dropout_p (float): The dropout probability of decoder. (default: 0.2)
        decoder_attn_mechanism (str): The attention mechanism for decoder. (default: dot)
        teacher_forcing_ratio (float): The ratio of teacher forcing. (default: 1.0)
        optimizer (str): Optimizer for training. (default: adam)
    """
    model_name: str = field(default="lstm_ce", metadata={"help": "Model name"})
    num_encoder_layers: int = field(default=3, metadata={"help": "The number of encoder layers."})
    num_decoder_layers: int = field(default=2, metadata={"help": "The number of decoder layers."})
    hidden_state_dim: int = field(default=512, metadata={"help": "The hidden state dimension of encoder."})
    encoder_dropout_p: float = field(default=0.3, metadata={"help": "The dropout probability of encoder."})
    encoder_bidirectional: bool = field(default=True, metadata={"help": "If True, becomes a bidirectional encoders"})
    rnn_type: str = field(default="lstm", metadata={"help": "Type of rnn cell (rnn, lstm, gru)"})
    extractor: str = field(default="ds2", metadata={"help": "The CNN feature extractor."})
    activation: str = field(default="hardtanh", metadata={"help": "Type of activation function"})
    joint_ctc_attention: bool = field(default=False, metadata={"help": "Flag indication joint ctc attention or not"})
    max_length: int = field(default=256, metadata={"help": "Max decoding length."})
    decoder_dropout_p: float = field(default=0.2, metadata={"help": "The dropout probability of decoder."})
    teacher_forcing_ratio: float = field(default=1.0, metadata={"help": "The ratio of teacher forcing. "})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})


@dataclass
class LstmCEAttentionConfigs(OpenspeechDataclass):
    r"""
    This is the configuration class to store the configuration of
    a :class:`~openspeech.models.LstmCEAttention`.

    It is used to initiated an `LstmCEAttention` model.

    Configuration objects inherit from :class: `~openspeech.dataclass.configs.OpenspeechDataclass`.

    Args:
        model_name (str): Model name (default: joint_ctc_lstm)
        num_encoder_layers (int): The number of encoder layers. (default: 3)
        num_decoder_layers (int): The number of decoder layers. (default: 2)
        hidden_state_dim (int): The hidden state dimension of encoder. (default: 768)
        encoder_dropout_p (float): The dropout probability of encoder. (default: 0.3)
        encoder_bidirectional (bool): If True, becomes a bidirectional encoders (default: True)
        rnn_type (str): Type of rnn cell (rnn, lstm, gru) (default: lstm)
        joint_ctc_attention (bool): Flag indication joint ctc attention or not (default: True)
        max_length (int): Max decoding length. (default: 128)
        num_attention_heads (int): The number of attention heads. (default: 1)
        decoder_dropout_p (float): The dropout probability of decoder. (default: 0.2)
        decoder_attn_mechanism (str): The attention mechanism for decoder. (default: loc)
        teacher_forcing_ratio (float): The ratio of teacher forcing. (default: 1.0)
        optimizer (str): Optimizer for training. (default: adam)
    """
    model_name: str = field(default="lstm_joint_ctc_attention", metadata={"help": "Model name"})
    num_encoder_layers: int = field(default=3, metadata={"help": "The number of encoder layers."})
    num_decoder_layers: int = field(default=2, metadata={"help": "The number of decoder layers."})
    hidden_state_dim: int = field(default=768, metadata={"help": "The hidden state dimension of encoder."})
    encoder_dropout_p: float = field(default=0.3, metadata={"help": "The dropout probability of encoder."})
    encoder_bidirectional: bool = field(default=True, metadata={"help": "If True, becomes a bidirectional encoders"})
    rnn_type: str = field(default="lstm", metadata={"help": "Type of rnn cell (rnn, lstm, gru)"})
    joint_ctc_attention: bool = field(default=True, metadata={"help": "Flag indication joint ctc attention or not"})
    max_length: int = field(default=256, metadata={"help": "Max decoding length."})
    num_attention_heads: int = field(default=8, metadata={"help": "The number of attention heads."})
    decoder_dropout_p: float = field(default=0.2, metadata={"help": "The dropout probability of decoder."})
    decoder_attn_mechanism: str = field(default="multi-head", metadata={"help": "The attention mechanism for decoder."})
    teacher_forcing_ratio: float = field(default=1.0, metadata={"help": "The ratio of teacher forcing. "})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})
