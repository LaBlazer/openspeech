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

from omegaconf import DictConfig
import torch
from typing import Dict, OrderedDict

from openspeech.decoders import LSTMAttentionDecoder, LSTMDecoder
from openspeech.encoders import ConvolutionalLSTMEncoder
from openspeech.models import OpenspeechEncoderDecoderModel, OpenspeechCTCModel, register_model
from openspeech.models.lstm.configurations import (
    LstmCTCConfigs,
    LstmCEConfigs,
    LstmCEAttentionConfigs
)
from openspeech.tokenizers.tokenizer import Tokenizer
from openspeech.modules.wrapper import Linear


@register_model("lstm_ctc", dataclass=LstmCTCConfigs)
class LstmCTCModel(OpenspeechCTCModel):
    r"""
    Args:
        configs (DictConfig): configuration set.
        tokenizer (Tokeizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs:
        - **inputs** (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        - **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        outputs (dict): Result of model predictions.
    """

    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(LstmCTCModel, self).__init__(configs, tokenizer)

        self.encoder = ConvolutionalLSTMEncoder(
            input_dim=self.configs.audio.num_mels,
            num_layers=self.configs.model.num_encoder_layers,
            num_classes=self.num_classes,
            hidden_state_dim=self.configs.model.hidden_state_dim,
            dropout_p=self.configs.model.encoder_dropout_p,
            bidirectional=self.configs.model.encoder_bidirectional,
            rnn_type=self.configs.model.rnn_type,
            joint_ctc_attention=self.configs.model.joint_ctc_attention,
            extractor=self.configs.model.extractor,
            conv_activation=self.configs.model.activation,
            output_hidden_states=False
        )
        self.fc = Linear(self.encoder.get_output_dim(), self.num_classes, bias=False)

    def training_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` and `targets` pair for training.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        inputs, targets, input_lengths, target_lengths = batch
        outputs, _, output_lengths = self.encoder(inputs, input_lengths)
        logits = self.fc(outputs).log_softmax(dim=-1)

        return self.collect_outputs(
            stage="train",
            logits=logits,
            output_lengths=output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )
    
    def validation_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` for evaluation.

        Inputs:
            valid_batch (tuple): A valid batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for validation
        """
        inputs, targets, input_lengths, target_lengths = batch
        outputs, _, output_lengths = self.encoder(inputs, input_lengths)
        logits = self.fc(outputs).log_softmax(dim=-1)

        return self.collect_outputs(
            stage="val",
            logits=logits,
            output_lengths=output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

    def test_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` for evaluation.

        Inputs:
            test_batch (tuple): A test batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for test
        """
        inputs, targets, input_lengths, target_lengths = batch
        outputs, _, output_lengths = self.encoder(inputs, input_lengths)
        logits = self.fc(outputs).log_softmax(dim=-1)

        return self.collect_outputs(
            stage="test",
            logits=logits,
            output_lengths=output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

@register_model("lstm_ce", dataclass=LstmCEConfigs)
class LstmCEModel(OpenspeechEncoderDecoderModel):

    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(LstmCEModel, self).__init__(configs, tokenizer)

        self.encoder = ConvolutionalLSTMEncoder(
            input_dim=self.configs.audio.num_mels,
            num_layers=self.configs.model.num_encoder_layers,
            num_classes=self.num_classes,
            hidden_state_dim=self.configs.model.hidden_state_dim,
            dropout_p=self.configs.model.encoder_dropout_p,
            bidirectional=self.configs.model.encoder_bidirectional,
            rnn_type=self.configs.model.rnn_type,
            joint_ctc_attention=self.configs.model.joint_ctc_attention,
            extractor=self.configs.model.extractor,
            conv_activation=self.configs.model.activation,
            output_hidden_states=True
        )
        
        self.decoder = LSTMDecoder(
            num_classes=self.num_classes,
            max_length=self.configs.model.max_length,
            hidden_state_dim=self.encoder.get_output_dim(),
            pad_id=self.tokenizer.pad_id,
            sos_id=self.tokenizer.sos_id,
            eos_id=self.tokenizer.eos_id,
            dropout_p=self.configs.model.decoder_dropout_p,
            num_layers=self.configs.model.num_decoder_layers,
            rnn_type=self.configs.model.rnn_type,
        )

    def set_beam_decoder(self, beam_size: int = 3):
        from openspeech.search import BeamSearchLSTM
        self.decoder = BeamSearchLSTM(
            decoder=self.decoder,
            beam_size=beam_size,
        )
    
@register_model("lstm_ce_attention", dataclass=LstmCEAttentionConfigs)
class LstmCEAttentionModel(OpenspeechEncoderDecoderModel):

    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(LstmCEAttentionModel, self).__init__(configs, tokenizer)

        self.encoder = ConvolutionalLSTMEncoder(
            input_dim=self.configs.audio.num_mels,
            num_layers=self.configs.model.num_encoder_layers,
            num_classes=self.num_classes,
            hidden_state_dim=self.configs.model.hidden_state_dim,
            dropout_p=self.configs.model.encoder_dropout_p,
            bidirectional=self.configs.model.encoder_bidirectional,
            rnn_type=self.configs.model.rnn_type,
            joint_ctc_attention=self.configs.model.joint_ctc_attention,
            extractor=self.configs.model.extractor,
            conv_activation=self.configs.model.activation,
            output_hidden_states=True
        )
        
        self.decoder = LSTMAttentionDecoder(
            num_classes=self.num_classes,
            max_length=self.configs.model.max_length,
            hidden_state_dim=self.encoder.get_output_dim(),
            pad_id=self.tokenizer.pad_id,
            sos_id=self.tokenizer.sos_id,
            eos_id=self.tokenizer.eos_id,
            dropout_p=self.configs.model.decoder_dropout_p,
            num_layers=self.configs.model.num_decoder_layers,
            rnn_type=self.configs.model.rnn_type,
            num_heads=self.configs.model.num_attention_heads,
            attn_mechanism=self.configs.model.decoder_attn_mechanism,
        )

    def set_beam_decoder(self, beam_size: int = 3):
        from openspeech.search import BeamSearchLSTM
        self.decoder = BeamSearchLSTM(
            decoder=self.decoder,
            beam_size=beam_size,
        )

