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

from typing import List, Tuple, Union

from jiwer import wer, cer
import torch

from openspeech.tokenizers.tokenizer import Tokenizer

class ErrorRate(object):
    r"""
    Provides inteface of error rate calcuation.

    Note:
        Do not use this class directly, use one of the sub classes.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, targets, y_hats):
        r"""
        Calculating error rate.

        Args:
            targets (torch.Tensor): set of ground truth
            y_hats (torch.Tensor): predicted y values (y_hat) by the model

        Returns:
            - **metric**: error rate metric
        """
        # if we are using subword tokenization, we need to remove the
        # subword token markers
        if self.tokenizer.subword_token:
            targets = [t.replace(self.tokenizer.subword_token, '') for t in targets]
            y_hats = [y.replace(self.tokenizer.subword_token, '') for y in y_hats]

        targets_decoded = [self.tokenizer.decode(t) for t in targets]
        y_hats_decoded = [self.tokenizer.decode(y) for y in y_hats]

        return self.metric(targets_decoded, y_hats_decoded) 

    def metric(self, *args, **kwargs) -> Tuple[float, int]:
        raise NotImplementedError


class CharacterErrorRate(ErrorRate):
    r"""
    Computes the Character Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to characters.
    """

    def __init__(self, tokenizer: Tokenizer):
        super(CharacterErrorRate, self).__init__(tokenizer)

    def metric(self, s1: Union[str, List[str]], s2: Union[str, List[str]]) -> float:
        r"""
        Computes the Character Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to characters.

        Args: s1, s2
            s1 (string): space-separated target sentences
            s2 (string): space-separated predicted sentences

        Returns: metric
            - **metric**: cer between target & y_hat
        """
        return cer(s1, s2)


class WordErrorRate(ErrorRate):
    r"""
    Computes the Word Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to words.
    """

    def __init__(self, tokenizer: Tokenizer):
        super(WordErrorRate, self).__init__(tokenizer)

    def metric(self, s1: Union[str, List[str]], s2: Union[str, List[str]]) -> float:
        r"""
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.

        Args: s1, s2
            s1 (string): space-separated target sentences
            s2 (string): space-separated predicted sentences

        Returns: metric
            - **metric**: wer between target & y_hat
        """
        return wer(s1, s2)
