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

import glob
import os
import re
import tarfile

import pandas as pd

from openspeech.data.audio.load import get_audio_length

SLOVAK_CHARACTERS = "aáäbcčdďeéfghiíjklĺľmnňoóôpqrŕsštťuúvwxyýzž"
ADDITIONAL_CHARACTERS = " " #" ,.?!"

VOCAB_REGEX = re.compile("[^{}]+".format(''.join(list(SLOVAK_CHARACTERS + ADDITIONAL_CHARACTERS))))
WHITESPACE_REGEX = re.compile(r'\s{2,}')  # multiple spaces

def _load_label(filepath):
    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv(filepath, encoding="utf-8")

    id_list = ch_labels["id"]
    char_list = ch_labels["char"]

    for (id_, char) in zip(id_list, char_list):
        char2id[char] = id_
        id2char[id_] = char
    return char2id, id2char

def _sanitize_text(text):
    text = VOCAB_REGEX.sub(" ", text.lower())
    text = WHITESPACE_REGEX.sub(" ", text)
    return text.strip()

def read_transcripts(dataset_path: str, text_min_len: int = 4, text_max_len: int = 200, 
                     audio_min_len: float = 0.0, audio_max_len: float = 20.0):
    """
    Returns:
        transcripts (dict): All the transcripts from datasets. They are represented
                            by {audio path: transcript}.
    """
    transcripts_dict = dict()
    skipped = 0

    for dataset in ["tedx_sk", "vox_sk", "cv12_sk"]:
        with open(os.path.join(dataset_path, "slovakspeech", dataset, "transcripts.tsv")) as f:
            # skip header
            f.readline()
            for line in f.readlines():
                audio_path, transcript = line.split('\t', maxsplit=1)

                transcript = _sanitize_text(transcript)

                if len(transcript) < text_min_len or len(transcript) > text_max_len:
                    skipped += 1
                    continue

                audio_path = os.path.join("slovakspeech", dataset, "clips", audio_path)

                audio_length = get_audio_length(os.path.join(dataset_path, audio_path), sample_rate=16000)

                if audio_length < audio_min_len or audio_length > audio_max_len:
                    skipped += 1
                    continue

                transcripts_dict[audio_path] = transcript

    return transcripts_dict, skipped


def _sentence_to_target(sentence, char2id):
    target = str()

    for ch in sentence:
        try:
            target += str(char2id[ch]) + " "
        except KeyError:
            continue

    return target[:-1]


def generate_vocab_file(transcripts: dict, vocab_path: str):
    label_freq_dict = {}

    for ch in SLOVAK_CHARACTERS + ADDITIONAL_CHARACTERS:
        label_freq_dict[ch] = 0

    for transcript in transcripts.values():
        for ch in transcript:
            label_freq_dict[ch] += 1

    # sort together Using zip
    label_freq, label_list = zip(*sorted(zip(label_freq_dict.values(), label_freq_dict.keys()), reverse=True))

    label = {"id": [0, 1, 2, 3], "char": ["<pad>", "<sos>", "<eos>", "<blank>"], "freq": [0, 0, 0, 0]}

    for idx, (ch, freq) in enumerate(zip(label_list, label_freq)):
        label["id"].append(idx + 4)
        label["char"].append(ch)
        label["freq"].append(freq)

    label_df = pd.DataFrame(label)
    label_df.to_csv(vocab_path, encoding="utf-8", index=False)


def generate_manifest_file(transcripts: dict, manifest_file_path: str, vocab_path: str):
    char2id, id2char = _load_label(vocab_path)

    with open(manifest_file_path, "w") as f:
        labels = [_sentence_to_target(ts, char2id) for ts in transcripts.values()]

        for (audio_path, ts, label) in zip(transcripts.keys(), transcripts.values(), labels):
            f.write(f"{audio_path}\t{ts}\t{label}\n")
