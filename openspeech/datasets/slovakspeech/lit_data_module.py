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

import logging
import os
import tarfile
import random
import urllib
from typing import Optional, Tuple

import pytorch_lightning as pl
from omegaconf import DictConfig

from openspeech.data.audio.data_loader import AudioDataLoader
from openspeech.data.audio.dataset import SpeechToTextDataset
from openspeech.data.sampler import RandomSampler, SmartBatchingSampler, SmartBatchingDistributedSampler
from openspeech.datasets import register_data_module
from openspeech.datasets.slovakspeech.preprocess import read_transcripts, generate_manifest_file, generate_vocab_file
from openspeech.datasets.slovakspeech.downloader import download


@register_data_module("slovakspeech")
class LightningSlovakSpeechDataModule(pl.LightningDataModule):
    r"""
    Lightning data module for combined slovak speech dataset consisting of TEDxSK, CommonVoice and VoxPopuli.

    Args:
        configs (DictConfig): configuration set.
    """
    SLOVAKSPEECH_TRAIN_SPLIT = 0.7
    SLOVAKSPEECH_VALID_SPLIT = 0.25

    assert SLOVAKSPEECH_TRAIN_SPLIT + SLOVAKSPEECH_VALID_SPLIT <= 1.0

    def __init__(self, configs: DictConfig):
        super(LightningSlovakSpeechDataModule, self).__init__()
        self.configs = configs
        self.dataset = dict()
        self.logger = logging.getLogger(__name__)

        if self.configs.trainer.strategy is not None:
            self.sampler = SmartBatchingDistributedSampler
            self.configs.trainer.replace_sampler_ddp = False
        else:
            self.sampler = SmartBatchingSampler if self.configs.trainer.sampler == "smart" else RandomSampler

    def _download_dataset(self) -> None:
        r"""Download datasets."""
        url = "https://lblzr.com/data/slovakspeech.tgz"

        if not os.path.exists(self.configs.dataset.dataset_path):
            os.mkdir(self.configs.dataset.dataset_path)
        elif os.path.exists(os.path.join(self.configs.dataset.dataset_path, "slovakspeech")):
            self.logger.info(f"Dataset already exists at {self.configs.dataset.dataset_path}, skipping download.")
            return
        
        archive_file = url.split("/")[-1]
        archive_path = os.path.join(self.configs.dataset.dataset_path, archive_file)


        self.logger.info(f"Downloading {url} to {archive_path}")

        download(url, archive_path)

        self.logger.info(f"Un-tarring archive {archive_path} to {self.configs.dataset.dataset_path}")

        tar = tarfile.open(archive_path, mode="r:gz")
        tar.extractall(self.configs.dataset.dataset_path)
        tar.close()
        os.remove(archive_path)

    def _generate_manifest_files(self, manifest_file_path: str) -> None:
        r"""Generate manifest files."""
        self.logger.info(f"Generating manifest files to {manifest_file_path}")
        transcripts, skipped = read_transcripts(
            dataset_path=self.configs.dataset.dataset_path
        )

        self.logger.info(f"Skipping {skipped} transcripts due to length constraints.")

        generate_vocab_file(
            transcripts=transcripts,
            vocab_path=self.configs.tokenizer.vocab_path,
        )
        generate_manifest_file(
            transcripts=transcripts,
            manifest_file_path=manifest_file_path,
            vocab_path=self.configs.tokenizer.vocab_path,
        )

    def _parse_manifest_file(self, manifest_file_path: str) -> Tuple[list, list]:
        """Parsing manifest file"""
        audio_paths = list()
        transcripts = list()

        with open(manifest_file_path) as f:
            for idx, line in enumerate(f.readlines()):
                audio_path, _, transcript = line.split("\t")
                transcript = transcript.replace("\n", "")

                audio_paths.append(audio_path)
                transcripts.append(transcript)

        return audio_paths, transcripts

    def prepare_data(self) -> None:
        r"""
        Prepare AI-Shell manifest file. If there is not exist manifest file, generate manifest file.

        Returns:
            tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.
        """
        if self.configs.dataset.dataset_download:
            self._download_dataset()

        if not os.path.exists(self.configs.dataset.manifest_file_path):
            self.logger.info("Manifest file does not exist !!\n" "Generating manifest files..")
            if not os.path.exists(self.configs.dataset.dataset_path):
                raise ValueError("Dataset path is not valid.")
            self._generate_manifest_files(self.configs.dataset.manifest_file_path)

    def setup(self, stage: Optional[str] = None) -> None:
        r"""
        Split `train` and `valid` dataset for training.

        Args:
            stage (str): stage of training. `train` or `valid`
            tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.

        Returns:
            None
        """
        audio_paths, transcripts = self._parse_manifest_file(self.configs.dataset.manifest_file_path)

        rnd = random.Random(42)
        tmp = list(zip(audio_paths, transcripts))
        rnd.shuffle(tmp)
        audio_paths, transcripts = zip(*tmp)
        
        train_end_idx = int(self.SLOVAKSPEECH_TRAIN_SPLIT * len(audio_paths))
        valid_end_idx = int((self.SLOVAKSPEECH_TRAIN_SPLIT + self.SLOVAKSPEECH_VALID_SPLIT) * len(audio_paths))
        test_end_idx = len(audio_paths)

        audio_paths = {
            "train": audio_paths[: train_end_idx],
            "valid": audio_paths[train_end_idx : valid_end_idx],
            "test": audio_paths[valid_end_idx:test_end_idx],
        }
        transcripts = {
            "train": transcripts[: train_end_idx],
            "valid": transcripts[train_end_idx : valid_end_idx],
            "test": transcripts[valid_end_idx:test_end_idx],
        }

        for stage in audio_paths.keys():
            self.dataset[stage] = SpeechToTextDataset(
                configs=self.configs,
                dataset_path=self.configs.dataset.dataset_path,
                audio_paths=audio_paths[stage],
                transcripts=transcripts[stage],
                apply_spec_augment=self.configs.audio.apply_spec_augment if stage == "train" else False,
                apply_noise_augment=self.configs.audio.apply_noise_augment if stage == "train" else False,
                apply_time_stretch_augment=self.configs.audio.apply_time_stretch_augment if stage == "train" else False,
                apply_joining_augment=self.configs.audio.apply_joining_augment if stage == "train" else False,
                del_silence=self.configs.audio.del_silence if stage == "train" else False,
            )

    def train_dataloader(self) -> AudioDataLoader:
        train_sampler = self.sampler(data_source=self.dataset["train"], batch_size=self.configs.trainer.batch_size)
        return AudioDataLoader(
            dataset=self.dataset["train"],
            num_workers=self.configs.trainer.num_workers,
            batch_sampler=train_sampler,
            persistent_workers=True,
        )

    def val_dataloader(self) -> AudioDataLoader:
        valid_sampler = self.sampler(self.dataset["valid"], batch_size=self.configs.trainer.batch_size)
        return AudioDataLoader(
            dataset=self.dataset["valid"],
            num_workers=self.configs.trainer.num_workers,
            batch_sampler=valid_sampler,
            persistent_workers=True,
        )

    def test_dataloader(self) -> AudioDataLoader:
        test_sampler = self.sampler(self.dataset["test"], batch_size=self.configs.trainer.batch_size)
        return AudioDataLoader(
            dataset=self.dataset["test"],
            num_workers=self.configs.trainer.num_workers,
            batch_sampler=test_sampler,
            persistent_workers=True,
        )
