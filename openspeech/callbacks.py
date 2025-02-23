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
# LIABILITY, W

import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


class DatasetShuffler(pl.Callback):
    """Shuffle the dataset after each epoch"""

    def _shuffle_dataloader(self, dataloader):
        """Shuffle the dataset"""
        if hasattr(dataloader, "batch_sampler") and \
            hasattr(dataloader.batch_sampler, 'shuffle') and \
            callable(dataloader.batch_sampler.shuffle):
            dataloader.batch_sampler.shuffle()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module):
        self._shuffle_dataloader(trainer.train_dataloader)
        for dl in trainer.val_dataloaders:
            self._shuffle_dataloader(dl)


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.

    Args:
        save_step_frequency: how often to save in steps
        use_modelcheckpoint_filename: just use the ModelCheckpoint callback's default filename, don't use ours.
    """

    def __init__(
        self,
        save_step_frequency,
        use_modelcheckpoint_filename=False,
    ) -> None:
        self.save_step_frequency = save_step_frequency
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx):
        """Check if we should save a checkpoint after every train batch"""
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            epoch = trainer.current_epoch
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{epoch}_{global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


class CheckpointMonitor(pl.Callback):
    "Prints checkpoint path when a checkpoint is saved."

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module, checkpoint):
        if trainer.checkpoint_callback:
            print(f"Saved checkpoint to: {trainer.checkpoint_callback.best_model_path}")

            if isinstance(trainer.logger, WandbLogger):
                trainer.logger.log_text("best_model", columns=["path", "score"], 
                                        data=[[trainer.checkpoint_callback.best_model_path, 
                                               trainer.checkpoint_callback.best_model_score]])
    

class DatasetShuffler(pl.Callback):
    """Shuffle the dataset after each epoch"""

    def _shuffle_dataloader(self, dataloader):
        """Shuffle the dataset"""
        if hasattr(dataloader, "batch_sampler") and \
            hasattr(dataloader.batch_sampler, 'shuffle') and \
            callable(dataloader.batch_sampler.shuffle):
            dataloader.batch_sampler.shuffle()
    
    def _shuffle_dataloaders(self, trainer: pl.Trainer):
        """Shuffle the dataset"""
        self._shuffle_dataloader(trainer.train_dataloader)
        for dl in trainer.val_dataloaders:
            self._shuffle_dataloader(dl)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # if we are on last batch
        if trainer.is_last_batch:
            self._shuffle_dataloaders(trainer)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # if we are on last batch
        if trainer.is_last_batch:
            self._shuffle_dataloaders(trainer)

class GpuProfilerCallback(pl.Callback):
    def __init__(self, gpu_id=0):
        import sys
        from openspeech.gpu_profiler import gpu_profile, gpu_eval
        os.environ['GPU_DEBUG']=str(gpu_id)
        print('Debugging gpu: ', os.environ['GPU_DEBUG'])
        sys.settrace(gpu_profile)
        self.gpu_eval = gpu_eval
        
    def on_after_backward(self, trainer: pl.Trainer, pl_module):
        self.gpu_eval()