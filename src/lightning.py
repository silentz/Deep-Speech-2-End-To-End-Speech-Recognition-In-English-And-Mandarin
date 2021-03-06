import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import wandb
import pytorch_lightning as pl
from wandb.sdk.data_types import Image

from . import text
from . import metrics
from . import decoder
from .model import ASRModel
from .dataset import collate_fn

from typing import (
    Any,
    Dict,
    List,
)


class DataModule(pl.LightningDataModule):

    def __init__(self, train_dataset: Dataset,
                       train_batch_size: int,
                       train_num_workers: int,
                       val_dataset: Dataset,
                       val_batch_size: int,
                       val_num_workers: int,
                       test_dataset: Dataset,
                       test_batch_size: int,
                       test_num_workers: int):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.train_dataloader_kwargs = {
                'batch_size': train_batch_size,
                'num_workers': train_num_workers,
                'collate_fn': collate_fn,
            }
        self.val_dataloader_kwargs = {
                'batch_size': val_batch_size,
                'num_workers': val_num_workers,
                'collate_fn': collate_fn,
            }
        self.test_dataloader_kwargs = {
                'batch_size': test_batch_size,
                'num_workers': test_num_workers,
                'collate_fn': collate_fn,
            }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, **self.train_dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, **self.val_dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, **self.test_dataloader_kwargs)


class Module(pl.LightningModule):

    def __init__(self, model: ASRModel,
                       criterion: nn.Module,
                       optimizer_lr: float,
                       n_examples: int):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer_lr = optimizer_lr
        self.n_examples = n_examples

    def configure_optimizers(self) -> Dict[str, Any]:
        optim = torch.optim.Adam(
                self.model.parameters(),
                lr=self.optimizer_lr,
                weight_decay=1e-5,
            )
        sched = torch.optim.lr_scheduler.LambdaLR(
                optim,
                lr_lambda=lambda epoch: (0.99 ** epoch),
            )
        return {
                'optimizer': optim,
                'scheduler': sched,
            }

    @staticmethod
    def _decode_batch(model_out: torch.Tensor,
                      texts: torch.LongTensor,
                      texts_len: torch.LongTensor):
        predicted_lines = decoder.ctc_decode(model_out)
        target_lines = [text.decode(x[:tlen]) for x, tlen in zip(texts, texts_len)]
        return predicted_lines, target_lines

    def training_step(self, batch, batch_idx) -> Dict[str, Any]:
        waves, texts = batch['waves'], batch['texts']
        waves_len, texts_len = batch['waves_len'], batch['texts_len']
        spectrograms = batch['spectrograms']

        model_out = self.model(spectrograms)

        ctc_reshaped = torch.transpose(model_out, 0, 1) # swap time and bn dim
        waves_len_coefs = waves_len / waves.shape[1]
        ctc_waves_len = waves_len_coefs * ctc_reshaped.shape[0]
        loss = self.criterion(ctc_reshaped, texts, ctc_waves_len.long(), texts_len)

        if batch_idx % 500 == 0:
            batch_size = waves.shape[0]
            rand_ids = torch.randperm(batch_size)[:self.n_examples]

            sample_waves = waves[rand_ids]
            sample_spectrograms = spectrograms[rand_ids]
            sample_waves_len = waves_len[rand_ids]
            sample_out = model_out[rand_ids]
            sample_texts = texts[rand_ids]
            sample_texts_len = texts_len[rand_ids]

            pred_lines, target_lines = self._decode_batch(
                    sample_out,
                    sample_texts,
                    sample_texts_len,
                )

            table_columns = [
                    'audio', 'spectrogram', 'target', 'prediction', 'CER', 'WER',
                ]

            table_lines = []

            for idx in range(len(rand_ids)):
                wave = sample_waves[idx][:sample_waves_len[idx]].detach().cpu()
                spectrogram = sample_spectrograms[idx]
                target = target_lines[idx]
                prediction = pred_lines[idx]
                cer = metrics.cer(pred=[prediction,], target=[target,])
                wer = metrics.wer(pred=[prediction,], target=[target,])

                table_lines.append([
                        wandb.Audio(wave, sample_rate=16000),
                        wandb.Image(torch.transpose(spectrogram, 0, 1)),
                        target,
                        prediction,
                        cer,
                        wer,
                    ])

            table = wandb.Table(columns=table_columns, data=table_lines)
            table_name = f'samples_{self.current_epoch}_{batch_idx}'
            self.logger.experiment.log({table_name: table}, commit=True)

        self.log('loss', loss.item(), on_step=True, logger=True)
        return {
                'loss': loss,
            }

    def validation_step(self, batch, batch_idx) -> Dict[str, Any]:
        waves, texts = batch['waves'], batch['texts']
        waves_len, texts_len = batch['waves_len'], batch['texts_len']
        spectrograms = batch['spectrograms']

        model_out = self.model(spectrograms)

        ctc_reshaped = torch.transpose(model_out, 0, 1) # swap time and bs dim
        waves_len_coefs = waves_len / waves.shape[1]
        ctc_waves_len = waves_len_coefs * ctc_reshaped.shape[0]
        loss = self.criterion(ctc_reshaped, texts, ctc_waves_len.long(), texts_len)

        predicted_lines, target_lines = self._decode_batch(model_out, texts, texts_len)

        return {
                'loss': loss.item(),
                'pred_lines': predicted_lines,
                'target_lines': target_lines,
            }

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        pred = []
        target = []

        for batch in outputs:
            pred.extend(batch['pred_lines'])
            target.extend(batch['target_lines'])

        wer = metrics.wer(pred=pred, target=target)
        cer = metrics.cer(pred=pred, target=target)
        self.log('val_cer', cer, logger=True)
        self.log('val_wer', wer, logger=True)

    def test_step(self, batch, batch_idx) -> Dict[str, Any]:
        waves, texts = batch['waves'], batch['texts']
        waves_len, texts_len = batch['waves_len'], batch['texts_len']
        spectrograms = batch['spectrograms']

        model_out = self.model(spectrograms)

        ctc_reshaped = torch.transpose(model_out, 0, 1) # swap time and bs dim
        waves_len_coefs = waves_len / waves.shape[1]
        ctc_waves_len = waves_len_coefs * ctc_reshaped.shape[0]
        loss = self.criterion(ctc_reshaped, texts, ctc_waves_len.long(), texts_len)

        predicted_lines, target_lines = self._decode_batch(model_out, texts, texts_len)

        return {
                'loss': loss.item(),
                'pred_lines': predicted_lines,
                'target_lines': target_lines,
            }

    def test_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        pred = []
        target = []

        for batch in outputs:
            pred.extend(batch['pred_lines'])
            target.extend(batch['target_lines'])

        wer = metrics.wer(pred=pred, target=target)
        cer = metrics.cer(pred=pred, target=target)

        print('CER:', cer)
        print('WER:', wer)

