import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI

from src import text
from src import metrics
from src.model import ASRModel
from src.dataset import LibriSpeechDataset
from src.dataset import librispeech_collate_fn

from typing import (
    Any,
    Dict,
    List,
)


class LibrispeechDataModule(pl.LightningDataModule):

    def __init__(self, train_root: str,
                       train_df_url: str,
                       train_batch_size: int,
                       train_num_workers: int,
                       val_root: str,
                       val_df_url: str,
                       val_batch_size: int,
                       val_num_workers: int):
        super().__init__()
        self.train_dataset_kwargs = {
                'root': train_root,
                'url': train_df_url,
            }
        self.train_dataloader_kwargs = {
                'batch_size': train_batch_size,
                'num_workers': train_num_workers,
                'collate_fn': librispeech_collate_fn,
            }
        self.val_dataset_kwargs = {
                'root': val_root,
                'url': val_df_url,
            }
        self.val_dataloader_kwargs = {
                'batch_size': val_batch_size,
                'num_workers': val_num_workers,
                'collate_fn': librispeech_collate_fn,
            }

    def train_dataloader(self) -> DataLoader:
        dataset = LibriSpeechDataset(**self.train_dataset_kwargs)
        return DataLoader(dataset, **self.train_dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        dataset = LibriSpeechDataset(**self.val_dataset_kwargs)
        return DataLoader(dataset, **self.val_dataloader_kwargs)


class ASRLightningModule(pl.LightningModule):

    def __init__(self, model: ASRModel,
                       criterion: nn.Module,
                       optimizer_lr: float):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer_lr = optimizer_lr

    def configure_optimizers(self) -> Dict[str, Any]:
        optim = torch.optim.Adam(self.model.parameters(), lr=self.optimizer_lr)
        return {
                'optimizer': optim,
            }

    def training_step(self, batch, batch_idx) -> Dict[str, Any]:
        waves, texts = batch['waves'], batch['texts']
        waves_len, texts_len = batch['waves_len'], batch['texts_len']

        model_out = self.model(waves)

        ctc_reshaped = torch.transpose(model_out, 0, 1) # swap time and bn dim
        waves_len_coefs = waves_len / waves.shape[1]
        ctc_waves_len = waves_len_coefs * ctc_reshaped.shape[0]
        loss = self.criterion(ctc_reshaped, texts, ctc_waves_len.long(), texts_len)

        return {
                'loss': loss,
            }

    def training_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        pass

    def validation_step(self, batch, batch_idx) -> Dict[str, Any]:
        batch_size = len(batch)
        waves, texts = batch['waves'], batch['texts']
        waves_len, texts_len = batch['waves_len'], batch['texts_len']

        model_out = self.model(waves)

        ctc_reshaped = torch.transpose(model_out, 0, 1) # swap time and bs dim
        waves_len_coefs = waves_len / waves.shape[1]
        ctc_waves_len = waves_len_coefs * ctc_reshaped.shape[0]
        loss = self.criterion(ctc_reshaped, texts, ctc_waves_len.long(), texts_len)

        probs = torch.exp(model_out.detach())
        predicted_lines = []
        target_lines = []

        for idx in range(batch_size):
            pred_line = text.ctc_decode(probs[idx], size=100)
            predicted_lines.append(pred_line)

            targ_line = texts[idx][:texts_len[idx]]
            targ_line = text.decode(targ_line)
            target_lines.append(targ_line)

        return {
                'loss': loss,
                'pred_lines': predicted_lines,
                'target_lines': target_lines,
            }

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        if isinstance(outputs, dict):
            outputs = [outputs,]

        pred = []
        target = []

        for batch in outputs:
            pred.extend(batch['pred_lines'])
            target.extend(batch['target_lines'])

        wer = metrics.wer(pred=pred, target=target)
        cer = metrics.cer(pred=pred, target=target)
        print('WER:', wer)
        print('CER:', cer)


if __name__ == '__main__':
    LightningCLI(
            ASRLightningModule,
            LibrispeechDataModule,
        )
