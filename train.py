import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI

from src.dataset import LibriSpeechDataset
from src.model import ASRModel

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
            }
        self.val_dataset_kwargs = {
                'root': val_root,
                'url': val_df_url,
            }
        self.val_dataloader_kwargs = {
                'batch_size': val_batch_size,
                'num_workers': val_num_workers,
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
        print(batch)

    def training_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        pass

    def validation_step(self, batch, batch_idx) -> Dict[str, Any]:
        print(batch)

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        pass


if __name__ == '__main__':
    LightningCLI(
            ASRLightningModule,
            LibrispeechDataModule,
        )
