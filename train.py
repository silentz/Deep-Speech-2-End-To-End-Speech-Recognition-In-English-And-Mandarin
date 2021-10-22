import warnings
warnings.filterwarnings('ignore')

from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.loggers import WandbLogger

from src.lightning import (
    Module,
    DataModule,
)


if __name__ == '__main__':
    logger = WandbLogger(
            project='hw1-asr',
            save_dir='lightning_logs',
        )

    LightningCLI(
            model_class=Module,
            datamodule_class=DataModule,
            trainer_defaults={
                'logger': logger,
            },
        )
