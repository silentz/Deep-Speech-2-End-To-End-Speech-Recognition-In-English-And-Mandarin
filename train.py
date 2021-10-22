import warnings
warnings.filterwarnings('ignore')

from pytorch_lightning.utilities.cli import LightningCLI
from src.lightning import (
    Module,
    DataModule,
)


if __name__ == '__main__':
    LightningCLI(
            model_class=Module,
            datamodule_class=DataModule,
            save_config_overwrite=True,
        )
