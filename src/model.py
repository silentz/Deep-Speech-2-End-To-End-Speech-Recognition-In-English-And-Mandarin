import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType


class ASRModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.rnn_hidden_size = 64
        self.n_classes = 64

        self.conv_filter = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(32, 8), bias=False),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(16, 8), bias=False),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
            )

        self.rnn_filter = nn.GRU(
                input_size=32,
                hidden_size=self.rnn_hidden_size,
                num_layers=2,
                batch_first=True,
                bidirectional=False,
            )

        self.head = nn.Sequential(
                nn.Linear(self.rnn_hidden_size, self.n_classes, bias=True),
            )

    def forward(self, input: TensorType['batch', 'time', 'n_mels']):
        pass

