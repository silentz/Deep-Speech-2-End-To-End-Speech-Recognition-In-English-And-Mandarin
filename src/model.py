import torch
import torch.nn as nn
from torchtyping import TensorType


class ASRModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.rnn_hidden_size = 64
        self.rnn_input_size = 352
        self.rnn_num_layers = 2

        self.n_classes = 29

        self.conv_filter = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(8, 32),
                          stride=(2, 4), bias=False),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(8, 16),
                          stride=(2, 4), bias=False),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(inplace=True),
            )

        self.rnn_filter = nn.GRU(
                input_size=self.rnn_input_size,
                hidden_size=self.rnn_hidden_size,
                num_layers=self.rnn_num_layers,
                batch_first=True,
                bidirectional=False,
            )

        self.head = nn.Sequential(
                nn.Linear(in_features=self.rnn_hidden_size, out_features=self.n_classes, bias=True),
            )

    def forward(self, X: TensorType['batch', 'n_mels', 'time']):
        X = torch.unsqueeze(X, dim=1) # add channel dim
        X = self.conv_filter(X)

        X = torch.transpose(X, 1, 3) # bs, ch, n_mels, time -> bs, time, n_mels, ch
        batch_size, time, _, _ = X.shape
        X = X.reshape(batch_size, time, -1) # bs, time, n_mels * ch
        X, _ = self.rnn_filter(X) # bs, time, hidden_size

        X = self.head(X)
        return X

