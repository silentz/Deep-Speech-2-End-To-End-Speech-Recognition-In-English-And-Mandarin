import torch
import torch.nn as nn
from torchtyping import TensorType


class ASRModel(nn.Module):

    def __init__(self, n_classes: int,
                       rnn_num_layers: int,
                       rnn_hidden_size: int,
                       head_hidden_size: int,
                       n_mels: int):
        super().__init__()

        self.conv_filter = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(11, 41),
                          stride=(2, 2), bias=False),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(11, 21),
                          stride=(1, 2), bias=False),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(inplace=True),
            )

        out_channels = 1
        out_n_mels = n_mels

        for layer in self.conv_filter.children():
            if isinstance(layer, nn.Conv2d):
                out_n_mels -= (layer.kernel_size[1] - 1)
                out_n_mels //= layer.stride[1]
                out_channels = layer.out_channels

        self.rnn_filter = nn.GRU(
                input_size=out_channels * out_n_mels,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True,
                bidirectional=False,
            )

        self.head = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=rnn_hidden_size, out_features=head_hidden_size, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(in_features=head_hidden_size, out_features=n_classes, bias=True),
            )

    def forward(self, X: TensorType['batch', 'time', 'n_mels']):
        X = torch.unsqueeze(X, dim=1) # add channel dim
        X = self.conv_filter(X)

        X = torch.transpose(X, 1, 2) # bs, ch, time, n_mels -> bs, time, ch, n_mels
        batch_size, time, _, _ = X.shape
        X = X.reshape(batch_size, time, -1) # bs, time, n_mels * ch
        X, _ = self.rnn_filter(X) # bs, time, hidden_size

        X = self.head(X)
        X = torch.log_softmax(X, dim=2)
        return X

