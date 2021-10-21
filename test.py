import torch
from src.model import ASRModel
from src.dataset import LibrispeechDataset, LJSpeechDataset


#  model = ASRModel(
#      n_classes=29,
#      rnn_num_layers=5,
#      rnn_hidden_size=1024,
#      sample_rate=16000,
#      n_fft=320,
#      win_length=320,
#      hop_length=160,
#      n_mels=160,
#  )

#  params = sum(p.numel() for p in model.parameters())
#  print(params)

#  df = LibrispeechDataset(root='data/', url='test-clean')
#  line = df[0]
#  wave = line['wave']
#  text = line['text']

#  model_out = model(torch.unsqueeze(wave, dim=0))
#  print(model_out.shape)
from src import text

df = LJSpeechDataset(root='data/')
sample = df[1]
print(sample)
print(text.decode(sample['text']))
