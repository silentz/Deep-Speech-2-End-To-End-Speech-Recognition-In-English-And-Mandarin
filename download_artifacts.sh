#!/bin/bash

mkdir -p checkpoints/
mkdir -p data/

wget http://silentz.ml/stage1.ckpt -O ./checkpoints/stage1.ckpt
wget http://silentz.ml/stage2.ckpt -O ./checkpoints/stage2.ckpt
wget http://silentz.ml/4gram.bin -O ./data/4gram.bin
