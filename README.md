# Deep Speech 2: End-to-End Speech Recognition in English and Mandarin

LibriSpeech CER test-clean: `0.260732`

LibriSpeech WER test-clean: `0.618628`

## How to reproduce model

#### 1. Setup virtualenv
Execute following commands from project root:
```bash
virtualenv --python=python3.8 venv
source venv/bin/activate
pip install pybind11
pip install -r requirements.txt

git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install . && cd ..
```

#### 2. Download artifacts (model weights and LM)
Execute following commands from project root:
```bash
./download_artifacts.sh
```

#### 3. Launch inference on librispeech test-clean
From project root:
```bash
./test.sh
```

#### 4. Learn model from scratch
From project root:
```bash
./train.sh
```

## Training logs

#### Stage 1 (LJSpeech)

<img src="images/stage1_loss.png" width="500" />
<img src="images/stage1_cer.png" width="500" /> 
<img src="images/stage1_wer.png" width="500" />

#### Stage 2 (Librispeech train-clean-360)

<img src="images/stage2_loss.png" width="500" />
<img src="images/stage2_cer.png" width="500" /> 
<img src="images/stage2_wer.png" width="500" />
