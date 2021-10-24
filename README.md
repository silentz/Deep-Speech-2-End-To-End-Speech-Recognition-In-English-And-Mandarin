# HW #1: ASR

### Как вопспроизвести модель

#### 1. Настраиваем виртуальное окружение 
```bash
virtualenv --python=python3.8 venv
source venv/bin/activate

# нужно установить эту штуку перед тем, как устанавливать другие пакеты
pip install pybind11

pip install -r requirements.txt

# очень быстрая и удобная библа, но нет pip-пакета :c
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
```
