# One-Cycle-Pruning


## Install fasterai

[fasterai](https://nathanhubens.github.io/fasterai/) is a library for fastai, created to create sparse networks.

```
pip install git+https://github.com/nathanhubens/fasterai.git
```

## Run experiments

```
python train.py --schedule 'sched_onecycle'
```

## Requirements

- fastai >= 2.1
- torch >= 1.8
