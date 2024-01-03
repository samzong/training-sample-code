# tf-fashion-minst-sample

This is a sample project to demonstrate how to use TensorFlow to train a model and use it to predict the fashion mnist dataset.

## Prerequisites

- Notebook with tensorflow installed
- Dataset [fashion mnist dataset](https://github.com/zalandoresearch/fashion-mnist)
  - Preload the dataset into a folder named `data`
- sample code [fashion mnist sample code](train.py)

## Pre-setup

### Dataset

- Create with Bazie UI

### Notebook

- Create with Bazie UI

## Train

### Develop with notebook

- Open the notebook
- Run the train.ipynb
- Check the result `models` directory

### Submit a TfJob with UI

- Create Job with Bazie UI

### Submit a TfJob with YAML

- Create Job with kubectl

### Submit a TfJob with CLI

use arena to submit a tfjob

```bash
arena submit tfjob --name tf-fashion-mnist-sample --gpus 1 --workers 1 --ps 1 --image tensorflow/tensorflow:1.14.0-gpu-py3 --tensorboard -- python train.py
```

## Tensorboard

- Create an Tensorboard with YAML
- Runing an Tensorboard
