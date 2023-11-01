# CIFAR10-with-PyTorch
Train CIFAR 10 with PyTorch

## Data
you can download data by execute
```
python data.py
```
Download the data is optional. The training still works if you don't.

## Training
train your model with my setting by execute
```
python train.py --model RseNet18 --lr 0.001 --epochs 30
```
You can modify the parameters. The models provided include ['ResNet18', 'ResNet50', 'EfficientNetB0'] for now. After the training, the test result will be showed automatically, and your model will also be saved in your directory.
