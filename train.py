# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import argparse

import torchvision
import torchvision.transforms as transforms

from models import *
# %%
# setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
parser = argparse.ArgumentParser(description='train CIFAR 10 with PyTorch')
parser.add_argument('--lr', help='input learning rate')
parser.add_argument('--model', choices=['ResNet18', 'ResNet50', 'EfficientNetB0'], required=True,)
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
args = parser.parse_args()
# %%
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0
# %%
# preparing data
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

data_path = './data'
training_set = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=train_transform)
train_loader = torch.utils.data.DataLoader(training_set, batch_size=128, shuffle=True, num_workers=2)
testing_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=test_transform)
test_loader = torch.utils.data.DataLoader(testing_set, batch_size=128, shuffle=False, num_workers=2)
# %%
model_name = args.model
if model_name == 'ResNet18':
    model = ResNet18()

elif model_name == 'ResNet50':
    model = ResNet50()

elif model_name == 'EfficientNetB0':
     model = EfficientNetB0()
