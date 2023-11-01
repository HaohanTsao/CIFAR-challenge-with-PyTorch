'''
download data in your directory.
'''
from torchvision import datasets

data_path = './data'

cifar10 =  datasets.CIFAR10(data_path, train=True, download=True)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)
