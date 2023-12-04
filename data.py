import argparse
from torchvision import datasets

def download_dataset(dataset_name, data_path):
    data_path_10 = data_path + '/cifar10'
    data_path_100 = data_path + '/cifar100'
    if dataset_name == 'cifar10':
        datasets.CIFAR10(data_path_10, train=True, download=True)
        datasets.CIFAR10(data_path_10, train=False, download=True)
        print("CIFAR-10 downloaded.")
    elif dataset_name == 'cifar100':
        data_path = data_path + '/cifar100'
        datasets.CIFAR100(data_path_100, train=True, download=True)
        datasets.CIFAR100(data_path_100, train=False, download=True)
        print("CIFAR-100 downloaded.")
    else:
        datasets.CIFAR10(data_path_10, train=True, download=True)
        datasets.CIFAR10(data_path_10, train=False, download=True)
        datasets.CIFAR100(data_path_100, train=True, download=True)
        datasets.CIFAR100(data_path_100, train=False, download=True)
        print("Both CIFAR-10 and CIFAR-100 downloaded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download CIFAR datasets.')
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default=None, help='Choose dataset to download: cifar10 or cifar100')
    args = parser.parse_args()

    data_path = './data'

    if args.dataset:
        download_dataset(args.dataset, data_path)
    else:
        download_dataset(None, data_path)
