# %%
import os
from tqdm import tqdm
import argparse
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import random_split, DataLoader
import torchvision
import torchvision.transforms as transforms

from models import *
# %%
# setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
parser = argparse.ArgumentParser(description='train CIFAR 10/100 with PyTorch')
parser.add_argument('--data', choices=['cifar10', 'cifar100'], required=True)
parser.add_argument('--model', choices=['ResNet18', 'ResNet50', 'EfficientNetB0'], required=True,)
parser.add_argument('--lr', help='input learning rate')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
args = parser.parse_args()
# %%
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0
# %%
# preparing data
data_path = './data'
if args.data == 'cifar10':
    num_classes = 10
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
    data_path += '/cifar10'
    training_set = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False)
    testing_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=test_transform)

elif args.data == 'cifar100':
    num_classes = 100
    train_transform = transforms.Compose([
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
    ])
    data_path += '/cifar100'
    training_set = torchvision.datasets.CIFAR100(root=data_path, train=True, download=False)
    testing_set = torchvision.datasets.CIFAR100(root=data_path, train=False, download=False, transform=test_transform)

torch.manual_seed(42)
val_size = int(len(training_set)*0.1)
train_size = len(training_set) - val_size
train_ds, val_ds = random_split(training_set, [train_size, val_size])

train_ds.dataset.transform = train_transform
val_ds.dataset.transform = test_transform
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)
test_loader = DataLoader(testing_set, batch_size=128, shuffle=False, num_workers=2)

classes = testing_set.classes

# %%
# choose model
model_name = args.model
VALID_MODELS = (
    'EfficientNetB0', )

available_models = [
    'ResNet18', 'ResNet50', 'ResNet101', 'ResNet152', 
    'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
    'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7',
    ]

if model_name == 'ResNet18':
    model = ResNet18(num_classes)

elif model_name == 'ResNet34':
    model = ResNet34(num_classes)

elif model_name == 'ResNet50':
    model = ResNet50(num_classes)

elif model_name == 'ResNet101':
    model = ResNet101(num_classes)

elif model_name == 'ResNet152':
    model = ResNet152(num_classes)

elif model_name.startwith("EfficientNet"):
    model = efficientnet(model_name, num_classes=num_classes)
else:
    raise ValueError(f"{model_name} is not available please use one of models below \n{', '.join(available_models)}")

if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

# %%
epochs = args.epochs
milestones = [epochs*0.3, epochs*0.6, epochs*0.8]
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2) #learning rate decay

# %%
# traning def
def training_loop(n_epochs, optimizer, scheduler, model, loss_fn, train_loader, val_loader):
    for epoch in range(n_epochs):
        loss_train = 0.0
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}', unit='batch') as t:
             for images, labels in t:
                images = images.to(device=device)
                labels = labels.to(device=device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                loss_train += loss.item()
            
        val_accuracy, val_loss, train_accuracy = validate(model, train_loader, val_loader, loss_fn)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        print(f'loss: {loss_train/(len(train_loader))}, acc: {train_accuracy}, val_loss: {val_loss/(len(val_loader))}, val_acc: {val_accuracy}, lr:{current_lr}')

# validation def
def validate(model, train_loader, val_loader, loss_fn, testing = False):
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        if name == "train" and testing == True:
            train_accuracy = None
            continue
        else:
            correct = 0
            total = 0
            total_loss = 0
            with torch.no_grad():
                for imgs, labels in loader:
                    imgs = imgs.to(device=device)
                    labels = labels.to(device=device)
                    outputs = model(imgs)
                    _, predicted = torch.max(outputs, dim=1)

                    total += labels.shape[0]
                    correct += int((predicted == labels).sum())
                    loss = loss_fn(outputs, labels)
                    total_loss += loss.item()

            if name == "val":
                val_accuracy = correct / total
                val_loss = total_loss
            
            if name == "train":
                train_accuracy = correct / total
    
    return val_accuracy, val_loss, train_accuracy     

# %%
# strat training
model.train()
training_loop(
    n_epochs=args.epochs,
    optimizer=optimizer,
    scheduler=scheduler,
    model=model,
    loss_fn=loss_fn,
    train_loader=train_loader,
    val_loader=val_loader,
    patience=5
)

print('Training completed, start testing on test_dataset...')

# %%
model.eval()
test_loss = 0
test_accuracy = 0
test_accuracy, test_loss, _ = validate(model, train_loader, test_loader, loss_fn, testing=True)

print(f"Average Test Loss: {test_loss / len(test_loader)}, Test Accuracy: {test_accuracy}")

# %%
# save model
current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
model_folder = f'trained_models/{model_name}_{current_datetime}'

os.makedirs(model_folder, exist_ok=True)

model_path = os.path.join(model_folder, 'model.pth')
torch.save(model.state_dict(), model_path)

print(f"Saved your model in {model_path}")
