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
from torch.utils.data import random_split

import torchvision
import torchvision.transforms as transforms

from models import *
# %%
# setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
parser = argparse.ArgumentParser(description='train CIFAR 10 with PyTorch')
parser.add_argument('--model', choices=['ResNet18', 'ResNet50', 'EfficientNetB0'], required=True,)
parser.add_argument('--lr', help='input learning rate')
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
training_set = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False)

torch.manual_seed(42)
val_size = int(len(training_set)*0.1)
train_size = len(training_set) - val_size
train_ds, val_ds = random_split(training_set, [train_size, val_size])

train_ds.dataset.transform = train_transform
val_ds.dataset.transform = test_transform
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)

testing_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=test_transform)
test_loader = torch.utils.data.DataLoader(testing_set, batch_size=128, shuffle=False, num_workers=2)

classes = testing_set.classes

# %%
# choose model
model_name = args.model
available_models = ['ResNet18', 'ResNet50', 'EfficientNetB0']

if model_name == 'ResNet18':
    model = ResNet18()

elif model_name == 'ResNet50':
    model = ResNet50()

elif model_name == 'EfficientNetB0':
     model = EfficientNetB0()

else:
    raise ValueError(f"{model_name} is not available please use one of models below \n{', '.join(available_models)}")

if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True


# %%
epochs = args.epochs
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

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
    val_loader=valid_loader,
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
model_folder = f'models/model_{current_datetime}'

os.makedirs(model_folder, exist_ok=True)

model_path = os.path.join(model_folder, 'model.pth')
torch.save(model.state_dict(), model_path)

print(f"Saved your model in {model_path}")
