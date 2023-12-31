# %%
import torch
import torch.nn as nn
from typing import Type

# %%
class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        ) # same size
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return  out

class Bottleneck(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride = 1,
            expansion = 4,
            downsample: nn.Module = None,
    ):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels,
            kernel_size=3, 
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, 
            out_channels*self.expansion,
            kernel_size=1, 
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return  out

        
class ResNet(nn.Module):
    def __init__(
        self, 
        num_layers: int,
        block: Type[BasicBlock],
        img_channels: int = 3,
        num_classes: int  = 10
    ) -> None:
        super(ResNet, self).__init__()
        '''
        以下的 layers list宣告Residual Block的數量，以及stack的數量。
        '''
        if num_layers == 18:
            layers = [2, 2, 2, 2]
            self.expansion = 1

        if num_layers == 50:
            layers = [3, 4, 6, 3]
            self.expansion = 4
        
        if num_layers == 34:
            layers = [3, 4, 6, 3]
            self.expansion = 1

        if num_layers == 101:
            layers = [3, 4, 23, 3]
            self.expansion = 4
        
        if num_layers == 152:
            layers = [3, 8, 36, 3]
            self.expansion = 4

        self.in_channels = 64
        # 所有 ResNet（18到152）都在前三層包含一個 Conv2d => BN => ReLU。
        # kernel size為7。
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7, 
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)
    def _make_layer(
        self, 
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels*self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # The spatial dimension of the final layer's output 
        # size should be (7, 7) for all ResNets.
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
# %%

def ResNet18():
    model = ResNet(18,BasicBlock)
    test(model)
    return model

def ResNet34():
    model = ResNet(34,BasicBlock)
    test(model)
    return model

def ResNet50():
    model = ResNet(50,Bottleneck)
    test(model)
    return model

def ResNet101():
    model = ResNet(101,Bottleneck)
    test(model)
    return model

def ResNet152():
    model = ResNet(152,Bottleneck)
    test(model)
    return model

def test(model_fn):
    model = model_fn
    try:
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        print('Your model is ready!')
    except Exception as e:
        error_message = "There are problems with the model. Please check the model's architecture."
        raise Exception(error_message) from e
# %%
# model = ResNet50()
