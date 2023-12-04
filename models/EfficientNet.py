# %%
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

# %%

class MBConvBlock(nn.Module):
    '''
    在efficientNet中的論文使用Mobile Inverted Residual Bottleneck Block作為basic block
    '''
    


class EfficientNet():
    pass

def EfficientNetB0():
    return EfficientNet(0)

def EfficientNetB1():
    return EfficientNet(1)

def EfficientNetB2():
    return EfficientNet(2)

def EfficientNetB3():
    return EfficientNet(3)

def EfficientNetB4():
    return EfficientNet(4)

def EfficientNetB5():
    return EfficientNet(5)

def EfficientNetB6():
    return EfficientNet(6)

def EfficientNetB7():
    return EfficientNet(7)