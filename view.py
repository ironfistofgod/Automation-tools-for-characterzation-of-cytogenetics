import os
import glob
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
from tqdm import tqdm
import torch.optim.lr_scheduler
import itertools
import matplotlib.pyplot as plt
import torchnet as tnt
from torchsummary import summary
import numpy as np
from torchsummary import summary
import ConvNet1
import Rs1
import Rs2
import tools

model=Rs1.resnet18(1,100,5)
#model = Myresnet1.resnet18(1,4)
#model=tools.to_cuda(ConvNet1.MyConvNet())

summary(model.cuda(), (1, 570, 690))




