from skimage import io
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
import torchnet as tnt
from sklearn.metrics import confusion_matrix
import ConvNet1
import Rs2
import Rs1
import Myresnet1
import new_u
import tools
import un
##################################################################################################
test_IDSv=np.load('./trainvaltest/IDSv_test.npy')
test_LABSv=np.load('./trainvaltest/LABSv_test.npy')

#mean=np.load('./trainvaltest/mean.npy')
#stdv=np.load('./trainvaltest/stdv.npy')
#####################################################################################################

#confusion_matrix = tnt.meter.ConfusionMeter(5)

#model=ConvNet1.MyConvNet()
#model=Rs2.resnet18(1,100,5)
model=tools.to_cuda(un.UNet())
model.load_state_dict(torch.load("./models/model_5.pt"))
model.eval()

batchsize=1

labels = []
preds = []

for i in range(0,len(test_IDSv)):
        id = test_IDSv[i]
        input=io.imread(id)
#        input=input[115:840,100:1005,:]
        input=input[200:760,200:888,:]
        input=input[:,:,0]
        input = np.reshape(input, (1, 1, input.shape[0], input.shape[1]))
        input=input/255.0
        lab=test_LABSv[i]
        input=tools.to_cuda(torch.from_numpy(input).float())
        output1, output2=model(input)
        _, ind = torch.max(output2,1)
        preds.append(ind.data.cpu().numpy()[0])
        labels.append(lab)


conf = confusion_matrix(labels, preds)

print(conf)
test_acc=(np.trace(conf)/float(np.ndarray.sum(conf))) *100
print('TEST_OA', '%.3f' % test_acc)

   
