import ConvNet1_int
import Myresnet1
import numpy as np
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
import matplotlib.pyplot as plt
import torchnet as tnt
import ConvNet1
import new_u
import tools

test_IDSv=np.load('./trainvaltest/IDSv_test.npy')
test_LABSv=np.load('./trainvaltest/LABSv_test.npy')

#model=ConvNet1.MyConvNet()
#model=Myresnet1.resnet18(1,5)
model=tools.to_cuda(new_u.U_Net2())
model.load_state_dict(torch.load("./models/model_11.pt"))
model.cuda()

criterion=nn.CrossEntropyLoss()
confusion_matrix = tnt.meter.ConfusionMeter(5)

with torch.no_grad():
    model.eval()
    test_losses = []


for i in range(0,len(test_IDSv)):
    input = io.imread(test_IDSv[i])
    input = input[:,:,0]
    input = np.reshape(input, (1, input.shape[0], input.shape[1]))
    input=input/255.0
    input=np.reshape(input, (1,input.shape[0], input.shape[1], input.shape[2]))
#            input[:,0,:,:]=(input[:,0,:,:]-mean[0])/float(stdv[0])
    lab = test_LABSv[i]
    label = torch.Tensor(1)
    label[0]=int(lab)

    inputV=Variable(torch.from_numpy(input).cuda())
    labelV=Variable(label).cuda()
    output=model(inputV.float())
    loss=criterion(output, labelV.long())
    test_losses.append(loss.item())
    confusion_matrix.add(output.data, labelV)
#    if cnt % 10 == 0:
#    print('Test (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                epoch, epochs, cnt, len(test_IDSv),100.*cnt/len(test_IDSv), loss.item()))
#    cnt=cnt+1
    del(inputV, labelV, loss)
#            del(inputs, targets, loss)

print(confusion_matrix.conf)
test_Loss = np.mean(test_losses)
test_acc=(np.trace(confusion_matrix.conf)/float(np.ndarray.sum(confusion_matrix.conf))) *100
print('TEST_OA', '%.3f' % test_acc)
