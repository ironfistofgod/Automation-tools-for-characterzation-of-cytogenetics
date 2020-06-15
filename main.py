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
import Myresnet1
import torchvision
from torch.utils.tensorboard import SummaryWriter
import ConvNet1_int
#import Myresnet3
from torchsummary import summary
#from imblearn.over_sampling import SMOTE
#import smote_variants as sv
#summary(model, (1, 224, 224))

##################################################################################################
IDSv=np.load('./trainvaltest/IDSv_train.npy')
LABSv=np.load('./trainvaltest/LABSv_train.npy')

val_IDSv = np.load('./trainvaltest/IDSv_val.npy')
val_LABSv = np.load('./trainvaltest/LABSv_val.npy')

#mean=np.load('./trainvaltest/mean.npy')
#stdv=np.load('./trainvaltest/stdv.npy')
#####################################################################################################
#w0 -- 0.9051724137931034
#w1 -- 0.7481527093596059
#w2 -- 0.47536945812807885
#w3 -- 0.8713054187192119

weight_tensor=torch.FloatTensor(4)
weight_tensor[0]= 0.9
weight_tensor[1]= 0.2
weight_tensor[2]= 0.2
weight_tensor[3]= 0.8
criterion=nn.CrossEntropyLoss(weight_tensor.cuda())
#criterion=nn.CrossEntropyLoss().cuda(2)

model=ConvNet1_int.MyConvNet(1,4)
#model = Myresnet1.resnet18(1, 4)
model=model.cuda()
#summary(model.cuda(1), input_size=(1, 725, 905))

optimizer = optim.Adam(model.parameters(), lr=0.0001)
#optimizer = optim.SGD(model.parameters(), lr=0.0001)
epochs=15
#ff=open('./models/progress.txt','w')
batchsize=2
confusion_matrix = tnt.meter.ConfusionMeter(4)



def shuffle(train):
    p = np.random.permutation(len(train))
    return p

def write_results(train_loss, val_loss, train_acc, val_acc, epoch):
    ff=open('./models/progress.txt','a')
    ff.write(' E: ')
    ff.write(str(epoch))
    ff.write('         ')
    ff.write(' TRAIN_OA: ')
    ff.write(str('%.3f' % train_acc))
    ff.write(' VAL_OA: ')
    ff.write(str('%.3f' % val_acc))
    ff.write('         ')
    ff.write(' TRAIN_LOSS: ')
    ff.write(str('%.3f' % train_loss))
    ff.write(' VAL_LOSS: ')
    ff.write(str('%.3f' % val_loss))

    ff.write('\n')

def make_conf(output, label):
  output_conf=(output.data).transpose(1,3)
  output_conf=output_conf.transpose(1,2)
  output_conf=(output_conf.contiguous()).view(output_conf.size(0)*output_conf.size(1)*output_conf.size(2), output_conf.size(3))
  target_conf=label.data
  target_conf=(target_conf.contiguous()).view(target_conf.size(0)*target_conf.size(1)*target_conf.size(2))

  return output_conf, target_conf

def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size


for epoch in range(1,epochs+1):
        model.train()

        p = shuffle(IDSv)
        IDSv = IDSv[p]
        LABSv = LABSv[p]

        train_losses = []

        cnt=0

        for t in tqdm(range(0, len(IDSv)-batchsize, batchsize)):
            iter=t/batchsize
            inputs=[]
            targets=[]
            for i in range(t, min(t+batchsize, len(IDSv))):
#        for i in p:
                input = io.imread(IDSv[i])
                input = input[:,:,0]
                input = np.reshape(input, (1, input.shape[0], input.shape[1]))
                input=input/255.0
                input=np.reshape(input, (1,input.shape[0], input.shape[1], input.shape[2]))
#            input[:,0,:,:]=(input[:,0,:,:]-mean[0])/float(stdv[0])
                lab = LABSv[i]
                label = torch.Tensor(1)
                label[0]=int(lab)
                inputV=Variable(torch.from_numpy(input).cuda())
                labelV=Variable(label).cuda()
                optimizer.zero_grad()
                output=model(inputV.float())
                loss=criterion(output, labelV.long())
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
                confusion_matrix.add(output.data, labelV)
                if cnt % 10 == 0:
                    print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, epochs, cnt, len(IDSv),100.*cnt/len(IDSv), loss.item()))
                cnt=cnt+1
                del(inputV, labelV, loss)
#            inputV.detach()
#            labelV.detach()
            
#            del(inputs,targets,loss)

            train_Loss = np.mean(train_losses)
            print(confusion_matrix.conf)
            train_acc=(np.trace(confusion_matrix.conf)/float(np.ndarray.sum(confusion_matrix.conf))) *100
            print('TRAIN_OA', '%.3f' % train_acc)
        
#        tb= SummaryWriter()
#        tb.add_scalar("loss",train_Loss, epoch)
#        tb.add_scalar("Accuracy",train_acc, epoch)

            torch.save(model.state_dict(), './models/model_{}.pt'.format(epoch))
            confusion_matrix.reset()

        ##VALIDATION
        with torch.no_grad():
            model.eval()
            val_losses = []

        for t in tqdm(range(0, len(IDSv)-batchsize, batchsize)):
            iter=t/batchsize
            inputs=[]
            targets=[]
            for i in range(t, min(t+batchsize, len(IDSv))):
#        for i in range(0,len(val_IDSv)):
                input = io.imread(val_IDSv[i])
                input = input[:,:,0]
                input = np.reshape(input, (1, input.shape[0], input.shape[1]))
                input=input/255.0
                input=np.reshape(input, (1,input.shape[0], input.shape[1], input.shape[2]))
#            input[:,0,:,:]=(input[:,0,:,:]-mean[0])/float(stdv[0])
                lab =val_LABSv[i]
                label = torch.Tensor(1)
                label[0]=int(lab)

                inputV=Variable(torch.from_numpy(input).cuda())
                labelV=Variable(label).cuda()
                output=model(inputV.float())
                loss=criterion(output, labelV.long())
                val_losses.append(loss.item())
                confusion_matrix.add(output.data, labelV)
                if cnt % 10 == 0:
                    print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, epochs, cnt, len(val_IDSv),100.*cnt/len(val_IDSv), loss.item()))
                cnt=cnt+1
                del(inputV, labelV, loss)
#            del(inputs, targets, loss)

            print(confusion_matrix.conf)
            val_Loss = np.mean(val_losses)
            val_acc=(np.trace(confusion_matrix.conf)/float(np.ndarray.sum(confusion_matrix.conf))) *100
            print('VAL_OA', '%.3f' % val_acc)
   

       # write_results(train_Loss, val_Loss, train_acc, val_acc, epoch)
            confusion_matrix.reset()
