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
import torchnet as tnt
import ConvNet1_int
import tools
import Myresnet1
import Rs2
import Rs1
import ConvNet1
import matplotlib.pyplot as plt
import unet
import unet1
import new_u
##################################################################################################
IDSv=np.load('./trainvaltest/IDSv_train.npy')
LABSv=np.load('./trainvaltest/LABSv_train.npy')

val_IDSv = np.load('./trainvaltest/IDSv_val.npy')
val_LABSv = np.load('./trainvaltest/LABSv_val.npy')

weight_tensor=torch.FloatTensor(5)
weight_tensor[0]= 1.1
weight_tensor[1]= 0.4
weight_tensor[2]= 0.4
weight_tensor[3]= 0.9
weight_tensor[4]= 0.3
criterion=tools.to_cuda(nn.CrossEntropyLoss(tools.to_cuda(weight_tensor))) #with weights
#criterion=nn.CrossEntropyLoss().cuda() #without weights

#model=tools.to_cuda(Rs2.resnet18(1,2000,4))
model=tools.to_cuda(ConvNet1.MyConvNet())
#model=tools.to_cuda(unet.UNet(padding=True, up_mode='upsample'))
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs=15
ff=open('./models/progress.txt','w')
batchsize=2
confusion_matrix = tnt.meter.ConfusionMeter(5)

t_acc=[]
t_Loss=[]
v_acc=[]
v_Loss=[]


for epoch in range(1,epochs+1):
        model.train()

        p = tools.shuffle(IDSv)
        IDSv = IDSv[p]
        LABSv = LABSv[p] 
        train_losses = []
        cnt=0
###############################################################################

        for t in tqdm(range(0, len(IDSv)-batchsize, batchsize)):
            iter=t/batchsize
            inputs=[]
            targets=[]
            for i in range(t, min(t+batchsize, len(IDSv))):
                  input = tools.make_patch(IDSv[i])
                  lab=LABSv[i] 

                  inputs.append(input)
                  targets.append(lab)

            inputs=np.asarray(inputs)
            targets=np.asarray(targets)
            inputs=tools.to_cuda(torch.from_numpy(inputs).float())
            targets=tools.to_cuda(torch.from_numpy(targets).long())

            optimizer.zero_grad()
            outputs=model(inputs)
            confusion_matrix.add(outputs.data.squeeze(), targets)
            loss=criterion(outputs, targets)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
##################################################################################
            if cnt % 10 == 0:
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, epochs, cnt, len(IDSv),100.*cnt/len(IDSv), loss.item()))
            cnt=cnt+1
            del(inputs,targets,loss)

        train_Loss = np.mean(train_losses)
        t_Loss.append(train_Loss)
        print(confusion_matrix.conf)
        train_acc=(np.trace(confusion_matrix.conf)/float(np.ndarray.sum(confusion_matrix.conf))) *100
        t_acc.append(train_acc)        
        print('TRAIN_OA', '%.3f' % train_acc)

        confusion_matrix.reset()

        ##VALIDATION
        with torch.no_grad():
         model.eval()
         val_losses = []
##########################################################################################
         for t in tqdm(range(0, len(val_IDSv)-batchsize, batchsize)):
            iter=t/batchsize
            inputs=[]
            targets=[]
            for i in range(t, min(t+batchsize, len(val_IDSv))):
                  input = tools.make_patch(val_IDSv[i])
                  lab=val_LABSv[i]

                  inputs.append(input)
                  targets.append(lab)

            inputs=np.asarray(inputs)
            targets=np.asarray(targets)
            inputs=tools.to_cuda(torch.from_numpy(inputs).float())
            targets=tools.to_cuda(torch.from_numpy(targets).long())

            outputs=model(inputs)
            confusion_matrix.add(outputs.data.squeeze(), targets)
            loss=criterion(outputs, targets)
            val_losses.append(loss.item())
######################################################################

            if cnt % 10 == 0:
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, epochs, cnt, len(val_IDSv),100.*cnt/len(val_IDSv), loss.item()))
            cnt=cnt+1
            del(inputs, targets, loss)

        print(confusion_matrix.conf)
        val_Loss = np.mean(val_losses)
        v_Loss.append(val_Loss)
        val_acc=(np.trace(confusion_matrix.conf)/float(np.ndarray.sum(confusion_matrix.conf))) *100
        v_acc.append(val_acc)
        print('VAL_OA', '%.3f' % val_acc)
        print('Train_Loss: ', np.mean(train_Loss))
        print('Val_Loss: ', np.mean(val_Loss))
        
        tools.write_results(train_Loss, val_Loss, train_acc, val_acc, epoch)
        torch.save(model.state_dict(), './models/model_{}.pt'.format(epoch))
        confusion_matrix.reset()

plt.figure(1)
plt.plot(t_Loss)
plt.plot(v_Loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss','val loss'], loc='upper left')
plt.savefig("plots/Myfile.png",format="png")


plt.figure(2)
plt.plot(t_acc)
plt.plot(v_acc)
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train acc', 'val acc'], loc='upper left')
plt.show()
plt.savefig("plots/Myfile1.png",format="png")
