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


##################################################################################################
HyperD=list(glob.glob(os.path.join("/workspace/data/HyperD/*.JPG"))) #0
T1114=list(glob.glob(os.path.join("/workspace/data/T1114/*.JPG"))) #1
T922=list(glob.glob(os.path.join("/workspace/data/T922/*.JPG")))   #2
TRI=list(glob.glob(os.path.join("/workspace/data/TRI/*.JPG")))   #3


len_HyperD = len(HyperD)
len_T1114 = len(T1114)
len_T922 = len(T922)
len_TRI = len(TRI)
#####################################################################
train_HyperD = HyperD[0:int(len_HyperD*0.6)]
val_HyperD   = HyperD[int(len_HyperD*0.6):int(len_HyperD*0.8)]
test_HyperD  = HyperD[int(len_HyperD*0.8):]

train_T1114 = T1114[0:int(len_T1114*0.6)]
val_T1114   = T1114[int(len_T1114*0.6):int(len_T1114*0.8)]
test_T1114  = T1114[int(len_T1114*0.8):]

train_T922 = T922[0:int(len_T922*0.6)]
val_T922   = T922[int(len_T922*0.6):int(len_T922*0.8)]
test_T922  = T922[int(len_T922*0.8):]

train_TRI = TRI[0:int(len_TRI*0.6)]
val_TRI   = TRI[int(len_TRI*0.6):int(len_TRI*0.8)]
test_TRI  = TRI[int(len_TRI*0.8):]
#############################################################################
print('HyperD', len(HyperD))
print('T1114', len(T1114))
print('T922', len(T922))
print('TRI', len(TRI))
#############################################################################
l_HyperD_train=[0]*len(train_HyperD)
l_T1114_train=[1]*len(train_T1114)
l_T922_train=[2]*len(train_T922)
l_TRI_train=[3]*len(train_TRI)

l_HyperD_val=[0]*len(val_HyperD)
l_T1114_val=[1]*len(val_T1114)
l_T922_val=[2]*len(val_T922)
l_TRI_val=[3]*len(val_TRI)

l_HyperD_test=[0]*len(test_HyperD)
l_T1114_test=[1]*len(test_T1114)
l_T922_test=[2]*len(test_T922)
l_TRI_test=[3]*len(test_TRI)
##########################################################################

IDS_train = train_HyperD + train_T1114 + train_T922 + train_TRI
LABS_train = l_HyperD_train + l_T1114_train + l_T922_train + l_TRI_train
IDSv_train = np.asarray(IDS_train)
LABSv_train = np.asarray(LABS_train)

IDS_val = val_HyperD + val_T1114 + val_T922 + val_TRI
LABS_val = l_HyperD_val + l_T1114_val + l_T922_val + l_TRI_val
IDSv_val = np.asarray(IDS_val)
LABSv_val = np.asarray(LABS_val)

IDS_test = test_HyperD + test_T1114 + test_T922 + test_TRI
LABS_test = l_HyperD_test + l_T1114_test + l_T922_test + l_TRI_test
IDSv_test = np.asarray(IDS_test)
LABSv_test = np.asarray(LABS_test)


#################################################################################

p=np.random.permutation(len(IDSv_train))
IDSv_train=IDSv_train[p]
LABSv_train=LABSv_train[p]

np.save('./trainvaltest/IDSv_train.npy', IDSv_train)
np.save('./trainvaltest/LABSv_train.npy', LABSv_train)

p=np.random.permutation(len(IDSv_val))
IDSv_val=IDSv_val[p]
LABSv_val=LABSv_val[p]
np.save('./trainvaltest/IDSv_val.npy', IDSv_val)
np.save('./trainvaltest/LABSv_val.npy', LABSv_val)

p=np.random.permutation(len(IDSv_test))
IDSv_test=IDSv_test[p]
LABSv_test=LABSv_test[p]
np.save('./trainvaltest/IDSv_test.npy', IDSv_test)
np.save('./trainvaltest/LABSv_test.npy', LABSv_test)

#####################################################################################################
