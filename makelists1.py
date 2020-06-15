from skimage import io
import os
import glob
import numpy as np

##################################################################################################
ANG=list(glob.glob(os.path.join("/workspace/data/ANG/*.JPG"))) #0
ANR=list(glob.glob(os.path.join("/workspace/data/ANR/*.JPG"))) #1
NA=list(glob.glob(os.path.join("/workspace/data/NA/*.JPG")))   #2
NG=list(glob.glob(os.path.join("/workspace/data/NG/*.JPG")))   #3
NR=list(glob.glob(os.path.join("/workspace/data/NR/*.JPG")))   #3


len_ANG = len(ANG)
len_ANR = len(ANR)
len_NA = len(NA)
len_NG = len(NG)
len_NR = len(NR)

#####################################################################
train_ANG = ANG[0:int(len_ANG*0.7)]
val_ANG   = ANG[int(len_ANG*0.7):int(len_ANG*0.8)]
test_ANG  = ANG[int(len_ANG*0.8):]

train_ANR = ANR[0:int(len_ANR*0.7)]
val_ANR   = ANR[int(len_ANR*0.7):int(len_ANR*0.8)]
test_ANR  = ANR[int(len_ANR*0.8):]

train_NA = NA[0:int(len_NA*0.7)]
val_NA   = NA[int(len_NA*0.7):int(len_NA*0.8)]
test_NA  = NA[int(len_NA*0.8):]

train_NG = NG[0:int(len_NG*0.7)]
val_NG   = NG[int(len_NG*0.7):int(len_NG*0.8)]
test_NG  = NG[int(len_NG*0.8):]

train_NR = NR[0:int(len_NR*0.7)]  
val_NR   = NR[int(len_NR*0.7):int(len_NR*0.8)]
test_NR  = NR[int(len_NR*0.8):]

#############################################################################
print('ANG', len(ANG))
print('ANR', len(ANR))
print('NA', len(NA))
print('NG', len(NG))
print('NR', len(NR))
#############################################################################
l_ANG_train=[0]*len(train_ANG)
l_ANR_train=[1]*len(train_ANR)
l_NA_train=[2]*len(train_NA)
l_NG_train=[3]*len(train_NG)
l_NR_train=[4]*len(train_NR)


l_ANG_val=[0]*len(val_ANG)
l_ANR_val=[1]*len(val_ANR)
l_NA_val=[2]*len(val_NA)
l_NG_val=[3]*len(val_NG)
l_NR_val=[4]*len(val_NR)


l_ANG_test=[0]*len(test_ANG)
l_ANR_test=[1]*len(test_ANR)
l_NA_test=[2]*len(test_NA)
l_NG_test=[3]*len(test_NG)
l_NR_test=[4]*len(test_NR)

##########################################################################

IDS_train = train_ANG + train_ANR + train_NA + train_NG + train_NR
LABS_train = l_ANG_train + l_ANR_train + l_NA_train + l_NG_train + l_NR_train
IDSv_train = np.asarray(IDS_train)
LABSv_train = np.asarray(LABS_train)

IDS_val = val_ANG + val_ANR + val_NA + val_NG + val_NR
LABS_val = l_ANG_val + l_ANR_val + l_NA_val + l_NG_val + l_NR_val
IDSv_val = np.asarray(IDS_val)
LABSv_val = np.asarray(LABS_val)

IDS_test = test_ANG + test_ANR + test_NA + test_NG + test_NR
LABS_test = l_ANG_test + l_ANR_test + l_NA_test + l_NG_test + l_NR_test
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

print( len(IDSv_train))
print(len(IDSv_val))
print( len(IDSv_test))
