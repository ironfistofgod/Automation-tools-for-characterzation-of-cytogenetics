import os
import torch
import numpy as np
from skimage import io

USE_CUDA = torch.cuda.is_available()
DEVICE = 1
def to_cuda(v):
    if USE_CUDA:
        return v.cuda(DEVICE)
    return v

def shuffle(train):
    p = np.random.permutation(len(train))
    return p

def shuffle(train):
    p = np.random.permutation(len(train))
    return p

def make_patch(id):
    input=io.imread(id)
#    input = input[115:840,100:1005,:]
    input=input[200:760,200:888,:]
    input=input[:,:,0]
    input = np.reshape(input, (1, input.shape[0], input.shape[1]))
    input=input/255.0

    return input

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

