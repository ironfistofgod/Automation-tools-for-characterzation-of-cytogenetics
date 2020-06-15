import numpy as np
from skimage import io
import glob
import os
import cv2

HyperD=list(glob.glob(os.path.join("/workspace/data/HyperD/*.JPG"))) #0
T1114=list(glob.glob(os.path.join("/workspace/data/T1114/*.JPG"))) #1
T922=list(glob.glob(os.path.join("/workspace/data/T922/*.JPG")))   #2
TRI=list(glob.glob(os.path.join("/workspace/data/TRI/*.JPG")))   #3

folder="TRI"

cnt=0
for id in TRI: ##
 print(cnt, ' from ', len(TRI)) ##
 #ss=id[id.find(folder+ '/')+7:id.find(".JPG")+4] ##
 ss=os.path.basename(id)
# print('ss', ss)
 im=io.imread(id)
 im=im[115:840,100:1005,:]
# imres=cv2.resize(im, (832,832))
 io.imsave('/workspace/data/NEW/' + folder +'/'+ ss, im) ##
 cnt=cnt+1
 
