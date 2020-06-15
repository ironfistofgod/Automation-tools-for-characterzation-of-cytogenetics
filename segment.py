import imageio
# import augmenters from imgaug
from imgaug import augmenters as iaa
import numpy as np
from skimage import io
import glob
import os
import cv2

from PIL import Image

# Import segmentation maps from imgaug
from imgaug.augmentables.segmaps import SegmentationMapOnImage
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
# Open image with mask

HyperD=list(glob.glob(os.path.join("/workspace/data/HyperD/*.JPG"))) #0
T1114=list(glob.glob(os.path.join("/workspace/data/T1114/*.JPG"))) #1
T922=list(glob.glob(os.path.join("/workspace/data/T922/*.JPG")))   #2
TRI=list(glob.glob(os.path.join("/workspace/data/TRI/*.JPG")))   #3

folder="T922"


cnt=0
for id in T922: ##
 print(cnt, ' from ', len(T922)) ##
 ss=os.path.basename(id)
 im=io.imread(id)
 im=im[115:840,100:1005,:]


# Initialize the bounding box for the original image
# using helpers from imgaug package
 bbs = BoundingBoxesOnImage([
    BoundingBox(x1=1, x2=980, y1=9, y2=535)
 ], shape=im.shape)


# Define a simple augmentations pipeline for the image with bounding box
 seq = iaa.Sequential([
     iaa.GammaContrast(1.5), # add contrast
     iaa.Affine(translate_percent={"x": 0.1}, scale=0.8), # translate the image
     iaa.Fliplr(p = 1.0) # apply horizontal flip
 ])


# Apply augmentations
 image_aug, bbs_aug = seq(image=im, bounding_boxes=bbs)
 io.imsave('/workspace/data/NEW/' + folder +'/'+ ss, image_aug) ##
 cnt=cnt+1
