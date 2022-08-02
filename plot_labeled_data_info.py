#!/usr/bin/env python3
import warnings
import numpy as np
import os
import time
from subprocess import call
import cv2
###############################################################################

# input raw video directory
#DataDir = '/Volumes/COMPA/upward_facing_cameras_data/raw_data'

image_path = '/home/ubuntu/work/labeled_data/0419_084547/0419-upcam_01_vi_0001_20190419_084547_trimmed-34614.jpg'

img = cv2.imread(image_path)
# Output img with window name as 'image'
cv2.imshow('image', img)
