#!/usr/bin/env python3
import warnings
import numpy as np
import os
import time
import json
import cv2
from fnmatch import fnmatch
import shutil
import re
###############################################################################
# input raw video directory
images_path = '/Users/fponce/Documents/cv4e/Skycam_annotations/0508_092022'
labels_path = '/Users/fponce/Documents/cv4e/Skycam_annotations/0508_092022/0508_092022.json'

#read in json file

labels_file = open(labels_path)
labels = json.load(labels_file)

# Iterating through the json to find frame names and visualize images
for i in sorted(labels.keys()):

    frame_code = i
    frame_path = images_path+'/'+str(frame_code)
    label_coordinates = labels[frame_code][0]

    x1 = int(label_coordinates[0])
    y1 = int(label_coordinates[1])
    x2 = int(label_coordinates[2])
    y2 = int(label_coordinates[3])
    img_unchanged = cv2.imread(frame_path,cv2.IMREAD_UNCHANGED)

    cv2.rectangle(img_unchanged, (x1, y1), (x2, y2), (0,0,0), 1)
    cv2.imshow(str(frame_code),img_unchanged)
    cv2.waitKey(0)

# Closing file
labels_file.close()
