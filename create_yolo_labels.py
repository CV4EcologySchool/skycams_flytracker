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
# input data directory
dataDir = '/Users/fponce/Documents/cv4e/Skycam_annotations'
training_dataDir = '/Users/fponce/Documents/cv4e/Skycam_annotations/training_frames'
validation_dataDir = '/Users/fponce/Documents/cv4e/Skycam_annotations/validation_frames'
test_dataDir = '/Users/fponce/Documents/cv4e/Skycam_annotations/test_frames'

#find json files
label_file_ext = '*.json'
all_label_paths = []
for path, subdirs, files in os.walk(dataDir):
    for name in files:
        if fnmatch(name, label_file_ext):
            all_label_paths.append(os.path.join(path, name))

#create a yolov5 format version of labels for all images
for i in range(len(all_label_paths)):
    labels_file = open(all_label_paths[i])
    labels = json.load(labels_file)
    head, tail = os.path.split(all_label_paths[i])
    directory_path = head
    print(directory_path)
    for j in labels.keys():

        label_coordinates = labels[j][0]
        x1 = int(label_coordinates[0])
        y1 = int(label_coordinates[1])
        x2 = int(label_coordinates[2])
        y2 = int(label_coordinates[3])
        (cx,cy) = (x2 + x1)/2, (y2+y1)/2
        width = x2-x1
        height = y2-y1

        f_name = directory_path+'/'+str(j[:-4]+'.txt')

        with open(f_name, 'w') as f:
            try:
                data = [cx, cy, width, height]
                f.write('{}\t{}\t{}\t{}'.format(data[0], data[1], data[2], data[3]))
            except FileNotFoundError:
                print("The 'docs' directory does not exist")
