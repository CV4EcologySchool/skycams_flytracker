#!/usr/bin/env python3
import warnings
import numpy as np
import os
import time
import json
import cv2
from fnmatch import fnmatch
import shutil
###############################################################################
# input raw video directory
dataDir = '/Users/fponce/Documents/cv4e/Skycam_annotations'
test_dataDir = '/Users/fponce/Documents/cv4e/Skycam_annotations/test_frames'

#find json files
label_file_ext = '*.json'
all_label_files = []
for path, subdirs, files in os.walk(dataDir):
    for name in files:
        if fnmatch(name, label_file_ext):
            all_label_files.append(os.path.join(path, name))
# print(all_label_files)

# read in json file and get a list of a random subset of frames
all_rand_frames = []
all_leftover_frames = []
for i in range(len(all_label_files)):
    labels_path = all_label_files[i]
    labels_file = open(labels_path)
    labels = json.load(labels_file)
    labels_file_size = len(labels.keys())
    rand_frames = np.random.randint(1,labels_file_size,int(0.3*labels_file_size))
    leftover_frames = list(set(np.arange(0,labels_file_size, 1)) - set(rand_frames))
    all_rand_frames.append(np.sort(rand_frames))
    all_leftover_frames.append(np.sort(leftover_frames))
labels_file.close()

# get the frame paths that have been selected as test data
test_frames_image_paths = []
for i in range(len(all_label_files)):
    labels_path = all_label_files[i]
    labels_file = open(labels_path)
    labels = json.load(labels_file)

    for j in range(len(all_rand_frames[i])):
        test_frames = (list(labels.keys())[j])
        head, tail = os.path.split(labels_path)
        test_frames_image_paths.append(head+'/'+test_frames)
labels_file.close()
print((test_frames_image_paths))
###############################################################################

#find frames and copy them and move them

for f in test_frames_image_paths:
    head, tail = os.path.split(f)
    shutil.copy(f, test_dataDir+'/'+tail)
