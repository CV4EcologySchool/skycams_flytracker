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

#find json files
label_file_ext = '*.json'
all_label_paths = []
for path, subdirs, files in os.walk(dataDir):
    for name in files:
        if fnmatch(name, label_file_ext):
            all_label_paths.append(os.path.join(path, name))

#get all names of labeled frames from the json files
all_frame_code_numbers = []
for i in [0]:#range(len(all_label_paths)):
    labels_path = all_label_paths[i]
    labels_file = open(labels_path)
    labels = json.load(labels_file)
    labels_file_size = len(labels.keys())

    frame_code_numbers = []
    for j in labels.keys():
        frame_code = j
        frame_code_number = re.split('\-', frame_code)[-1][:-4]
        frame_code_numbers.append(int(frame_code_number))
    end_seqs = np.where(np.diff(frame_code_numbers)>1)[0]
    all_traces = []
    for k in range(len(end_seqs)):
        if k==0:
            trace = np.zeros(end_seqs[k]-0)+k
        elif k>0:
            trace = np.zeros(end_seqs[k]-end_seqs[k-1])+k
        all_traces.append(list(trace))

flat_list = [item for sublist in all_traces for item in sublist]
for i in range(len(all_traces)):

    print(flat_list[i],frame_code_numbers[i])

    #
    #
    # print(end_seqs)
    # print(frame_code_numbers[0:end_seqs[0]+1])
    # print(frame_code_numbers[end_seqs[0]+1:end_seqs[1]])
    # print('oo')
    # all_frame_code_numbers.append(frame_code_numbers)

#print(np.diff(all_frame_code_numbers[2]))
