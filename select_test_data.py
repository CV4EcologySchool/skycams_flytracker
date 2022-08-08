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
dataDir = '/Users/fponce/Documents/cv4e/Skycam_annotations'

training_dataDir = '/Users/fponce/Documents/cv4e/Skycam_annotations/training_frames'
validation_dataDir = '/Users/fponce/Documents/cv4e/Skycam_annotations/validation_frames'
test_dataDir = '/Users/fponce/Documents/cv4e/Skycam_annotations/test_frames'

training_labelDir = '/Users/fponce/Documents/cv4e/Skycam_annotations/training_frames'
validation_labelDir = '/Users/fponce/Documents/cv4e/Skycam_annotations/validation_frames'
test_labelDir = '/Users/fponce/Documents/cv4e/Skycam_annotations/test_frames'

#find json files
label_file_ext = '*.json'
all_label_files = []
for path, subdirs, files in os.walk(dataDir):
    for name in files:
        if fnmatch(name, label_file_ext):
            all_label_files.append(os.path.join(path, name))
#print(all_label_files)

###############################################################################

# read in json file and get a list of a subset of frames as a training set
# getting the first 70% of frames from each video as training set
# the next 30% is split in 2 for validation and test frames

all_training_frames_nums = []
all_validation_frames_nums = []
all_test_frames_nums = []
all_frame_codes = []
for i in range(len(all_label_files)):
    labels_path = all_label_files[i]
    labels_file = open(labels_path)
    labels = json.load(labels_file)
    labels_file_size = len(labels.keys())

    frame_codes = []
    frame_code_numbers = []
    for j in labels.keys():
        frame_code = j
        frame_code_number = re.split('\-', frame_code)[-1][:-4]
        frame_code_numbers.append(int(frame_code_number))
        frame_codes.append(frame_code)

    sorted_frame_code_numbers = list(np.sort(frame_code_numbers))
    last_training_frame = int(len(sorted_frame_code_numbers)/(1/0.7))
    training_frame_code_numbers = sorted_frame_code_numbers[0:last_training_frame]
    last_validation_frame = int((len(sorted_frame_code_numbers) - last_training_frame)/2)+last_training_frame
    validation_frame_code_numbers = sorted_frame_code_numbers[last_training_frame:last_validation_frame]
    test_frame_code_numbers = sorted_frame_code_numbers[last_validation_frame:-1]

    all_training_frames_nums.append(training_frame_code_numbers)
    all_test_frames_nums.append(test_frame_code_numbers)
    all_validation_frames_nums.append(validation_frame_code_numbers)
    all_frame_codes.append(frame_codes)

labels_file.close()
#(all_training_frames_nums)

#################################################################################
#find the frame codes that correspond to training, validation, test frames

all_training_frame_codes = []
for i in range(len(all_training_frames_nums)):
    training_frame_codes = []
    for j in range(len(all_training_frames_nums[i])):
        sub_string = sub = str('-')+str(all_training_frames_nums[i][j])+str('.')
        training_frame_code = "\n".join(s for s in all_frame_codes[i] if sub_string.lower() in s.lower())
        training_frame_codes.append(training_frame_code)
    all_training_frame_codes.append(training_frame_codes)

all_validation_frame_codes = []
for i in range(len(all_validation_frames_nums)):
    validation_frame_codes = []
    for j in range(len(all_validation_frames_nums[i])):
        sub_string = sub = str('-')+str(all_validation_frames_nums[i][j])+str('.')
        validation_frame_code = "\n".join(s for s in all_frame_codes[i] if sub_string.lower() in s.lower())
        validation_frame_codes.append(validation_frame_code)
    all_validation_frame_codes.append(validation_frame_codes)

all_test_frame_codes = []
for i in range(len(all_test_frames_nums)):
    test_frame_codes = []
    for j in range(len(all_test_frames_nums[i])):
        sub_string = sub = str('-')+str(all_test_frames_nums[i][j])+str('.')
        test_frame_code = "\n".join(s for s in all_frame_codes[i] if sub_string.lower() in s.lower())
        test_frame_codes.append(test_frame_code)
    all_test_frame_codes.append(test_frame_codes)

###############################################################################
#create the paths to frames so I can copy and move them

training_frames_image_paths = []
for i in range(len(all_training_frame_codes)):
    for j in range(len(all_training_frame_codes[i])):
        labels_path = all_label_files[i]
        head, tail = os.path.split(labels_path)
        training_frame_code_path = head+'/'+all_training_frame_codes[i][j]
        training_frames_image_paths.append(training_frame_code_path)

validation_frames_image_paths = []
for i in range(len(all_validation_frame_codes)):
    for j in range(len(all_validation_frame_codes[i])):
        labels_path = all_label_files[i]
        head, tail = os.path.split(labels_path)
        validation_frame_code_path = head+'/'+all_validation_frame_codes[i][j]
        validation_frames_image_paths.append(validation_frame_code_path)

test_frames_image_paths = []
for i in range(len(all_test_frame_codes)):
    for j in range(len(all_test_frame_codes[i])):
        labels_path = all_label_files[i]
        head, tail = os.path.split(labels_path)
        test_frame_code_path = head+'/'+all_test_frame_codes[i][j]
        test_frames_image_paths.append(test_frame_code_path)

print(len(training_frames_image_paths))
print(len(validation_frames_image_paths))
print(len(test_frames_image_paths))

###############################################################################
#get the paths for the .txt files
training_frames_label_paths = []
for i in range(len(training_frames_image_paths)):
    txt_path_training = training_frames_image_paths[i][:-4]+'.txt'
    training_frames_label_paths.append(txt_path_training)

validation_frames_label_paths = []
for i in range(len(validation_frames_image_paths)):
    txt_path_validation = validation_frames_image_paths[i][:-4]+'.txt'
    validation_frames_label_paths.append(txt_path_validation)

test_frames_label_paths = []
for i in range(len(test_frames_image_paths)):
    txt_path_test = test_frames_image_paths[i][:-4]+'.txt'
    test_frames_label_paths.append(txt_path_test)

# ###############################################################################
# #find test frames and copy them and move them
# for f in training_frames_image_paths:
#     head, tail = os.path.split(f)
#     shutil.copy(f, training_dataDir+'/'+tail)
# # ###############################################################################
#
# #find non-test frames and copy them and move them
# for ff in validation_frames_image_paths:
#     head, tail = os.path.split(ff)
#     shutil.copy(f, validation_dataDir+'/'+tail)
# # ###############################################################################
#
# #find non-test frames and copy them and move them
# for ff in test_frames_image_paths:
#     head, tail = os.path.split(ff)
#     shutil.copy(f, test_dataDir+'/'+tail)

###############################################################################
#find test labels and copy them and move them
for f in training_frames_label_paths:
    head, tail = os.path.split(f)
    shutil.copy(f, training_labelDir+'/'+tail)
# ###############################################################################

#find non-test labels and copy them and move them
for ff in validation_frames_label_paths:
    head, tail = os.path.split(ff)
    shutil.copy(f, validation_labelDir+'/'+tail)
# ###############################################################################

#find non-test labels and copy them and move them
for ff in test_frames_label_paths:
    head, tail = os.path.split(ff)
    shutil.copy(f, test_labelDir+'/'+tail)
