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
dataDir = '/Users/fponce/Documents/cv4e/Skycam_annotations_tempoinfo'

# label data directory
dataDir_labels = '/Users/fponce/Documents/cv4e/Skycam_annotations'

training_dataDir = '/Users/fponce/Documents/cv4e/Skycam_annotations_tempoinfo/training/images'
validation_dataDir = '/Users/fponce/Documents/cv4e/Skycam_annotations_tempoinfo/validation/images'
test_dataDir = '/Users/fponce/Documents/cv4e/Skycam_annotations_tempoinfo/test/images'

training_labelDir = '/Users/fponce/Documents/cv4e/Skycam_annotations_tempoinfo/training/labels'
validation_labelDir = '/Users/fponce/Documents/cv4e/Skycam_annotations_tempoinfo/validation/labels'
test_labelDir = '/Users/fponce/Documents/cv4e/Skycam_annotations_tempoinfo/test/labels'
###############################################################################
def unique(list1):
    # initialize a null list
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list
###############################################################################
#find json files
label_file_ext = '*.json'
all_label_paths = []
for path, subdirs, files in os.walk(dataDir_labels):
    for name in files:
        if fnmatch(name, label_file_ext):
            all_label_paths.append(path)

frame_file_ext = '*.jpg'
all_label_paths = []
for path, subdirs, files in os.walk(dataDir):
    for name in files:
        if fnmatch(name, frame_file_ext):
            p = os.path.join(path, name)
            ps= re.split('\/', p)
            pss = ps[-1][:-4]
            dd = dataDir_labels+'/'+ps[-2]+'/'+pss+'.txt'
            all_label_paths.append(dd)

# #find labels and copy them and move them
all_frame_pathss = []
for lp in all_label_paths:
    head, tail = os.path.split(lp)
    head_output_dir, tail2 = os.path.split(head)
    all_frame_pathss.append(dataDir+'/'+tail2)
    ##shutil.copy(lp, dataDir+'/'+tail2)
all_m_frame_paths = unique(all_frame_pathss)
# ###############################################################################
# getting the first 70% of frames from each video as training set
# the next 30% is split in 2 for validation and test frames
all_frame_paths = []
for i in range(len(all_m_frame_paths)):
    frame_paths = []
    for path, subdirs, files in os.walk(all_m_frame_paths[i]):
        for name in files:
            if fnmatch(name, frame_file_ext):
                frame_paths.append(os.path.join(path, name))
    all_frame_paths.append(frame_paths)

all_training_frames_nums = []
all_validation_frames_nums = []
all_test_frames_nums = []
all_frame_codes = []
for i in range(len(all_frame_paths)):

    frame_codes = []
    frame_code_numbers = []
    for j in range(len(all_frame_paths[i])):
        frame_code = all_frame_paths[i][j]
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

print(all_training_frames_nums[0])
# #################################################################################
#find the frame codes that correspond to training, validation, test frames

training_frames_image_paths = []
for i in range(len(all_training_frames_nums)):
    training_frame_codes = []
    for j in range(len(all_training_frames_nums[i])):
        sub_string = sub = str('-')+str(all_training_frames_nums[i][j])+str('.')
        training_frame_code = "\n".join(s for s in all_frame_codes[i] if sub_string.lower() in s.lower())
        #training_frame_codes.append(training_frame_code)
        training_frames_image_paths.append(training_frame_code)

validation_frames_image_paths = []
for i in range(len(all_validation_frames_nums)):
    validation_frame_codes = []
    for j in range(len(all_validation_frames_nums[i])):
        sub_string = sub = str('-')+str(all_validation_frames_nums[i][j])+str('.')
        validation_frame_code = "\n".join(s for s in all_frame_codes[i] if sub_string.lower() in s.lower())
        #validation_frame_codes.append(validation_frame_code)
        validation_frames_image_paths.append(validation_frame_code)

test_frames_image_paths = []
for i in range(len(all_test_frames_nums)):
    test_frame_codes = []
    for j in range(len(all_test_frames_nums[i])):
        sub_string = sub = str('-')+str(all_test_frames_nums[i][j])+str('.')
        test_frame_code = "\n".join(s for s in all_frame_codes[i] if sub_string.lower() in s.lower())
        #test_frame_codes.append(test_frame_code)
        test_frames_image_paths.append(test_frame_code)

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


###############################################################################
#find test frames and copy them and move them
for f in training_frames_image_paths:
    head, tail = os.path.split(f)
    shutil.copy(f, training_dataDir+'/'+tail)
# ###############################################################################

#find non-test frames and copy them and move them
for ff in validation_frames_image_paths:
    head, tail = os.path.split(ff)
    shutil.copy(ff, validation_dataDir+'/'+tail)
# ###############################################################################

#find non-test frames and copy them and move them
for fff in test_frames_image_paths:
    head, tail = os.path.split(fff)
    shutil.copy(fff, test_dataDir+'/'+tail)

###############################################################################
#find test labels and copy them and move them
for lf in training_frames_label_paths:
    head, tail = os.path.split(lf)
    shutil.copy(lf, training_labelDir+'/'+tail)
# ###############################################################################

#find non-test labels and copy them and move them
for lff in validation_frames_label_paths:
    head, tail = os.path.split(lff)
    shutil.copy(lff, validation_labelDir+'/'+tail)
# ###############################################################################

#find non-test labels and copy them and move them
for lfff in test_frames_label_paths:
    head, tail = os.path.split(lfff)
    shutil.copy(lfff, test_labelDir+'/'+tail)
