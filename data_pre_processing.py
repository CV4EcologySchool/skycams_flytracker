#!/usr/bin/env python3
import warnings
import numpy as np
import os
import time
from subprocess import call
###############################################################################

# input raw video directory
DataDir = '/Volumes/COMPA/upward_facing_cameras_data/raw_data'
# output trimmed video directory
OutputDir_str = '/Volumes/COMPA/upward_facing_cameras_data/pre_processed_data'

t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
#experiment_dates = ['0629']
experiment_dates = ['0611']
# release time to cut videos
#release_times = ['17_43_20']
release_times = ['07_38_00']
# ###############################################################################

all_release_t_in_secs = []
for i in range(len(release_times)):
    release_t = release_times[i]
    release_t_in_secs = (int(release_t[0:2])*3600+(int(release_t[3:5])*60)+(int(release_t[6:8])))
    all_release_t_in_secs.append(release_t_in_secs)
    print('release in secs', release_t_in_secs)

# times to cut before and after video
min_pre_release = 12
min_post_release = 1

min_pre_release = int(min_pre_release)
min_post_release = int(min_post_release)
#
for e in range(len(experiment_dates)):
    OutputDir = OutputDir_str+'/'+'pre_processed_data_'+str(experiment_dates[e])
    try:
        os.makedirs(OutputDir)
    except FileExistsError:
        print('directory already exists')
        pass
    for dirpath, dirnames, filenames in os.walk(DataDir):
        for filename in [f for f in filenames if f.endswith(".mp4") and experiment_dates[e] in f]:
            file_name = os.path.join(dirpath, filename)
            basename = os.path.basename(file_name)

            if basename[0]=='.':
                print('skipping ', basename)
            else:
                cn = dirpath.split('/')[-1] #camera name
                bn =  basename[:basename.find(".mp4")]

                hour_start = int((basename.split('.')[0]).split('_')[-1][0:2])
                min_start = int((basename.split('.')[0]).split('_')[-1][2:4])
                sec_start = int((basename.split('.')[0]).split('_')[-1][4:6])

                vid_start_in_secs = (hour_start*3600+min_start*60+sec_start)
                release_t_in_secs = int(all_release_t_in_secs[e])

                sec_start = release_t_in_secs - vid_start_in_secs - (min_pre_release)*60
                sec_end = release_t_in_secs - vid_start_in_secs + (min_pre_release + min_post_release)*60
                #sec_end = vid_start_in_secs + (min_pre_release + min_post_release)*60
                print(sec_start, sec_end)
                splitx_string = str(sec_start) +":" + str(sec_end)
                output_name = cn + '_' + bn +'_trimmed'+'.mp4'
                output_name = OutputDir+'/'+output_name
                try:
                    call(["MP4Box", "-splitx", splitx_string, file_name, "-out", output_name])
                except:
                    print('something wrong here', basename)
                print('sliced', basename)
