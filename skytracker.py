from __future__ import print_function
import sys
import cv2
#import cv
import json
import numpy as np

from median_background import MedianBackground
from blob_finder import BlobFinder

class SkyTracker:
    default_param = {
            'bg_window_size': 11,
            'fg_threshold': 10,
            'datetime_mask': {'x': 410, 'y': 20, 'w': 500, 'h': 40},
            'min_area': 0,
            'max_area': 100000,
            'open_kernel_size': (3,3),
            'close_kernel_size': (3,3),
            'kernel_shape': 'ellipse',
            'output_video_name': 'tracking_video.avi',
            'output_video_fps': 20.0,
            'blob_file_name': 'blob_data.json',
            'show_dev_images' : False,
            'min_interblob_spacing' : 2}

    def __init__(self, input_video_name, param=default_param):
        self.input_video_name = input_video_name
        self.param = self.default_param
        if param is not None:
            self.param.update(param)

    def apply_datetime_mask(self,img):
        x = self.param['datetime_mask']['x']
        y = self.param['datetime_mask']['y']
        w = self.param['datetime_mask']['w']
        h = self.param['datetime_mask']['h']
        img_masked = np.array(img)
        img_masked[y:y+h, x:x+w] = np.zeros([h,w,3])
        return img_masked

    def run(self):

        cap = cv2.VideoCapture(self.input_video_name)

        bg_model = MedianBackground(
                window_size=self.param['bg_window_size'],
                threshold=self.param['fg_threshold']
                )

        blob_finder = BlobFinder(
                filter_by_area=True,
                min_area=self.param['min_area'],
                max_area=self.param['max_area'],
                open_kernel_size = self.param['open_kernel_size'],
                close_kernel_size = self.param['close_kernel_size'],
                kernel_shape = self.param['kernel_shape'],
                min_interblob_spacing = self.param['min_interblob_spacing'])

        # Output files
        vid = None
        blob_fid = None

        if self.param['blob_file_name'] is not None:
            blob_fid = open(self.param['blob_file_name'], 'w')

        frame_count = -1

        annotations = {}

        while True:
            if frame_count%1000 == 0:
                print ('frame count: {0}'.format(frame_count))
            if frame_count > 84000:
                break
            # Get frame, mask and convert to gray scale
            ret, frame = cap.read()
            #frame = cv2.resize(frame, (500,500))

            ### 0706 -
            blob_frame_counts = [55139.0, 29382.0, 48400.0, 26174.0, 48401.0, 64577.0, 29746.0, 22380.0, 48397.0, 58896.0, 2188.0, 72512.0, 29749.0, 5134.0, 74495.0, 72515.0, 58688.0, 58690.0, 13103.0, 48722.0, 72503.0, 13102.0, 73765.0, 50029.0, 72511.0, 50028.0, 29380.0, 58686.0, 11382.0, 22378.0, 74494.0, 50032.0, 72513.0, 58897.0, 83385.0, 22372.0, 72509.0, 55141.0, 26171.0, 58894.0, 50030.0, 48723.0, 72504.0, 29383.0, 22374.0, 58893.0, 26173.0, 29747.0, 74493.0, 72505.0, 22377.0, 26172.0, 50031.0, 50035.0, 28897.0, 27613.0, 43126.0, 50034.0, 64578.0, 11384.0, 72506.0, 72507.0, 52159.0, 22376.0, 43124.0, 26170.0, 29748.0, 72514.0, 72510.0, 13105.0, 22375.0, 73766.0, 48721.0, 29744.0, 29384.0, 29750.0, 43125.0, 22379.0, 29745.0, 27614.0, 13101.0, 13100.0, 55142.0, 55140.0, 13104.0, 64579.0, 43122.0, 58687.0, 43127.0, 48396.0, 29743.0, 28898.0, 48399.0, 58895.0, 52160.0, 11380.0, 11381.0, 43123.0, 58892.0, 50033.0, 29381.0, 83384.0, 72508.0, 58689.0, 28900.0, 3505.0, 50036.0, 28899.0]
            ### 0508 - blob_frame_counts = [25208.0, 3358.0, 3352.0, 25209.0, 642.0, 2169.0, 4403.0, 3758.0, 3663.0, 3284.0, 2168.0, 3660.0, 26099.0, 3657.0, 3429.0, 3173.0, 3322.0, 806.0, 807.0, 643.0, 3764.0, 7638.0, 802.0, 669.0, 3616.0, 800.0, 3420.0, 3669.0, 3010.0, 3171.0, 3320.0, 3664.0, 3195.0, 637.0, 3350.0, 3662.0, 3170.0, 3351.0, 26098.0, 7637.0, 805.0, 25206.0, 3321.0, 3319.0, 641.0, 3765.0, 636.0, 3348.0, 3762.0, 3184.0, 3011.0, 3760.0, 3431.0, 639.0, 3252.0, 638.0, 623.0, 4404.0, 3359.0, 5383.0, 2167.0, 2163.0, 3249.0, 2166.0, 3661.0, 809.0, 3419.0, 3425.0, 4180.0, 2164.0, 3009.0, 3617.0, 3172.0, 810.0, 3183.0, 7639.0, 3659.0, 803.0, 26096.0, 3668.0, 3169.0, 3119.0, 808.0, 3120.0, 801.0, 3182.0, 3763.0, 3250.0, 3444.0, 25207.0, 3197.0, 4401.0, 3619.0, 3286.0, 3285.0, 3251.0, 3445.0, 640.0, 4402.0, 2170.0, 3168.0, 3008.0, 3666.0, 670.0, 3665.0, 2162.0, 3349.0, 26097.0, 7636.0, 3618.0, 4400.0, 5382.0, 3667.0, 3196.0, 2165.0, 3007.0]
            ### 0611 - blob_frame_counts = [23608.0, 9799.0, 9379.0, 23612.0, 23788.0, 9062.0, 15362.0, 23614.0, 949.0, 23733.0, 23606.0, 3244.0, 3246.0, 23796.0, 1040.0, 15363.0, 9784.0, 9793.0, 18418.0, 9775.0, 23800.0, 1043.0, 9377.0, 23790.0, 9375.0, 23903.0, 23601.0, 15361.0, 952.0, 18417.0, 24195.0, 9801.0, 23789.0, 1038.0, 9056.0, 9060.0, 23225.0, 24194.0, 9058.0, 23786.0, 23613.0, 9779.0, 23902.0, 1044.0, 18419.0, 956.0, 23791.0, 9063.0, 9059.0, 23603.0, 9782.0, 24192.0, 23798.0, 23598.0, 15360.0, 9791.0, 23793.0, 15365.0, 23932.0, 9378.0, 23926.0, 9776.0, 23792.0, 23732.0, 935.0, 9805.0, 9774.0, 23227.0, 9777.0, 23787.0, 9804.0, 9781.0, 9773.0, 944.0, 23730.0, 23795.0, 18420.0, 9783.0, 951.0, 23604.0, 939.0, 23931.0, 24193.0, 9792.0, 9798.0, 9800.0, 1039.0, 24196.0, 9789.0, 23610.0, 9788.0, 9797.0, 950.0, 23599.0, 18422.0, 1037.0, 9785.0, 23794.0, 23226.0, 9787.0, 23731.0, 9802.0, 942.0, 945.0, 9796.0, 15366.0, 18415.0, 11638.0, 23224.0, 23927.0, 23930.0, 1041.0, 23729.0, 23797.0, 23728.0, 18416.0, 943.0, 937.0, 9794.0, 9376.0, 23933.0, 9780.0, 953.0, 23929.0, 18423.0, 940.0, 936.0, 11640.0, 11639.0, 941.0, 9795.0, 23609.0, 9778.0, 948.0, 3245.0, 9803.0, 15359.0, 9061.0, 3243.0, 23602.0, 11637.0, 1042.0, 934.0, 938.0, 9772.0, 23600.0, 23607.0, 23799.0, 15364.0, 23904.0, 954.0, 946.0, 23223.0, 23605.0, 23611.0, 18421.0, 23928.0, 9790.0, 955.0, 9786.0, 947.0]
            ### 0629 - blob_frame_counts = [12907.0, 27500.0, 19506.0, 27504.0, 27496.0, 4381.0, 27512.0, 3096.0, 19406.0, 27503.0, 19503.0, 27770.0, 23353.0, 3098.0, 12699.0, 6996.0, 20393.0, 9039.0, 12696.0, 27437.0, 19942.0, 27487.0, 23354.0, 12698.0, 27436.0, 8031.0, 27769.0, 12911.0, 19509.0, 12904.0, 17074.0, 4382.0, 3100.0, 27776.0, 27519.0, 12695.0, 9037.0, 27438.0, 4388.0, 12913.0, 20397.0, 27501.0, 23351.0, 3097.0, 20396.0, 20394.0, 15381.0, 27502.0, 23352.0, 9038.0, 27517.0, 9035.0, 27509.0, 19510.0, 27490.0, 4385.0, 19590.0, 19589.0, 23356.0, 23357.0, 19411.0, 12697.0, 27439.0, 27497.0, 8028.0, 19410.0, 27499.0, 6998.0, 19507.0, 27495.0, 27508.0, 19404.0, 27489.0, 12912.0, 14278.0, 4383.0, 15376.0, 15377.0, 8026.0, 12908.0, 27492.0, 27506.0, 19588.0, 15045.0, 17073.0, 27515.0, 15382.0, 9036.0, 8029.0, 15379.0, 20399.0, 9033.0, 27493.0, 23350.0, 12914.0, 27507.0, 27441.0, 4380.0, 27777.0, 9040.0, 23358.0, 20392.0, 19502.0, 12910.0, 27440.0, 27520.0, 19504.0, 27442.0, 23355.0, 27778.0, 15385.0, 27505.0, 27511.0, 20395.0, 19508.0, 6997.0, 20391.0, 19595.0, 27779.0, 19594.0, 19409.0, 27494.0, 4387.0, 27510.0, 4386.0, 8027.0, 27775.0, 15044.0, 4384.0, 12905.0, 27491.0, 27518.0, 8030.0, 19940.0, 27771.0, 27434.0, 27435.0, 19408.0, 19407.0, 27488.0, 27513.0, 6846.0, 27514.0, 27498.0, 27772.0, 15043.0, 19941.0, 15378.0, 19591.0, 27443.0, 19412.0, 15386.0, 9034.0, 15042.0, 15383.0, 12909.0, 19505.0, 27773.0, 19405.0, 19596.0, 3095.0, 15046.0, 27516.0, 15375.0, 15380.0, 19943.0, 19592.0, 12700.0, 17071.0, 3099.0, 27774.0, 6847.0, 19593.0, 12906.0]
            ## 0419 -blob_frame_counts = [15024.0, 2985.0, 3447.0, 15109.0, 14875.0, 142.0, 16345.0, 2995.0, 30740.0, 16342.0, 16334.0, 15105.0, 14870.0, 15124.0, 27284.0, 130.0, 15068.0, 16344.0, 15026.0, 2997.0, 146.0, 34610.0, 3454.0, 34613.0, 15106.0, 16333.0, 2992.0, 15119.0, 28461.0, 34611.0, 131.0, 30736.0, 16308.0, 28464.0, 15118.0, 34591.0, 34609.0, 147.0, 16348.0, 15115.0, 14884.0, 28468.0, 15107.0, 34590.0, 15111.0, 32239.0, 15117.0, 143.0, 14883.0, 30738.0, 15052.0, 16309.0, 28467.0, 140.0, 133.0, 14871.0, 28456.0, 144.0, 14877.0, 28469.0, 136.0, 15121.0, 15073.0, 129.0, 14873.0, 16337.0, 15112.0, 27291.0, 16340.0, 28470.0, 148.0, 34614.0, 2993.0, 24536.0, 28471.0, 15051.0, 139.0, 16350.0, 15123.0, 28465.0, 16341.0, 30737.0, 3000.0, 24537.0, 28463.0, 30735.0, 3450.0, 151.0, 3452.0, 16357.0, 14878.0, 127.0, 3453.0, 14881.0, 141.0, 124.0, 3449.0, 16335.0, 30739.0, 34606.0, 15116.0, 14874.0, 15067.0, 3451.0, 15072.0, 16343.0, 16349.0, 27285.0, 16358.0, 28466.0, 3448.0, 16310.0, 14880.0, 15023.0, 15113.0, 15120.0, 14882.0, 34592.0, 128.0, 14876.0, 2983.0, 15108.0, 2984.0, 34608.0, 145.0, 2994.0, 150.0, 14885.0, 14872.0, 149.0, 134.0, 28462.0, 16347.0, 3005.0, 34612.0, 28457.0, 34607.0, 14869.0, 2998.0, 16346.0, 2996.0, 2999.0, 125.0, 15071.0, 135.0, 132.0, 15070.0, 32240.0, 15110.0, 15122.0, 16336.0, 16338.0, 14879.0, 126.0, 15114.0]
            
            if not ret:
                break
            frame_count += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            ## Tarun : crop out the top portion where the time stamp is also crop the trees
            #frame = frame[180::,50::]
            frame = frame[180::,:]
            
            ## to wait for kellan to exit video
            #if frame_count < 3000:
            #    continue




            #if frame_count == 0 and self.param['output_video_name'] is not None:
                # vid = cv2.VideoWriter(
                #         self.param['output_video_name'],
                #         0x00000021,    # hack for cv.VideoWriter_fourcc(*'MP4V')
                #         self.param['output_video_fps'],
                #         (frame.shape[1], frame.shape[0]),
                #         )
                # #vid = cv2.VideoWriter(
                #        self.param['output_video_name'],
		        #        cv2.FOURCC('X','V','I','D'),
                #        self.param['output_video_fps'],
                #        (frame.shape[1], frame.shape[0]),
                #        )

            # Update background model
            bg_model.update(frame)
            if not bg_model.ready:
                continue

            #import ipdb;ipdb.set_trace()
            # Find blobs and add data to blob file
            blob_list, blob_image, circ_image = blob_finder.find(frame, bg_model.foreground_mask)

            #if vid is not None and len(blob_list)>0:
                #cv2.imwrite('/home/tarun/Downloads/upward_facing_camera_project/flytracker_demo-master/skytracker/output_frames/'+str(frame_count)+'.png', circ_image)
                #vid.write(circ_image)

            if blob_fid is not None and len(blob_list)>0:
                frame_data = {'frame': frame_count, 'blobs' : blob_list}
                frame_data_json = json.dumps(frame_data)
                blob_fid.write('{0}\n'.format(frame_data_json))
                
                orig_frame_width, orig_frame_height = frame.shape 

                frame = cv2.resize(frame, (500,500))
                circ_image = cv2.resize(circ_image, (500,500))
                if frame_count in blob_frame_counts:
                    annotations[self.input_video_name.split('/')[-3].split('_')[1] + '-' + self.input_video_name.split('/')[-1][:-4] +'-'+str(frame_count)+'.jpg'] = [] 
                    for b in blob_list:
                        ## make rectangle out of centroids of blobs
                        ## determine rectangle size based on blob coordinates
                        rect_size = int(round(max((b['max_x'] - b['min_x'])*500/orig_frame_height, (b['max_y'] - b['min_y'])*500/orig_frame_width))) + 2

                        annotations[self.input_video_name.split('/')[-3].split('_')[1] + '-' + self.input_video_name.split('/')[-1][:-4] +'-'+str(frame_count)+'.jpg'].append([(b['centroid_x']*500/orig_frame_height)-rect_size, (b['centroid_y']*500/orig_frame_width)-rect_size,(b['centroid_x']*500/orig_frame_height)+rect_size, (b['centroid_y']*500/orig_frame_width)+rect_size])
                    cv2.imwrite('/home/tarun/Downloads/upward_facing_camera_project/flytracker_demo-master/skytracker/output_frames_blob_locations/'+self.input_video_name.split('/')[-3].split('_')[1] + '-' + self.input_video_name.split('/')[-1][:-4] +'-'+str(frame_count)+'.jpg', circ_image)
                    cv2.imwrite('/home/tarun/Downloads/upward_facing_camera_project/flytracker_demo-master/skytracker/output_frames/'+self.input_video_name.split('/')[-3].split('_')[1] + '-' + self.input_video_name.split('/')[-1][:-4] +'-'+str(frame_count)+'.jpg', frame)


            # Display preview images
            if self.param['show_dev_images']:
                cv2.imshow('original',frame)
                cv2.imshow('background', bg_model.background)
                cv2.imshow('foreground mask', bg_model.foreground_mask)
                cv2.imshow('blob_image', blob_image)
                cv2.imshow('circ_image', circ_image)
            else:
                #cv2.imshow('circ_image', circ_image)
                pass

            wait_key_val = cv2.waitKey(1) & 0xFF
            if wait_key_val == ord('q'):
                break
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

        with open('/home/tarun/Downloads/upward_facing_camera_project/flytracker_demo-master/skytracker/'+self.input_video_name.split('/')[-3].split('_')[1]+'.json', 'w') as outfile:
          json.dump(annotations, outfile)

        if vid is not None:
            vid.release()

        if blob_fid is not None:
            blob_fid.close()

# ---------------------------------------------------------------------------------------

#if __name__ == '__main__':
#
#    input_video_name = sys.argv[1]
#
#    param = {
#            'bg_window_size': 11,
#            'fg_threshold': 10,
#            'datetime_mask': {'x': 430, 'y': 15, 'w': 500, 'h': 40},
#            'min_area': 1,
#            'max_area': 100000,
#            'open_kernel_size': (3,3),
#            }
#
#    tracker = SkyTracker(input_video_name=input_video_name, param=param)
#    tracker.run()
