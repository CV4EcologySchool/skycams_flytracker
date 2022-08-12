python3 train.py --img 500 --batch 16 --epochs 5 --device cpu --data '/Users/fponce/Documents/cv4e/Skycam_annotations/skycams_flytracker_file_foryolo.yaml' --weights '/Users/fponce/Documents/cv4e/Skycam_annotations/yolov5l.pt'

python detect.py --source '/Users/fponce/Documents/cv4e/Skycam_annotations/training/images/0706-vi_0001_20190706_062254_trimmed-26172.jpg' --weights /Users/fponce/Documents/cv4e/yolov5/runs/train/exp9/weights/best.pt  --conf-thres 0.1 --iou-thres 0.1
