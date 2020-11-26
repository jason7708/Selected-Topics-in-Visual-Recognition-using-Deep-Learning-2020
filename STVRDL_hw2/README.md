# Digits detection  
Street View House Numbers dataset (train: 33402, test: 13068)  
# Getting start  
### Download data  
<https://drive.google.com/drive/u/0/folders/1Ob5oT9Lcmz7g5mVOcYH3QugA7tV3WsSl>  
put train/ in STVRDL_hw2/  
### change annotation file (.mat) to yolov3 form (txt file each row with[class x_center y_center width height](0~1 range))  
`python3 To_yolo_1.py`  (get the label data in .mat)  
`python3 To_yolo_2.py`  (transform to 0 ~ 1 range)  
then we can get all .txt training label file  
put them in STVRDL_hw2/PyTorch-YOLOv3/data/custom/labels
### decide train and valid image  
`python3 add_filename.py`  
get train.txt (the path of training images)  
    valid.txt (the path of valid images)  
put train.txt and valid.txt to STVRDL_hw2/PyTorch-YOLOv3/data/custom/  
put all training images to STVRDL_hw2/PyTorch-YOLOv3/data/custom/images/  
### custom model  
`cd STVRDL_hw2/PyTorch-YOLOv3/config`  
`bash create_custom_model.sh <num-classes>` (num-class = 10)  
### class  
add class names to data/custom/classes.names (one row per class name)  
### train  
`cd PyTorch-YOLOv3`  
`python3 train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --epochs 10`  
if you have pretrained weights add --pretrained_weights <your weights>  
### test  
put test data to STVRDL_hw2/PyTorch-YOLOv3/data/custom/images/test  
`python3 detect.py --image_folder data/custom/images/test --model_def config/yolov3-custom.cfg --weights_path <your weights> --class_path data/custom/classes.names`  
you can get the output image in output/ , output json in output/test/
### speed benchmark
see PyTorch-YOLOv3/hw2.ipynb
# Reference  
yolov3 code from (https://github.com/eriklindernoren/PyTorch-YOLOv3)
data preprocess (https://github.com/StephenEkaputra/SVHN-YOLOV3-CUSTOM)
