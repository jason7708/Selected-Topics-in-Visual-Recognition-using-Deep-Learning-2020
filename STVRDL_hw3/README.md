# Instance segmentation  
Tiny PASCAL VOC dataset (1,349 training images, 100 test images with 20 common object classes)  

# Getting start  
###  Data
put your data in `AdelaiDet/datasets/`  
```
datasets/  
  img_data/  
    annotations/  
       instance_train.json  
       test.json  
    train/  
       xxx.jpg  
    test_images/  
       xxx.jpg  
```  
In `AdelaiDet/datasets/`  
`python3 prepare_thing_sem_from_instance.py --dataset-name img_data`  

###  Train  
In `AdelaiDet/datasets/`  
```  
OMP_NUM_THREAD=1 python3 tools/train_net.py \  
--config-file configs/BlendMask/R_50_1x_my.yaml \  
--num-gpus 1 \  
OUTPUT_DIR training_dir/blendmask_R_50_1x  
```  
OUTPUT_DIR: path to save the result  
###  Test  
In `AdelaiDet/datasets/`  
```  
OMP_NUM_THREAD=1 python3 tools/train_net.py \  
--config-file configs/BlendMask/R_50_1x_my.yaml \  
--num-gpus 1 \  
--eval-only \  
OUTPUT_DIR training_dir/blendmask_R_50_1x_test \  
MODEL.WEIGHTS training_dir/blendmask_R_50_1x/model_final.pth  
```  
OUTPUT_DIR: path to save the result  
MODEL.WEIGHTS: weights
