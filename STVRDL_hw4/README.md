# Image Super Resolution  
Training set: 291 high-resolution images  
Testing set: 14 low-resolution images  

# Getting Start
###  Generate Training Data  
change the save directory and image path in `SRFBN_CVPR19/scripts/Prepare_TrainData_HR_LR.py` at line 18 19  
`python3 Prepare_TrainData_HR_LR.py` and generate augmentation data  

###  Train  
1. Edit SRFBN_CVPR19/options/train/train_SRFBN_example.json for your needs according to SRFBN_CVPR19/options/train/README.md  
2. `python train.py -opt options/train/train_SRFBN_example.json`  

###  Test  
1. place your LR test images to SRFBN_CVPR19/results/LR/MyImage  
2. Edit SRFBN_CVPR19/options/test/test_SRFBN_example.json for your needs according to SRFBN_CVPR19/options/test/README.md  
3. `python test.py -opt options/test/test_SRFBN_example.json`  
4. You can find the reconstruction images in SRFBN_CVPR19/results  
