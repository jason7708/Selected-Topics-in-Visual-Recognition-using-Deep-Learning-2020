# car brand classification  
16,185 car images belonging to 196 classes (train: 11,185, test:5000)  

# Getting start  
## Download data from Kaggle  
<https://www.kaggle.com/c/cs-t0828-2020-hw1/data>  
put them and code in the same directory  
## Change directory  
run change_dir.py to change the directory of data for the directory read by torchvision imagefolder
`python3 change_dir.py`  
``` Data  
 | train  
 |   | category1  
 |   |     | xxx.jpg  
 |   |     | ...  
 |   | category2  
 |   |     | xxx.jpg  
 |   |     | ...  
 |   | ...  
 | valid  
 |   | category1  
 |   |     | xxx.jpg  
 |   |     | ...  
 |   | category2  
 |   |     | xxx.jpg  
 |   |     | ...  
 |   | ...  
 | train_valid  
 |   | category1  
 |   |     | xxx.jpg  
 |   |     | ...  
 |   | category2  
 |   |     | xxx.jpg  
 |   |     | ...  
 |   | ...  
 | test  
 |   | unknown  
 |   |     | xxx.jpg  
 |   |     | ...  
 ```
 # Train the model  
 run hw1.py  
 `python3 hw1.py`
 
 # Test the model  
 run test.py  
 `python3 test.py`
