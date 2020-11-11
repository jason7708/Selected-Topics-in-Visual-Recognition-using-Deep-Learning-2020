# car brand classification  
16,185 car images belonging to 196 classes (train: 11,185, test:5000)  

# Getting start  
### Download data from Kaggle  
<https://www.kaggle.com/c/cs-t0828-2020-hw1/data>  
put them and code in the same directory  
### Change directory  
run change_dir.py to change the directory of data for the directory read by torchvision imagefolder  
`python3 change_dir.py`  
``` Data  
 | train(80% training data)  
 |   | category1  
 |   |     | xxx.jpg  
 |   |     | ...  
 |   | category2  
 |   |     | xxx.jpg  
 |   |     | ...  
 |   | ...  
 | valid(20% training data)  
 |   | category1  
 |   |     | xxx.jpg  
 |   |     | ...  
 |   | category2  
 |   |     | xxx.jpg  
 |   |     | ...  
 |   | ...  
 | train_valid(100% training data)  
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
 ### Train the model  
 run hw1.py  
 `python3 hw1.py`  
 if you want to close vaildation set and train on 100% training data  
 delete the code about validation set (line 46 47 50 51 96~126)  
 change line 44 45 to `train_set = torchvision.datasets.ImageFolder(root='Data/train_valid', transform=train_transform)`  
 ### Test the model  
 run test.py  
 `python3 test.py`  
 make sure the file name to load your model at line19  
 `net.load_state_dict(torch.load('./my_model.pth'))`  
 
