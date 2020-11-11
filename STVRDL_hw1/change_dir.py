import pandas as pd
import os
import shutil


def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))
# Data
# | train
# |   | category1
# |   |     | xxx.jpg
# |   |     | ...
# |   | category2
# |   |     | xxx.jpg
# |   |     | ...
# |   | ...
# | valid
# |   | category1
# |   |     | xxx.jpg
# |   |     | ...
# |   | category2
# |   |     | xxx.jpg
# |   |     | ...
# |   | ...
# | train_valid
# |   | category1
# |   |     | xxx.jpg
# |   |     | ...
# |   | category2
# |   |     | xxx.jpg
# |   |     | ...
# |   | ...
# | test
# |   | unknown
# |   |     | xxx.jpg
# |   |     | ...
new_data_dir = './Data'

labels = pd.read_csv('./training_labels.csv')
id2label = {Id: label for Id, label in labels.values}

train_imgs = os.listdir('./training_data/training_data')

valid_size = len(train_imgs) * 0.2   # 11185 * 0.2
for i, file_name in enumerate(train_imgs):
    image_id = file_name.split('.')[0]
    image_lbl = id2label[int(image_id)]
    if i < valid_size:  # valid data
        mkdir_if_not_exist([new_data_dir, 'valid', image_lbl])
        shutil.copy(os.path.join('training_data/training_data', file_name),
                    os.path.join(new_data_dir, 'valid', image_lbl))
    else:   # train data
        mkdir_if_not_exist([new_data_dir, 'train', image_lbl])
        shutil.copy(os.path.join('training_data/training_data', file_name),
                    os.path.join(new_data_dir, 'train', image_lbl))
    # train+valid data
    mkdir_if_not_exist([new_data_dir, 'train_valid', image_lbl])
    shutil.copy(os.path.join('training_data/training_data', file_name),
                os.path.join(new_data_dir, 'train_valid', image_lbl))

# test data
mkdir_if_not_exist([new_data_dir, 'test', 'unknown'])
for test_imgs in os.listdir('./testing_data/testing_data'):
    shutil.copy(os.path.join('testing_data/testing_data', test_imgs),
                os.path.join(new_data_dir, 'test', 'unknown'))
