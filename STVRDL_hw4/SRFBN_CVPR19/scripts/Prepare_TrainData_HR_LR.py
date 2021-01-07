# -*- coding: utf-8 -*-
# @Time    : 2019-05-21 19:55
# @Author  : LeeHW
# @File    : Prepare_data.py
# @Software: PyCharm
from glob import glob
from flags import *
import os
from scipy import misc
import numpy as np
import datetime
import imageio
from multiprocessing.dummy import Pool as ThreadPool
import cv2

starttime = datetime.datetime.now()

save_dir = './data/291/'
train_HR_dir = '../training_hr_images/'
save_HR_path = os.path.join(save_dir, 'HR_x3')
save_LR_path = os.path.join(save_dir, 'LR_x3')
os.mkdir(save_HR_path)
os.mkdir(save_LR_path)
file_list = sorted(glob(os.path.join(train_HR_dir, '*.png')))
HR_size = [1, 0.8, 0.7, 0.6, 0.5]
#HR_size = [1]


def save_HR_LR(img, size, path, idx):
	
	rows, cols, channels = img.shape
	HR_img = cv2.resize(img, (int(size*rows), int(size*cols)), interpolation=cv2.INTER_CUBIC)
	HR_img = modcrop(HR_img, 3)
	
	rot180_img = cv2.rotate(HR_img, cv2.ROTATE_180)
	
	rows, cols, channels = HR_img.shape
	x3_img = cv2.resize(HR_img, (int(cols/3),int(rows/3)), interpolation=cv2.INTER_CUBIC)
	
	rows, cols, channels = rot180_img.shape
	x3_rot180_img = cv2.resize(rot180_img, (int(cols/3),int(rows/3)), interpolation=cv2.INTER_CUBIC)


	img_path = path.split('/')[-1].split('.')[0] + '_rot0_' + 'ds' + str(idx) + '.png'
	rot180img_path = path.split('/')[-1].split('.')[0] + '_rot180_' + 'ds' + str(idx) + '.png'
	x3_img_path = path.split('/')[-1].split('.')[0] + '_rot0_' + 'ds' + str(idx) + '.png'
	x3_rot180img_path = path.split('/')[-1].split('.')[0] + '_rot180_' + 'ds' + str(idx) + '.png'

	cv2.imwrite(save_HR_path + '/' + img_path, HR_img)
	cv2.imwrite(save_HR_path + '/' + rot180img_path, rot180_img)
	cv2.imwrite(save_LR_path + '/' + x3_img_path, x3_img)
	cv2.imwrite(save_LR_path + '/' + x3_rot180img_path, x3_rot180_img)


def modcrop(image, scale=3):
	if len(image.shape) == 3:
		h, w, _ = image.shape
		h = h - np.mod(h, scale)
		w = w - np.mod(w, scale)
		image = image[0:h, 0:w, :]
	else:
		h, w = image.shape
		h = h - np.mod(h, scale)
		w = w - np.mod(w, scale)
		image = image[0:h, 0:w]
	return image


def main(path):
	print('Processing-----{}/0800'.format(path.split('/')[-1].split('.')[0]))
	#img = imageio.imread(path)
	img = cv2.imread(path)
	idx = 0
	for size in HR_size:
		save_HR_LR(img, size, path, idx)
		idx += 1

items = file_list
pool = ThreadPool()
pool.map(main, items)
pool.close()
pool.join()
endtime = datetime.datetime.now()
print((endtime - starttime).seconds)
