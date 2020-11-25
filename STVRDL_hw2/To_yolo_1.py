import h5py
import os

imgs = []
for img_file in os.listdir('train'):
    if img_file.endswith('.png'):
        imgs.append(img_file)

imgs = sorted(imgs, key=lambda x: int(os.path.splitext(x)[0]))

file = open("train.txt", "w")
for i in range(len(imgs)):
    file.write('train/'+str(imgs[i])+'\n')
file.close()

hdf5_data = h5py.File("train/digitStruct.mat", 'r')
attrs = {}

for index in range(len(imgs)):
    file = open("temp_train/"+str(index+1)+".txt", "w")
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [hdf5_data[attr.value[i].item()].value[0][0]
                  for i in range(len(attr))] if len(
                      attr) > 1 else [attr.value[0][0]]
        attrs[key] = values
    file.write(str(len(attrs['label'])))
    for j in range(len(attrs['label'])):
        file.write("\n" + str(int(attrs['left'][j])) +
                   " " + str(int(attrs['top'][j])) +
                   " " + str(int(attrs['width'][j])) +
                   " " + str(int(attrs['height'][j])) +
                   " " + str(int(attrs['label'][j])) + " ")

file.close()
