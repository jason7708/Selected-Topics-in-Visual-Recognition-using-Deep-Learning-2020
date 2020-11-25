from os import listdir

data_path = './train/'

files = listdir(data_path)

with open('./train.txt', 'w') as f:
    for i in files[:30062]:
        f.write('data/custom/images/' + str(i) + '\n')

with open('./valid.txt', 'w') as f:
    for i in files[30062:]:
        f.write('data/custom/images/' + str(i) + '\n')
