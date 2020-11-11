from torch.utils.data.dataset import Dataset
import torch
import torchvision
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = models.resnet18(pretrained=False)

in_fea = net.fc.in_features
net.fc = nn.Linear(in_fea, 196)

# net = models.vgg16_bn(pretrained=True)
# last_layer = nn.Linear(net.classifier[6].in_features, 196)
net.load_state_dict(torch.load('./res_novalid.pth'))
net.to(device)

train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.RandomCrop((224, 224)),
        torchvision.transforms.ColorJitter(brightness=0.5),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4559, 0.4425, 0.4395),
                                         (0.2623, 0.2608, 0.2648))
        ])
test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.RandomCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4559, 0.4425, 0.4395),
                                         (0.2623, 0.2608, 0.2648))
        ])
train_set = torchvision.datasets.ImageFolder(
        root='Data/train_valid', transform=train_transform)


test_set = torchvision.datasets.ImageFolder(
        root='Data/test', transform=test_transform)
test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=2)

# print(train_set.class_to_idx)
net.eval()
preds = []
for X, _ in test_loader:
    X = X.to(device)
    output = net(X)
    output = torch.softmax(output, dim=1)
    _, pred = output.max(1)
    preds += pred.tolist()
ids = sorted(os.listdir(os.path.join('./Data', 'test/unknown')))

with open('submission_res_f.csv', 'w') as f:
    f.write('id,label' + '\n')
    for i, output in zip(ids, preds):
        for cat, idx in train_set.class_to_idx.items():
            if idx == output:
                class_name = cat
        f.write(str(int(i.split('.')[0])) + ',' + class_name + '\n')
