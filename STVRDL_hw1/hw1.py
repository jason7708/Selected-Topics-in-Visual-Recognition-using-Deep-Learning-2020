from torch.utils.data.dataset import Dataset
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter('./res_novalid_graph_data')

if __name__ == '__main__':

    #net = Net().to(device)
    net = models.resnet18(pretrained=True)
    
    #net = models.vgg16_bn(pretrained=True)
    
    # freeze the para
    '''
    # Freeze training for all "features" layers
    for param in net.features.parameters():
        param.requires_grad = False
    '''
    # change resnet last layer
    
    in_fea = net.fc.in_features
    net.fc = nn.Linear(in_fea, 196)
    
    # change vgg last layer
    #last_layer = nn.Linear(net.classifier[6].in_features, 196)
    
    # Load model
    #net.load_state_dict(torch.load('./test2.pth'))

    # if GPU is available, move the model to GPU
    net.to(device)

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.RandomCrop((224, 224)),
        torchvision.transforms.ColorJitter(brightness=0.5),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4559, 0.4425, 0.4395), (0.2623, 0.2608, 0.2648))
        ])
    valid_transform = torchvision.transforms.Compose([ 
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.RandomCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4559, 0.4425, 0.4395), (0.2623, 0.2608, 0.2648))
        ])

    #train_set = MyCustomDataset(img_path=img_path[:8948], label=label_dig[:8948], transform=train_transform)
    #valid_set = MyCustomDataset(img_path=img_path[8948:], label=label_dig[8948:],transform=valid_transform)
    train_set = torchvision.datasets.ImageFolder(root='Data/train_valid',transform=train_transform)
    #valid_set = torchvision.datasets.ImageFolder(root='Data/valid',transform=valid_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    #valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=32, shuffle=False, num_workers=2)
   
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0.01)

    # res
    optimizer = optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-1)#3e-4
    
    # vgg16_bn
    #optimizer = optim.AdamW(net.parameters(), lr=3e-4)
        
    for epoch in range(25):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        correct = 0
        if epoch % 7 == 6:
            for g in optimizer.param_groups:
                if g['lr'] > 0.00001:
                    g['lr'] = g['lr'] * 0.5
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            #train accu compute (correct num)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Train:')
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (i+1)))
        print('train accuracy: %.4f%%' % (100.*correct/len(train_set)))
        print('train error: %.4f%%' % (100.*(1-correct/len(train_set))))
        print('')

        # Tensor board
        writer.add_scalar('Train/Loss', running_loss / (i+1), epoch)
        writer.add_scalar('Train/Accuracy', 100.*correct/len(train_set), epoch)
        writer.add_scalar('Train/error' , 100.*(1-correct/len(train_set)), epoch)
        writer.flush()
        
        '''
        net.eval()
        correct = 0
        running_loss = 0.0
        l = 0
        with torch.no_grad():
            for data in valid_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                #top1
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                l = l + 1
        running_loss = running_loss / l
        print('Validation:')
        print('epoch: %d' % (epoch+1))
        print('Validation accuracy: %.4f%%, loss: %.3f' % (100.*correct/len(valid_set), running_loss))
        print('Validation error: %.4f%%' % (100.*(1-(correct/len(valid_set)))))
        print('--------------')
        # Tensor board
        writer.add_scalar('Validation/Loss', running_loss, epoch)
        writer.add_scalar('Validation/accuracy', 100.*correct/len(valid_set), epoch)
        writer.add_scalar('Validation/error', 100.*(1-(correct/len(valid_set))), epoch)
        writer.flush()
        '''

    PATH = './res_novalid.pth'
    torch.save(net.state_dict(), PATH)
