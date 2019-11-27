# pytorch packege
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
# torchvision packege
import torchvision.transforms as transforms
# other package
import math
import os
import numpy as np
from PIL import Image
from logger import logger

# Tiny ImangeNet Dataloader
def default_loader(path):
    return Image.open(path).convert('RGB')
class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_list, transform = None, loader = default_loader):
        # root: your_path/TinyImageNet/
        # data_list: your_path/TinyImageNet/train.txt etc.
        images = []
        image_name = []
        labels = open(data_list).readlines()
        for line in labels:
            items = line.strip('\n').split()
            img_name = items[0]

            # test list contains only image name
            test_flag = True if len(items) == 1 else False
            label = None if test_flag == True else np.array(int(items[1]))
            if test_flag == True:
                image_name.append(img_name)
            if os.path.isfile(os.path.join(root, img_name)):
                images.append((img_name, label))
            else:
                print(os.path.join(root, img_name) + 'Not Found.')

        self.root = root
        self.images = images
        self.transform = transform
        self.loader = loader
        self.image_name = image_name

    def __getitem__(self, index):
        img_name, label = self.images[index]
        img = self.loader(os.path.join(self.root, img_name))
        raw_img = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        return (img, label) if label is not None else img

    def __len__(self):
        return len(self.images)

# --- HELPERS ---

def conv3x3(in_planes, out_planes, stride=1):
    '''
        3x3 convolution with padding
    '''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# --- COMPONENTS ---

class BasicBlock(nn.Module):
    
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# --- ResNet-50 ---

class ResNet(nn.Module):
    
    def __init__(self, block, layers, num_classes=100):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,  bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# --- MAIN ---

if __name__ == "__main__":
    # construct a model
    net = ResNet(Bottleneck, [3, 4, 6, 3])
    net.cuda()

    # log the result
    writer = SummaryWriter('./log/')

    # loss function + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    # load train&val data set
    logger.info("Reading data...")
    root = './TinyImageNet/'
    train_list = './TinyImageNet/train.txt'
    val_list = './TinyImageNet/val.txt'
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = TinyImageNetDataset(root, train_list, data_transforms)
    train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    logger.info("Loaded: %s", root + 'train')
    val_dataset = TinyImageNetDataset(root, val_list, data_transforms)
    val_loader = data.DataLoader(val_dataset, batch_size=32, shuffle=True)
    logger.info("Loaded: %s", root + 'val')

    # train&val the model
    for epoch in range(10):
        logger.info("-- EPOCH: %s", epoch)

        # train the model
        running_loss = 0.0
        total_loss = 0.0
        correct = torch.zeros(1).cuda()
        total = torch.zeros(1).cuda()
        for i, train_data in enumerate(train_loader, 0):
            if i % 50 == 49: 
                logger.info("-- ITERATION: %s", i)
            inputs, target = train_data

            # wrap input + target into variables
            inputs_var = Variable(inputs).cuda()
            target_var = Variable(target).cuda()

            # compute output
            output = net(inputs_var)
            prediction = torch.argmax(output, 1)
            correct += (prediction == target_var).sum().float()
            total += len(target_var)

            #compute loss
            loss = criterion(output, target_var.long())

            # computer gradient + sgd step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print progress
            running_loss += loss.data
            total_loss += loss.data
            
            if i % 50 == 49:  # print every 50 mini-batches
                logger.info('-- TRAIN_RUNNING_ACC: %s', (correct/total).cpu().data.numpy()[0])
                logger.info("-- TRAIN_RUNNING_LOSS: %s", (running_loss / 50).cpu().numpy())
                writer.add_scalar('Trian/Acc/' + str(epoch), (correct/total).cpu().data.numpy()[0], i)
                writer.add_scalar('Train/Loss/' + str(epoch), (running_loss / 50).cpu().numpy(), i)
                writer.flush()
                running_loss = 0.0
        # log the result
        writer.add_scalar('Trian/Acc', (correct/total).cpu().data.numpy()[0], epoch)
        writer.add_scalar('Train/Loss', (total_loss / train_dataset.__len__()).cpu().numpy(), epoch)
        writer.flush()
        
        # validate the model
        total_loss = 0.0
        correct = torch.zeros(1).cuda()
        total = torch.zeros(1).cuda()
        for val_data in val_loader:
            inputs, target = val_data
            inputs_var = Variable(inputs).cuda()
            target_var = Variable(target).cuda()

            output = net(inputs_var)
            prediction = torch.argmax(output, 1)
            correct += (prediction == target_var).sum().float()
            total += len(target_var)

            loss = criterion(output, target_var.long())
            total_loss += loss.data
        # print the result
        logger.info('-- VAL_TOTAL_ACC: %s', (correct/total).cpu().data.numpy()[0])
        logger.info("-- VAL_AVERAGE_LOSS: %s", (total_loss / val_dataset.__len__()).cpu().numpy())
        # log the result
        writer.add_scalar('Val/Acc', (correct/total).cpu().data.numpy()[0], epoch)
        writer.add_scalar('Val/Loss', (total_loss / val_dataset.__len__()).cpu().numpy(), epoch)
        writer.flush()
    
    # save the model
    logger.info('Finished Training')
    torch.save(net.state_dict(), "./models/baseline-resnet50.pt")


