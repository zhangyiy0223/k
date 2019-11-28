import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from dataset.TinyImageDataSet import TinyImageNetDataset
from models.resnet import ResNet, Bottleneck
from functions import train_model, val_model, test_model
from logger import logger

if __name__ == '__main__':
    # 加载训练模型
    net_name = 'resnet'
    net = ResNet(Bottleneck, [3, 4, 6, 3])
    net.cuda()
    # 设置超参数
    epoch = 1
    batch_size = 128
    lr = 0.01
    # 设置训练方法
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # 标志，0表示训练并验证，1表示测试
    root = './TinyImageNet/'
    transforms = transforms.Compose([
            transforms.ToTensor()
    ])
    flag = 1
    # 训练并验证模型
    if flag == 0:
        # 读入数据，设置dataset
        train_list = './TinyImageNet/train.txt'
        val_list = './TinyImageNet/val.txt'
        logger.info("Reading data...")
        logger.info("Loaded: %s", root + 'train')
        train_dataset = TinyImageNetDataset(root, train_list, transforms)
        logger.info("Loaded: %s", root + 'val')
        val_dataset = TinyImageNetDataset(root, val_list, transforms)
        for i in range(epoch):
            logger.info("-- EPOCH: %s", i)
            # 训练模型
            net.train(mode=True)
            train_model(net, train_dataset, batch_size, criterion, optimizer, i)
            # 验证模型
            net.eval()
            val_model(net, val_dataset, batch_size, criterion, i)
        # 保存模型
        logger.info('Finished Training')
        torch.save(net.state_dict(), "./model_para/"+net_name+'.pt')
    # 测试模型
    elif flag == 1:
        # 读取测试数据
        test_name = 'test'
        test_list = './TinyImageNet/test.txt'
        logger.info("Reading data...")
        logger.info("Loaded: %s", root + 'test')
        test_dataset = TinyImageNetDataset(root, test_list, transforms)
        # 加载测试模型
        net.load_state_dict(torch.load("./model_para/"+net_name+'.pt'))
        net.eval()
        test_model(net, test_dataset, test_name)
    # 观察验证集结果使用，方便调整
    else :
        # 读取测试数据
        test_name = 'val'
        test_list = './TinyImageNet/val.txt'
        logger.info("Reading data...")
        logger.info("Loaded: %s", root + 'val')
        test_dataset = TinyImageNetDataset(root, test_list, transforms)
        # 加载测试模型
        net.load_state_dict(torch.load("./model_para/"+net_name+'.pt'))
        net.eval()
        test_model(net, test_dataset, test_name)