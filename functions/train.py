# pytorch packege
import torch
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
# torchvision packege
import torchvision.transforms as transforms
# other package
import numpy as np
from logger import logger

def train_model(model, train_dataset, batch_size, criterion, optimizer, epoch):
    # log the result
    writer = SummaryWriter('./log/')

    # load train data set
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
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
        output = model(inputs_var)
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



