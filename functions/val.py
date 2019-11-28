import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
# torchvision package
import torchvision.transforms as transforms
# other package
import numpy as np
from logger import logger

def val_model(model, val_dataset, batch_size, criterion, epoch):
    # log the result
    writer = SummaryWriter('./log/')

    # load val data set
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size)

    # validate the model
    total_loss = 0.0
    correct = torch.zeros(1).cuda()
    total = torch.zeros(1).cuda()
    for i, val_data in enumerate(val_loader, 0):
        inputs, target = val_data
        inputs_var = Variable(inputs).cuda()
        target_var = Variable(target).cuda()
        output = model(inputs_var)
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