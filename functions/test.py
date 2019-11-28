import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import torchvision.transforms as transforms

import numpy as np

from logger import logger

def test_model(model, test_dataset, test_name):
    # load test data set
    test_loader = data.DataLoader(test_dataset, batch_size=1)

    # log the result
    record_file = './result/result_'+test_name+'.csv'
    fd = open(record_file, 'a+')
    fd.write('Id,Category\n')
    fd.close()
    # test the model
    for i, test_data in enumerate(test_loader, 0):
        logger.info("-- NUM: %s", i)
        if len(test_data) == 1:
            inputs = test_data
        else:
            inputs, target = test_data
        inputs_var = Variable(inputs).cuda()
        # compute output
        output = model(inputs_var)
        prediction = torch.argmax(output, 1).cpu().numpy()
        # log the result
        fd = open(record_file, 'a+')
        fd.write(test_dataset.image_name[i].lstrip(test_name+'/')+','+str(prediction[0])+'\n')
        fd.close()