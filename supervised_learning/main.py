import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import sys

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms

from utils import CNN_Actor, data_selection


def run(init_lr=0.0001, max_epoch=64e3, batch_size=128*10, save_model=''):

    use_color = True 
    train_set, valid_set, train_data_size, valid_data_size = data_selection(use_color)
    print('training data size: ', train_data_size, 'validation data size: ', valid_data_size)
    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} 
    train_loader = torch.utils.data.DataLoader(train_set, **params)
    valid_loader = torch.utils.data.DataLoader(valid_set, **params)

    dataloaders = {'train': train_loader, 'val': valid_loader}
  
    model = CNN_Actor(num_inputs = 8*2*3)
    # print(model)
    model.cuda()
    model = nn.DataParallel(model)

    lr = init_lr
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])
    loss_function = nn.MSELoss() 

    num_steps_per_update = 1 # accum gradient
    epoch = 0
    gap = 1
    train_error_list = []
    valid_error_list = [1.]
    smallest_valid_error = 1.

    # train it
    while epoch < max_epoch:#for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, max_epoch))
        print('-' * 10)
        steps = 0
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()  # Set model to evaluate mode
                
            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            acc_old= 0.
            sample_number = 0
            optimizer.zero_grad()

            # Iterate over data.
            for (X, y) in dataloaders[phase]:
                num_iter += 1
                steps += 1 
                #print(X.size(),y.size())
                # get the inputs
                # inputs, labels = data
                # inputs_1_temp,labels = X,y
                # inputs_1 = torch.from_numpy(np.array(inputs_1)).type(torch.FloatTensor)
                # inputs_2 = torch.from_numpy(np.array(inputs_2)).type(torch.FloatTensor)
                # wrap them in Variable
                # print inputs.shape 
                inputs = Variable(X.cuda())
                labels = Variable(y.cuda())

                error_predicted = model(inputs)
                # print 'predict_error',per_frame_logits.squeeze().size()
                # print 'gt error',labels.size()
                cls_loss = F.smooth_l1_loss(error_predicted.squeeze(),labels)
                sample_number += y.size(0)


                # print cls_loss.size()
                tot_cls_loss += cls_loss.data

                loss = (cls_loss)/num_steps_per_update
                #loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update

                tot_loss += loss.data
                loss.backward()
             
                if num_iter == num_steps_per_update and phase == 'train':
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    # print steps%gap 
                    if steps % gap == 0:
                        print('{} {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, \
                            sample_number*1.0/train_data_size,tot_cls_loss/(gap*num_steps_per_update), tot_loss/gap))
                        # save model
                        train_error_list.append([epoch,tot_loss/gap])
                        np.save(save_model+'train_error_list.npy',train_error_list)
                        # torch.save(model.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                        tot_loss = tot_cls_loss = 0.
                        # label_true_list = []
                        # label_predict_list = []


            if phase == 'val':
                print('{} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_cls_loss/num_iter, \
                    (tot_loss*num_steps_per_update)/num_iter)) 
                if tot_cls_loss/num_iter < smallest_valid_error:
                    torch.save(model.state_dict(), save_model+'best_model_color_small_decrease_bnorm.pt')
                    smallest_valid_error = tot_cls_loss/num_iter
                valid_error_list.append(tot_cls_loss/num_iter)
                np.save(save_model+'valid_error_list.npy',valid_error_list)
            epoch += 1



if __name__ == '__main__':
    # need to add argparse
    run(save_model='/home/ubuntu/packing/weights/')
