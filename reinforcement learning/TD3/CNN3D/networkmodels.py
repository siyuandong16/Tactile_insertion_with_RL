##pytorch cnn+ lstm

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from utils_network import *

########################## Model for CNN LSTM ##########################################################


# 2D CNN encoder train from scratch (no transfer learning)
class EncoderCNN(nn.Module):
    def __init__(self,
                 img_x=84,
                 img_y=84,
                 fc_hidden1=512,
                 fc_hidden2=512,
                 drop_p=0.3,
                 CNN_embed_dim=300):
        super(EncoderCNN, self).__init__()

        self.img_x = img_x
        self.img_y = img_y
        self.CNN_embed_dim = CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 32, 64, 128, 256
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (
            3, 3)  # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (
            2, 2)  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (
            0, 0)  # 2d padding

        # conv2D output shapes
        self.conv1_outshape = conv2D_output_size((self.img_x, self.img_y),
                                                 self.pd1, self.k1,
                                                 self.s1)  # Conv1 output shape
        self.conv2_outshape = conv2D_output_size(self.conv1_outshape, self.pd2,
                                                 self.k2, self.s2)
        self.conv3_outshape = conv2D_output_size(self.conv2_outshape, self.pd3,
                                                 self.k3, self.s3)
        self.conv4_outshape = conv2D_output_size(self.conv3_outshape, self.pd4,
                                                 self.k4, self.s4)

        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=self.ch1,
                      kernel_size=self.k1,
                      stride=self.s1,
                      padding=self.pd1),
            # nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1,
                      out_channels=self.ch2,
                      kernel_size=self.k2,
                      stride=self.s2,
                      padding=self.pd2),
            # nn.BatchNorm2d(self.ch2, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2,
                      out_channels=self.ch3,
                      kernel_size=self.k3,
                      stride=self.s3,
                      padding=self.pd3),
            # nn.BatchNorm2d(self.ch3, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch3,
                      out_channels=self.ch4,
                      kernel_size=self.k4,
                      stride=self.s4,
                      padding=self.pd4),
            # nn.BatchNorm2d(self.ch4, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.drop = nn.Dropout2d(self.drop_p)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(
            self.ch4 * self.conv4_outshape[0] * self.conv4_outshape[1],
            self.fc_hidden1)  # fully connected layer, output k classes
        # self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc2 = nn.Linear(
            self.fc_hidden1,
            self.CNN_embed_dim)  # output = CNN embedding latent variables

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # CNNs
            x = self.conv1(x_3d[:, t, :, :, :])
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.view(x.size(0), -1)  # flatten the output of conv

            # FC layers
            x = F.relu(self.fc1(x))
            # x = F.dropout(x, p=self.drop_p, training=self.training)
            # x = F.relu(self.fc2(x))
            # x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc2(x)
            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self,
                 CNN_embed_dim=300,
                 h_RNN_layers=2,
                 h_RNN=256,
                 h_FC_dim=128,
                 drop_p=0.3,
                 output_dim=3):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.output_dim = output_dim

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=
            True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN * 2, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.output_dim)

    def forward(self, x_RNN_left, x_RNN_right):

        self.LSTM.flatten_parameters()
        # RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        RNN_out_left, (h_n, h_c) = self.LSTM(x_RNN_left, None)
        RNN_out_right, (h_n, h_c) = self.LSTM(x_RNN_right, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """
        RNN_out = torch.cat((RNN_out_left[:, -1, :], RNN_out_right[:, -1, :]),
                            1)
        # print(RNN_out_left[:, -1, :].size())
        # FC layers
        # x = self.fc1(RNN_out[:, -1, :])  # choose RNN_out at the last time step
        x = self.fc1(RNN_out)
        x = F.relu(x)
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        x = torch.tanh(self.fc2(x))

        return x


###### End of CNNLSTM Network #####

######################## ConvolutionLSTM Network Model ########################################


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels,
                             self.hidden_channels,
                             self.kernel_size,
                             1,
                             self.padding,
                             bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels,
                             self.hidden_channels,
                             self.kernel_size,
                             1,
                             self.padding,
                             bias=False)
        self.Wxf = nn.Conv2d(self.input_channels,
                             self.hidden_channels,
                             self.kernel_size,
                             1,
                             self.padding,
                             bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels,
                             self.hidden_channels,
                             self.kernel_size,
                             1,
                             self.padding,
                             bias=False)
        self.Wxc = nn.Conv2d(self.input_channels,
                             self.hidden_channels,
                             self.kernel_size,
                             1,
                             self.padding,
                             bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels,
                             self.hidden_channels,
                             self.kernel_size,
                             1,
                             self.padding,
                             bias=False)
        self.Wxo = nn.Conv2d(self.input_channels,
                             self.hidden_channels,
                             self.kernel_size,
                             1,
                             self.padding,
                             bias=True)
        self.Who = nn.Conv2d(self.hidden_channels,
                             self.hidden_channels,
                             self.kernel_size,
                             1,
                             self.padding,
                             bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0],
                                            shape[1])).cuda()
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0],
                                            shape[1])).cuda()
            self.Wco = Variable(torch.zeros(1, hidden, shape[0],
                                            shape[1])).cuda()

        else:

            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'

        return (Variable(torch.zeros(batch_size, hidden, shape[0],
                                     shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0],
                                     shape[1])).cuda())


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size,
                 step=1,
                 effective_step=[1]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i],
                                self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(
                        batch_size=bsize,
                        hidden=self.hidden_channels[i],
                        shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)


######################## End of ConvolutionLSTM Network Model ########################################

######################## CNN3D Model ################################################################


class CNN3D(nn.Module):
    def __init__(self,
                 t_dim=45,
                 img_x=128,
                 img_y=128,
                 drop_p=0.2,
                 fc_hidden1=256,
                 fc_hidden2=128,
                 output_dim=3):
        super(CNN3D, self).__init__()

        # set video dimension
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.output_dim = output_dim
        self.ch1, self.ch2 = 32, 48
        self.k1, self.k2 = (5, 5, 5), (3, 3, 3)  # 3d kernel size
        self.s1, self.s2 = (2, 2, 2), (2, 2, 2)  # 3d strides
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)  # 3d padding

        # compute conv1 & conv2 output shape
        self.conv1_outshape = conv3D_output_size(
            (self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        self.conv2_outshape = conv3D_output_size(self.conv1_outshape, self.pd2,
                                                 self.k2, self.s2)

        self.conv1 = nn.Conv3d(in_channels=1,
                               out_channels=self.ch1,
                               kernel_size=self.k1,
                               stride=self.s1,
                               padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1,
                               out_channels=self.ch2,
                               kernel_size=self.k2,
                               stride=self.s2,
                               padding=self.pd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(self.ch2 * self.conv2_outshape[0] *
                             self.conv2_outshape[1] * self.conv2_outshape[2],
                             self.fc_hidden1)  # fully connected hidden layer
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(
            self.fc_hidden2,
            self.output_dim)  # fully connected layer, output = multi-classes

    def forward(self, x_3d):
        # Conv 1
        x = self.conv1(x_3d)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        # FC 1 and 2
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)

        return x


## --------------------- end of 3D CNN module ---------------- ##
