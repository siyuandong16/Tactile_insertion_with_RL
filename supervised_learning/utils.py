import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import shuffle 
import random 
from torch.utils import data
import cv2
import scipy.misc 
import os 

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.33, 0.33, 0.34])

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CNN_Actor(nn.Module):
    def __init__(self, num_inputs = 8, hidden_size=256, num_classes = 3):
        super(CNN_Actor, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 64, 7, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 5, stride=2)), nn.ReLU(), 
            init_(nn.Conv2d(32, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            # init_(nn.Linear(32 * 10 * 10, hidden_size)), nn.ReLU()
            init_(nn.Linear(32 * 6 * 6, hidden_size)), nn.ReLU()
            )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))#, nn.init.calculate_gain('tanh'))

        self.critic_linear = init_(nn.Linear(hidden_size, num_classes))


    def forward(self, inputs):
        x = self.main(inputs)
        x = self.critic_linear(x)
        x = torch.tanh(x)
        return x



class Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, root, file_folder, use_color):
        "Initialization"
        self.root = root 
        self.file_folder = file_folder
        self.use_color = use_color
        
    def __len__(self):
        "Denotes the total number of samples"
        return len(self.file_folder)

    def label_correction(self, label):
        
        label[:2] /= 10.
        label[2] /= 15.

        return label 

    def load_data(self, data_path):
        img2_seq = []
        range_list = list(range(0, 45, 6)) + list(range(45, 90, 6))
        for i in range_list:
            img = cv2.imread(data_path+str(i)+'.jpg')
            img = img[30:-30, 30:-30, :]
            if not self.use_color:
                imgwc_gray = rgb2gray(img)
            else:
                imgwc_gray = np.array(img).astype(np.float32)
            img2_temp = scipy.misc.imresize(imgwc_gray,(84,84))
            if i == 0 or i == 45:  
                mean_2 = np.mean(img2_temp)
                std_2 = np.std(img2_temp)

            img2_temp = (img2_temp-mean_2)/std_2
            if not self.use_color:
                img2_seq.append(img2_temp)   
            else:
                if i == 0: 
                    img2_seq = img2_temp.copy()
                    img2_seq = img2_seq.transpose(2,0,1)
                else: 
                    img2_seq = np.concatenate((img2_seq, img2_temp.transpose(2,0,1)), axis=0)
        img2_temp = np.array(img2_seq)
        X = torch.from_numpy(img2_temp).type(torch.FloatTensor)  

        label = np.array(np.load(data_path + 'label_true.npy'))  
        label = self.label_correction(label)
        Y = torch.from_numpy(label).type(torch.FloatTensor)
        return X, Y 

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        filename = self.file_folder[index]
        # Load data
        X, Y = self.load_data(filename)                 
        return X, Y



def data_selection(use_color):
    root = []
    root.append("/media/siyuan/data/Data_packing_RL/data_newsensor_3/")

    num_data = 5000

    file_folder = []  

    range_list = list(range(0, 45, 6)) + list(range(45, 90, 6))
    for k in range(len(root)):
        root_folder = root[k]
        for i in range(num_data):
            path = root_folder+str(i)+'/'
            break_sign = False
            if os.path.isdir(path):
                label = np.array(np.load(path + 'label_true.npy'))  
                if (label[0] < 0 and label[1] < 1.0) or (label[0] > -1 and label[1] > 0):
                    file_folder.append(path)

    num_of_data = len(file_folder)
    num_of_train = int(num_of_data*0.8) 
    num_of_valid = num_of_data - num_of_train 

    random.seed(40)
    shuffle(file_folder)

    train_folder = file_folder[:num_of_train]
    valid_folder = file_folder[num_of_train:]

    train_set, valid_set = Dataset_CRNN(root, train_folder, use_color), Dataset_CRNN(root, valid_folder, use_color)

    return train_set, valid_set, num_of_train, num_of_valid

