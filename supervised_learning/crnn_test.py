import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='10,11'
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
import cv2
from utils_siyuan import data_selection
from networkmodels import DecoderRNN, EncoderCNN
import matplotlib
gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
for gui in gui_env:
    try:
        print("testing", gui)
        matplotlib.use(gui,warn=False, force=True)
        from matplotlib import pyplot as plt
        break
    except:
        continue
print("Using:",matplotlib.get_backend())
import matplotlib.pyplot as plt




def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.33, 0.33, 0.34])

def read_img(data_path):
    img2_seq = []
    use_color = True 
        # range_list = list(range(9,24,2)) #+ list(range(33,48,2))
    range_list = list(range(0, 45, 6)) + list(range(45, 90, 6))
    for i in range_list:
        img = cv2.imread(data_path+str(i)+'.jpg')
        img = img[30:-30, 30:-30, :]
        if not use_color:
            imgwc_gray = rgb2gray(img)
        else:
            imgwc_gray = np.array(img).astype(np.float32)
        # img2_temp = scipy.misc.imresize(imgwc_gray,(84,84))
        img2_temp = cv2.resize(imgwc_gray, (84, 84)) 
        # print(img2_temp.shape)
        # cv2.imwrite('image_test.jpg', (img2_temp.astype(np.uint8)))
        if i == 0 or i == 45:
            mean_2 = np.mean(img2_temp)
            std_2 = np.std(img2_temp)

        img2_temp = (img2_temp-mean_2)/std_2
        if not use_color:
            img2_seq.append(img2_temp)   
        else:
            img2_seq.append(img2_temp.transpose(2,0,1))
    # print(img2_seq.shape)
    img2_temp = np.array(img2_seq)
    img2_temp = np.expand_dims(img2_temp, axis = 0)
    # print("shape", img2_temp.shape)
    # cv2.imwrite('image_test.jpg', ((img2_temp[:3,:,:].transpose(1,2,0)+1)*255/2.).astype(np.uint8))
    #~ img2_temp1 = np.expand_dims(img2_temp, axis=0)
    X = torch.from_numpy(img2_temp).type(torch.FloatTensor) 
    return X


def run():
    CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
    CNN_embed_dim = 512      # latent dim extracted by 2D CNN
    img_x, img_y = 84,84  # 
    dropout_p = 0.3          # dropout probability

    # DecoderRNN architecture
    RNN_hidden_layers = 3
    RNN_hidden_nodes = 512
    RNN_FC_dim = 256
    k = 3           

    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda:0" if use_cuda else "cpu")   # use CPU or GPU

    cnn_encoder = EncoderCNN(img_x=img_x, img_y=img_y, fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2,
                        drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)

    rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                            h_FC_dim=RNN_FC_dim, drop_p=dropout_p, output_dim=k).to(device)
    cnn_encoder = nn.DataParallel(cnn_encoder)
    rnn_decoder = nn.DataParallel(rnn_decoder)

    # model = torch.load('/home/ubuntu/packing/weights/best_model_color_small.pt')
    cnn_encoder.load_state_dict(torch.load('weights/cnn_encoder_epoch20.pth')) #
    rnn_decoder.load_state_dict(torch.load('weights/rnn_decoder_epoch20.pth')) #
    cnn_encoder.eval()
    rnn_decoder.eval()
    loss_function = nn.MSELoss() 

    # model = nn.DataParallel(model)
    loss_list = []
    label_list = []
    prediction_list = []
    for i in range(1300,1400):
        try: 
            root = "/home/ubuntu/packing/data/data_newsensor_3/"
            index = i
            label_true = np.load(root + str(index) + '/label_true.npy')
            # r_matrix = np.load(root + str(index) + '/r_matrix.npy')

            # if label_true[0] > 0:
            #     label_true[0] = 0
            # if label_true[1] < 0:
            #     label_true[1] = 0

            # label_correct = r_matrix.dot(np.array([-label[0], -label[1]]))
            # label_true = np.array([label_correct[0], label_correct[1], -label[2]])
            label_true[:2] /= 10.
            label_true[2] /= 25.
            labels = Variable(torch.from_numpy(label_true).type(torch.FloatTensor).cuda())
            # print("corrected label", label_true)


            X = read_img(root + str(index) + '/')
            inputs = Variable(X.cuda())
            error_predicted = rnn_decoder(cnn_encoder(inputs))
            loss = loss_function(error_predicted.squeeze(),labels)
            label_list.append(np.array(label_true)*10)
            prediction_list.append(error_predicted.cpu().data[0].numpy()*10)
            print(error_predicted.cpu().data[0].numpy()*10)
            print(np.array(label_true)*10)
            loss_list.append(loss)
        except:
            pass
        # print(loss)
    print(sum(loss_list)/len(loss_list))
    label_list = np.array(label_list)
    prediction_list = np.array(prediction_list)
    plt.figure()
    plt.plot(label_list[:,1],'ro')
    plt.hold = True
    plt.plot(prediction_list[:,1],'go')
    plt.show()
             


if __name__ == '__main__':
    # need to add argparse
    run()
