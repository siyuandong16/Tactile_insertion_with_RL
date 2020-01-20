import cv2
import numpy as np 


# folder = '/media/siyuan/data/Data_packing_RL/data_newsensor_3/'
folder = '/media/siyuan/data/Data_packing_RL/data_newsensor_3/'
folder = '/home/siyuan/Dropbox (MIT)/2020_RSS_RL_packing/data/data_newsensor_3/'
num = 5000
# num = 1

for i in range(num):
    try:
        r_matrxi = np.load(folder+str(i)+'/r_matrix.npy')
        label = np.load(folder+str(i)+'/label.npy')
        if label[0] > 0:
            label[0] = 0 
        if label[1] < 0:
            label[1] = 0 
        label_true = np.linalg.inv(r_matrxi).dot(label[:2]).tolist()
        label_true.append(label[2])
        label_true = np.array(label_true)
        np.save(folder+str(i)+'/label_true.npy', label_true)
    except:
        pass