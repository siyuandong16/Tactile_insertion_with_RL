# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt

def hole_error():
    error_px = [-5., -3., -1., 5., 7., 9.]
    error_py = [5., 3., 1., -5., -7., -9.]
    error_theta = [-10., -6., -2., 0., 2., 6., 10.]
    x_error_temp, y_error_temp, theta_error_temp = np.meshgrid(
        error_px, error_py, error_theta)
    x_error = x_error_temp.flatten()
    y_error = y_error_temp.flatten()
    theta_error = theta_error_temp.flatten()
    return x_error, y_error, theta_error

if __name__ == "__main__":

    ob = 'circle'
    x_error, y_error, theta_error = hole_error()
    data_folder = 'C:/Users/siyua/Documents/research/tactile_packing/packing/'
    object_folder = 'policy_test_' + ob + '/'
    reward = np.load(data_folder+object_folder+'reward_log.npy')
    object_id = np.load(data_folder+object_folder+'object_log.npy')
    trial_num = np.load(data_folder+object_folder+'trialnum_log.npy')
    success = np.load(data_folder+object_folder+'success_log.npy')
    x_error_log = np.load(data_folder+object_folder+'x_error_log.npy')
    y_error_log = np.load(data_folder+object_folder+'y_error_log.npy')
    theta_error_log = np.load(data_folder+object_folder+'theta_error_log.npy')
    len_error = len(x_error_log)


    if save:
        np.save('data/'+ob+'/x_error.npy', x_error_log)
        np.save('data/'+ob+'/y_error.npy', y_error_log)
        np.save('data/'+ob+'/theta_error.npy', theta_error_log)
        np.save('data/'+ob+'/trial_num.npy', trial_num[:len_error])
        np.save('data/'+ob+'/success.npy', success[:len_error])

    display = False 
    # print('x error len', len(x_error))
    # print('reward', len(object_id))
    # print('x_error', len(x_error_log))
    # print('success rate', sum(success)*1.0/len(success))

    if display:
        plt.figure(0)
        plt.plot(trial_num[:len_error])
        plt.figure(1)
        plt.plot(success[:len_error])
        plt.figure(2) 
        plt.subplot(1,3,1)
        plt.plot(x_error_log)
        plt.subplot(1,3,2)
        plt.plot(y_error_log)
        plt.subplot(1,3,3)
        plt.plot(theta_error_log)
        plt.show()

    # plt.figure(0)
    # plt.plot(x_error)
    # plt.hold = True 
    # plt.plot(x_error_log)
    # plt.show()  

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    