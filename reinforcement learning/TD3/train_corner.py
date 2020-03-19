#!/usr/bin/env python

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import numpy as np
from TD3_corner import TD3
from utils import ReplayBuffer
import os
from packing_enviroment_corner_v3 import Robot_motion, slip_detector, Packing_env
import scipy
import rospy
import cv2
import os.path
import random
from marker_detection import marker_detection
import time

data_folder = "/media/mcube/SERVER_HD/siyuan/policy_finetune/"
test_mode = True
test_object = 'vitamin'


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.33, 0.33, 0.34])


def preprocess(state_raw, rows, cols, num_frame, folder_num, state_full, mode,
               done, r_matrix, object_name, marker_maker):

    if test_mode:
        save_data = False
    else:
        save_data = True
    img2_seq = []
    use_color = True
    m, n = 320, 427
    pad_m = 145
    pad_n = 200

    for i in range(24):

        # if not use_color:
        #     imgwc_gray = rgb2gray(state_raw[i, :, :, :])
        # else:
        imgwc_gray = np.array(state_raw[i, :, :, :]).astype(np.uint8)
        # cv2.imshow('imgwc', imgwc.astype(np.uint8))
        # cv2.waitKey(0)
        # t0 = time.time()
        _, marker_image = marker_maker.marker_detection(imgwc_gray)
        # print('max value', np.max(marker_image), marker_image.shape)
        # print('time', time.time() - t0)
        if not done and save_data:
            save_folder = data_folder + object_name + '/' + str(folder_num)
            if not os.path.isdir(save_folder):
                os.mkdir(save_folder)
            cv2.imwrite(save_folder + '/' + str(i) + '.jpg',
                        state_raw[i, :, :, :])
            cv2.imwrite(save_folder + '/' + 'marker' + str(i) + '.jpg',
                        marker_image * 255)

        # img2_temp = imgwc_gray[int(m / 2) - pad_m:int(m / 2) + pad_m,
        #                        int(n / 2) - pad_n:int(n / 2) + pad_n, :]

        marker_image = marker_image[int(m / 2) - pad_m:int(m / 2) + pad_m,
                                    int(n / 2) - pad_n:int(n / 2) + pad_n]

        # img2_temp = cv2.resize(img2_temp, (200, 200)).astype(np.float32)
        img2_temp = cv2.resize(marker_image, (218, 300)).astype(np.float32)

        img2_temp = np.expand_dims(img2_temp, 2)
        # if i == 0 or i == 12:
        #     mean_2 = np.mean(img2_temp)
        #     std_2 = np.std(img2_temp)

        # img2_temp = (img2_temp - mean_2) / std_2
        if not use_color:
            img2_seq.append(img2_temp)
        else:
            # if i == 0:  #or i == 40:
            #     img2_seq = img2_temp.copy()
            #     img2_seq = img2_seq.transpose(2, 0, 1)
            # else:
            #     img2_seq = np.concatenate(
            #         (img2_seq, img2_temp.transpose(2, 0, 1)), axis=0)

            img2_seq.append(img2_temp.transpose(2, 0, 1))

    img2_temp = np.array(img2_seq)
    img2_temp1 = np.expand_dims(img2_temp, axis=0)
    # img = np.concatenate((img1,img2),axis = 0)
    if not done and save_data:
        np.save(save_folder + '/' + 'label.npy', state_full)
        np.save(save_folder + '/' + 'r_matrix.npy', r_matrix)

    # signal_quality = (np.mean(img2_temp[:12, :, :, :] * std_2 + mean_2) <
    #                   40) or (np.mean(img2_temp[12:, :, :, :] * std_2 + mean_2)
    #                           < 40)
    signal_quality = False

    X = torch.from_numpy(img2_temp1).type(torch.FloatTensor)
    return X, signal_quality


def save_buffer(data1, data2, data3, data4, data5, data6, data7, number):
    np.savez('/media/mcube/data/Data_packing_RL/buffer/'+str(number)+'.npz', state = data1, action = data2, \
        reward = data3, next_state = data4, done = data5, state_full = data6, next_state_full = data7)


def load_buffer(replay_buffer):
    folder = '/media/mcube/data/Data_packing_RL/buffer/'
    path, dirs, files = next(os.walk(folder))
    for i in range(min(len(files), 950)):
        data = np.load(path + files[i])
        replay_buffer.add((torch.from_numpy(data['state']), data['action'], data['reward'], \
            torch.from_numpy(data['next_state']), data['done'], data['state_full'], \
            data['next_state_full']))
    return replay_buffer


def check_regrasp(mode, state_full):
    if state_full[0] >= 0 and state_full[1] >= 0 and mode == 0:
        need_regrasp = True
    elif state_full[0] <= 0 and state_full[1] <= 0 and mode > 0:
        need_regrasp = True
    else:
        need_regrasp = False
    return need_regrasp


def hole_error():
    add_on_x = 2
    add_on_y = 2
    error_px = [-5., -3., -1., 5., 7., 9.]
    # error_px = [2., -3.]
    error_py = [5., 3., 1., -5., -7., -9.]
    # error_py = [-10., 3.]
    error_theta = [-10., -6., -2., 0., 2., 6., 10.]
    # error_theta = [-2., 0., 2.]
    x_error_temp, y_error_temp, theta_error_temp = np.meshgrid(
        error_px, error_py, error_theta)
    x_error = x_error_temp.flatten()
    y_error = y_error_temp.flatten()
    theta_error = theta_error_temp.flatten()
    return x_error, y_error, theta_error


def train():
    ######### Hyperparameters #########
    env_name = 'policy_test_vitamin_2.1'
    log_interval = 1  # print avg reward after interval
    random_seed = 0
    gamma = 0.99  # discount for future rewards
    batch_size = 30  # num of transitions sampled from replay buffer
    lr = 1e-5

    if not test_mode:
        exploration_noise = 0.1
        # exploration_noise = 1.
        exploration_noise_min = 0.1
    else:
        exploration_noise = 0.001
        # exploration_noise = 1.
        exploration_noise_min = 0.001
    polyak = 0.995  # target policy update parameter (1-tau)
    policy_noise = 0.2  # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 4  # delayed policy updates parameter
    max_episodes = 1500  # max num of episodes
    max_timesteps = 15  # max timesteps in one episode
    directory = "./preTrained/{}".format(env_name)  # save trained models
    if not os.path.isdir(directory):
        os.mkdir(directory)
    filename = "TD3_{}_{}".format(env_name, random_seed)
    graspForce = 10

    cols, rows = 320, 427

    if not test_mode:
        rgrasp_threshold = {
            'circle': 5,
            'ellipse': 5,
            'rectangle': 5,
            'hexagon': 5,
            'vitamin': 10
        }
    else:
        rgrasp_threshold = {
            'circle': 15,
            'ellipse': 15,
            'rectangle': 15,
            'hexagon': 15,
            'vitamin': 15
        }
    num_frame = 8

    ###################################

    env = Packing_env(num_frame)
    robot = Robot_motion()

    state_dim = 8 * 3 * 2
    action_dim = 3
    max_action = 5.0

    if not (os.path.isfile(directory + '/actor_loss.npy')
            and os.path.isfile(directory + '/critic_loss.npy')):
        actor_loss_list = []
        critic_loss_list = []
        np.save(directory + '/actor_loss.npy', actor_loss_list)
        np.save(directory + '/critic_loss.npy', critic_loss_list)

    policy = TD3(lr, state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(max_size=800)
    marker_maker = marker_detection()
    # replay_buffer = load_buffer(replay_buffer)
    print('buffer size', replay_buffer.size)
    # policy.load(directory, filename)
    policy.freeze_cnnlayer()
    # policy.print_param()

    # if random_seed:
    #     print("Random Seed: {}".format(random_seed))
    #     env.seed(random_seed)
    #     torch.manual_seed(random_seed)
    #     np.random.seed(random_seed)

    # logging variables:
    avg_reward = 0
    ep_reward = 0
    ep_reward_list = []
    ep_object_list = []
    ep_success_list = []
    ep_trialnum_list = []
    ep_model_list = []
    # ep_reward_list = np.load('reward_log.npy').tolist()
    inposition = False
    success_sign = False
    object_name = ''
    bad_data = False
    # log_f = open("log.txt", "w+")

    x_error_list, y_error_list, theta_error_list = hole_error()

    if test_mode:
        max_episodes = len(x_error_list)
        x_error_list_ = []
        y_error_list_ = []
        theta_error_list_ = []
    # ep_reward_list = np.load('reward_log.npy').tolist()
    # ep_object_list = np.load('object_log.npy').tolist()
    # ep_success_list = np.load('success_log.npy').tolist()
    # ep_trialnum_list = np.load('trialnum_log.npy').tolist()
    # ep_model_list = np.load('mode_log.npy').tolist()

    robot.setSpeed(600, 200)
    robot.robot_reset()
    robot.open_gripper()

    # training procedure:
    sample_counter = 0

    for episode in range(249, max_episodes):
        print(
            '###########################################################################'
        )
        num_trial = 0
        if object_name != env.target_object:
            inposition = False
        if test_mode:
            env.target_object = test_object
        object_name = env.target_object

        if not test_mode:
            rand_pose = np.array([0., 0., random.random() * 15. - 10.])
            graspForce = random.random() * 20 + 10.
        else:
            rand_pose = np.array([0., 0., 15.])
            graspForce = 10.
        robot.pick_up_object(env.target_object, graspForce, inposition,
                             env.mode, rand_pose)
        inposition = True
        if test_mode:
            env.x_error, env.y_error, env.theta_error = x_error_list[
                episode], y_error_list[episode], theta_error_list[episode]
        # env.x_error, env.y_error, env.theta_error = 9., -9., 10.
        env_x_error, env_y_error = robot.error_converter(
            env.x_error, env.y_error)
        # print('converted error', env_x_error, env_y_error)

        state_full = np.array([env.x_error, env.y_error, env.theta_error])
        state, _, done, _, _, _, r_matrix_next = env.step(
            [env_x_error, env_y_error, env.theta_error], False)
        # raw_input("Press Enter to continue...")
        need_regrasp = check_regrasp(robot.mode, state_full) and (not done)
        if done:
            env.reset(rand_pose)
        else:
            state, signal_quality = preprocess(state, cols, rows, num_frame,
                                               sample_counter, state_full,
                                               robot.mode, done, r_matrix_next,
                                               object_name, marker_maker)

            trial_number_rgrasp = 0
            if sample_counter > 300:
                batch_size = 30
            # if sample_counter >= 500 and sample_counter < 1000:
            #     batch_size = 200
            # elif sample_counter >= 1000:
            #     batch_size = 500

            # exploration_noise += -0.01
            for t in range(max_timesteps):
                # select action and add exploration noise:

                trial_number_rgrasp += 1
                num_trial += 1
                action = policy.select_action(state)
                # action[2] = 0
                original_action = action

                action = action + np.random.normal(
                    0,
                    max(exploration_noise, exploration_noise_min),
                    size=action_dim) * max_action
                action = action.clip(-max_action,
                                     max_action)  # in gripper frame
                # action[2] = 0
                # print action.shape

                # take action in env:
                # if need_regrasp:
                #     trial_number_rgrasp += 2

                if trial_number_rgrasp > rgrasp_threshold[env.target_object]:
                    env.regrasp(graspForce, rand_pose)
                    # print('regrasp++++++++++++++++++++++++++++++')
                    trial_number_rgrasp = 0
                    env.rgrasp_counter = 0

                next_state, reward, done, next_state_full, action_world, r_matrix, r_matrix_next = env.step(
                    action, True)

                sample_counter += 1
                need_regrasp = check_regrasp(robot.mode,
                                             next_state_full) and (not done)

                print('Ep', episode, 'Num', sample_counter,
                      'Name', object_name, 'mode', robot.mode, 'A.',
                      list(action), 'N. A.', list(action_world), 'R.', reward)
                print('state_full', state_full)
                # raw_input("Press Enter to continue...")

                next_state, signal_quality = preprocess(
                    next_state, cols, rows, num_frame, sample_counter,
                    next_state_full, robot.mode, done, r_matrix_next,
                    object_name, marker_maker)
                replay_buffer.add(
                    (torch.squeeze(state,
                                   0), np.array(action) / max_action, reward,
                     torch.squeeze(next_state, 0), float(done),
                     np.linalg.inv(r_matrix).dot(state_full / max_action),
                     np.linalg.inv(r_matrix_next).dot(next_state_full /
                                                      max_action)))

                # save_buffer(torch.squeeze(state, 0).numpy(), np.array(action) / max_action, reward, torch.squeeze(next_state, 0).numpy(),\
                #     float(done), np.linalg.inv(r_matrix).dot(state_full / max_action), np.linalg.inv(r_matrix_next).dot(next_state_full / max_action), sample_counter-1)

                state = next_state
                state_full = next_state_full

                # print('reward', reward)
                avg_reward += reward
                ep_reward += reward

                # if episode is done then update policy:
                if done or t == (max_timesteps - 1):
                    env.reset(rand_pose)
                    if sample_counter > 100 and not test_mode:
                        policy.update(replay_buffer, t + 1, batch_size, gamma,
                                      polyak, policy_noise, noise_clip,
                                      policy_delay, directory)
                        print('NN updated')
                    trial_number_rgrasp = 0
                    ep_reward_list.append(500)
                    ep_object_list.append('gap')
                    ep_success_list.append(True)
                    ep_trialnum_list.append(0)
                    ep_model_list.append(100)
                    break

                if sample_counter > 100:
                    policy.unfreeze_cnnlayer()

                if signal_quality:
                    break

            # logging updates:
            # log_f.write('{},{},{},{}\n'.format(episode, ep_reward, t,
            #                                    object_name))
            # log_f.flush()

            if reward > 0:
                success_sign = True
            else:
                success_sign = False

            if test_mode:
                x_error_list_.append(x_error_list[episode])
                y_error_list_.append(y_error_list[episode])
                theta_error_list_.append(theta_error_list[episode])
            ep_reward_list.append(ep_reward)
            ep_object_list.append(object_name)
            ep_success_list.append(success_sign)
            ep_trialnum_list.append(num_trial)
            ep_model_list.append(robot.mode)
            np.save(directory + '/reward_log.npy', ep_reward_list)
            np.save(directory + '/object_log.npy', ep_object_list)
            np.save(directory + '/success_log.npy', ep_success_list)
            np.save(directory + '/trialnum_log.npy', ep_trialnum_list)
            np.save(directory + '/mode_log.npy', ep_model_list)
            if test_mode:
                np.save(directory + '/x_error_log.npy', x_error_list_)
                np.save(directory + '/y_error_log.npy', y_error_list_)
                np.save(directory + '/theta_error_log.npy', theta_error_list_)

            ep_reward = 0
            # if avg reward > 300 then save and stop traning:
            if (avg_reward / log_interval) >= 800:
                print("########## Solved! ###########")
                name = filename + '_solved'
                policy.save(directory, name)
                # log_f.close()
                break

            if episode % 10 == 0 and not test_mode:
                policy.save(directory, filename)

            # print avg reward every log interval:
            if episode % log_interval == 0:
                avg_reward = int(avg_reward / log_interval)
                print("Episode: {}\tAverage Reward: {}".format(
                    episode, avg_reward))
                avg_reward = 0

            if signal_quality:
                break


if __name__ == '__main__':
    rospy.init_node('Dense_packing_with_RL', anonymous=True)
    train()