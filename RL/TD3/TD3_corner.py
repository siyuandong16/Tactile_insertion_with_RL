import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# from i3d.pytorch_i3d import InceptionI3d
# from CNN3D.cnn3d import CNN3D_actor, CNN3D_critic
from CNN3D.cnn import CNN_Actor_new, CNN_Critic
from CNN3D.nn import MLP_Critic_3
from CNN3D.networkmodels import DecoderRNN, EncoderCNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        # self.actor = InceptionI3d(400, in_channels=3)
        # self.actor = CNN3D_actor()
        self.actor_cnn = EncoderCNN(img_x=218,
                                    img_y=300,
                                    fc_hidden1=1024,
                                    fc_hidden2=768,
                                    drop_p=0,
                                    CNN_embed_dim=512)

        self.actor_rnn = DecoderRNN(CNN_embed_dim=512,
                                    h_RNN_layers=1,
                                    h_RNN=512,
                                    h_FC_dim=256,
                                    drop_p=0,
                                    output_dim=action_dim)

        # self.actor_cnn = nn.DataParallel(self.actor_cnn)
        # self.actor_rnn = nn.DataParallel(self.actor_rnn)
        # self.actor_cnn.load_state_dict(
        #     torch.load(
        #         'preTrained/supervised_learned_policy/cnn_encoder_epoch21.pth')
        # )
        # strict=False)  #
        # self.actor_rnn.load_state_dict(
        #     torch.load(
        #         'preTrained/supervised_learned_policy/rnn_decoder_epoch21.pth')
        # )
        # strict=False)  #
        # self.actor_cnn.eval()
        # self.actor_rnn.eval()
        # self.actor = CNN_Actor_new(num_inputs=state_dim,
        #                            num_classes=action_dim)
        # self.actor_cnn = nn.DataParallel(self.actor)
        # self.actor.load_state_dict(
        #     torch.load(
        #         'preTrained/Tactile_packing_corner_4_directions_new_gelsight/best_model_color_small_decrease.pt'
        #     ))
        # self.actor.freeze_cnnlayer()
        # self.actor.load_state_dict(torch.load('i3d/models/rgb_imagenet.pt'))
        # self.actor.replace_logits(action_dim, nn.Tanh())
        # self.actor.load_state_dict(torch.load('i3d/weights/error_000065.pt'))
        self.max_action = max_action

    def forward(self, state):
        # action = self.actor(state) * self.max_action
        action = self.actor_rnn(self.actor_cnn(state[:, :12, :, :, :]),
                                self.actor_cnn(
                                    state[:, 12:, :, :, :])) * self.max_action
        # print("action",action.size())
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # self.critic = InceptionI3d(400, in_channels=3)
        # self.critic = CNN3D_critic()
        # self.critic = CNN_Critic()
        self.critic = MLP_Critic_3()
        # self.critic.load_state_dict(torch.load('i3d/models/rgb_imagenet.pt'))
        # self.critic.replace_logits(action_dim, None)
        # self.critic.load_state_dict(torch.load('i3d/weights/error_000065.pt'))
        # self.critic.replace_logits(1)
        # self.last_layer = nn.Linear(402, 1)

    def forward(self, state, action):
        # feature = self.critic(state)
        value = self.critic(state, torch.squeeze(action))
        # print 'feature and action size', feature.size(), action.size()
        # state_action = torch.cat([feature, action.unsqueeze(2)], 1).view(-1,402)
        # print 'state action', state_action.size()

        # value = self.last_layer(state_action)
        # q = F.relu(self.l1(state_action))
        # q = F.relu(self.l2(q))
        # q = self.l3(q)
        return value


class TD3:
    def __init__(self, lr, state_dim, action_dim, max_action):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor.load_state_dict(
            torch.load(
                'preTrained/policy_finetune_6/TD3_policy_finetune_6_0_actor.pth'
                # 'preTrained/tactile_packing_corner_4object_new/TD3_tactile_packing_corner_4object_new_0_actor.pth'
            ))
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1.load_state_dict(
            torch.load(
                'preTrained/policy_finetune_6/TD3_policy_finetune_6_0_crtic_1.pth'
            ))
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(),
                                             lr=0.1)

        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2.load_state_dict(
            torch.load(
                'preTrained/policy_finetune_6/TD3_policy_finetune_6_0_crtic_1.pth'
            ))
        self.critic_2_target = Critic(state_dim, action_dim).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(),
                                             lr=0.1)

        self.max_action = max_action

    def select_action(self, state):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # self.actor.eval()
        state = state.to(device)
        # self.actor.eval()
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, replay_buffer, n_iter, batch_size, gamma, polyak,
               policy_noise, noise_clip, policy_delay, directory):

        if replay_buffer.size > 200:
            n_iter = (n_iter + 1)
        use_full_state = True
        actor_loss_list = np.load(directory + '/actor_loss.npy').tolist()
        critic_loss_list = np.load(directory + '/critic_loss.npy').tolist()
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state, action_, reward, next_state, done, state_full, next_state_full = replay_buffer.sample(
                batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action_).to(device)
            reward = torch.FloatTensor(reward).reshape(
                (batch_size, 1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).reshape((batch_size, 1)).to(device)
            state_full = torch.FloatTensor(state_full).to(device)
            next_state_full = torch.FloatTensor(next_state_full).to(device)

            noise = torch.FloatTensor(action_).data.normal_(
                0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) / self.max_action)
            # print(r_matrix_next.size(), next_action.size())
            next_action = next_action + noise
            next_action = next_action.clamp(-1, 1)
            # next_action[2] = 0.
            # print 'action', action.size(), 'next action', next_action.size()
            # Compute target Q-value:
            if not use_full_state:
                target_Q1 = self.critic_1_target(next_state, next_action)
                target_Q2 = self.critic_2_target(next_state, next_action)
            else:
                target_Q1 = self.critic_1_target(next_state_full, next_action)
                target_Q2 = self.critic_2_target(next_state_full, next_action)

            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * gamma * target_Q).detach()

            # Optimize Critic 1:
            if not use_full_state:
                current_Q1 = self.critic_1(state, action)
            else:
                current_Q1 = self.critic_1(state_full, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            print('critic loss 1', loss_Q1)
            critic_loss_list.append(loss_Q1.cpu().detach().numpy())
            # Optimize Critic 2:
            if not use_full_state:
                current_Q2 = self.critic_2(state, action)
            else:
                current_Q2 = self.critic_2(state_full, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            print('critic loss 2', loss_Q2)

            # Delayed policy updates:
            if i % policy_delay == 0:
                self.actor.train()
                # Compute actor loss:
                if not use_full_state:
                    actor_loss = -self.critic_1(
                        state,
                        self.actor(state) / self.max_action).mean()
                else:
                    action_predict = self.actor(state) / self.max_action
                    torch.clamp(action_predict, min=-1.0, max=1.0)
                    actor_loss = -self.critic_1(state_full,
                                                action_predict).mean()
                # print('actor loss', actor_loss, 'action', self.actor(state)/self.max_action, 'reward', reward, 'target_Q', target_Q)
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                print('actor loss', actor_loss)
                actor_loss_list.append(actor_loss.cpu().detach().numpy())
                self.actor_optimizer.step()

                # Polyak averaging update:
                for param, target_param in zip(self.actor.parameters(),
                                               self.actor_target.parameters()):
                    target_param.data.copy_((polyak * target_param.data) +
                                            ((1 - polyak) * param.data))

                for param, target_param in zip(
                        self.critic_1.parameters(),
                        self.critic_1_target.parameters()):
                    target_param.data.copy_((polyak * target_param.data) +
                                            ((1 - polyak) * param.data))

                for param, target_param in zip(
                        self.critic_2.parameters(),
                        self.critic_2_target.parameters()):
                    target_param.data.copy_((polyak * target_param.data) +
                                            ((1 - polyak) * param.data))
            np.save(directory + '/actor_loss.npy', actor_loss_list)
            np.save(directory + '/critic_loss.npy', critic_loss_list)

    def save(self, directory, name):
        torch.save(self.actor.state_dict(),
                   '%s/%s_actor.pth' % (directory, name))
        torch.save(self.actor_target.state_dict(),
                   '%s/%s_actor_target.pth' % (directory, name))

        torch.save(self.critic_1.state_dict(),
                   '%s/%s_crtic_1.pth' % (directory, name))
        torch.save(self.critic_1_target.state_dict(),
                   '%s/%s_critic_1_target.pth' % (directory, name))

        torch.save(self.critic_2.state_dict(),
                   '%s/%s_crtic_2.pth' % (directory, name))
        torch.save(self.critic_2_target.state_dict(),
                   '%s/%s_critic_2_target.pth' % (directory, name))

    def load(self, directory, name):
        self.actor.load_state_dict(
            torch.load('%s/%s_actor.pth' % (directory, name),
                       map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(
            torch.load('%s/%s_actor_target.pth' % (directory, name),
                       map_location=lambda storage, loc: storage))

        self.critic_1.load_state_dict(
            torch.load('%s/%s_crtic_1.pth' % (directory, name),
                       map_location=lambda storage, loc: storage))
        self.critic_1_target.load_state_dict(
            torch.load('%s/%s_critic_1_target.pth' % (directory, name),
                       map_location=lambda storage, loc: storage))

        self.critic_2.load_state_dict(
            torch.load('%s/%s_crtic_2.pth' % (directory, name),
                       map_location=lambda storage, loc: storage))
        self.critic_2_target.load_state_dict(
            torch.load('%s/%s_critic_2_target.pth' % (directory, name),
                       map_location=lambda storage, loc: storage))

    def load_actor(self, directory, name):
        self.actor.load_state_dict(
            torch.load('%s/%s_actor.pth' % (directory, name),
                       map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(
            torch.load('%s/%s_actor_target.pth' % (directory, name),
                       map_location=lambda storage, loc: storage))

    def freeze_cnnlayer(self):
        for i, param in enumerate(self.actor.parameters()):
            # if i < 6:
            param.requires_grad = False

        for i, param in enumerate(self.actor_target.parameters()):
            # if i < 6:
            param.requires_grad = False

    def print_param(self):
        for i, param in enumerate(self.actor.parameters()):
            print param.requires_grad

        for i, param in enumerate(self.actor_target.parameters()):
            print param.requires_grad

    def unfreeze_cnnlayer(self):
        for i, param in enumerate(self.actor.parameters()):
            # if i < 6:
            param.requires_grad = True

        for i, param in enumerate(self.actor_target.parameters()):
            # if i < 6:
            param.requires_grad = True

    def unfreeze_rnnlayer(self):
        for i, param in enumerate(self.actor.actor_rnn.parameters()):
            # if i < 6:
            param.requires_grad = True

        for i, param in enumerate(self.actor_target.actor_rnn.parameters()):
            # if i < 6:
            param.requires_grad = True