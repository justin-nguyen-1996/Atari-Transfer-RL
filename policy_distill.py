# policy_distill.py

#===================
# Imports
#===================

import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import torch.nn as nn
import warnings # For ignoring pytorch warnings
import time
import datetime
import collections
import random
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

from dqn import DQN_Network

#===================
# Config Stuff
#===================

log_dir = "logs_distill/"

# Reference - https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
env_names = ["AssaultNoFrameskip-v4", "DemonAttackNoFrameskip-v4"]
# Teacher environments
teacher_envs = []
for env_name in env_names:
    env = make_atari(env_name)
    env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=True)
    env.observation_space = gym.spaces.box.Box(
        env.observation_space.low[0, 0, 0],
        env.observation_space.high[0, 0, 0],
        [env.observation_space.shape[2], env.observation_space.shape[1], env.observation_space.shape[0]],
        dtype=env.observation_space.dtype)
    teacher_envs.append(env)
# Student environments
student_envs = []
for env_name in env_names:
    env = make_atari(env_name)
    env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=True)
    env.observation_space = gym.spaces.box.Box(
        env.observation_space.low[0, 0, 0],
        env.observation_space.high[0, 0, 0],
        [env.observation_space.shape[2], env.observation_space.shape[1], env.observation_space.shape[0]],
        dtype=env.observation_space.dtype)
    student_envs.append(env)

#epsilon = 0.1
epsilon = 0.02
#total_steps = 500000
total_steps = 100000
num_game_eps = 100
state = env.reset()
experience_replay_size = 10000
alpha = 3e-4
#gamma = 0.99
gamma = 0.9
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_network_update_freq = 500

#=====================
# Network Declaration
#=====================

class DQN_Distill_Network(nn.Module):
    def __init__(self):
        # Initialize parent's constructor
        super(DQN_Distill_Network, self).__init__()
        # Initialize object variables
        self.state_dims = student_envs[0].observation_space.shape
        self.num_actions = [env.num_actions for env in student_envs]
        # Convolution layers
        self.conv1 = nn.Conv2d(self.state_dims[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size(), 512)
        # Different fully connected layer for each game
        self.fc2s = [nn.Linear(512, num_actions) for num_actions in self.num_actions]
        self.fc2s = torch.nn.ModuleList(self.fc2s)

    def forward(self, x, env_index):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # Reshape to 1D vector for fully connected layers
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2s[env_index](x)
        return x

    def feature_size(self):
        state_dim_zeros = torch.zeros(1, *self.state_dims)
        conv_output = self.conv3(self.conv2(self.conv1(state_dim_zeros)))
        feature_size = conv_output.view(1, -1).size(1)
        return feature_size

#===============================
# Policy Distillation Algorithm
#===============================

class Policy_Distill(nn.Module):
    def __init__(self):
        # Initialize parent's constructor
        super(PolicyDistill, self).__init__()
        self.student_experience_replays = [collections.deque(maxlen=experience_replay_size) for env in student_envs]
        self.teacher_experience_replays = [collections.deque(maxlen=experience_replay_size) for env in teacher_envs]
        self.state_dims = student_envs[0].observation_space.shape
        self.num_actions = [env.action_space.n for env in student_envs]
        self.rewards = [[] for env in student_envs] # List of rewards for each game
        self.target_network_update_count = 0
        # Network and target network
        self.network = DQN_Distill_Network()
        self.target_network = DQN_Distill_Network()
        # Load teacher network weights
        self.teacher_networks = []
        for i in range(len(teacher_envs)):
            env = teacher_envs[i]
            env_name = env_names[i]
            network = DQN_Network(env.observation_space.shape, env.action_space.n)
            weights = os.path.join('teacher_weights', env_name+'.pt')
            network.load_state_dict(torch.load(weights))
            network.to(device) # Move stuff to the correct device (cuda gpu or cpu)
            self.teacher_networks.append(network)
        # Optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=alpha)
        # Move stuff to the correct device (cuda gpu or cpu)
        self.network = self.network.to(device)
        self.target_network = self.target_network.to(device)

    # Return Greedy Action
    def get_epsilon_greedy_action(self, s, epsilon, env):
        if (np.random.uniform(0, 1) < epsilon):
            return np.random.randint(0, env.action_space.n)
        else:
            with torch.no_grad():
                s = torch.tensor([s], device=device, dtype=torch.float)
                q_values = self.network(s)
#                print(f'q_values: {q_values}')
                a = q_values.argmax()
                return a

    # Keep Track of Rewards
    def save_reward(self, r):
        self.rewards.append(r)

    # Save Network
    def save_network(self, env_name):
        torch.save(self.network.state_dict(), 'logs_distill/{}_saved_network.pt'.format(env_name))

    def get_loss_dqn(self, experience_replay):
        # Grab random trajectories from experience replay
        trajectories = random.sample(experience_replay, batch_size)
        # Unpack trajectories
        states, actions, rewards, next_states = zip(*trajectories)
        # Reshape and create tensors
        states = torch.tensor(states, device=device, dtype=torch.float).view((batch_size,)+self.state_dims)
        actions = torch.tensor(actions, device=device, dtype=torch.long).squeeze().view(-1,1)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float).squeeze().view(-1,1)
        next_states = torch.tensor(next_states, device=device, dtype=torch.float).view((batch_size,)+self.state_dims)
        # Compute the TD error
        q_values = self.network(states).gather(1, actions)
        with torch.no_grad():
            target_action = self.target_network(next_states).argmax(dim=1).view(-1,1)
            target_q_values = self.target_network(next_states).gather(1, target_action)
        td_error = rewards + gamma*target_q_values - q_values
        loss = (0.5 * td_error**2).mean()
        return loss

    def get_loss_kl_divergence(self):
        pass

    def get_loss(self):
        return kl_divergence_ratio*get_loss_kl_divergence() + get_loss_dqn()

    # Update Model Based on state
    # TODO: fix this
    def update(self, s, a, r, sp):
        # Get loss
        loss = self.get_loss()
#        print(f'loss: {loss}')
        # Backprop and update the weights
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # Update the target network every so often
        self.target_network_update_count += 1
        if self.target_network_update_count % target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

#==================
# Training Loop
#==================

if __name__ == '__main__':
    start = time.time()
    model = Policy_Distill()
    total_reward = 0
    cur_steps = 0
    while cur_steps < total_steps:
        # Go through each game and grab the corresponding student and teacher environment
        # Switch to training on the other game
        for env_i in student_envs:
            student_env = student_envs[env_i]
            teacher_env = teacher_envs[env_i]
            state = env.reset()
            state = state.transpose(2, 0, 1)
            episode_counter = 0
            # Train on each game, one at a time, for num_game_eps episodes
            while episode_counter < num_game_eps:
                # Take a step for the student and save experience
                student_action = model.get_epsilon_greedy_action(student_state, epsilon)
                student_prev_state = student_state
                student_state, student_reward, student_done, _ = env.step(student_action)
                student_state = student_state.transpose(2, 0, 1)
                total_reward += student_reward
                cur_steps += 1
                model.student_experience_replays[env_i].append(
                    (student_prev_state, student_action, student_reward, student_state))
                # Take a step for the teacher and save experience
                teacher_action = model.get_epsilon_greedy_action(teacher_state, epsilon)
                teacher_prev_state = teacher_state
                teacher_state, teacher_reward, teacher_done, _ = env.step(teacher_action)
                teacher_state = teacher_state.transpose(2, 0, 1)
                model.teacher_experience_replays[env_i].append(
                    (teacher_prev_state, teacher_action, teacher_reward, teacher_state))
                # Skip some frames to get some experiences in replay buffer
                if cur_steps >= 100:
                    model.update(prev_state, action, reward, state) # TODO: fix this
                # Debug print
                if cur_steps % 100 == 0:
                    time_delta = int(time.time() - start)
                    time_delta = datetime.timedelta(seconds=time_delta)
                    print(f'cur_steps: {cur_steps}, time: {time_delta}')
                # Reset the environment for the next episode if we're done with the current episode
                if done:
                    episode_counter += 1
                    state = env.reset()
                    state = state.transpose(2, 0, 1)
                    model.rewards[env_i].append(total_reward)
                    print(f'total_reward: {total_reward}')
                    total_reward = 0
            # Every time we're done training on one game (i.e. before switch to other game) save plots and network weights
            model.save_network(env_names[env_i])
            ax = plt.subplot(111)
            ax.plot(range(len(model.rewards[env_i])), model.rewards[env_i], label='y = Reward')
            plt.title('Total Reward per Episode')
            ax.legend()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.savefig('{}.png'.format(env_names[env_i]))
            plt.clf()
            plt.close()
