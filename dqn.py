
#===================
# Imports
#===================
import matplotlib
matplotlib.use('Agg')

import numpy as np
import gym
import matplotlib
import torch
import warnings # For ignoring pytorch warnings
import time
import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocatorimport collections


#=====================
# Network Declaration
#=====================

class DQN_Network(nn.Module):
    # TODO: change state_dims or pass in as correct size after state aggregation
    def __init__(self, state_dims, num_actions):
        # Initialize parent's constructor
        super(DQN, self).__init__()
        # Initialize object variables
        self.state_dims = state_dims
        self.num_actions = num_actions
        # Convolution layers
        self.conv1 = nn.Conv2d(self.state_dims, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size(), 512)
        self.fc2 = nn.Linear(512, self.num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # Reshape to 1D vector for fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def feature_size(self):
        state_dim_zeros = torch.zeros(1, self.state_dims)
        conv_output = self.conv3(self.conv2(self.conv1(state_dim_zeros)))
        feature_size = conv_output.view(1, -1).size(1)
        return feature_size

#===================
# DQN Algorithm
#===================

class DQN():
    def __init__(self):
        self.experience_replay = # TODO
        self.network = DQN_Network()
        self.target_network = DQN_Network()
        self.rewards = []
        pass

    # Return the Q-value of (s,a)
    def __call__(self, s, a):
        pass
    
    # Return Greedy Action
    def get_action(self, s, epsilon):
        pass
    
    # Keep Track of Rewards
    def save_reward(self, r):
        self.rewards.append(r)

    # Save Network
    def save_network(self):
        pass

    # Update Model Based on state
    def update(self, state)
    # TODO: change reward function
    # TODO: use environment wrappers to clip rewards, skip frames, stack frames
    #       - https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
 
#==================
# Training Loop
#==================

#%% md

 ## Training Loop

#%%

# Run Time
start=timer()

log_dir = "log/"

env_id = "PongNoFrameskip-v4"
env    = make_atari(env_id)
env    = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=True)

# TODO DQN
model = DQN()

total_reward = 0
epsilon = 0.1
total_steps = 500000
state = env.reset()
for i_steps in range(1, total_steps):
    action = model.get_action(observation, epsilon)
    
    prev_state = state
    state, reward, done, _ = env.step(action)
    state = None if done else observation

    model.update(state)
    
    #Reward for Surviving (Therefore just 1 per time step)
    total_reward += 1

    if done:
        state = env.reset()
        model.save_reward(total_reward)
        total_reward = 0

    if i_steps % 10000 == 0:
        model.save_network()
        ax = plt.subplot(111)
        ax.plot(range(len(model.rewards)), model.rewards, label='y = Reward')
        plt.title('Total Reward per Episode')
        ax.legend()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.savefig('plot.png')

