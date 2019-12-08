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
        self.experience_replays = [collections.deque(maxlen=experience_replay_size) for env in student_envs]
        self.state_dims = student_envs[0].observation_space.shape
        self.num_actions = [env.action_space.n for env in student_envs]
        self.rewards = [] # TODO
        self.target_network_update_count = 0
        # Network and target network
        self.network = DQN_Distill_Network()
        self.target_network = DQN_Distill_Network()
        # Load teacher network weights
        self.teachers = []
        for i in range(len(teacher_envs)):
            env = teacher_envs[i]
            env_name = env_names[i]
            network = DQN_Network(env.observation_space.shape, env.action_space.n)
            weights = os.path.join('teacher_weights', env_name+'.pt')
            network.load_state_dict(torch.load(weights))
            network.to(device) # Move stuff to the correct device (cuda gpu or cpu)
            self.teachers.append(network)
        # Optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=alpha)
        # Move stuff to the correct device (cuda gpu or cpu)
        self.network = self.network.to(device)
        self.target_network = self.target_network.to(device)

