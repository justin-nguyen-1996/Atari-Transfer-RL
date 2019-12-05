
#===================
# Imports
#===================

import numpy as np
import gym
import matplotlib
import torch
import warnings # For ignoring pytorch warnings

#=====================
# Network Declaration
#=====================

class DQN(nn.Module):
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

