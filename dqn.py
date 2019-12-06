
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

#===================
# Config Stuff
#===================

log_dir = "logs/"

# Reference - https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
env_name = "PongNoFrameskip-v4"
env = make_atari(env_name)
env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=True)
env.observation_space = gym.spaces.box.Box(
    env.observation_space.low[0, 0, 0],
    env.observation_space.high[0, 0, 0],
    [env.observation_space.shape[2], env.observation_space.shape[1], env.observation_space.shape[0]],
    dtype=env.observation_space.dtype)

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

class DQN_Network(nn.Module):
    def __init__(self, state_dims, num_actions):
        # Initialize parent's constructor
        super(DQN_Network, self).__init__()
        # Initialize object variables
        self.state_dims = state_dims
        self.num_actions = num_actions
        # Convolution layers
        self.conv1 = nn.Conv2d(self.state_dims[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size(), 512)
        self.fc2 = nn.Linear(512, self.num_actions)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # Reshape to 1D vector for fully connected layers
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def feature_size(self):
        state_dim_zeros = torch.zeros(1, *self.state_dims)
        conv_output = self.conv3(self.conv2(self.conv1(state_dim_zeros)))
        feature_size = conv_output.view(1, -1).size(1)
        return feature_size

#===================
# DQN Algorithm
#===================

class DQN():
    def __init__(self):
        self.experience_replay = collections.deque(maxlen=experience_replay_size)
        self.state_dims = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.network = DQN_Network(self.state_dims, self.num_actions)
        self.target_network = DQN_Network(self.state_dims, self.num_actions)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=alpha)
        self.rewards = []
        self.target_network_update_count = 0
        # Move stuff to the correct device (cuda gpu or cpu)
        self.network = self.network.to(device)
        self.target_network = self.target_network.to(device)

    # Return the Q-value of (s,a)
    def __call__(self, s, a):
        with torch.no_grad():
            return self.network(s)[a]

    # Return Greedy Action
    def get_epsilon_greedy_action(self, s, epsilon):
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
    def save_network(self):
        torch.save(self.network.state_dict(), 'logs/{}_saved_network.pt'.format(env_name))
        torch.save(self.optimizer.state_dict(), 'logs/{}_saved_network_optimizer.pt'.format(env_name))

    def get_loss(self):
        # Grab random trajectories from experience replay
        trajectories = random.sample(self.experience_replay, batch_size)
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

    # Update Model Based on state
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
    model = DQN()
    state = env.reset()
    state = state.transpose(2, 0, 1)
    total_reward = 0
    for i_steps in range(1, total_steps):
        action = model.get_epsilon_greedy_action(state, epsilon)
        prev_state = state
        state, reward, done, _ = env.step(action)
        state = state.transpose(2, 0, 1)
        print(f'action: {action}')

        # TODO: change state_dims or pass in as correct size after state aggregation

#        if done:
#            state = None

        # Save experience
        model.experience_replay.append((prev_state, action, reward, state))

        # Skip some frames to get some experiences in replay buffer
        if i_steps >= 100:
            model.update(prev_state, action, reward, state)

        # Reward for Surviving (Therefore just 1 per time step)
#        total_reward += 1 # TODO: change reward function
        total_reward += reward

        if done:
            state = env.reset()
            state = state.transpose(2, 0, 1)
            model.save_reward(total_reward)
            print(f'total_reward: {total_reward}')
            total_reward = 0
            model.save_network()
            ax = plt.subplot(111)
            ax.plot(range(len(model.rewards)), model.rewards, label='y = Reward')
            plt.title('Total Reward per Episode')
            ax.legend()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.savefig('plot.png')
            plt.clf()
            plt.close()

        if i_steps % 100 == 0:
            time_delta = int(time.time() - start)
            time_delta = datetime.timedelta(seconds=time_delta)
            print(f'i_steps: {i_steps}, time: {time_delta}')

#        if i_steps % 10000 == 0:
#            model.save_network()
#            ax = plt.subplot(111)
#            ax.plot(range(len(model.rewards)), model.rewards, label='y = Reward')
#            plt.title('Total Reward per Episode')
#            ax.legend()
#            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#            fig.savefig('plot.png')
