
#===================
# Imports
#===================

import numpy as np
import gym
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import torch.nn as nn
import warnings # For ignoring pytorch warnings
import time
import datetime
import os

from dqn import DQN

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

num_episodes = 20

epsilon = 0
#epsilon = 0.02

#==================
# Testing Loop
#==================

start = time.time()

# Load the saved network weights
model = DQN()
network_weights = os.path.join(log_dir, env_name + '_saved_network.pt')
if torch.cuda.is_available():
    model.network.load_state_dict(torch.load(network_weights))
else:
    model.network.load_state_dict(torch.load(network_weights, map_location=torch.device('cpu')))

state = env.reset()
state = state.transpose(2, 0, 1)
env.render()
cur_episode = 0
total_reward = 0

# Run for num_episodes episodes
while cur_episode < num_episodes:
    action = model.get_epsilon_greedy_action(state, epsilon)
#    print(f'action: {action}')
    prev_state = state
    state, reward, done, _ = env.step(action)
    env.render()
    state = state.transpose(2, 0, 1)
    total_reward += reward

    if done:
        # Save the reward earned during the episode to the list of rewards
        model.save_reward(total_reward)
        print(f'total_reward: {total_reward}')
        # Save the reward plot
        ax = plt.subplot(111)
        ax.plot(range(len(model.rewards)), model.rewards, label='y = Reward')
        plt.title('Total Reward per Episode')
        ax.legend()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig('plot_test_dqn.png')
        plt.clf()
        plt.close()
        # Reset some variables for the next episode
        state = env.reset()
        state = state.transpose(2, 0, 1)
        total_reward = 0
        cur_episode += 1
        # Print to the screen that the episode is over
        time_delta = int(time.time() - start)
        time_delta = datetime.timedelta(seconds=time_delta)
        print(f'cur_episode: {cur_episode}, time: {time_delta}')
