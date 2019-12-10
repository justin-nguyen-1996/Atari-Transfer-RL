# Atari-Transfer-RL
CS 394R - The purpose of this repository is to showcase multitask learning and state abstraction methods with the goal of compressing value approximation networks while still performing comparable to non compressed agents.

## How to run 
This code was built to run on python 3.6+

Run the following to install the required libraries 
```
pip3 install --user numpy torch gym gym[atari] matplotlib ipython baselines
```

To train a teacher network using DQN:
```
python3 dqn.py
```

To train a student using policy distillation:
```
python3 policy_distill.py
```

In order to change Atari environments, just change the value of ``env_name`` in the top of `dqn.py` and `policy_distill.py`.

For the state abstraction code, run `git checkout state_abstraction` and follow a similar process as above.

