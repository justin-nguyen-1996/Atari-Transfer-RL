# Atari-Transfer-RL
CS 394R - The Purpose of this repository is to showcase multitask learning and state abstraction methods with the goal of compressing value approximation networks while still performing comparable to non compressed agents.

## How to run 
This code was built to run on python 3.6+

Run the following to have all the required libraries 
```
pip3 install --user numpy torch gym gym[atari] matplotlib ipython baselines
```

To run code as is just use
```
python3 dqn.py
```

In order to change environments just change the value of ``env_id`` in the top of the code

