import gym
import numpy as np
import random
import matplotlib.pyplot as plt 
from gym.envs.registration import register

env = gym.make("FrozenLake-v1").env

# Make environment deterministic
register(
    id = "FrozenLakeNotSlippery-v1",
    entry_point = "gym.envs.toy_text:FrozenLakeEnv",
    kwargs = {"map_name" : "4x4", "is_slippery" : False},
    max_episode_steps = 100,
    reward_threshold = 0.78,
)

"""
SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)
"""

"""
ACTIONS
0	Move Left
1	Move Down
2	Move Right
3	Move Up
"""

# Creating Q_Table
q_table = np.zeros([env.observation_space.n, env.action_space.n])
print("Q_Table Shape = ", q_table.shape)

# Hyperparameter
alpha = 0.1
gama = 0.9
epsilon = 0.2

# Matrix for plotting
dropout_list = []
reward_list = []

# Episodes 
episodes = 10000


for i in range(1, episodes):
    
    # Initialize State and variables
    state = env.reset()
    reward_num = 0
    dropout_num = 0

    while True:

        # Exploit vs Explore to find an action
        # %10 explore, %90 exploit
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])


        #Take Action
        next_state, reward, done, _ = env.step(action)
        
        old_value = q_table[state, action]
        next_max = np.argmax(q_table[next_state])
        next_value = ((1 - alpha) * old_value) + (alpha * (reward + gama + next_max))

        q_table[state, action] = next_value
        state = next_state

        if reward == -10:
            dropout += 1

        reward_num += reward

        if done:
            break

    if i % 10 == 0:
        dropout_list.append(dropout_num)
        reward_list.append(reward_num)
        print("Episode : {}, Reward : {}, Wrong Dropout : {}".format(i, reward_num, dropout_num))



