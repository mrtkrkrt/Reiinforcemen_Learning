import gym
import numpy as np 
import matplotlib.pyplot as plt 
import random

env = gym.make("Taxi-v3").env

# Q-Table
q_table = np.zeros([env.observation_space.n, env.action_space.n])
print("Q_Table Shape = ", q_table.shape)
# Hyperparameter
alpha = 0.1
gama = 0.9
epsilon = 0.1

# Plotting Metrix
reward_list = []
dropout_list = []

# Episodes
epiode_num = 1000

for i in range(1, epiode_num):

    # Initialize enviroments
    state = env.reset()
    reward_num = 0
    dropout = 0 
    # Train one Episode
    while True:

        # Exploit vs Explore to find an action
        # %10 explore, %90 exploit
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Action process and take reward
        next_state, reward, done, _ = env.step(action)
        # Q-Learning function and old value
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        next_value = ((1 - alpha) * old_value) + (alpha * (reward + gama * next_max))
        # Q-Learning update
        q_table[state, action] = next_value
        # Update state
        state = next_state
        # find wrong dropouts
        if dropout == -10:
            dropout += 1

        reward_num += reward

        if done:
            break

    if i % 10 == 0:
        dropout_list.append(dropout)
        reward_list.append(reward_num)
        print("Episode : {}, Reward : {}, Wrong Dropout : {}".format(i, reward_num, dropout))

fig, axs = plt.subplots(1, 2)
axs[0].plot(reward_list)
axs[0].set_xlabel("Episode") 
axs[0].set_ylabel("Reward")

axs[0].plot(dropout_list)
axs[0].set_xlabel("Episode") 
axs[0].set_ylabel("Dropout")

axs[0].grid(True)
axs[1].grid(True)

plt.show()