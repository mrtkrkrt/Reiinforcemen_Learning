import gym

env = gym.make("Taxi-v3").env

env.render()

"""
blue = passenger
purple = destination
yellow/red = empty taxi
green = full taxi
RGBY = location for destination and passenger 
"""

# env.reset() => reset env and return random initial state
#%%
print("State Space =  ", env.observation_space) #500
print("Action Spaace = ", env.action_space) #6

#%%
env.reset()

time_step = 0
total_reward = 0
list_v = []

while True:
    
    time_step += 1
    
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    
    total_reward += reward
    
    list_v.append({"frame" : env,
                   "state" : state, "action" : action,
                   "reward" : reward, "total_reward": total_reward
        })
    
    env.render()
    
    if done:
        break
    
#%%
import time

for i, frame in enumerate(list_v):
    print(frame["frame"].render)
    print("Tİme Step = ", i+1)
    print("State = ", frame["state"])
    print("Action = ", frame["action"])
    print("Reward = ", frame["reward"])
    print("Total Reward = ", frame["total_reward"])
    
    time.sleep(1)
    
    
# %%
