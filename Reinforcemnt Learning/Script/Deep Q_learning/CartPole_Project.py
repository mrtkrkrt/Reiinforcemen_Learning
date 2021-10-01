#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
import random
from collections import deque
import gym


class DQLAgent:

    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.gamma = 0.95
        self.learning_rate = 0.001

        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.memory = deque(maxlen=1000)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(48, input_dim = self.state_size, activation = "tanh"))
        model.add(Dense(self.action_size, activation = "linear"))
        model.compile(loss = "mse", optimizer = Adam(learning_rate = self.learning_rate))
        return model

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()

        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])

    def replay(self, batch_size):

        if len(self.memory) < batch_size:
            return
        mini_batch = random.sample(self.memory, batch_size)
        print(mini_batch[0])

        for (state, action, reward, next_state, done) in mini_batch:
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            train_target = self.model.predict(state)
            train_target[0][action] = target
            self.model.fit(state, train_target, verbose = 0)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def adaptiveEpsilonGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":

    env = gym.make("CartPole-v1")
    agent = DQLAgent(env)

    episodes = 4

    for i in range(episodes):

        state = env.reset()
        time = 0
        batch_size = 16

        state = np.reshape(state, (1, 4))

        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            next_state = np.reshape(next_state, (1, 4))

            agent.remember(state, action, reward, next_state, done)

            state = next_state

            agent.replay(batch_size)

            agent.adaptiveEpsilonGreedy()

            time += 1

            if done:
                print("Episode {} => Time : {}".format(i, time))
                break
# %%
import time
trained_model = agent
state = env.reset()
state = np.reshape(state, (1,4))
time_t = 0

while True:
    env.render()
    action = trained_model.act(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    next_state = np.reshape(next_state, (1,4))
    time_t += 1
    time.sleep(0.4)
    print(time_t)

    if done:
        break

print("Done")

