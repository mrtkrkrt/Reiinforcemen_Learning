import os
import pygame
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import tensorflow as tf
import numpy as np
from keras import Sequential
from collections import deque
from keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from Sprites import Paddle, Enemy

"""
0 => LEFT
1 => RIGHT
2 => DO NOTHING
"""

class Agent():
    
    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space

        self.epsilon = 1
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 16
        self.gamma = 0.95
        
        self.memory = deque(maxlen=10000)
        self.learning_rate = 0.001
        #self.model = self.get_model()
        self.model = tf.keras.models.load_model("Model.model")
    def get_model(self):
        model = Sequential()
        model.add(Dense(64, input_shape=(self.state_space,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_space)
        result = self.model.predict(state)
        return np.argmax(result[0])

    def replay(self):
        
        if len(self.memory) < self.batch_size:
            return

        mini_batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for state, action, reward , next_state, done in mini_batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        states, actions, rewards, next_states, dones = np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

        not_done_indices = np.where(dones == False)[0]
        
        target = rewards
        #print("target")
        target[not_done_indices] = target[not_done_indices] +  self.gamma * np.amax(self.model.predict(next_states[not_done_indices].reshape(len(not_done_indices), self.state_space)), axis = 1) 
        target_array = self.model.predict(states)

        target_array = target_array.reshape(self.batch_size, 3)
        for i in range(len(actions)):
            target_array[i, actions[i]] = target[i]

        #target_array[np.arange(self.batch_size), actions] = target
    
        self.model.fit(states, target_array, verbose = 0)

        """for state, action, reward, next_state, done in mini_batch:
            state = np.array(state)
            next_state = np.array(next_state)
            if done:
                target = reward 
            else:
                target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])
            train_target = self.model.predict(state)
            train_target[0][action] = target
            self.model.fit(state,train_target, verbose = 0)"""


    def adaptiveEGreedy(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay



        
