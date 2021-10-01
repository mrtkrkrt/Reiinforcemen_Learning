import gym
import tensorflow as tf 
import random
import numpy as np 
from collections import deque
import time
from tensorflow import keras

env = gym.make("CartPole-v1")
print("Action Space : {}".format(env.action_space.n))
print("State Space : {}".format(env.observation_space.shape[0]))

train_episode = 100
test_size = 20

def agent(state_shape, action_shape):

    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model

def get_qs(model, state, step):
    return model.predict(state.reshape([1, state.shape[0]]))[0]

def train(env, replay_memory, model, target_model, done):

    learning_rate = 0.7
    discount_rate = 0.68

    MIN_REPLAY_SIZE = 1000

    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = 64
    mini_batch = random.sample(replay_memory, batch_size)
    #print(mini_batch[0])
    current_states = np.array([batch[0] for batch in mini_batch])
    current_qs_list = model.predict(current_states)

    next_current_states = np.array([batch[3] for batch in mini_batch])
    next_current_qs = target_model.predict(next_current_states)

    X = []
    Y = []

    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        
        if not done:
            max_reward = reward + discount_rate * np.max(next_current_qs[index])
        else:
            max_reward = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_reward

        X.append(observation)
        Y.append(current_qs)
    
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose = 0, shuffle = True)


def main():
    epsilon = 1
    max_epsilon = 1
    min_epsilon = 0.01
    decay = 0.01

    model = agent(env.observation_space.shape, env.action_space.n)
    target_model = agent(env.observation_space.shape, env.action_space.n)

    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=1000)
    target_update_counter = 0

    X = [] #states
    y = [] #actions

    steps_to_update_target = 0

    for episode in range(train_episode):
        total_reward = 0
        state = env.reset()
        done =False

        while not done:
            steps_to_update_target += 1

            if True:
                env.render()
            
            num = random.uniform(0, 1)
            if num < epsilon:
                action = env.action_space.sample()

            else:
                temp = state
                temp = np.reshape(temp, (1, state.shape[0]))
                action = model.predict(temp).flatten()
                action = np.argmax(action) 
            
            next_state, reward, done, _ = env.step(action)

            replay_memory.append([state, action, reward, next_state, done])

            if steps_to_update_target % 4 == 0:
                train(env, replay_memory, model, target_model, done)

            state = next_state
            total_reward += reward

            if done:
                print('Total training rewards: {} after n steps = {} with final reward = {}'.format(total_reward, episode, reward))
                total_reward += 1

                if steps_to_update_target >= 100:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target = 0
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
    env.close()

if __name__ == "__main__":
    main()



        