import gym 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import random
import tensorflow as tf
from tensorflow import keras
from collections import deque
import numpy as np

env = gym.make("LunarLander-v2")

print("Action Space : {}".format(env.action_space.n))
print("State Spae : {}".format(env.observation_space.shape[0]))

train_episode = 300
test_episode = 100

def agent(state_shape, action_shape):

    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model

def predict(model, state, step):
    return model.predict(state.reshape([1, state.shape[0]]))[0]

def train(env, replay_memory, model, target_model, done):

    learning_rate = 0.7
    discount_rate = 0.68

    min_len_memory = 1000
    batch_size = 32

    if len(replay_memory) < min_len_memory:
        return

    mini_batch = random.sample(replay_memory, batch_size)
    current_state_list = np.array([batch[0] for batch in mini_batch])
    current_qs_list = model.predict(current_state_list)

    next_state_list = np.array([batch[3] for batch in mini_batch])
    next_qs_list = target_model.predict(next_state_list)

    X = []
    y = []

    for index, (state, action, reward, new_observation, done) in enumerate(mini_batch):

        if not done:
            max_future_q = reward + discount_rate * np.max(next_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        X.append(observation)
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)

def main():
    epsilon = 1
    min_epsilon = 0.01
    decay = 0.01
    max_epslion = 1

    model = agent(env.observation_space.shape, env.action_space.n)
    target_model = agent(env.observation_space.shape, env.action_space.n)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=50000)

    target_update_counter = 0
    steps_to_update_target = 0

    X = []
    y = []

    for episode in range(train_episode):

        total_reward = 0
        state = env.reset()
        done = False

        while not done:
            steps_to_update_target += 1

            env.render()

            num = np.random.rand()
            if num < epsilon:
                act = env.action_space.sample()
            else:
                temp = state
                temp = np.reshape(temp, (1, temp.shape[0]))
                act = model.predict(temp).flatten()
                act = np.argmax(act)

            next_state, reward, done, _ = env.step(act)
            replay_memory.append((state, reward, done, next_state, _))

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
        epsilon = min_epsilon + (max_epslion - min_epsilon) * np.exp(-decay * episode)
    env.close()

if __name__ == "__main__":
    main()

            

