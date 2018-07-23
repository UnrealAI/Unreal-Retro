import tensorflow as tf
from collections import deque
import numpy as np
import retro
import random
import matplotlib.pyplot as plt


class DQNAgent: # https://keon.io/deep-q-learning/
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size[0]
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        input_shape = self.state_size
        print(self.state_size)
        model.add(tf.keras.layers.Conv2D(24, (3, 3), padding='same', input_shape = input_shape, activation='relu')) # https://github.com/keon/deep-q-learning/blob/master/dqn.py
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(24, activation='relu'))# input_dim = self.state_size,
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.train.AdamOptimizer(self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        act = np.zeros((self.action_size,))
        if np.random.rand() <= self.epsilon:
            act_values = random.randrange(0,self.action_size)
            act[act_values] = 1.
            return act
        state = state.reshape((-1,)+self.state_size)
        act_values = self.model.predict(state)

        #print(act)
        act[np.argmax(act_values)] = 1.
        #print(act)
        return act

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        #print(minibatch)#np.concatenate(minibatch,axis=2).shape)
        for state, action, reward, next_state, done in minibatch:
            #print(state.shape)
            #
            target = reward
            state = state.reshape((-1,)+self.state_size)
            if not done:
                next_state = next_state.reshape((-1,)+self.state_size)#expand_dims(state, axis=0)
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][np.argmax(action)] = target

            self.model.fit(state, target_f, epochs = 1, verbose = 0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train():
    game, state, record, scenario, players  =  ('Airstriker-Genesis', 'Level1.state', False, 'scenario', 1)
    verbose, quiet = 1, 0
    env = retro.make(game, state, scenario=scenario, record=record)
    verbosity = verbose - quiet
    frame_step = 200
    episodes = 70
    batch_size = 150
    game_time = 10000
    eps, scores = [], []

    agent = DQNAgent(env.observation_space.shape,env.action_space.shape) # [0] .n , change reshape https://github.com/keon/deep-q-learning/blob/master/dqn.py

    for e in range(episodes):

        state = env.reset()
        #state = np.reshape(state, [1, 4])
        total_reward = 0
        for time_t in range(game_time):

            if time_t % frame_step == 0:
                env.render()

            action = agent.act(state)

            #print(action)

            next_state, reward, done, _ = env.step(action)
            #next_state = np.reshape(next_state, [1, 4])
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:

                env.render()

                break

        print("episode: {}/{}, score: {}"
              .format(e, episodes, time_t))
        scores.append(total_reward)#time_t)
        agent.replay(batch_size)

    plt.figure()
    plt.plot(scores)
    plt.savefig('scores.png', dpi=300)

if __name__ == '__main__':
    train()
