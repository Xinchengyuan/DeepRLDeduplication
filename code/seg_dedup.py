import gym
import random
import pandas as pd
import numpy as np
import os
import time
from collections import deque
import tensorflow as tf
import gym_dedup
# from argparse import ArgumentParser
import data_preprocessing as dp
from keras.callbacks import TensorBoard

import progressbar


# Own Tensorboard class

class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    # Added because of version
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()


class SegDedup:
    """
       data: file chunk data
       size: segment size threshold
       perc: sample ratio
       expl_rt exploration_rate:
       hidden: number of hidden_units
    """

    def __init__(self, data, size, perc, max_episodes, expl_rt, discount, hidden):
        # environment
        self.data = data
        self.size = size
        self.perc = perc
        self.segments = dp.get_all_segments(self.data, self.size)
        self.sample = dp.get_all_sample(self.perc, self.segments)
        self.sample = np.asarray(self.sample, dtype=np.int64)
        self.discount = discount

        self.env = gym.make('dedup-v0', seg=self.sample)

        # hyper-parameters
        self.max_episodes = max_episodes
        self.expl_rt = expl_rt  # epsilon
        self.exploration_decay = 0.9995
        self.discount = float(discount)  # gamma

        # model
        self.in_units = self.env.observation_space.shape[0]
        print(self.in_units)
        self.out_units = self.env.action_space.n
        self.hidden = hidden
        # self.saver = tf.compat.v1.train.Saver()

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{int(time.time())}")
        self.experience_replay = deque(maxlen=2000)
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.align_models()

    def _build_model(self):
        model = tf.keras.models.Sequential()
        # input layer
        model.add(
            tf.keras.layers.Dense(units=32, activation='relu', name="input", input_dim=self.in_units))
        # hidden layer
        model.add(tf.keras.layers.Dense(units=self.hidden, activation='relu', name="hidden"))
        # output layer
        model.add(tf.keras.layers.Dense(units=self.out_units, activation='linear', name="output"))
        # compile model
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def memory(self, state, action, reward, next_state, done):
        self.experience_replay.append((state, action, reward, next_state, done))

    def align_models(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def act(self, state):
        state = tf.reshape(state, [1, 4])
        if np.random.rand() <= self.expl_rt:
            return self.env.action_space.sample()
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def retrain(self, batch_size):
        minibatch = random.sample(self.experience_replay, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = tf.reshape(state, [1, 4])
            next_state = tf.reshape (next_state, [1, 4])
            target = self.q_network.predict(state)

            if done:
                target[0][action] = reward
            else:
                ns = self.target_network.predict(next_state)
                target[0][action] = reward + self.discount * np.amax(ns)
            self.q_network.fit(state, target, epochs=1, verbose=0)

    def train(self, batch_size):
        self.q_network.summary()
        rewards = [0]
        EPISODE_INTERVAL = 100
        MIN_EXPLORATION = 0.001  # min epsilon threshold

        for ep in range(0, self.max_episodes):

            # update tensorboard step every episode
            self.tensorboard.step = ep

            eps_reward = 0

            time_steps_per_episode = self.sample.shape[0]
            # reset environment
            state = self.env.reset()
            state = np.reshape(state, (4,))
            reward = 0
            done = False

            bar = progressbar.ProgressBar(maxval=time_steps_per_episode / 10, widgets=[
                progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()

            for time_step in range(time_steps_per_episode):
                action = self.act(state)
                # Take action
                next_state, reward, done = self.env.step(action)
                self.memory(state, action, reward, next_state, done)

                state = next_state
                eps_reward += reward

                if done:
                    self.align_models()
                    print("episode: {}/{}, score: {}, e: {:.2}".format(ep, self.max_episodes, time_step, self.expl_rt))
                    break

                if len(self.experience_replay) > batch_size:
                    self.retrain(batch_size)

                if time_step % 10 == 0:
                    bar.update(time_step / 10 + 1)

                # Append episode reward to a list and log stats (every given number of episodes)
                rewards.append(eps_reward)
                if not ep % EPISODE_INTERVAL or ep == 1:
                    average_reward = sum(rewards[-EPISODE_INTERVAL:]) / len(rewards[-EPISODE_INTERVAL:])
                    min_reward = min(rewards[-EPISODE_INTERVAL:])
                    max_reward = max(rewards[-EPISODE_INTERVAL:])
                    self.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward,
                                                  reward_max=max_reward, epsilon=self.expl_rt)

                    # Save model, but only when min reward is greater or equal a set value
                    if min_reward >= 0:
                        self.q_network.save(
                            f'models/{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__'
                            f'{int(time.time())}.model')

            # decay the epsilon

            if self.expl_rt > MIN_EXPLORATION:
                self.expl_rt *= self.exploration_decay
                self.expl_rt = max(MIN_EXPLORATION, self.expl_rt)

            bar.finish()
            if (ep + 1) % 10 == 0:
                print("*****************************")
                print("Episode: {}".format(ep + 1))
                print("*****************************")


"""
def arg_parse():
    parser = ArgumentParser()
    parser.add_argument("--data", help="data")
    parser.add_argument("--seg_size", help="segment size in bytes", default=4096)
    parser.add_argument("--max_episodes", default=10000)
    parser.add_argument("--hidden_units", default=10)
    parser.add_argument("--exploration_rate", default=1.0)
    parser.add_argument("--discount_factor", default=0.95)
    return parser.parse_args()
"""

if __name__ == '__main__':
    # args = arg_parse()
    if not os.path.isdir('models'):
        os.makedirs('models')
    df = "../data/out.csv"
    ds = pd.read_csv(df, dtype=np.int64, chunksize=1000, nrows=20000)
    seg_size = 4096
    percentage = 0.01
    max_ep = 1000
    exploration_rate = 1.00  # going to be decayed later
    discount_factor = 0.99
    hidden_units = 32

    agent = SegDedup(ds, seg_size, percentage, max_ep, exploration_rate, discount_factor, hidden_units)
    print (len(agent.segments))
    print (len(agent.sample))
    #agent.train(batch_size=32)
    # print (agent.sample)
