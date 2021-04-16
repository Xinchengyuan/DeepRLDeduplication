import random
import gym
import numpy as np
from gym import spaces
from statistics import variance
import pandas as pd


# reward_threshold = 475.0
# whole_chunk_hash_index = 0
# whole_file_hash_index = 1
# whole_chunk_hash_scaled_index = 2


class DedupEnv(gym.Env):
    """deduplication environment """

    def __init__(self, seg):
        """df: a pandas data frame passed in for chunk info
           cache: cache storage which is a list contains lists of
           [chunk hash, chunk size, whole_file_hash_index]}
        """
        super(DedupEnv, self).__init__()

        # self.chunk_hsh = 0
        self.seg = seg # sample segment list
        self.cache = []
        # self.thresh = thresh
        self.done_flag = 0
        self.state = [0, 0, 0, 0]
        self.data_length = len(self.seg)
        self.current_step = 0
        self.min_chunk = 0  # minimum chunk hash of the current segment
        self.max_chunk = 0  # maximum chunk hash of the current segment
        self.seg_size = 0
        # Action space: two actions, 0: select a chunk 1: skip chunk
        self.action_space = spaces.Discrete(2)
        # observation: Box(4) ('Whole _File_Hash','seg min fingerprint', 'seg max fingerprint','chunk size')
        self.high = 0xffffffffffff
        self.observation_space = spaces.Box(low=0, high=self.high, shape=(4,), dtype=np.int64)
        # n chunk fingerprints
        # self.observation_space = spaces.Box(
        # low=0, high=1, shape=(1,), dtype=np.float32)


    """get the next observation, a row (a segment from segment list)"""

    def _next_row(self):
        row = np.array(self.seg[self.current_step])
        self.min_chunk = row[1]
        self.max_chunk = row[2]
        self.seg_size = row[3]
        self.state = row.reshape(4, )

    """check if the maximum and minimum chunk hash of current segment exists in the cache"""

    def _check_seg(self):
        for subLst in self.cache:
            if all(x in subLst for x in [self.min_chunk, self.max_chunk]):
                return True
        return False

    """How the agent should take action. 
       Action: action type, 0 or 1
       thresh: the cache size threshold
       return the corresponding state after the action is taken
       1. A unique chunk is stored
       2. A duplicated chunk is stored
       3  A duplicated chunk is removed
       4. A unique chunk is removed
       """

    def _act_(self, action):
        # state = 0  # initial state
        reward = 0
        if action == 1:  # storing a chunk to cache
            if self.cache:  # if cache is not empty
                if self._check_seg():
                    self.done_flag = 1  # terminate since duplicate segment selected
                    print("terminated")
                    return -1 * int(self.seg_size)  # wasted this much of space
                self.cache.append(self.state)
        else:
            if not self._check_seg() or not self.cache:  # a unique segment is omitted:
                self.done_flag = 1  # terminate
                return -1 * int(self.seg_size)  # lost this much of info

        reward = self.seg_size

        # if any(self.chunk_hsh in subl for subl in self.cache):
        # state = 2
        # else:
        #   state = 1
        # if len(self.cache) <= self.thresh:
        # store this row into cache

        # delete this row from data frame
        # self.df.drop(self.current_step)
        # self.data_length = self.data_length - 1
        # elif action == 1:  # removing a random chunk from cache
        # if len(self.cache) == 0:
        # state = 0
        # else:
        # rmv = random.randint(0, len(self.cache) - 1)
        # current_ck = self.cache.pop(rmv)
        # if any(current_ck in sub for sub in self.cache):
        # state = 3
        # else:
        # state = 4
        # return this row to data frame
        # curr_series = pd.Series(current_ck)
        # self.df.append(curr_series, ignore_index=True)
        # self.data_length = self.data_length + 1
        return reward

    def step(self, action):
        self._next_row()
        reward = self._act_(action)
        reward = reward
        # done = (len(self.cache) == self.thresh)
        # update current step
        curr_state = self.state
        self.current_step = self.current_step + 1
        if self.done_flag == 0:
            if self.current_step == self.data_length-1:
                self.done_flag = 1
                print(self.current_step)
        done = True if self.done_flag == 1 else False
        # info = f' state: {curr_state}, reward: {reward}, cache size: {len(self.cache)}'

        return curr_state, reward, done

    def reset(self):
        # self.chunk_hsh = 0
        # reset current chunk hash
        # self.current_step = random.randint(
        # 0, self.data_length - 1
        # )
        # empty the cache
        self.cache = []
        self.state = [0, 0, 0, 0]
        self.current_step = 0
        self.done_flag == 0
        return self.state

    # return cache
    def get_cache(self):
        return self.cache

    def render(self, mode='human'):
        if self.seg is not None:
            print("Data successfully loaded")
            print("Deduplication Started:")
        else:
            print("Please provide data file required for deduplication")
