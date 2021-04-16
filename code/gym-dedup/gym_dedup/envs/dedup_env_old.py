import random
import gym
import numpy as np
from gym import spaces
import pandas as pd


# reward_threshold = 475.0
# whole_chunk_hash_index = 0
# whole_file_hash_index = 1
# whole_chunk_hash_scaled_index = 2


class DedupEnv(gym.Env):
    """deduplication environment """

    def __init__(self, df1, df2):
        """df1: a pandas data frame passed in for chunk info
           df2: a pandas data frame fpr file info containing unique whole file fingerprints
           cache: cache storage which is a list contains lists of
           [chunk hash, chunk size, whole_file_hash_index]}
        """
        super(DedupEnv, self).__init__()

        # self.chunk_hsh = 0
        self.df1 = df1
        self.df2 = df2
        self.cache = []
        # self.thresh = thresh
        self.state = 0
        self.data_length = self.df1.shape[0]
        self.current_step = 0
        # Action space: two actions, 0: select a chunk 1: do not select chunk
        self.action_space = spaces.Discrete(2)

        # n chunk fingerprints
        # self.observation_space = spaces.Box(
        # low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Discrete(self.data_length)

    """get a random row from data frame"""

    def _next_row(self):
        # get the next observation, a row from data frame containing a chunk hash and visited flag
        row = np.array(self.df1.iloc[self.current_step].values)
        return row

    """Count of chunks , modification time and  cluster id of a file with whole file hash f"""

    def _file_info(self, f):
        entry = self.df2.loc[self.df2['Whole _File_Hash'] == f].values[0]
        count = entry[1]
        mt = entry[2]  # modification time
        clst = entry[3]
        return count, mt, clst

    """Assign a weight to chunk b according to chunk a. 
       la: a list that is a row in data set containing info of chunk a
       lb: a list that is a row in data set containing info of chunk b
    """

    @staticmethod
    def _compare_chunks(la, lb, fa, fb):
        wa = fa[0]
        wb = fb[0]
        ma = fa[1]
        mb = fb[1]
        ca = fa[2]
        cb = fb[2]
        #print("wb:{}, wa:{}, lb0:{}, la0:{}, cb:{}, ca:{}, lb2:{}, la2:{}".format(wb, wa, lb[0], la[0], cb, ca, lb[2],
                                                                                  #la[2]))
        return float((wb/wa + mb/ma + abs(cb - ca) + abs(lb[2] - la[2])*1E-14) * abs(lb[0] - la[0])*1E-14)

    """Assign a weight to chunk b according to a list of chunks
        lb: a list that is a row in data set containing info of chunk b
        """

    def _weight_chunks(self, cklst, lb):
        weight = 1
        fb = self._file_info(lb[2])
        for ck in cklst:
            fa = self._file_info(ck[2])
            weight = float(weight * self._compare_chunks(ck, lb, fa, fb))
            #print (weight)
        return weight

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
        next_r = self._next_row()

        if action == 1:  # storing a chunk to cache
            # update state
            self.state = self.current_step

            # update chunk weights
            if self.cache:  # if cache is not empty
                reward = self._weight_chunks(self.cache, next_r)

            # if any(self.chunk_hsh in subl for subl in self.cache):
            # state = 2
            # else:
            #   state = 1
            # if len(self.cache) <= self.thresh:
            # store this row into cache
            self.cache.append(next_r)

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
        reward = self._act_(action)
        reward = reward
        # done = (len(self.cache) == self.thresh)
        # update current step
        curr_state = self.state
        self.current_step = self.current_step + 1
        done = self.current_step == self.data_length
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
        self.state = 0
        self.current_step = 0

    # return cache
    def get_cache(self):
        return self.cache

    def render(self, mode='human'):
        if self.df1 is not None and self.df2 is not None:
            print("Data successfully loaded")
            print("Deduplication Started:")
        else:
            print("Please provide both data files required for deduplication")
