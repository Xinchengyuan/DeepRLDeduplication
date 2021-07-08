import random
import gym
import numpy as np
from gym import spaces

# reward_threshold = 475.0
# whole_chunk_hash_index = 0
# whole_file_hash_index = 1
# whole_chunk_hash_scaled_index = 2


class DedupEnv(gym.Env):
    """deduplication environment """

    def __init__(self, size, feature_extractor, data_preprocessor, cache_thresh):
        """
           size: segmentation size threshold
           cache_thresh: cache length threshold
           feature_extractor: a feature extractor object
           data_preprocessor: a data preprocessor object
           [chunk hash, chunk size, whole_file_hash_index]}
        """
        super(DedupEnv, self).__init__()
        self.size = size
        self.cache_thresh = cache_thresh
        self.feature_extractor = feature_extractor
        self.data_preprocessor = data_preprocessor
        self.cache = {}
        self.bins = {}
        self.seg_info = {} # store previous features, segments and hits from cache
        # self.thresh = thresh
        #self.done_flag = 0
        self.state = [0, 0, 0]
        self.chunks = []
        self.feature = 0
        self.file = []
        # Action space: two actions, 0: select a chunk 1: skip chunk
        self.action_space = spaces.Discrete(2)
        # observation: Box(4) ('Whole _File_Hash','seg min fingerprint', 'seg max fingerprint','chunk size')
        self.high = 0xffffffffffff
        self.observation_space = spaces.Box(low=0, high=self.high, shape=(3,), dtype=np.int64)
        # n chunk fingerprints
        # self.observation_space = spaces.Box(
        # low=0, high=1, shape=(1,), dtype=np.float32)


    """get the next observation, a row (a segment from segment list)"""

    def _next_row(self):
        if not self.feature_extractor.done:
            f_next = ()
            segments = []
            if not f_next:
                f_next = self.feature_extractor.next_file()
                segments = self.data_preprocessor.segment(f_next, self.size)
            segment = segments.pop(0)
            min_chunk = segment[1]
            max_chunk = segment[2]
            self.chunks = segment[4]
            self.feature = hash(self.min_chunk, self.max_chunk)
            row = np.array(segment[1:4])
            self.state = row.reshape(3,)
        else:
          self.chunks = []

    """How the agent should take action. 
       Action: action type, 0 or 1s
       0: Do not pick the segment
       1: Pick the segment
       Return the corresponding state after the action is taken
       Feedback to the agent with reward
       """

    def _act_(self, action):
        # state = 0  # initial state
        reward = 0

        if action == 1:  # storing a segment to cache
            try:
                self.seg_info[self.feature][0] = self.seg_info[self.feature][0]+1 # Hits in seg_info
                reward = self.seg_info[self.feature][0]
            except KeyError:
                try:
                    # see if there's hit in cache
                    self.cache[self.feature][0] = self.cache[self.feature][0] + 1
                    reward = -1 * self.cache[self.feature][0]
                except KeyError:
                    #create new entry in cache
                    self.cache[self.feature] = (0, self.chunks)
                    reward = 0
        else: # skip a segment
            #Check if any punish needed
            try:
                #get number of hit in seg_info, if any, if this segment were selected
                hits = self.seg_info[self.feature][0]
            except KeyError:
                hits = 0
            reward = hits*(-1)

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

    def _step_(self, action):
        done = False
        self._next_row()
        reward = self._act_(action)
        curr_state = self.state

        if not self.chunks or len(self.cache) == self.cache_thresh:
            done = True
            #check out cache content to seg_info
            self.seg_info.update(self.cache)

        # info = f' state: {curr_state}, reward: {reward}, cache size: {len(self.cache)}'

        return curr_state, reward, done

    def _reset_(self):
        # self.chunk_hsh = 0
        # reset current chunk hash
        # self.current_step = random.randint(
        # 0, self.data_length - 1
        #
        # empty the cache
        self.cache = {}
        self.state = [0, 0, 0]
        #self.done_flag = 0
        return self.state

    def render(self, mode='human'):
        if self.seg is not None:
            print("Data successfully loaded")
            print("Deduplication Started:")
        else:
            print("Please provide data file required for deduplication")
