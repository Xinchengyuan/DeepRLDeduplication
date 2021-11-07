import random
import gym
import numpy as np
from gym import spaces
from .bloom_filter import bloom_filter

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
        self.cache_size = 0
        self.feature_extractor = feature_extractor
        self.data_preprocessor = data_preprocessor
        #containers
        self.cache = {}
        self.bins = {}
        self.bin_num = 0
        # Bloom filter properties
        self.bf_length = 500000000 #length of bloom filter
        self.num_zero = self.bf_length
        self.bloom_filter= bloom_filter(num_elements=self.bf_length, fp_prob= 0.01)
        print ("Bloom_Filter Setup")
        print ("False positive rate: ", self.bloom_filter.fp_prob)
        print("Total length: ", self.bloom_filter.m)
        print ("Number of hash functions: ",self.bloom_filter.hash_count)
        self.state = [0, 0, 0]
        self.file_fp = 0
        #self.chunks = []
        self.segments = []
        self.seg_size = 0 #segment size
        self.accum_seg_size = 0 #accumulated segment size
        #self.feature = 0
        self.file = []
        #self.eviction = 0 # feature to evict
        #self.max_hit = 0 # max number of hits so far
        # Action space: two actions, 0: select a chunk 1: skip chunk
        self.action_space = spaces.Discrete(2)
        # observation: Box(4) ('Whole _File_Hash','seg min fingerprint', 'seg max fingerprint','chunk size')
        self.high = 0xffffffffffff
        self.observation_space = spaces.Box(low=0, high=self.high, shape=(3,), dtype=np.int64)

    """
    Helper function to uniquely encode two integers into given that a >= b,
    usinh the Szudzik function,
    """
    def encode(self, a, b):
        x = int(a)
        y = int(b)
        return x*x+x+y

    """
    get the segmens from next file
    segments format: [whole_file_chunk, 
    (min_chunk, max_chunk, seg_size, [chunk1, chunk2, ...]) # segment 1,
    (min_chunk, max_chunk, seg_size, [chunk1, chunk2, ...]) # segment 2,
    ......
    ]
    """
    def _get_segments(self):
        f_next = ()
        #print("Obtaining next file")
        f_next = self.feature_extractor.next_file()
        #print(f_next)
        self.segments = self.data_preprocessor.segment(f_next, self.size)
        #print("Segments: ", self.segments)
        try:
            self.file_fp = self.segments.pop(0)
            #record first segment as state
            row = np.array(self.segments[0][:3])
        except IndexError:
            row = np.array([0, 0, 0])
        self.state = row.reshape(3, )
    """
      load a segment to cache, given feature and factor to time the reward
    """

    def _load_to_cache(self, seg, feature, factor):
        #ft=0
        #min_chunk = min(seg[0] for seg in self.segments)
        #max_chunk = max(seg[1] for seg in self.segments)
        #for seg in self.segments:
        #self.feature = self.encode(seg[1], seg[0])
        #self.feature = int(ft/len(self.segments))
        try:
            # see if there's hit in cache
            self.cache[feature][0] = self.cache[feature][0] + 1
            # accumulate segment size
            self.cache[feature][1] = self.cache[feature][1] + seg[2]
            # update chunks
            self.cache[feature][2].extend(seg[3])
            reward = factor * self.cache[feature][0]
        except KeyError:
            # create new entry in cache
            #print("Adding to cache")
            self.cache[feature] = [0, seg[2], seg[3]]
            reward = 1
        self.cache_size = self.cache_size + seg[2]
        return reward
        #else:
            #print("No more rows left to fetch")
            #self.chunks = []

    """How the agent should take action. 
       Action: action type, 0 or 1s
       0: Do not pick the segment
       1: Pick the segment
       Return the corresponding state after the action is taken
       Feedback to the agent with reward
       cache format:
       {feature:(score, size, [chunks])}
       """

    def act(self, action):
        # state = 0  # initial state
        self._get_segments()
        reward = 0
        if self.segments:
            if action == 1:
                for seg in self.segments:
                    feature = self.encode(seg[1], seg[0])
                #check bloom filter
                    if self.bloom_filter.does_exist(feature):
                       reward = reward-1
                    else:
                       reward=self._load_to_cache(seg, feature, -1)

            else: # skip a segment
                for seg in self.segments:
                    feature = self.encode(seg[1], seg[0])
                    if self.bloom_filter.does_exist(feature):
                       reward = reward+1
                    else:
                        # try lookup feature
                       reward=self._load_to_cache(seg, feature, 1)



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
        done = self.feature_extractor.done
        #self._next_row()
        #print(self.feature)
        #print(self.cache)
        #print("csh_szie", self.cache_size)
        while self.cache_size >= self.cache_thresh:
            # evict feature with highest hits score and put corresponding chunks to bin
            eviction = max(self.cache, key=lambda k: self.cache[k][0])
            #print ("Evicting", eviction)
            #print(self.cache[eviction])
            self.bloom_filter.add(eviction)
            self.bins[self.bin_num] = {}
            #Load to bins the containing chunks of eviction feature
            #print ("Loading to bin ", self.bin_num)
            with open ("rec.txt","a") as f:
             f.write(str({self.bin_num: set(self.cache[eviction][2])}))
             #ins[self.bin_num] = )
            self.accum_seg_size = self.accum_seg_size + self.cache[eviction][1]
            self.bin_num = self.bin_num+1
            self.cache_size = self.cache_size - self.cache[eviction][1]
            del self.cache[eviction]

        if done:
            #print(done)
            # evict rest of segments from cache:
            if self.cache:
                with open("recipe.txt", "a") as f:
                    for key in self.cache:
                        #self.bins[self.bin_num] = {}
                        f.write(str({self.bin_num: set(self.cache[key][2])}))
                        #self.bins[self.bin_num] = set(self.cache[key][2])
                        self.accum_seg_size = self.accum_seg_size + self.cache[key][1]
                        self.bin_num = self.bin_num + 1
                        #del self.cache[key]
            reward = 0
        else:
            reward = self.act(action)

        curr_state = self.state



            #check out cache content to seg_info
            #self.seg_info.update(self.cache)

        # info = f' state: {curr_state}, reward: {reward}, cache size: {len(self.cache)}'

        return curr_state, reward, done

    def reset(self):
        # self.chunk_hsh = 0
        # reset current chunk hash
        # self.current_step = random.randint(
        # 0, self.data_length - 1
        #
        # empty the cache
        self.cache = {}
        self.state = [0, 0, 0]
        self.bloom_filter.reset()
        self.bins = {}
        self.bin_num = 0
        #self.max_hit = 0
        self.eviction  = 0
        #self.done_flag = 0
        return self.state

    def render(self, mode='human'):
        if self.seg is not None:
            print("Data successfully loaded")
            print("Deduplication Started:")
        else:
            print("Please provide data file required for deduplication")
