import os
import time
from dedup_feature_extractor import featureExtractor
import psutil

start_time = time.time()

path = "/Users/test/Documents/SegDedup/data/user5"
# data = pd.read_csv("../data/pt.csv", dtype=np.int64, chunksize=5)
primary = {}  # primary index {representative index, bin number}
bin = {}  # {bin num: {ck1:size1, ck2:size2, ck3:size3, ...}}

curr_bin_num = -1
repre = {}  # {file: representative index}
rid = ""
feature_extractor = featureExtractor(path=path, mode='sample')
complete = False

while not feature_extractor.done:
    # get the next file
    file_next = feature_extractor.next_file()
    if file_next[0] and file_next[1]:
        curr_file = file_next[0]  # current file fingerprint
        rid = min([int(item[0], 16) for item in file_next[1]])
        rid = hex(rid)
        if curr_file not in repre:  # means this file is not duplicate
            repre[curr_file] = rid
            if rid in primary.keys():
                rel_bin_num = primary[rid]
            else:
                curr_bin_num = curr_bin_num + 1
                rel_bin_num = curr_bin_num
                # log bin number in primary index
                primary[rid] = rel_bin_num
                bin[rel_bin_num] = {}
            # load chunks to bins
            for cks in file_next[1]:
                bin[rel_bin_num][cks[0]] = cks[1]

# compute memory overhead
process = psutil.Process(os.getpid())
mem = process.memory_info()[0] / float(2 ** 20)

# compute deduplication ratio
duration = time.time() - start_time
print("Calculating deduplication ratio")
s = 0
for dic in bin.values():
    for v in dic.values():
        s = s + v
dedup_ratio = feature_extractor.total_chunk_size / s
print("--- %s seconds ---" % duration)
print("Deduplication ratio: {}".format(dedup_ratio))
print("Memory Usage: {} mib".format(mem))
