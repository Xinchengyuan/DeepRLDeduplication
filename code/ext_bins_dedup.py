import pandas as pd
import numpy as np
import time
import psutil
import os

start_time = time.time()
data = pd.read_csv("../data/out.csv", dtype=np.int64, chunksize=100000)
# data = pd.read_csv("../data/pt.csv", dtype=np.int64, chunksize=5)
primary = {}  # primary index {representative index, bin number}
bin = {}
curr_file = 0
curr_bin_num = -1
repre = {}  # {file: representative index}
rid = 0

for data_chunk in data:
    curr = data_chunk.groupby('Whole _File_Hash', as_index=False, sort=False).apply(pd.DataFrame.sort_values, 'Chunk '
                                                                                                              '_Hash')
    # temp = dict([(a, b) for a, b in zip(curr.iloc[:, 0], curr.iloc[:, 1])])
    # first row
    first_row = curr.iloc[0]
    if first_row['Whole _File_Hash'] != curr_file:
        curr_file = first_row['Whole _File_Hash']
        if curr_file not in repre:  # means this file is not duplicate
            rid = first_row['Chunk _Hash']
            repre[curr_file] = rid
            if rid in primary.keys():
                rel_bin_num = primary[rid]
                bin[rel_bin_num][first_row['Chunk _Hash']] = first_row['Chunk_Size']
            else:
                curr_bin_num = curr_bin_num + 1
                primary[rid] = curr_bin_num
                bin[curr_bin_num] = {}
                bin[curr_bin_num][first_row['Chunk _Hash']] = first_row['Chunk_Size']
    else:
        curr_min = first_row['Chunk _Hash']
        old_rid = repre[curr_file]
        old_bin_num = primary[old_rid]
        if curr_min < old_rid:
            rid = curr_min
            # update primary:
            primary.pop(old_rid)
            if curr_min in primary.keys():
                rel_bin_num = primary[curr_min]
                bin_value = bin[old_bin_num]
                print("Deleting bin {} ".format(old_bin_num))
                bin.pop(old_bin_num)
                bin[rel_bin_num].update(bin_value)
                print("Bin {} updated ".format(rel_bin_num))
            else:
                primary[curr_min] = old_bin_num
            # update repre:
            repre[curr_file] = curr_min

    # rest of data chunk
    for i in range(1, curr.shape[0]):
        row = curr.iloc[i]
        if row['Whole _File_Hash'] != curr_file:  # Dealing with a different file
            curr_file = row['Whole _File_Hash']
            rid = row['Chunk _Hash']
            repre[curr_file] = rid
            if rid in primary.keys():
                rel_bin_num = primary[rid]
                # if rel_bin_num >= 3256:
                # print(bin[3256])
                bin[rel_bin_num][row['Chunk _Hash']] = row['Chunk_Size']
            else:  # create a new bin
                curr_bin_num = curr_bin_num + 1
                primary[rid] = curr_bin_num
                bin[curr_bin_num] = {}
                bin[curr_bin_num][row['Chunk _Hash']] = row['Chunk_Size']
        else:  # put chunk directly into bin
            # put chunk into correct bin
            rel_bin_num = primary[rid]
            bin[rel_bin_num][row['Chunk _Hash']] = row['Chunk_Size']

# compute memory overhead
process = psutil.Process(os.getpid())
mem = process.memory_info()[0] / float(2 ** 20)

# compute deduplication ratio
sum = 0
for dic in bin.values():
    for v in dic.values():
        sum = sum + v
dedup_ratio = 149090736688 / sum
print("--- %s seconds ---" % (time.time() - start_time))
print("Deduplication ratio: {}".format(dedup_ratio))
print("Memory Usage: {} mib".format(mem))
