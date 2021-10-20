import os
import time
from dedup_feature_extractor import featureExtractor
import psutil
import matplotlib.pyplot as plt

start_time = time.time()

path = "/Users/test/Documents/SegDedup/data/user5"
feature_extractor = featureExtractor(path=path, mode='sample')
num_snap = feature_extractor.snapshot_num
original_size = 0
unique = {}  # {chunk_hash: chunk_size}
dup_counts = []
i = 0  # number of chunks that's already in unique chunk container

while not feature_extractor.done:
    f_next = feature_extractor.next_file()
    if feature_extractor.new_snapshot:
        # check out i
        dup_counts.append(i)
        # reinitialize i
        i = 0

    chunks = f_next[1]

    original_size = original_size + sum(ck[1] for ck in chunks)
    for cks in chunks:
        if cks[0] not in unique:
            unique[cks[0]] = cks[1]
        else:
            i = i + 1

# check out i of last snapshot
dup_counts.append(i)
# remove first i inserted before starting first snapshot:
del dup_counts[0]

print(len(unique.keys()))
process = psutil.Process(os.getpid())
mem = process.memory_info()[0] / float(2 ** 20)
print(dup_counts)
# compute deduplication ratio
duration = time.time() - start_time
print("Calculating deduplication ratio")
sum = 0
for v in unique.values():
    sum = sum + v
dedup_ratio = 149090736688 / sum
print("--- %s seconds ---" % duration)
print("Deduplication ratio: {}".format(dedup_ratio))
print("Memory Usage: {} mib".format(mem))

# Plot duplicate chunk count vs snapshot num
X = list(range(1, num_snap + 1))
plt.plot(X, dup_counts)
plt.title('Duplicate Chunk Count vs Snapshot Version')
plt.xlabel('Snapshot Version')
plt.ylabel('Duplicate Chunk Count')
plt.show()
