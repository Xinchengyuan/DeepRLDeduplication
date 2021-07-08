import os
import time
from threading import Thread, Lock
from dedup_feature_extractor import featureExtractor
import psutil
from queue import Queue

# data = pd.read_csv("../data/pt.csv", dtype=np.int64, chunksize=5)
primary = {}  # primary index {representative index, bin number}
bins = {}  # {bin num: {ck1:size1, ck2:size2, ck3:size3, ...}}
curr_bin_num = -1
repre = {}  # {file: representative index}
exit_flag = 0
file_queue = Queue()
path = "/Users/test/Documents/SegDedup/data/t"
feature_extractor = featureExtractor(path)


def get_file(lock):
    lock.acquire()
    global file_queue
    file_queue.put(feature_extractor.next_file())
    lock.release()


def process(lock):
    global curr_bin_num, bins, repre
    f_next = ("", [])
    lock.acquire()
    if not file_queue.empty():
        f_next = file_queue.get()
    lock.release()

    if f_next[0] and f_next[1]:
        curr_file = f_next[0]  # current file fingerprint
        rid = min([int(item[0], 16) for item in f_next[1]])
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
                bins[rel_bin_num] = {}
            # load chunks to bins
            for cks in f_next[1]:
                bins[rel_bin_num][cks[0]] = cks[1]


def main_task():
    lock = Lock()
    while not feature_extractor.done:
        thread1 = Thread(target=get_file, args=(lock,))
        thread2 = Thread(target=process, args=(lock,))
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

if __name__ == "__main__":
    start_time = time.time()
    main_task()
    # compute memory overhead
    duration = time.time() - start_time
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    # compute deduplication ratio
    print("Calculating deduplication ratio")
    sum = 0
    for dic in bins.values():
        for v in dic.values():
            sum = sum + v
    dedup_ratio = 149090736688 / sum
    print("--- %s seconds ---" % duration)
    print("Deduplication ratio: {}".format(dedup_ratio))
    print("Memory Usage: {} mib".format(mem))
