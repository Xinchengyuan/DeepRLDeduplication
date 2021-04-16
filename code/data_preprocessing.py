from zlib import crc32
import numpy as np
import multiprocessing as mp

"""
Helper function to split list into equal parts
"""


def split_lst(ls, n):
    k, m = divmod(len(ls), n)
    return (ls[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


"""
   Split coming data stream into segment of an average size
   d: current data stream
   seglist:
   [[whole_file_chunk, min_chunk, max_chunk, seg_size, [chunk1, chunk2, ...]] # segment 1,
    [whole_file_chunk, min_chunk, max_chunk, seg_size, [chunk1, chunk2, ...]] # segment 2,
     ......
   ]
   return a dafaframe containing segment info
   """


def segment(d, size):
    seg_list = []
    cks = d['Chunk _Hash'].values
    ck_sizes = d['Chunk_Size'].values
    wfh = d['Whole _File_Hash'].values

    seg_size = 0
    curr_file = 0
    file_fp = 0
    curr_cks = []
    max_chunk = 0
    min_chunk = 0

    for i in range(len(wfh)):

        file_fp = wfh[i]

        if file_fp == curr_file:
            seg_size = seg_size + ck_sizes[i]
            if seg_size <= size:
                curr_cks.append(cks[i])
            else:  # same file, but segment size exceeds limit
                seg_size = seg_size - ck_sizes[i]

                # record last segment
                max_chunk = np.max(curr_cks)
                min_chunk = np.min(curr_cks)
                seg_list.append([file_fp, min_chunk, max_chunk, seg_size, curr_cks])

                # start new segment
                # reset segment size, chunk list and starting index

                curr_cks = [cks[i]]
                seg_size = ck_sizes[i]

        else:  # new file encountered
            if i > 0:
                # record last segment
                max_chunk = np.max(curr_cks)
                min_chunk = np.min(curr_cks)
                seg_list.append([file_fp, min_chunk, max_chunk, seg_size, curr_cks])

            # start new segment
            # reset file fingerprint, segment size, chunk list and starting index
            curr_file = file_fp
            seg_size = ck_sizes[i]
            curr_cks.append(cks[i])

    # record remaining segments before EOF
    seg_list.append([file_fp, min_chunk, max_chunk, seg_size, curr_cks])

    # seg_list = np.array(seg_list)
    return seg_list


def bytes_to_float(b):
    return float(crc32(b) & 0xffffffff) / 2 ** 32


"""
   get all segments
"""


def get_all_segments(data, size):
    print("Segmenting Data")
    segs = []
    for cks in data:
        segs.extend(segment(cks, size))
    return segs


"""
   get sample from a segment list
   sl: segment list
"""


def sample_segment(percentage, sl):
    if percentage == 0:
        return sl
    else:
        sample_ls = []
        for k in range(len(sl)):
            # prob = float(crc32(str(k).encode()) & 0xffffffff) / 2 ** 32
            prob = bytes_to_float(str(k).encode())
            if 0 <= prob < percentage:
                sample_ls.append(sl[k][:4])  # append only first 4 elements as features

    return sample_ls


"""
get sample from entire data chunk
sl: segment list
return the sample and original segments
"""


def get_all_sample(percentage, sl):
    print("sampling")

    if not sl:
        print("No segment list provided")
    else:
        sample = []
        n = mp.cpu_count() - 1
        seg_splt = list(split_lst(sl, n))
        arg_pairs = list(zip([percentage] * n, seg_splt))
        pool = mp.Pool(n)
        sp = pool.starmap(sample_segment, arg_pairs)
        for seg in sp:
            sample.extend(seg)
        # sample.append([r.get() for r in sp])

        # sp = self.sample_segment(sl)
        # pool.close()
        # print(type(sp))
        return sample

