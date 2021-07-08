class DataPreprocessor:
    def __init__(self):
        pass

    """
    Split coming data stream into segment of an average size
    f: current file from data stream:  (file_hash, [(ck1,size1), (ck2,size2),...])
    seglist:
    [[whole_file_chunk, min_chunk, max_chunk, seg_size, [chunk1, chunk2, ...]] # segment 1,
    [whole_file_chunk, min_chunk, max_chunk, seg_size, [chunk1, chunk2, ...]] # segment 2,
     ......
    ]
    return a list of segments from the file f
    """

    @staticmethod
    def segment(f, size):
        seg_list = []
        cks = f[1]  # [(ck1, size1), (ck2,size2),...]

        seg_size = 0
        file_fp = 0
        curr_cks = []
        max_chunk = 0
        min_chunk = 0

        for elem in cks:
            if seg_size <= size:
                seg_size = seg_size + elem[1]
                curr_cks.append(int(elem[0], 16))
            else:  # segment size exceeds limit, then subtract the extra size
                seg_size = seg_size - elem[1]

                # record last segment
                print(curr_cks)
                max_chunk = max(curr_cks)
                min_chunk = min(curr_cks)
                seg_list.append([file_fp, min_chunk, max_chunk, seg_size, curr_cks])

                # start new segment
                # reset segment size, chunk list and starting index
                curr_cks = [int(elem[0], 16)]
                seg_size = elem[1]

        # record remaining segment
        seg_list.append([file_fp, min_chunk, max_chunk, seg_size, curr_cks])
        # seg_list = np.array(seg_list)
        return seg_list


