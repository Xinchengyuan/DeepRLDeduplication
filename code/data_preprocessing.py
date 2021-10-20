class DataPreprocessor:
    def __init__(self):
        pass

    """
    Split coming data stream into segment of an average size
    f: current file from data stream:  (file_hash, [(ck1,size1), (ck2,size2),...])
    seglist:
    [whole_file_chunk, 
     (min_chunk, max_chunk, seg_size, [chunk1, chunk2, ...]) # segment 1,
     (min_chunk, max_chunk, seg_size, [chunk1, chunk2, ...]) # segment 2,
     ......
    ]
    return a list of segments from the file f
    """

    @staticmethod
    def segment(f, size):
        seg_list = [f[0]]  # append file fingerprint at first
        #cks = f[1]  # [(ck1, size1), (ck2,size2),...]
        # print(cks)
        curr_cks = []
        temp_size = 0
        if len(f[1]) > 1:
            for elem in f[1] :
                if temp_size + elem[1] <= size:
                    temp_size = temp_size + elem[1]
                    curr_cks.append(int(elem[0], 16))
                else:
                    if not curr_cks:
                        curr_cks.append(int(elem[0], 16))
                    # record last segment
                    max_chunk = max(curr_cks)
                    min_chunk = min(curr_cks)
                    seg_list.append((min_chunk, max_chunk, temp_size, curr_cks))
                    # reset temp size
                    temp_size = int(elem[1])
                    # start new segment
                    curr_cks = [int(elem[0], 16)]

            # insert the remaining segment
            if curr_cks:
                max_chunk = max(curr_cks)
                min_chunk = min(curr_cks)
                seg_list.append((min_chunk, max_chunk, temp_size, curr_cks))
            # seg_list = np.array(seg_list)
        elif f[1]:
            # directly insert the only chunk
            seg_list.append((int(f[1][0][0], 16), int(f[1][0][0], 16), int(f[1][0][1]), [int(f[1][0][0], 16)]))
        return seg_list
