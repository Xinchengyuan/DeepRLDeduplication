from dedup_feature_extractor import featureExtractor
import data_preprocessing as dp

"""
Script to test if segmentation breaks up duplication in one version of snapshot
"""

path = "/Users/test/Documents/SegDedup/data/t/sub"
feature_extractor = featureExtractor(path)
i = 0  # number of chunks that's already in unique chunk container
unique = {}
unique_seg = {}
seg_unique = set()

while not feature_extractor.done:
    f_next = feature_extractor.next_file()
    segments = dp.segment(f_next, 4096)

    chunks = f_next[1]
    for cks in chunks:
        if cks[0] not in unique:
            unique[cks[0]] = cks[1]
        else:
            i = i+1

    for seg in segments:
        min_max = (seg[1], seg[2])
        if min_max not in unique_seg:
            unique_seg[min_max] = seg[3]
        else:
            seg_unique = seg_unique.union(seg[4])



print("i= ", i)
print("j= ", len(seg_unique))
