from data_preprocessing import DataPreprocessor
from dedup_feature_extractor import featureExtractor

file_path = "/Users/test/Documents/SegDedup/data/user5"
"""
file_path_a = "/Users/test/Documents/SegDedup/data/a"
file_path_b = "/Users/test/Documents/SegDedup/data/b"
file_path_c = "/Users/test/Documents/SegDedup/data/c"
file_path_d = "/Users/test/Documents/SegDedup/data/d"
file_path_e = "/Users/test/Documents/SegDedup/data/e"
file_path_f = "/Users/test/Documents/SegDedup/data/f"
"""

data_Preprocessor = DataPreprocessor()
feature_Extractor = featureExtractor(path=file_path, mode='default')
#feature_Extractor_a = featureExtractor(path=file_path_a, mode='default')
#feature_Extractor_b = featureExtractor(path=file_path_b, mode='default')

#feature_Extractors = [feature_Extractor_a, feature_Extractor_b]
num_segments = 0
segment = ()
#for fe in feature_Extractors:
while not feature_Extractor.done:
    f_next = feature_Extractor.next_file()
    segment = data_Preprocessor.segment(f_next, 12288)
    num_segments = num_segments + len(segment) - 1

print(num_segments)
