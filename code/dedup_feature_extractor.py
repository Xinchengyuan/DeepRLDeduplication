# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:22:39 2020

@author: xinch
"""
import os
# import csv
import subprocess


class featureExtractor():
    """
      Feature Extractor
      Parameters:
         - Path: path to directory containing hash.anon data files
    """

    def __init__(self, path):
        self.path = path
        self.files = sorted(
            [f for f in os.listdir(self.path) if f.endswith('.hash.anon')])  # a queue of file names sorted
        self.file_num = len(self.files)
        self.done = False
        self.info = ""  # self.information text to extract data from

    """ Run Command"""

    def hf_stat(self, file):
        hf_path = '/Users/test/Documents/SegDedup/fs-hasher-0.9.5/hf-stat '  # Path to directory where hf-stat is installed
        cmd = hf_path + '-h -w -f ' + file
        out = subprocess.check_output(cmd, shell=True, text=True)
        # remove empty lines
        out = "".join([s for s in out.strip().splitlines(True) if s.strip()])
        lines = out.split('\n')
        return lines

    """Get next file self.info, return: 
        1. File Hash
        2. Chunk self.info: [(chunk1, size), (chunk2, size2)...] ]
    """

    def next_file(self):
        file_hash = 0
        chunk = []
        wfh = ""
        i = 0

        if not self.info:
            if self.files:
                i = self.file_num - len(self.files) + 1
                print("processing file {n} / {t}" .format(n=i, t=self.file_num))
                # retrieve a new document, which is the first document in queue
                #print(self.files)
                curr_file = self.files.pop(0)
                # parse the file with hf-stat
                complete_path = self.path + "/" + curr_file
                self.info = self.hf_stat(complete_path)
            ## Extract the fingerprint of the next file and chunks contained ##
            else:
                self.done = True
                return "", []
            # delete unuseful lines
        while self.info and not self.info[0].startswith('Chunk Hash'):
            del (self.info[0])

        if self.info:
            # delete the header line "Chunk Hash   Chunk Size (bytes) 	Compression Ratio (tenth)"
            del (self.info[0])
            # load chunk hashes and sizes
            while not self.info[0].startswith('Whole File'):  # Stop before the whole file hash line
                # record all chunk hashes
                temp = self.info.pop(0).split()
                current_chunk = temp[0].replace(":", "").strip()
                current_chunk = hex(int(current_chunk, 16))
                curr_size = int(temp[1])
                chunk.append((current_chunk, curr_size))
            # get whole file chunk
            fc = self.info.pop(0).split(":")
            wfh = hex(int(fc[1], 16))
        return wfh, chunk

    def is_done(self):
        return self.done
"""
def main():
    path = "/Users/test/Documents/SegDedup/data/user5"
    fe = FeatureExtractor(path)
    for i in range(3):
        print(fe.next_file())
    print("Done !!")


if __name__ == "__main__":
    main()
"""
