# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:22:39 2020

@author: xinch
"""
import os
import csv


#def extract(path, files, dataset1, dataset2):
def extract(path, files, dataset):
    total_lines = 2667130717
    features = ["Chunk _Hash", "Chunk_Size", "Whole _File_Hash"]
    # features_1 = ["Whole _File_Hash", "Num_Cks", "Modification_Time"]
    current_file_hash = []
    #ck_num = 0  # total number of chunks in a file
    #md_time = ""  # File modification time
    count = 0
    # with open(dataset, "w") as output1, open(dataset2, "w") as output2:
    with open(dataset, "w") as output1:
        writer1 = csv.writer(output1, lineterminator='\n')
        #writer2 = csv.writer(output2, lineterminator='\n')
        writer1.writerow(features)
        #writer2.writerow(features_1)
        print("0% done")
        for file in files:
            if file.endswith(".txt"):
                file_in = path + "/" + file
                inputf = open(file_in, "r")
                for line in inputf:
                    #if line.startswith("Chunks"):
                        #ck_num = int(line.split(":")[1])
                    #elif ck_num > 0 and line.startswith("Modification"):  # skip files with no chunks
                        #md_time = str(line.split(": ")[1])  # don't remove space after colon
                       # md_time = md_time.split(' ', 1)[1]
                    #elif line.startswith("Chunk Hash"):
                    if line.startswith("Chunk Hash"):
                        curr_line = next(inputf, None)  # skip the current line
                        while curr_line.strip():  # while the following lines are non-empty,
                            # record all chunk hashes
                            nlst = curr_line.split()
                            current_file_hash.append((str(nlst[0]).replace(":", "").strip(), nlst[1].strip()))
                            curr_line = next(inputf, None)
                            count = count + 1
                    elif line.startswith("Whole"):
                        wlst = line.split(":")
                        wfc = str(wlst[1].strip())
                        # Writing
                        for fh in current_file_hash:
                            writer1.writerow([fh[0], fh[1], wfc])
                            current_file_hash = []
                        #writer2.writerow([wfc, ck_num, md_time])
                    count = count + 1
                    if count % 26671300 == 0:
                        print("{:.1f}% Done".format(100 * (count / total_lines)))
        inputf.close()


def main():
    path = "C:/Users/xinch/Documents/298Proj/data/2012-user3And5/User3/tar/extract/2kb/txt"
    data1 = "user03_chunks.csv"
    #data2 = "user05_files.csv"
    all_files = os.listdir(path)
    #extract(path, all_files, data1, data2)
    extract(path, all_files, data1)
    print("Done !!")


if __name__ == "__main__":
    main()
