#!/usr/bin/env python

"""Part-Of-Speech Preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs

# Your preprocessing, features construction, and word2vec code.




FILE_PATHS = {"PTB": ("data/train_chars.txt",
                      "data/valid_chars.txt",
                      "data/valid_chars_kaggle.txt",
                      "data/valid_chars_kaggle_answer.txt",
                      "data/test_chars.txt")}
args = {}

char_dict = {}

def get_char(s):
    if s in char_dict:
        return char_dict[s]
    ind = len(char_dict) + 1  # 1-indexed
    char_dict[s] = ind
    return ind

SPACE_CHAR = "<space>"
START_CHAR = "<s>"
END_CHAR = "</s>"

for c in [SPACE_CHAR, START_CHAR, END_CHAR]:
    print c, get_char(c)

def read_chars(file_name):
    with codecs.open(file_name, 'r', encoding="utf-8") as f:
        chars = []
        for line in f:
            for token in line.split():
                chars.append(get_char(token))
    return np.array(chars, dtype=np.int32)

def read_counts(file_name):
    with codecs.open(file_name, 'r', encoding="utf-8") as f:
        f.readline()
        counts = []
        for line in f:
            ind, count = line.split(',')
            counts.append(int(count))
    return np.array(counts, dtype=np.int32)


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        const="PTB", nargs="?", default="PTB",
                        type=str)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    train_file, valid_file, valid_file_seg, valid_file_out, test_file_seg = FILE_PATHS[dataset]

    train_input = read_chars(train_file)
    valid_input = read_chars(valid_file)
    valid_seg_input = read_chars(valid_file_seg)
    valid_seg_output = read_counts(valid_file_out)
    test_seg_input = read_chars(test_file_seg)

    for value, ind in sorted(char_dict.items(), key=lambda v: v[1]):
        print value, ind

    print "train", len(train_input)
    print "valid", len(valid_input)
    print "valid_seg", len(valid_seg_input)
    print "valid_seg_output", len(valid_seg_output)
    print "test_seg", len(test_seg_input)

    filename = args.dataset + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['valid_input'] = valid_input
        f['valid_seg_input'] = valid_seg_input
        f['valid_seg_output'] = valid_seg_output
        f['test_seg_input'] = test_seg_input
        f['space_char'] = np.array([get_char(SPACE_CHAR)], dtype=np.int32)
        f['start_char'] = np.array([get_char(START_CHAR)], dtype=np.int32)
        f['end_char'] = np.array([get_char(END_CHAR)], dtype=np.int32)
        # Treated as words in the other file
        f['nwords'] = np.array([len(char_dict)], dtype=np.int32)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
