#!/usr/bin/env python

"""Language modeling preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs
import itertools

# Your preprocessing, features construction, and word2vec code.




FILE_PATHS = {"PTB": ("data/train.txt",
                      "data/valid.txt",
                      "data/valid_blanks.txt",
                      "data/test_blanks.txt",
                      "data/words.dict")}
args = {}

word_dict = {}
unk = None
start_word = "<s>"
end_word = "</s>"

def read_word_dict(file_name):
    global word_dict
    global unk
    with codecs.open(file_name, 'r', encoding="utf-8") as f:
        for line in f:
            index, word, freq = line.split()
            word_dict[word] = int(index)
    unk = word_dict["<unk>"]


def read_sentences(file_name):
    ngramsize = args.ngramsize
    pref = [start_word] * ngramsize

    features = []
    target = []
    with codecs.open(file_name, 'r', encoding="utf-8") as f:
        for line in f:
            tokens = line.split()
            if not tokens:
                continue
            cur_list = []
            for word in itertools.chain(pref, tokens):
                index = word_dict.get(word, unk)
                if word != start_word:
                    ngram_list = cur_list[-ngramsize:]
                    features.append(ngram_list)
                    target.append(index)
                cur_list.append(index)
    assert len(target) == len(features)
    features = np.array(features, dtype=np.int32)
    target = np.array(target, dtype=np.int32)
    return features, target


def read_blanks(file_name, replace_last):
    ngramsize = args.ngramsize
    pref = [start_word] * ngramsize

    queries = []
    ngrams = []
    with codecs.open(file_name, 'r', encoding="utf-8") as f:
        for line in f:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "Q":
                query_list = []
                for word in tokens[1:]:
                    index = word_dict.get(word, unk)
                    query_list.append(index)
                queries.append(query_list)
            elif tokens[0] == "C":
                cur_list = []
                if replace_last:
                    tokens[-1] = "_1_"
                for word in itertools.chain(pref, tokens[1:]):
                    if word.startswith("_") and word.endswith("_"):
                        ngram_list = cur_list[-ngramsize:]
                        ngrams.append(ngram_list)
                        break
                    index = word_dict.get(word, unk)
                    cur_list.append(index)
            else:
                print "Unrecognized line", line
    assert len(queries) == len(ngrams)
    ngrams = np.array(ngrams, dtype=np.int32)
    queries = np.array(queries, dtype=np.int32)
    return ngrams, queries


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-ngramsize', help="n-gram size", const=2, nargs="?", default=2,
                        type=int)
    parser.add_argument('-dataset', help="Data set", const="PTB", nargs="?", default="PTB",
                        type=str)
    args = parser.parse_args(arguments)
    ngramsize = args.ngramsize
    dataset = args.dataset
    print "ngramsize", ngramsize
    print "dataset", dataset
    train_file, valid_file, valid_blanks_file, test_blanks_file, word_dict_file = FILE_PATHS[dataset]

    read_word_dict(word_dict_file)
    nwords = len(word_dict)
    print "nwords", nwords

    train_input, train_output = read_sentences(train_file)
    valid_input, valid_output = read_sentences(valid_file)
    valid_blanks_input, valid_blanks_options = read_blanks(valid_blanks_file, True)
    test_blanks_input, test_blanks_options = read_blanks(test_blanks_file, False)

    print "train", train_input.shape
    print "valid", valid_input.shape
    print "valid_blanks", valid_blanks_input.shape
    print "test_blanks", test_blanks_input.shape

    filename = args.dataset + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_output'] = train_output
        if valid_file:
            f['valid_input'] = valid_input
            f['valid_output'] = valid_output
        if valid_blanks_file:
            f['valid_blanks_input'] = valid_blanks_input
            f['valid_blanks_options'] = valid_blanks_options
        if test_blanks_file:
            f['test_blanks_input'] = test_blanks_input
            f['test_blanks_options'] = test_blanks_options
        f['nwords'] = np.array([nwords], dtype=np.int32)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
