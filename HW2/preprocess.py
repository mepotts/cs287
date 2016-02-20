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


def get_caps_class(word):
    ncaps = sum(1 for c in word if c.isupper())
    if ncaps == 0:
        return 1
    elif ncaps > 1:
        return 2
    elif word[0].isupper():
        return 3
    elif ncaps == 1:
        return 4
    else:
        return 0

def convert_word(word):
    ret = word.lower()
    ret = re.sub("\d+", "NUMBER", ret)
    return ret

def convert_data(data_name, word_to_idx, tag_to_idx, window_size):
    word_features = []
    caps_features = []
    lbl = []
    
    with codecs.open(data_name, 'r', encoding="latin-1") as f:
        words = ["PADDING"] * (window_size / 2)
        caps = [0] * (window_size / 2)
        tags = [None] * (window_size / 2)
        for line in f:
            if line == "\n":
                for i in xrange((window_size-1)/2):
                    words.append("PADDING")
                    caps.append(0)
                    # tags.append(None)
                for i in xrange(len(words)-window_size+1):
                    word_features.append(words[i:i+window_size])
                    caps_features.append(caps[i:i+window_size])
                    lbl.append(tags[i+(window_size / 2)])
                words = []
                caps = []
            else:
                datum = line.split('\t')
                word = datum[2]
                caps.append(get_caps_class(word))
                word = convert_word(word)
                if word in word_to_idx:
                    words.append(word_to_idx[word])
                else:
                    words.append(word_to_idx["RARE"])
                # words.append(convert_word(word))
                if datum[3][:-1] in tag_to_idx:
                    tags.append(tag_to_idx[datum[3][:-1]])
                else:
                    tags.append(None)
    
    return word_features, caps_features, lbl


FILE_PATHS = {"PTB": ("data/train.tags.txt",
                      "data/dev.tags.txt",
                      "data/test.tags.txt",
                      "data/tags.dict")}
args = {}


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str)
    parser.add_argument('vocab', help="Vocabulary",
                        type=str)
    parser.add_argument('vsize', help="Vocabulary size", default=100000,
                        type=str)
    parser.add_argument('wsize', help="Window size", default=5,
                        type=str)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    train, valid, test, tag_dict = FILE_PATHS[dataset]
    window_size = int(args.wsize)
    
    # read from dictionary first
    word_to_idx = {}
    with codecs.open(args.vocab, 'r', encoding="latin-1") as f:
        word_to_idx["PADDING"] = 1
        word_to_idx["RARE"] = 2
        n = 3
        for line in f:
            if n >= int(args.vsize):
                break
            
            word = convert_word(line.split(None, 1)[0])
            
            if word not in word_to_idx:
                word_to_idx[word] = n
                n += 1
        
    # get tag ids
    tag_to_idx = {}
    with codecs.open(tag_dict, 'r', encoding="latin-1") as f:
        for line in f:
            a = line.split()
            tag_to_idx[a[0]] = int(a[1])
    
    
    # Dataset name
    train_words, train_caps, train_output = convert_data(train, word_to_idx, tag_to_idx, window_size)
    print('Train input:', len(train_words))
    
    if valid:
        valid_words, valid_caps, valid_output = convert_data(valid, word_to_idx, tag_to_idx, window_size)
        print('Valid input:', len(valid_words))
    if test:
        test_words, test_caps, _ = convert_data(test, word_to_idx, tag_to_idx, window_size)
        print('Test input:', len(test_words))
    
    V = int(args.vsize)
    print('Vocab size:', V)
    
    C = len(tag_to_idx)
    print('Classes:', C)
    
    filename = args.dataset + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input_word_windows'] = train_words
        f['train_input_cap_windows'] = train_caps
        f['train_output'] = train_output
        if valid:
            f['valid_input_word_windows'] = valid_words
            f['valid_input_cap_windows'] = valid_caps
            f['valid_output'] = valid_output
        if test:
            f['test_input_word_windows'] = test_words
            f['test_input_cap_windows'] = test_caps
        f['nclasses'] = np.array([C], dtype=np.int32)
        f['nwords'] = np.array([V], dtype=np.int32)
        # f['word_embeddings'] = ??


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
