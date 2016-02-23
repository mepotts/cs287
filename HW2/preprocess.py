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

    with codecs.open(data_name, 'r', encoding="utf-8") as f:
        words = [word_to_idx["PADDING"]] * (window_size / 2)
        caps = [2] * (window_size / 2)
        tags = [-1] * (window_size / 2)
        for line in f:
            if line == "\n":
                for i in xrange((window_size-1)/2):
                    words.append(word_to_idx["PADDING"])
                    caps.append(2)
                    # tags.append(None)
                for i in xrange(len(words)-window_size+1):
                    word_features.append(words[i:i+window_size])
                    caps_features.append(caps[i:i+window_size])
                    lbl.append(tags[i+(window_size / 2)])
                words = [word_to_idx["PADDING"]] * (window_size / 2)
                caps = [2] * (window_size / 2)
                tags = [-1] * (window_size / 2)
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
                    tags.append(-1)

    return np.array(word_features, dtype=np.int32), np.array(caps_features, dtype=np.int32), np.array(lbl, dtype=np.int32)


FILE_PATHS = {"PTB": ("data/train.tags.txt",
                      "data/dev.tags.txt",
                      "data/test.tags.txt",
                      "data/tags.dict",
                      "data/glove.6B.50d.txt")}
args = {}


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set", default="PTB", nargs="?",
                        type=str)
    parser.add_argument('vsize', help="Vocabulary size", default=100000, nargs="?",
                        type=str)
    parser.add_argument('wsize', help="Window size", default=5, nargs="?",
                        type=str)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    train, valid, test, tag_dict, embed_file = FILE_PATHS[dataset]
    vocab_size = int(args.vsize)
    window_size = int(args.wsize)

    # get embeddings
    word_to_idx = {}
    embedding_mat = []
    with codecs.open(embed_file, 'r', encoding="utf-8") as f:
        word_to_idx["PADDING"] = 1
        word_to_idx["RARE"] = 2
        n = 3
        for line in f:
            tokens = line.split()
            word = tokens[0]

            row = []
            for token in tokens[1:]:
                row.append(float(token))
            while len(embedding_mat) < n - 1:
                embedding_mat.append([0] * len(row))
            embedding_mat.append(row)
            assert len(embedding_mat) == n

            if word not in word_to_idx:
                word_to_idx[word] = n
                n += 1
            else:
                print word, "duplicate"
            if n > vocab_size:
                break
    embeddings = np.array(embedding_mat, dtype=np.float32)

    # get tag ids
    tag_to_idx = {}
    with codecs.open(tag_dict, 'r', encoding="utf-8") as f:
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

    print('Embeddings:', len(embedding_mat))
    print('Embedding len:', len(embedding_mat[0]))

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
        f['word_embeddings'] = embeddings
        f['nclasses'] = np.array([C], dtype=np.int32)
        f['nwords'] = np.array([V], dtype=np.int32)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
