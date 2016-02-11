#!/usr/bin/env python

"""Text Classification Preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs
import collections

BIGRAM_LIMIT = 10000


def line_to_words(line, dataset):
    # Different preprocessing is used for these datasets.
    if dataset in ['SST1', 'SST2']:
        clean_line = clean_str_sst(line.strip())
    else:
        clean_line = clean_str(line.strip())
    words = clean_line.split(' ')
    words = words[1:]
    return words

def get_vocab(file_list, dataset=''):
    """
    Construct index feature dictionary.
    EXTENSION: Change to allow for other word features, or bigrams.
    """
    max_sent_len = 0
    word_to_idx = {}
    bigram_freq = collections.Counter()
    # Start at 2 (1 is padding)
    idx = 2
    for filename in file_list:
        if filename:
            with codecs.open(filename, "r", encoding="latin-1") as f:
                for line in f:
                    words = line_to_words(line, dataset)
                    max_sent_len = max(max_sent_len, len(words))
                    prevword = None
                    for word in words:
                        if word not in word_to_idx:
                            word_to_idx[word] = idx
                            idx += 1
                        if prevword:
                            bigram = (prevword, word)
                            bigram_freq[bigram] += 1
                        prevword = word

    bigram_list = bigram_freq.most_common(BIGRAM_LIMIT)
    bigram_to_idx = {}
    for bigram, freq in bigram_list:
        bigram_to_idx[bigram] = idx
        idx += 1

    return max_sent_len, word_to_idx, bigram_to_idx


def convert_data(data_name, word_to_idx, bigram_to_idx, max_sent_len, dataset, start_padding=0):
    """
    Convert data to padded word index features.
    EXTENSION: Change to allow for other word features, or bigrams.
    """
    features = []
    lbl = []
    length = None
    with codecs.open(data_name, 'r', encoding="latin-1") as f:
        for line in f:
            words = line_to_words(line, dataset)

            row = [word_to_idx[word] for word in words]
            row = list(set(row))
            # end padding
            if len(row) < max_sent_len:
                row.extend([1] * (max_sent_len - len(row)))

            bigrams = [bigram_to_idx.get((prev, word), 1) for (prev, word) in zip(words, words[1:])]
            if len(bigrams) < max_sent_len:
                bigrams.extend([1] * (max_sent_len - len(bigrams)))
            row = row + bigrams

            # start padding
            row = [1]*start_padding + row
            length = len(row)
            features.append(row)

            y = int(line[0]) + 1
            lbl.append(y)
    print("Length of features:", length)
    return np.array(features, dtype=np.int32), np.array(lbl, dtype=np.int32)


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# Different data sets to try.
# Note: TREC has no development set.
# And SUBJ and MPQA have no splits (must use cross-validation)
FILE_PATHS = {"SST1": ("data/stsa.fine.phrases.train",
                       "data/stsa.fine.dev",
                       "data/stsa.fine.test"),
              "SST2": ("data/stsa.binary.phrases.train",
                       "data/stsa.binary.dev",
                       "data/stsa.binary.test"),
              "TREC": ("data/TREC.train.all", None,
                       "data/TREC.test.all"),
              "SUBJ": ("data/subj.all", None, None),
              "MPQA": ("data/mpqa.all", None, None)}
args = {}


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set", const="SST1", nargs="?", default="SST1",
                        type=str)
    parser.add_argument('bigram', help="Use bigrams?", const=1, nargs="?", default=1,
                        type=int)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    print dataset
    train, valid, test = FILE_PATHS[dataset]

    # Features are just the words.
    max_sent_len, word_to_idx, bigram_to_idx = get_vocab([train, valid, test], dataset)
    print('Max sent len:', max_sent_len)
    if not args.bigram:
        bigram_to_idx = {}

    # Dataset name
    train_input, train_output = convert_data(train, word_to_idx, bigram_to_idx, max_sent_len,
                                             dataset)
    print('Train input:', len(train_input))

    if valid:
        valid_input, valid_output = convert_data(valid, word_to_idx, bigram_to_idx, max_sent_len,
                                                 dataset)
        print('Valid input:', len(valid_input))
    if test:
        test_input, _ = convert_data(test, word_to_idx, bigram_to_idx, max_sent_len,
                                 dataset)
        print('Test input:', len(test_input))

    V = len(word_to_idx)
    print('Vocab size:', V)

    BV = len(bigram_to_idx)
    print('Bigram size:', BV)

    ALLV = V + BV + 1  # Padded
    print('Feature size:', ALLV)

    C = np.max(train_output)
    print('Classes:', C)

    filename = args.dataset + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_output'] = train_output
        if valid:
            f['valid_input'] = valid_input
            f['valid_output'] = valid_output
        if test:
            f['test_input'] = test_input
        f['nfeatures'] = np.array([ALLV], dtype=np.int32)
        f['nclasses'] = np.array([C], dtype=np.int32)
    print "Finished!"


if __name__ == '__main__':
    main(sys.argv[1:])
