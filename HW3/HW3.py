import sys
import time
import codecs
import h5py
import numpy as np
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

EMBED_FILE = "embedding4.hdf5"
WORDS_FILE = "data/words.dict"

data = None
word_dict = {}
inv_word_dict = {}

def dot(a, b):
    return np.dot(a, b)

def cosine(a, b):
    return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)

def nearest_neighbors(word, k, fn):
    worddat = data[word]
    l = []
    for i in xrange(len(data)):
        dist = fn(worddat, data[i])
        l.append((dist, i))
    l.sort(reverse=True)
    return l[:k]


def plot_embedding(X, Y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = ((X - x_min) / (x_max - x_min) - 0.5) * 0.9 + 0.5

    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], inv_word_dict[Y[i]],
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontdict={'size': 10})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    fig.canvas.manager.window.attributes('-topmost', 1)


def main(arguments):
    global data
    global word_dict
    with h5py.File(EMBED_FILE, "r") as f:
        data = np.array(f["embedding"])
        print data.shape

    word_dict = {}
    with codecs.open(WORDS_FILE, 'r', encoding="utf-8") as f:
        for line in f:
            index, word, freq = line.split()
            ind = int(index) - 1  # Zero indexed in Python
            word_dict[word] = ind
            inv_word_dict[ind] = word

    print("Done reading")

    plt.ion()
    while True:
        try:
            word = raw_input('Enter a word: ')
        except:
            break
        word_ind = word_dict.get(word, None)
        if word_ind is None:
            print "Not found"
            continue
        for fn in [dot, cosine]:
            print "%s:" % (fn)
            nearest = nearest_neighbors(word_ind, 20, fn)
            for d, w in nearest:
                print "%s: %f" % (inv_word_dict[w], d)
            words = [w for d, w in nearest]

        # LDA
        X = data[words]
        X.flat[::X.shape[1] + 1] += 0.01
        X_lda = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
        plot_embedding(X_lda,
                       words,
                       "Principal Components projection")

        # t-SNE
        X = data[words]
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(X)
        plot_embedding(X_tsne,
                       words,
                       "t-SNE embedding")



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
