import spacy
import numpy as np
from sklearn.datasets import load_files

from collections import OrderedDict
import numpy as np
import pickle
from fn import get_dataframe
import re
from fn import get_stop_words


def get_word2pos_map(text, pos_list, pattern, stopwords, min_length=4):
    """Get a dictionary with positional values from spacy doc object.

    Select only the word with target POS and not found in the pattern

    :param text:spacy doc object.
    :param pos_list:a list of target POS.
    :param pattern: regular expression or None.
    :param stopwords: a list of stopwords to filter.
    :return:A dictionary, where key is the lemma and the values are the positions in the text
     for example: {"vervoer": [4,6,19]}.
    """

    map = OrderedDict()
    for i, word in enumerate(text):
        if word.pos_ in pos_list:  # only select word from the target POS.
            # select only word that is neither in the stop word list or found in the pattern list
            if word.lemma_ not in stopwords and len(word.lemma_) >= min_length and (not pattern or not pattern.search(word.text)):
                if word.lemma_ not in map:
                    map[word.lemma_] = []
                map[word.lemma_].append(i)
    return map


def build_keywords(text):
    words = set()
    for i, word in enumerate(text):
        if word.pos_ == 'NOUN' or word.pos_ == 'PROPN':
            words.add(word.lemma_)
    return words


class Graph:
    """Graph object
        The data is stored with adjacency matrix.

        Attributes:
            nv: number of vertices.
            ne: number of edges.
            G: adjacency matrix for undirected weighted graph.
    """
    nv = 0
    ne = 0
    G = None

    def __init__(self, N):
        self.G = np.zeros((N, N), dtype='float')
        self.nv = N

    def insert_edge(self, v, w):
        """insert edge for undirected weighted graph
        """
        self.G[v][w] += 1
        self.G[w][v] += 1

    def __repr__(self):
        return self.G


def build_key2idx(map):
    """build a key to index mapping for array indexing.
    """
    m = {}
    for i, key in enumerate(map):
        m[key] = i
    return m

def build_edges(graph, text, m, window_size=2):
    """build edges for the graph.
    the edge is governed by the window size, if two words are within this size length in the context, then
    insert an edge between the two word vertices

    :param graph: the Graph object
    :param text: the spacy doc object
    :param m: the word to positions map, result of get_word2pos_map
    :param window_size: the context to consider whether two words are connected,
        the context refers to the original document, not the context of the selected words,
        the paper doesn't specify which context
    :return: void
    """
    key2idx = build_key2idx(m)
    for key in m:
        v = key2idx[key]
        for i in m[key]:  # list all positions of occurrences of the token
            for j in range(1, window_size + 1):  # scan through a size of window size
                if i + j > len(text) - 1:
                    break
                if text[i + j].lemma_ in m: #found a nearby word
                    w = key2idx[text[i + j].lemma_]
                    graph.insert_edge(v, w)


def normalize(M):
    """normalize a matrix along column
    """
    sums = np.sum(M, axis=0)
    out = np.copy(M)
    np.divide(M, sums, out=out, where=sums != 0)
    return out


debug = True


def _text_rank(W, iteration=1000, d=0.85, epsilon=1e-6):
    """the iteration step for text rank algorithm

    :param W: the transition matrix
    :param iteration: number of iteration
    :param d: coefficient for smoothing
    :param epsilon: criteria for stopping
    :return: the final state vector of shape (W.shape[1],1)
    """
    s = np.ones((W.shape[1], 1)) #initial state set to all 1
    for i in range(iteration):
        temp = (1 - d) + d * np.dot(W, s)

        # stop when absolute value differences between the old and new state are all below threshold
        flag = np.all(abs(s - temp) < epsilon)

        if debug and i % 20 == 0 or flag:
            # print(temp)
            square_loss = np.sum((s - temp) ** 2)
            if flag:
                print("finished in iteration %d, loss = %f" % (i, square_loss))
            else:
                print("iteration %d loss = %f" % (i, square_loss))

        s = temp

        if flag:
            break

    return s


class Text_rank:
    """Text rank implementation, interface similar to scikit-learn.
    Attributes:
        WINDOW_SIZE: A window for calculating the connection between two words.
        m: the return value of get_word2pos_map, for storing the positions of occurances of each word.
        Gn: normalized transition matrix
        doc: spacy document object
        filter_list: a list contains target POS
        reg_obj: a regular expression object
    """
    WINDOW_SIZE = 2
    m = None
    Gn = None
    doc = None
    filtered_list = None
    reg_obj = None

    def __init__(self, doc, window_size=2, iteration=1000, d=0.85, epsilon=1e-6, filtered_list=['NOUN', 'PROPN'],
                 pattern=None, stopwords=None):
        self.WINDOW_SIZE = window_size
        self.doc = doc
        self.iteration = iteration
        self.d = d
        self.epsilon = epsilon
        self.filtered_list = filtered_list
        self.m = get_word2pos_map(doc, filtered_list, pattern, stopwords)
        graph = Graph(len(self.m))
        build_edges(graph, doc, self.m, self.WINDOW_SIZE)
        self.Gn = normalize(graph.G)

    def fit(self):
        s = _text_rank(self.Gn, self.iteration, self.d, self.epsilon)
        self.s = s.ravel()

    def get_top_keywords(self, k=10):
        """get the top keywords according to the final state s of textrank algorithm

        :param k: the number of keywords to select, default to 10, pass -1 to get all keywords
        :return: a list of (key,value) where key is the keyword and the value is the final weight for the keyword
        """
        keys = [key for key in self.m]
        idx = np.argsort(self.s)[::-1][:k]
        return [(keys[i], self.s[i]) for i in idx]



if __name__ == '__main__':
    nlp = spacy.load('nl_core_news_md')
    path = '/Users/septem/Downloads/companies2/nl/'
    data = load_files(path, encoding='latin1', load_content=False)
    # data.filenames[0]
    path = "/Users/septem/Downloads/companies2/nl/others/32066236.txt"
    df = get_dataframe()

    idx = np.random.choice(df.index)
    print(df.loc[idx]['URL'])
    idx = str(idx)
    # idx = "10094911"
    path = [filename for filename in data.filenames if filename.find(idx) != -1][0]
    pattern = re.compile(r'[\d\W]')  # only keeps words that contain only alphabet characters

    stopwords = get_stop_words(200)
    # print(stopwords)

    with open(path, 'r') as f:
        doc = nlp(f.read()[:1000])
    #select only 200 words

    text_rank = Text_rank(doc, pattern=pattern, window_size=4, stopwords=stopwords)
    text_rank.fit()
    top_keywords = text_rank.get_top_keywords(-1)
    for word in top_keywords:
        print("%s: %f" % (word[0], word[1]))
