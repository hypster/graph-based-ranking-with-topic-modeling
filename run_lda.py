from gensim import corpora, models
import pickle
from collections import OrderedDict
from nltk.corpus import stopwords
from sklearn.datasets import load_files

from collections import defaultdict
import logging
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# logging.debug("test")
import pyLDAvis
from gensim.models import TfidfModel
from gensim.models import CoherenceModel


def get_dictionary_input(corpus):
    """get a format where each doc is a list of words. This is simply to conform
        to gensim.corpora.getDictionary API.

    :param corpus: a dictionary with items the result of running tex trank algorithm,
        for example, {12345: [("foo", 1.2), ("bar",1.1)...]}
    :return: a list where each element is a doc consists of a list of words.
    """
    words = []
    for idx in corpus:
        doc = []
        for word in corpus[idx]:
            # if len(word) == 2 and word[0] not in stopwords:
            doc.append(word[0])
        words.append(doc)
    return words


def get_selected_keywords(keywords, k=10):
    """get only the top k keywords from the text rank algorithm result.

    :param keywords: the full keyword list.
    :param k: the maximum number of keywords to choose from.
    :return: a truncated keyword list.
    """
    keywords_selected = OrderedDict()
    for idx in keywords:
        keywords_selected[idx] = keywords[idx][:k]
    return keywords_selected


if __name__ == '__main__':
    # runs the lda algorithm with the input dictionary and corpus, and number of topics.



    # corpus = [[(dictionary_reverse[k], v) for k, v in keywords[idx] if k not in stopwords] for idx in
    #           keywords]
    # model = TfidfModel(corpus)
    # corpus = model[corpus]
    # with open('./data/keywords.pkl', 'rb') as f:
    #     keywords = pickle.load(f)
    # stopwords = set(stopwords.words("english"))
    # dictionary = corpora.Dictionary([[word[0] for word in keywords[idx] if word not in stopwords] for idx in keywords])
    with open('./data/dictionary.pkl', 'rb') as f:
        dictionary = pickle.load(f)
    # with open('./data/corpus.pkl', 'rb') as f:
    #     corpus = pickle.load(f)
    with open('./data/keywords.pkl', 'rb') as f:
        keywords = pickle.load(f)
    with open('./data/dictionary_reverse.pkl', 'rb') as f:
        dr = pickle.load(f)

    corpus = []
    for idx in keywords:
        doc = []
        for word in keywords[idx]:
            doc.append((dr[word[0]], word[1]))
        corpus.append(doc)
    num_topics = 10
    # for num_topics in range(6,15):
    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, random_state=42, passes=10,chunksize=100, alpha='auto',eta='auto', num_topics = num_topics)
        # for topic in lda.print_topics(num_words=10):
        #     print(topic)

        #
    top_topics = lda.top_topics(corpus)  # , num_words=20)

        # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence (#topic = %d): %.4f.' %(num_topics,avg_topic_coherence))

    # from pprint import pprint
    #
    # pprint(top_topics)

    # for e, values in enumerate(lda.inference(corpus)[0]):
    #     topic_val = 0
    #     topic_id = 0
    #     for tid, val in enumerate(values):
    #         if val > topic_val:
    #             topic_val = val
    #             topic_id = tid
    #     print(topic_id, '->', documents[e])
