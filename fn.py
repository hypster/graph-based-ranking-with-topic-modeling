import numpy as np
import spacy
import pickle
nlp = spacy.load('nl_core_news_md')

def cosine_similarity(v,w):
    if v is np.NAN or w is np.NAN:
        return np.NAN
    return np.dot(v,w)/np.linalg.norm(v)/np.linalg.norm(w)


def find_similar_words(v, n):
    scores = []

    keys = [key for key in nlp.vocab.vectors.keys()]
    for i, key in enumerate(keys):
        # print(nlp.vocab[key].text)
        scores.append(cosine_similarity(v, nlp.vocab.vectors[key]))

    scores = np.array(scores)
    indices = np.argsort(scores)
    indices = indices[::-1]
    for i in indices[:n]:
        key = keys[i]
        print(nlp.vocab[key].text, scores[i])

def get_stop_words(k = 200):
    """get stop words specific to the corpus.

    :param k: the threshold for counting as stop word.
    :return: a stop word set.
    """
    with open('./data/stop_words_candidates.pkl', 'rb') as f:
        freq = pickle.load(f)
    return set(k for k, v in freq[:k])

def get_dataframe():
    import pickle
    import pandas as pd
    with open('/Users/septem/Documents/dke/2/ke@work/second_phrase/data/cluster_result.pkl', 'rb') as f:
        cluster_result = pickle.load(f)
    df = pd.read_csv('/Users/septem/Documents/dke/2/ke@work/code_for_ke@work/companies.csv')
    df = df[['BEID','URL']]
    df = df.set_index('BEID')

    indices = []
    # list only indices found in the cluster result
    for key in cluster_result:
        if key != '':
            indices.append(int(key))
    indices = np.array(indices)

    df = df.loc[set(df.index).intersection(indices)]

    df2 = pd.DataFrame(cluster_result,index=[0]).transpose()
    df2 = df2.drop(df2[df2.index == ''].index)
    df2.index = df2.index.astype(np.int32)
    df = df.join(df2)
    df = df.rename(columns={0:'label', 1:'prop'})
    return df


if __name__ == '__main__':
    v = nlp('dog').vector
    find_similar_words(v, 30)


