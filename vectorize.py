import pickle
import pprint
import numpy as np
import spacy
nlp = spacy.load('nl_core_news_md')
with open('./data/frequencies_keywords.pkl', 'rb') as f:
    data = pickle.load(f)
# pprint.pprint(data)
def sort_dict(d):
    keys = []
    counts = []
    for key in d:
        keys.append(key)
        counts.append(d[key])
    indices = np.argsort(counts)[::-1]
    counts = np.array(counts)[indices]
    keys = np.array(keys)[indices]
    res = []
    for i in range(len(keys)):
        res.append((keys[i], counts[i]))
    return res

vectors = []
for index, d in enumerate(data):
    sorted = sort_dict(d)
    sorted = sorted[:20]
    vec = []
    for i, kv in enumerate(sorted):
        k = str(kv[0])
        idx = nlp.vocab.strings[k]
        if idx in nlp.vocab.vectors:
            vec.append(nlp.vocab.vectors[idx])
    vectors.append(vec)

with open('./data/vectors.pkl', 'wb') as f:
    pickle.dump(vectors, f)

