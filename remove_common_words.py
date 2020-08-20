import spacy
import numpy as np
from sklearn.datasets import load_files

from collections import OrderedDict
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
print("loading language model...")
nlp = spacy.load('nl_core_news_sm')
from sklearn.datasets import load_files
from collections import Counter

path = '/Users/septem/Downloads/companies2/nl/'
data = load_files(path, encoding='latin1', load_content=False)


target_pos = {'NOUN', 'PROPN'}
pattern = re.compile(r'[\d\W]')

def special_stop_words(data, target_pos, pattern):
    counter = Counter()
    for filename in data.filenames:
        with open(filename, 'r', errors='ignore') as f:
            doc = nlp(f.read())
        print("processing doc: %s" %filename)
        words = []
        for word in doc:
            if word.pos_ in target_pos and not pattern.search(word.text):
                words.append(word.lemma_)
        counter.update(words)

    # total = sum(counter.values())
    return counter.most_common()
    # freq = OrderedDict()
    # for k, v in counter:
    #     freq[k] = v/total
    # return freq


def filter_docs(data, target_pos, pattern):
    words_per_doc = []
    for filename in data.filenames:
        with open(filename, 'r', errors='ignore') as f:
            doc = nlp(f.read())
            words = []
            for word in doc:
                if word.pos_ in target_pos and not pattern.search(word.text):
                    words.append(word.lemma_)
            words = " ".join(words)
            words_per_doc.append(words)
    return words_per_doc


print("processing word frequencies")
freq = special_stop_words(data, target_pos, pattern)
print("saving to files")
with open('./data/stop_words_candidates.pkl', 'wb') as f:
    pickle.dump(freq, f)
print("done")
# for i, k in enumerate(freq):
#     print(k, freq[k])
#     if i == 100:
#         break

# filtered_doc = filter_docs(data, target_pos, pattern)
# TfidfVectorizer(words_per_doc, min_df=)



