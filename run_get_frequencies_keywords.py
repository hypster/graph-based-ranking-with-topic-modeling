import numpy as np
from _collections import  defaultdict
import spacy
import pickle
print("loading language model...")
nlp = spacy.load('nl_core_news_sm')
from sklearn.datasets import load_files
path = '/Users/septem/Downloads/companies2/nl/'
data = load_files(path, encoding='latin1', load_content=False)


frequencies_keywords = []
print("tokenize text...")
for filename in data.filenames:
    frequencies = defaultdict(int)
    print("processing filename: ", filename)
    with open(filename, 'r', errors='ignore') as f:
        doc = nlp(f.read())
        for word in doc:
            if word.pos_ == 'NOUN' or word.pos_ == 'PROPN':
                frequencies[word.lemma_] +=1
    frequencies_keywords.append(frequencies)

with open('./data/frequencies_keywords.pkl', 'wb') as f:
    pickle.dump(frequencies_keywords, f)

"""Later added, for ease of later code use, basically add company id as identifier for each document
"""
data = load_files(path, encoding='latin1', load_content=False)
with open('./data/frequencies_keywords.pkl', 'rb') as f:
    keywords = pickle.load(f)


frequencies = {}
for i, filename in enumerate(data.filenames):
    key = filename.split("/")[-1].split(".")[0]
    frequencies[key] = keywords[i]

with open('./data/frequencies_keywords.pkl', 'wb') as f:
    pickle.dump(frequencies, f)


with open('./data/frequencies_keywords.pkl', 'wb') as f:
    pickle.dump(frequencies_keywords, f)
