import pickle
from sklearn.datasets import load_files
from nltk.corpus import stopwords
from gensim import corpora
from collections import defaultdict
import spacy
nlp = spacy.load('nl_core_news_sm')
with open('./data/keywords.pkl', 'rb') as f:
    keywords = pickle.load(f)

# with open('./data/frequencies_keywords.pkl', 'rb') as f:
#     frequencies = pickle.load(f)

stopwords = set(stopwords.words("english"))
# keywords_selected = get_selected_keywords(keywords, -1)
# gensim dictionary format {idx:token}
# dictionary = corpora.Dictionary(get_dictionary_input(keywords_selected))

dictionary = corpora.Dictionary([[word[0] for word in keywords[idx] if word not in stopwords] for idx in keywords])
with open('./data/dictionary.pkl', 'wb') as f:
    pickle.dump(dictionary, f)
# reverse the gensim dictionary key value pair, format: {token: id}
dictionary_reverse = {v: k for k, v in dictionary.items()}
with open('./data/dictionary_reverse.pkl', 'wb') as f:
    pickle.dump(dictionary_reverse, f)

# BOW model in gensim format, created directly from text rank result instead of go through gensim corpus api
# format is [[(id, weight)...]]

# frequencies[idx][k]
path = '/Users/septem/Downloads/companies2/nl/'
data = load_files(path, encoding='latin1', load_content=False)
texts = {}
corpus = []
for filename in data.filenames:
    with open(filename, 'r', errors='ignore') as f:
        print("processing %s" %filename)
        text = f.read()[:300]
        key = filename.split("/")[-1].split(".")[0]
        di = defaultdict(int)
        doc = nlp(text)
        for word in doc:
            if word.lemma_ in dictionary_reverse and word.lemma_ not in stopwords:
                di[word.lemma_] += 1
        corpus.append([(dictionary_reverse[k], di[k]) for k in di])
with open('./data/corpus.pkl', 'wb') as f:
    pickle.dump(corpus, f)
