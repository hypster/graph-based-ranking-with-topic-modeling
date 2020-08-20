from _collections import defaultdict
import spacy
import pickle
import re
"""run text rank algorithm on the whole corpus
"""
print("loading language model...")
nlp = spacy.load('nl_core_news_sm')
from sklearn.datasets import load_files
from graph_model import Text_rank
from graph_model import get_stop_words
from collections import OrderedDict

path = '/Users/septem/Downloads/companies2/nl/'
data = load_files(path, encoding='latin1', load_content=False)
pattern = re.compile(r'[\d\W]')  # only keeps words that contain only alphabet characters.

stopwords = get_stop_words()
write_path = './data/keywords.pkl'
with open(write_path, 'wb') as wf:
    D = OrderedDict()
    for filename in data.filenames:
        print("processing %s" %filename)
        with open(filename, 'r', errors='ignore') as f:
            doc = nlp(f.read()[:1000])
            model = Text_rank(doc, pattern=pattern, window_size=4, stopwords=stopwords)
            model.fit()
            keywords = model.get_top_keywords(-1)
            key = filename.split("/")[-1].split(".")[0]
            if key not in D:
                D[key] = keywords
    pickle.dump(D, wf)
