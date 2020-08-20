import pickle
from collections import OrderedDict
from gensim import corpora
from run_lda import get_dictionary_input
from run_lda import get_selected_keywords
from nltk.corpus import stopwords
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')




stopwords = set(stopwords.words("english"))

with open('./data/lda_model.pkl','rb') as f:
    lda = pickle.load(f)

with open('./data/keywords.pkl','rb') as f:
    keywords = pickle.load(f)
#
# keywords_selected = get_selected_keywords(keywords)
# dictionary = corpora.Dictionary(get_dictionary_input(keywords_selected)) #idx:token
# dictionary_reverse = {v:k for k,v in dictionary.items()} #token:idx

# corpus = [[(dictionary_reverse[k], v) for k,v in keywords_selected[idx] if k not in stopwords] for idx in keywords_selected]
with open('./data/dictionary_reverse.pkl', 'rb') as f:
    dr = pickle.load(f)
corpus = []
for idx in keywords:
    doc = []
    for word in keywords[idx]:
        doc.append((dr[word[0]], word[1]))
    corpus.append(doc)

dist = lda[corpus]
labels = []
for d in dist:
    labels.append(np.argmax([x[1] for x in d]))


key_label_pair = {}
for i, idx in enumerate(keywords):
    key_label_pair[idx] = labels[i]

with open('./data/cluster_result.pkl', 'wb') as f:
    pickle.dump(key_label_pair, f)


freq = defaultdict(int)
for label in labels:
    freq[label] += 1

heights = []
x = []
fig, ax = plt.subplots()
for label in freq:
    heights.append(freq[label])
    x.append(label)
rect = ax.bar(x, heights)
ax.set_xticks(x)
ax.set_xticklabels(x)
autolabel(rect)
plt.show()







