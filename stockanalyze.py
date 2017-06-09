# -*- coding: utf-8 -*-

import json

from collections import defaultdict

import jieba
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from newsscore import post_score

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC

sns.set(style='whitegrid')

path = '2330.json'

with open(path, encoding='utf-8') as f:
    posts = json.load(f)

words = []
scores = []


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


for idx, post in enumerate(posts):
    d = defaultdict(int)
    content = post['content']
    for w in jieba.cut(content):
        if len(w) > 1 and not isfloat(w):
            d[w] += 1
    if len(d) > 0:
        words.append(d)
        scores.append(post_score(post))

# convert to vectors
dvec = DictVectorizer()
tfidf = TfidfTransformer()
X = tfidf.fit_transform(dvec.fit_transform(words))

svc = LinearSVC()
svc.fit(X, scores)


def display_top_features(weights, names, top_n, select=abs):
    top_features = sorted(zip(weights, names),
                          key=lambda x: select(x[0]), reverse=True)[:top_n]
    top_weights = [x[0] for x in top_features]
    top_names = [x[1] for x in top_features]

    fig, ax = plt.subplots(figsize=(10, 8))
    ind = np.arange(top_n)
    bars = ax.bar(ind, top_weights, color='blue', edgecolor='black')
    for bar, w in zip(bars, top_weights):
        if w < 0:
            bar.set_facecolor('red')

    width = 0.30
    ax.set_xticks(ind + width)
    ax.set_xticklabels(top_names, rotation=45, fontsize=12,
                       fontdict={'fontname': 'Microsoft JhengHei',
                                 'fontsize': 12})

    plt.show(fig)


# top features for posts
display_top_features(svc.coef_[0],
                     dvec.get_feature_names(), 30, select=lambda x: -x)
# top positive features for posts
display_top_features(svc.coef_[0],
                     dvec.get_feature_names(), 30, select=lambda x: x)
