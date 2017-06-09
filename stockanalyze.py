# -*- coding: utf-8 -*-

import json
import random

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

RANDOM_SEED = 20170609
MAX_FAIL_ATTEMPT = 100
path = '2330.json'
training_ratio = 0.7

random.seed(RANDOM_SEED)

with open(path, encoding='utf-8') as f:
    posts = json.load(f)


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

post_words = []
post_scores = []
for post in posts:
    d = defaultdict(int)
    content = post['content']
    for w in jieba.cut(content):
        if len(w) > 1 and not isfloat(w):
            d[w] += 1
    post_words.append(d)
    post_scores.append(post_score(post))


def get_vector(index=[]):
    words = []
    scores = []
    if not len(index):
        index = range(len(posts))
    for i in index:
        words.append(post_words[i])
        scores.append(post_scores[i])
    return words, scores


dvec = DictVectorizer()
tfidf = TfidfTransformer()
svc = LinearSVC()
training_set = int(len(posts) * training_ratio)
post_index = list(range(len(posts)))
random.shuffle(post_index)
# Training
print("[Training] Start Training...")
prev_error = training_set
fail_attempt = 0
while fail_attempt < MAX_FAIL_ATTEMPT:
    sub_training_ratio = 0.5
    sub_training_set = int(training_set * sub_training_ratio)
    index = post_index[:training_set]
    random.shuffle(index)
    words, scores = get_vector(index[:sub_training_set])

    dvec_cur = DictVectorizer()
    tfidf_cur = TfidfTransformer()
    X = tfidf_cur.fit_transform(dvec_cur.fit_transform(words))
    svc_cur = LinearSVC()
    svc_cur.fit(X, scores)

    # verify
    words, scores = get_vector(index[sub_training_set:])
    X2 = tfidf_cur.transform(dvec_cur.transform(words))
    preds = svc_cur.predict(X2)
    error = 0
    for i in range(len(preds)):
        error += abs(preds[i] - scores[i])

    if error < prev_error:
        prev_error = error
        svc = svc_cur
        dvec = dvec_cur
        tfidf = tfidf_cur
        fail_attempt = 0
        print("[Training] Errors: {}/{}".format(error, len(index)))
    else:
        fail_attempt += 1
print("[Training] Training Finished")

# verify
words, scores = get_vector(post_index[training_set:])
X2 = tfidf.transform(dvec.transform(words))
preds = svc.predict(X2)
error = 0
for i in range(len(preds)):
    error += abs(preds[i] - scores[i])

print("Errors: {}/{}".format(error, len(posts) - training_set))


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
