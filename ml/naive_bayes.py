"""Naive Bayes: Gaussian likelihoods for continuous features (wine) and a
multinomial model with add-one smoothing over a bag of words (SMS spam)."""

import re

import numpy as np

from ml.datasets import load_sms, load_wine
from ml.metrics import accuracy, precision_recall_f1
from ml.preprocessing import train_test_split


def gaussian_fit(X, y):
    classes = np.unique(y)
    mu = np.array([X[y == c].mean(0) for c in classes])
    var = np.array([X[y == c].var(0) + 1e-9 for c in classes])  # floor variance
    prior = np.array([np.mean(y == c) for c in classes])
    return classes, mu, var, prior


def gaussian_predict(X, model):
    classes, mu, var, prior = model
    logp = np.empty((len(X), len(classes)))
    for j in range(len(classes)):
        ll = -0.5 * (np.log(2 * np.pi * var[j]) + (X - mu[j]) ** 2 / var[j]).sum(1)
        logp[:, j] = np.log(prior[j]) + ll
    return classes[logp.argmax(1)]


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def multinomial_fit(texts, y, k=2):
    vocab = {}
    for t in texts:
        for w in tokenize(t):
            vocab.setdefault(w, len(vocab))
    V = len(vocab)
    counts = np.zeros((k, V))
    doc = np.zeros(k)
    for text, c in zip(texts, y):
        doc[c] += 1
        for w in tokenize(text):
            counts[c, vocab[w]] += 1
    log_prior = np.log(doc / doc.sum())
    log_lik = np.log((counts + 1) / (counts.sum(1, keepdims=True) + V))  # Laplace add-one
    return vocab, log_prior, log_lik


def multinomial_predict(texts, model):
    vocab, log_prior, log_lik = model
    out = np.empty(len(texts), int)
    for i, text in enumerate(texts):
        score = log_prior.copy()
        for w in tokenize(text):
            j = vocab.get(w)
            if j is not None:
                score += log_lik[:, j]
        out[i] = score.argmax()
    return out


if __name__ == "__main__":
    X, y = load_wine()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, seed=0, stratify=True)
    acc = accuracy(yte, gaussian_predict(Xte, gaussian_fit(Xtr, ytr)))
    print(f"GaussianNB    wine       test accuracy {acc:.3f}")

    texts, ys = load_sms()
    idx = np.arange(len(texts))
    tr, te, ytr, yte = train_test_split(idx, ys, test_size=0.25, seed=0, stratify=True)
    model = multinomial_fit([texts[i] for i in tr], ytr)
    pred = multinomial_predict([texts[i] for i in te], model)
    acc_s = accuracy(yte, pred)
    _, _, f1 = precision_recall_f1(yte, pred, average="binary", pos=1)
    print(f"MultinomialNB SMS spam   test accuracy {acc_s:.3f}  spam F1 {f1:.3f}  vocab {len(model[0])}")

    assert acc > 0.95
    assert acc_s > 0.97 and f1 > 0.9
    print("naive_bayes ok")
