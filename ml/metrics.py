"""Classification, ranking, regression, and clustering metrics from their definitions.
AUC is the Mann-Whitney rank statistic; silhouette follows Rousseeuw (1987)."""

import numpy as np

from ml.datasets import load_breast_cancer
from ml.preprocessing import standardize


def confusion_matrix(y_true, y_pred, k=None):
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    k = int(max(y_true.max(), y_pred.max())) + 1 if k is None else k
    M = np.zeros((k, k), int)
    np.add.at(M, (y_true, y_pred), 1)  # rows = true, cols = predicted
    return M


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision_recall_f1(y_true, y_pred, average="macro", pos=1):
    """average='binary' scores class `pos`; 'macro' averages per-class scores."""
    C = confusion_matrix(y_true, y_pred)
    tp = np.diag(C).astype(float)
    fp = C.sum(0) - tp
    fn = C.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    if average == "binary":
        return float(prec[pos]), float(rec[pos]), float(f1[pos])
    return float(prec.mean()), float(rec.mean()), float(f1.mean())


def specificity(y_true, y_pred, pos=1):
    """True-negative rate TN / (TN + FP) for a binary problem with positive class `pos`."""
    C = confusion_matrix(y_true, y_pred)
    neg = [i for i in range(len(C)) if i != pos]
    tn = C[np.ix_(neg, neg)].sum()
    fp = C[np.ix_(neg, [pos])].sum()
    return float(tn / (tn + fp)) if tn + fp > 0 else 0.0


def _rankdata(a):
    # ranks with ties averaged (competition-safe for the U statistic)
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    return ranks


def roc_auc(y_true, scores):
    """AUC = P(score(pos) > score(neg)) via the Mann-Whitney U statistic."""
    y = np.asarray(y_true, int)
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rankdata(scores)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def log_loss(y_true, p, eps=1e-15):
    y_true = np.asarray(y_true)
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    if p.ndim == 1:
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y_true)), y_true.astype(int)] = 1
    return float(-np.mean(np.sum(onehot * np.log(p), axis=1)))


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    ss_res = np.sum((y_true - np.asarray(y_pred)) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def silhouette_score(X, labels):
    labels = np.asarray(labels)
    D = np.sqrt(np.maximum(((X[:, None] - X[None]) ** 2).sum(-1), 0))
    uniq = np.unique(labels)
    s = np.zeros(len(X))
    for i in range(len(X)):
        same = labels == labels[i]
        same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        b = min(D[i, labels == c].mean() for c in uniq if c != labels[i])
        s[i] = 0.0 if (same.sum() == 0 or max(a, b) == 0) else (b - a) / max(a, b)
    return float(s.mean())


def purity(labels_pred, labels_true):
    """Fraction correct if each cluster is labelled by its majority true class."""
    total = sum(np.bincount(labels_true[labels_pred == c]).max()
                for c in np.unique(labels_pred))
    return total / len(labels_true)


def adjusted_rand_index(labels_true, labels_pred):
    """Rand index corrected for chance agreement (Hubert & Arabie 1985)."""
    cont = np.zeros((labels_true.max() + 1, labels_pred.max() + 1), int)
    np.add.at(cont, (labels_true, labels_pred), 1)  # contingency table
    comb2 = lambda a: (a * (a - 1) / 2).sum()
    sum_ij = comb2(cont)
    sa, sb = comb2(cont.sum(1)), comb2(cont.sum(0))
    total = len(labels_true) * (len(labels_true) - 1) / 2
    expected = sa * sb / total
    return float((sum_ij - expected) / (0.5 * (sa + sb) - expected))


if __name__ == "__main__":
    X, y = load_breast_cancer()
    Xs, _, _ = standardize(X)
    # trivial diagonal classifier: project on the difference of class means
    w = Xs[y == 1].mean(0) - Xs[y == 0].mean(0)
    score = Xs @ w
    prob = 1 / (1 + np.exp(-(score - score.mean())))
    pred = (prob >= 0.5).astype(int)

    acc = accuracy(y, pred)
    C = confusion_matrix(y, pred)
    p, r, f1 = precision_recall_f1(y, pred, average="binary")
    spec = specificity(y, pred)
    auc = roc_auc(y, score)
    ll = log_loss(y, prob)
    print(f"breast cancer trivial  acc {acc:.3f}  prec {p:.3f}  rec {r:.3f}  spec {spec:.3f}  f1 {f1:.3f}  auc {auc:.3f}  logloss {ll:.3f}")
    print(f"confusion\n{C}")

    # identity checks against brute-force / definitions
    pos, neg = score[y == 1], score[y == 0]
    brute = np.mean((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None]))
    assert abs(auc - brute) < 1e-9
    assert abs(acc - np.trace(C) / C.sum()) < 1e-12
    assert abs(f1 - 2 * p * r / (p + r)) < 1e-12
    assert abs(spec - C[0, 0] / (C[0, 0] + C[0, 1])) < 1e-12

    # purity and adjusted Rand: identical labelings score 1, a random labeling near 0
    random_labels = np.random.default_rng(1).integers(0, 2, len(y))
    assert purity(y, y) == 1.0 and abs(adjusted_rand_index(y, y) - 1.0) < 1e-12
    assert adjusted_rand_index(y, random_labels) < 0.1

    # regression identities on controlled inputs
    yt = np.array([3.0, -0.5, 2.0, 7.0])
    yp = np.array([2.5, 0.0, 2.0, 8.0])
    assert abs(mse(yt, yp) - 0.375) < 1e-12 and abs(mae(yt, yp) - 0.5) < 1e-12
    assert abs(r2(yt, yt) - 1.0) < 1e-12 and abs(r2(yt, np.full(4, yt.mean()))) < 1e-12
    print(f"regression checks  mse {mse(yt, yp):.3f}  mae {mae(yt, yp):.3f}  r2(perfect) 1.000")

    # silhouette: well-separated blobs high, random labels near zero
    rng = np.random.default_rng(0)
    B = np.vstack([rng.normal(0, 0.3, (30, 2)), rng.normal(6, 0.3, (30, 2))])
    lab = np.r_[np.zeros(30), np.ones(30)].astype(int)
    sil = silhouette_score(B, lab)
    sil_rand = silhouette_score(B, rng.integers(0, 2, 60))
    print(f"silhouette  separated {sil:.3f}  random {sil_rand:.3f}")
    assert sil > 0.8 and sil_rand < 0.3
    assert acc > 0.9 and auc > 0.95
    print("metrics ok")
