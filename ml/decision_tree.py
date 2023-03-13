"""CART decision trees (Breiman et al. 1984): classification by gini/entropy and
regression by variance reduction, with depth and split-size limits. Reused by the
ensemble modules via the max_features random-subspace option."""

import numpy as np

from ml.datasets import load_breast_cancer, load_housing
from ml.metrics import accuracy, mse, r2
from ml.preprocessing import train_test_split


class DecisionTree:
    def __init__(self, task="classification", criterion="gini", max_depth=None,
                 min_samples_split=2, max_features=None, seed=0):
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth if max_depth is not None else 2 ** 31
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        self.k = int(y.max()) + 1 if self.task == "classification" else 0
        self.root = self._grow(np.asarray(X, float), np.asarray(y), 0)
        return self

    def _leaf(self, y):
        if self.task == "classification":
            return {"proba": np.bincount(y, minlength=self.k) / len(y)}
        return {"value": float(y.mean())}

    def _feature_subset(self, d):
        if self.max_features is None:
            return range(d)
        if self.max_features == "sqrt":
            m = max(1, int(np.sqrt(d)))
        elif isinstance(self.max_features, float):
            m = max(1, int(self.max_features * d))
        else:
            m = int(self.max_features)
        return self.rng.choice(d, m, replace=False)

    def _best_split(self, X, y):
        # minimise the child-size-weighted impurity; parent impurity is constant here
        n, d = X.shape
        best_f, best_thr, best_score = None, None, np.inf
        for f in self._feature_subset(d):
            order = np.argsort(X[:, f], kind="mergesort")
            xs, ys = X[order, f], y[order]
            ln = np.arange(1, n + 1)
            rn = n - ln
            if self.task == "classification":
                oh = np.zeros((n, self.k))
                oh[np.arange(n), ys] = 1.0
                cum = np.cumsum(oh, axis=0)
                pl = cum / ln[:, None]
                pr = (cum[-1] - cum) / np.maximum(rn, 1)[:, None]
                if self.criterion == "entropy":
                    il = -np.sum(np.where(pl > 0, pl * np.log2(pl + 1e-12), 0), 1)
                    ir = -np.sum(np.where(pr > 0, pr * np.log2(pr + 1e-12), 0), 1)
                else:
                    il, ir = 1 - (pl ** 2).sum(1), 1 - (pr ** 2).sum(1)
            else:
                cs, cq = np.cumsum(ys), np.cumsum(ys ** 2)
                il = cq / ln - (cs / ln) ** 2
                rs, rq = cs[-1] - cs, cq[-1] - cq
                ir = rq / np.maximum(rn, 1) - (rs / np.maximum(rn, 1)) ** 2
            score = (ln * il + rn * ir) / n
            score[:-1] = np.where(xs[:-1] != xs[1:], score[:-1], np.inf)  # split distinct values only
            score[-1] = np.inf
            i = int(score.argmin())
            if score[i] < best_score:
                best_f, best_thr, best_score = f, 0.5 * (xs[i] + xs[i + 1]), score[i]
        return best_f, best_thr

    def _grow(self, X, y, depth):
        if (len(y) < self.min_samples_split or depth >= self.max_depth
                or (self.task == "classification" and len(np.unique(y)) == 1)
                or (self.task == "regression" and np.ptp(y) == 0)):
            return self._leaf(y)
        f, thr = self._best_split(X, y)
        if f is None:
            return self._leaf(y)
        mask = X[:, f] <= thr
        if mask.all() or (~mask).all():
            return self._leaf(y)
        return {"f": f, "thr": thr,
                "left": self._grow(X[mask], y[mask], depth + 1),
                "right": self._grow(X[~mask], y[~mask], depth + 1)}

    def _route(self, x, node):
        while "f" in node:
            node = node["left"] if x[node["f"]] <= node["thr"] else node["right"]
        return node

    def predict_proba(self, X):
        return np.array([self._route(x, self.root)["proba"] for x in np.asarray(X, float)])

    def predict(self, X):
        X = np.asarray(X, float)
        if self.task == "classification":
            return self.predict_proba(X).argmax(1)
        return np.array([self._route(x, self.root)["value"] for x in X])


if __name__ == "__main__":
    X, y = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, seed=0, stratify=True)
    for crit in ("gini", "entropy"):
        t = DecisionTree("classification", crit, max_depth=4).fit(Xtr, ytr)
        print(f"classification breast cancer  {crit:<7} depth 4  accuracy {accuracy(yte, t.predict(Xte)):.3f}")

    Xc, yc = load_breast_cancer()
    Xctr, Xcte, yctr, ycte = train_test_split(Xc, yc, test_size=0.25, seed=0, stratify=True)
    clf = DecisionTree("classification", "gini", max_depth=4).fit(Xctr, yctr)
    acc = accuracy(ycte, clf.predict(Xcte))

    X, y = load_housing()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, seed=0)
    reg = DecisionTree("regression", max_depth=6, min_samples_split=10).fit(Xtr, ytr)
    pred = reg.predict(Xte)
    r2_reg = r2(yte, pred)
    print(f"regression     housing       depth 6  RMSE {mse(yte, pred) ** 0.5:.3f}  R2 {r2_reg:.3f}")

    assert acc > 0.9
    assert r2_reg > 0.7
    print("decision_tree ok")
