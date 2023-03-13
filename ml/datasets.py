"""Runtime download and parsing of the real datasets used across the curriculum.
Files cache under <repo>/data/. Loaders return NumPy arrays (X, y) unless noted."""

import gzip
import struct
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_UA = {"User-Agent": "ml-from-scratch (educational; urllib)"}

_SK = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/"
_JB = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
_SMS = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
_ML100K = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
_MNIST = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def download(url, filename):
    """Fetch url once into DATA_DIR/filename and return its path (cached)."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / filename
    if not path.exists():
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    return path


def _load_sklearn_csv(url, filename):
    # header line "n_samples,n_features,<class names>"; rows are features + int target
    lines = download(url, filename).read_text().splitlines()
    d = int(lines[0].split(",")[1])
    rows = [[float(v) for v in ln.split(",")] for ln in lines[1:] if ln]
    a = np.asarray(rows)
    return a[:, :d], a[:, d].astype(int)


def load_iris():
    return _load_sklearn_csv(_SK + "iris.csv", "iris.csv")


def load_wine():
    return _load_sklearn_csv(_SK + "wine_data.csv", "wine_data.csv")


def load_breast_cancer():
    return _load_sklearn_csv(_SK + "breast_cancer.csv", "breast_cancer.csv")


def load_pima():
    lines = download(_JB + "pima-indians-diabetes.data.csv", "pima.csv").read_text().splitlines()
    a = np.asarray([[float(v) for v in ln.split(",")] for ln in lines if ln])
    return a[:, :-1], a[:, -1].astype(int)


def load_banknote():
    lines = download(_JB + "banknote_authentication.csv", "banknote.csv").read_text().splitlines()
    a = np.asarray([[float(v) for v in ln.split(",")] for ln in lines if ln])
    return a[:, :-1], a[:, -1].astype(int)


def load_sonar():
    # 60 spectral bands, last field R (rock, 0) or M (mine, 1)
    lines = download(_JB + "sonar.csv", "sonar.csv").read_text().splitlines()
    X, y = [], []
    for ln in lines:
        if not ln:
            continue
        f = ln.split(",")
        X.append([float(v) for v in f[:-1]])
        y.append(1 if f[-1].strip() == "M" else 0)
    return np.asarray(X), np.asarray(y)


def load_housing():
    # Boston housing; 13 features, target MEDV (median home value, $1000s)
    lines = download(_JB + "housing.csv", "housing.csv").read_text().splitlines()
    a = np.asarray([[float(v) for v in ln.split(",")] for ln in lines if ln])
    return a[:, :-1], a[:, -1]


def load_abalone():
    # sex M/F/I one-hot (3) + 7 measurements; target = ring count (age proxy)
    lines = download(_JB + "abalone.csv", "abalone.csv").read_text().splitlines()
    sex = {"M": [1, 0, 0], "F": [0, 1, 0], "I": [0, 0, 1]}
    X, y = [], []
    for ln in lines:
        if not ln:
            continue
        f = ln.split(",")
        X.append(sex[f[0]] + [float(v) for v in f[1:-1]])
        y.append(int(f[-1]))
    return np.asarray(X), np.asarray(y)


def load_sms():
    """SMS spam corpus. Returns (texts list[str], y) with ham=0, spam=1."""
    lines = download(_SMS, "sms.tsv").read_text(encoding="utf-8").splitlines()
    texts, y = [], []
    for ln in lines:
        if "\t" not in ln:
            continue
        label, text = ln.split("\t", 1)
        texts.append(text)
        y.append(1 if label == "spam" else 0)
    return texts, np.asarray(y)


def load_movielens():
    """MovieLens 100k. Returns int array (100000, 4): user, item, rating, timestamp."""
    path = download(_ML100K, "ml-100k.zip")
    with zipfile.ZipFile(path) as z:
        raw = z.read("ml-100k/u.data").decode()
    return np.asarray([[int(v) for v in ln.split("\t")] for ln in raw.splitlines() if ln])


def load_mnist(n=None):
    """MNIST training split from IDX. Returns (X uint8 (N,784), y (N,)); first n if given."""
    ip = download(_MNIST + "train-images-idx3-ubyte.gz", "mnist-images.gz")
    lp = download(_MNIST + "train-labels-idx1-ubyte.gz", "mnist-labels.gz")
    with gzip.open(ip, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(), np.uint8).reshape(num, rows * cols)
    with gzip.open(lp, "rb") as f:
        struct.unpack(">II", f.read(8))
        y = np.frombuffer(f.read(), np.uint8)
    if n is not None:
        X, y = X[:n], y[:n]
    return X.copy(), y.copy()


if __name__ == "__main__":
    def balance(y):
        return dict(zip(*[c.tolist() for c in np.unique(y, return_counts=True)]))

    iris = load_iris()
    wine = load_wine()
    bc = load_breast_cancer()
    pima = load_pima()
    bank = load_banknote()
    sonar = load_sonar()
    house = load_housing()
    aba = load_abalone()
    for name, (X, y) in [("iris", iris), ("wine", wine), ("breast_cancer", bc),
                         ("pima", pima), ("banknote", bank), ("sonar", sonar)]:
        print(f"{name:<14} X {X.shape}  classes {balance(y)}")
    for name, (X, y) in [("housing", house), ("abalone", aba)]:
        print(f"{name:<14} X {X.shape}  y range [{y.min():.1f}, {y.max():.1f}] mean {y.mean():.2f}")

    texts, ys = load_sms()
    print(f"{'sms':<14} n {len(texts)}  classes {balance(ys)}")
    ml = load_movielens()
    print(f"{'movielens':<14} ratings {ml.shape[0]}  users {ml[:,0].max()}  items {ml[:,1].max()}")
    Xm, ym = load_mnist(2000)
    print(f"{'mnist':<14} X {Xm.shape}  classes {len(np.unique(ym))}  (subsample 2000)")

    assert iris[0].shape == (150, 4) and set(iris[1]) == {0, 1, 2}
    assert wine[0].shape == (178, 13) and set(wine[1]) == {0, 1, 2}
    assert bc[0].shape == (569, 30) and set(bc[1]) == {0, 1}
    assert pima[0].shape == (768, 8)
    assert bank[0].shape == (1372, 4)
    assert sonar[0].shape == (208, 60)
    assert house[0].shape == (506, 13)
    assert aba[0].shape == (4177, 10)
    assert len(texts) == 5574 and ys.sum() == 747
    assert ml.shape == (100000, 4)
    assert Xm.shape == (2000, 784)
    print("datasets ok")
