"""Principal component analysis as the eigendecomposition of the sample covariance
(ESL sec. 14.5): explained-variance ratios, a 2-D projection, and reconstruction error
on breast cancer. The eigensolver (ml.linalg.symmetric_eig) is implemented from scratch."""

import numpy as np

from ml.datasets import load_breast_cancer
from ml.linalg import symmetric_eig
from ml.preprocessing import standardize


def pca(X, n_components):
    """Return (scores, components, mean, explained_variance_ratio)."""
    mean = X.mean(0)
    Xc = X - mean
    cov = Xc.T @ Xc / (len(X) - 1)  # sample covariance assembled directly
    eigval, eigvec = symmetric_eig(cov)
    evr = eigval / eigval.sum()
    comp = eigvec[:, :n_components].T
    return Xc @ comp.T, comp, mean, evr[:n_components]


def reconstruct(scores, components, mean):
    return scores @ components + mean


if __name__ == "__main__":
    X, _ = load_breast_cancer()
    Xs, _, _ = standardize(X)
    scores, comp, mean, evr = pca(Xs, n_components=2)
    Xhat = reconstruct(scores, comp, mean)
    rel_err = np.linalg.norm(Xs - Xhat) / np.linalg.norm(Xs - Xs.mean(0))
    print(f"breast cancer 30 -> 2   explained variance ratio {evr.round(3).tolist()}  cumulative {evr.sum():.3f}")
    print(f"reconstruction relative error {rel_err:.3f}")

    # variance identity: fraction of variance lost equals 1 - cumulative explained ratio
    assert abs(rel_err ** 2 - (1 - evr.sum())) < 1e-8
    assert evr.sum() > 0.55
    print("pca ok")
