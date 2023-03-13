"""Numerical primitives assembled from first principles: a symmetric eigensolver by
cyclic Jacobi rotations (Golub & Van Loan, Matrix Computations, sec. 8.4), used by the
PCA and LDA modules instead of a library eigendecomposition."""

import numpy as np


def symmetric_eig(A, tol=1e-12, max_sweeps=100):
    """Eigenvalues (descending) and orthonormal eigenvectors (columns) of a symmetric A."""
    A = np.array(A, float)
    n = A.shape[0]
    V = np.eye(n)
    for _ in range(max_sweeps):
        off = A.copy()
        np.fill_diagonal(off, 0.0)
        if np.abs(off).max() < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(A[p, q]) < tol:
                    continue
                theta = (A[q, q] - A[p, p]) / (2 * A[p, q])
                t = np.sign(theta) / (abs(theta) + np.sqrt(theta ** 2 + 1))  # smaller root of t^2+2 theta t-1
                c = 1 / np.sqrt(t ** 2 + 1)
                s = t * c
                App, Aqq, Apq = A[p, p], A[q, q], A[p, q]
                A[p, p] = c * c * App - 2 * s * c * Apq + s * s * Aqq
                A[q, q] = s * s * App + 2 * s * c * Apq + c * c * Aqq
                A[p, q] = A[q, p] = 0.0
                m = np.ones(n, bool)
                m[[p, q]] = False
                aip, aiq = A[m, p].copy(), A[m, q].copy()
                A[m, p] = A[p, m] = c * aip - s * aiq
                A[m, q] = A[q, m] = s * aip + c * aiq
                vp, vq = V[:, p].copy(), V[:, q].copy()
                V[:, p] = c * vp - s * vq
                V[:, q] = s * vp + c * vq
    w = np.diag(A).copy()
    order = np.argsort(-w)
    return w[order], V[:, order]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    B = rng.normal(size=(6, 6))
    A = B + B.T  # arbitrary symmetric matrix
    w, V = symmetric_eig(A)
    recon = V @ np.diag(w) @ V.T
    print(f"symmetric_eig 6x6  reconstruction max error {np.abs(recon - A).max():.2e}"
          f"  orthonormality {np.abs(V.T @ V - np.eye(6)).max():.2e}")

    assert np.abs(recon - A).max() < 1e-9
    assert np.abs(V.T @ V - np.eye(6)).max() < 1e-9
    assert np.all(np.diff(w) <= 1e-12)  # returned in descending order
    print("linalg ok")
