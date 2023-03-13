# ml-from-scratch

Classical machine learning implemented from first principles as a reading curriculum.
Each module derives one method from its mathematics, trains it on a real dataset, prints
plain metrics, and asserts a quality bar so the demo doubles as a regression test.

## Scope and policy

The only third-party dependency is NumPy, used purely as an array and linear-algebra
kernel (matrix products, `solve` on systems assembled by hand, Cholesky for whitening).
Every estimator, loss, metric, and optimizer is written out explicitly: no scikit-learn,
SciPy, pandas, or plotting libraries. Where a decomposition *is* the topic (PCA, LDA) the
eigenproblem is solved by a hand-written cyclic-Jacobi routine (`ml/linalg.py`) rather than
a library call; `numpy.linalg.svd` appears only once, as a denoising tool inside t-SNE,
and is flagged there. Randomness is seeded with `numpy.random.default_rng(0)` throughout.

Datasets are downloaded at runtime and cached under `data/` (git-ignored). Synthetic inputs
appear only to verify numerical identities (metric definitions, the eigensolver); every
headline demo uses real data. The shared utilities are `datasets`, `preprocessing`,
`metrics`, and the numerical helper `linalg`; the remaining modules are the algorithms.

Run any module from the repository root:

```
python3 -m ml.<module>
```

Each demo completes in well under three minutes; several subsample and say so in their output.

## Curriculum

| # | module | topic | dataset | measured result | run |
|---|--------|-------|---------|-----------------|-----|
| 1 | datasets | loaders, caching, IDX/zip parsing | all | shapes asserted (iris 150x4, wine 178x13, ...) | `python3 -m ml.datasets` |
| 2 | preprocessing | scaling, one-hot, polynomial features, stratified split/k-fold | iris | 4 -> 14 poly features; folds balanced 10/10/10 | `python3 -m ml.preprocessing` |
| 3 | metrics | accuracy, P/R/specificity/F1, ROC-AUC, log-loss, MSE/MAE/R2, silhouette | breast cancer | AUC equals brute-force rank statistic (< 1e-9) | `python3 -m ml.metrics` |
| 4 | linear_regression | OLS: normal equation and batch gradient descent | housing | both solvers RMSE 4.564, R2 0.763 | `python3 -m ml.linear_regression` |
| 5 | ridge_lasso | ridge closed form; lasso/elastic-net coordinate descent | housing (deg-2) | ridge R2 0.863; lasso 72 -> 3 nonzero as lambda grows | `python3 -m ml.ridge_lasso` |
| 6 | logistic_regression | L2 logistic GD; multinomial softmax | breast cancer, wine | acc 0.986 / AUC 0.998; softmax acc 1.000 | `python3 -m ml.logistic_regression` |
| 7 | perceptron | Rosenblatt update with pocket variant | banknote | test accuracy 0.991 | `python3 -m ml.perceptron` |
| 8 | knn | k-NN classifier and regressor, distance weighting | iris, wine, housing | iris 0.917, wine 0.978; housing R2 0.914 (weighted) | `python3 -m ml.knn` |
| 9 | naive_bayes | Gaussian NB; multinomial NB with own tokenizer | wine, SMS spam | wine 1.000; spam acc 0.982, F1 0.931 | `python3 -m ml.naive_bayes` |
| 10 | decision_tree | CART: gini/entropy, variance reduction | breast cancer, housing | classification 0.958; regression R2 0.786 | `python3 -m ml.decision_tree` |
| 11 | random_forest | bagging + random subspace + OOB estimate | breast cancer, sonar | test 0.993, OOB 0.946 | `python3 -m ml.random_forest` |
| 12 | adaboost | decision stumps, binary SAMME | banknote | test accuracy 0.997 (50 stumps) | `python3 -m ml.adaboost` |
| 13 | gradient_boosting | least-squares and logistic-loss boosting | housing, breast cancer | R2 0.920; acc 0.958 / AUC 0.985 | `python3 -m ml.gradient_boosting` |
| 14 | svm | Pegasos linear; simplified-SMO RBF kernel | banknote, sonar | linear 0.997; kernel 0.806 | `python3 -m ml.svm` |
| 15 | kmeans | k-means++, silhouette model selection, purity/ARI | wine | k=3 chosen; purity 0.966, adjusted Rand 0.897 | `python3 -m ml.kmeans` |
| 16 | gmm | EM for Gaussian mixtures, full covariance | iris | log-likelihood monotone; purity 0.967 | `python3 -m ml.gmm` |
| 17 | hierarchical | agglomerative single/complete/average linkage | wine subset | complete-linkage purity 0.844 | `python3 -m ml.hierarchical` |
| 18 | dbscan | density clustering with noise labels | iris (2 PCs) | 2 clusters, 10 noise; setosa isolated (purity 1.000) | `python3 -m ml.dbscan` |
| 19 | pca | covariance eigendecomposition, reconstruction | breast cancer | 30 -> 2 explains 0.632 of variance | `python3 -m ml.pca` |
| 20 | lda | Fisher discriminant via symmetric whitening | wine | 13 -> 2, nearest-centroid accuracy 0.978 | `python3 -m ml.lda` |
| 21 | tsne | exact t-SNE, per-point perplexity calibration | 500 MNIST digits | 10-NN embedding purity 0.792 (chance ~0.10) | `python3 -m ml.tsne` |
| 22 | recommender | biased matrix factorization by SGD | MovieLens 100k | test RMSE 0.939 | `python3 -m ml.recommender` |

Shared numerical helper: `ml/linalg.py` implements the symmetric eigensolver used by
modules 19 and 20 (`python3 -m ml.linalg` verifies it against a reconstruction identity).

## Data

All files are fetched over HTTP with a User-Agent header and cached in `data/`.

- scikit-learn bundled CSVs (iris, wine, breast cancer): `raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/`
- Jason Brownlee dataset mirror (pima diabetes, banknote, sonar, Boston housing, abalone): `raw.githubusercontent.com/jbrownlee/Datasets/master/`
- SMS Spam Collection (tab-separated): `raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv`
- MovieLens 100k: `files.grouplens.org/datasets/movielens/ml-100k.zip` (uses `u.data`)
- MNIST training images/labels, IDX format: `ossci-datasets.s3.amazonaws.com/mnist/`

The first run of a demo downloads its data (a few seconds; MovieLens and MNIST are the
largest at ~5 MB and ~10 MB); subsequent runs read the cache.

## Dependencies

- Python >= 3.9 (developed on 3.10)
- NumPy (see `requirements.txt`)

## References

- T. Hastie, R. Tibshirani, J. Friedman. *The Elements of Statistical Learning*, 2nd ed., 2009.
- C. Bishop. *Pattern Recognition and Machine Learning*, 2006.
- K. Murphy. *Machine Learning: A Probabilistic Perspective*, 2012.
- F. Rosenblatt. The perceptron. *Psychological Review*, 1958; S. Gallant. Perceptron-based learning (pocket). *IEEE TNN*, 1990.
- L. Breiman, J. Friedman, R. Olshen, C. Stone. *Classification and Regression Trees*, 1984.
- L. Breiman. Random forests. *Machine Learning*, 2001.
- Y. Freund, R. Schapire. A decision-theoretic generalization of on-line learning. *JCSS*, 1997; J. Zhu et al. Multi-class AdaBoost (SAMME). *Statistics and Its Interface*, 2009.
- J. Friedman. Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 2001.
- J. Friedman, T. Hastie, R. Tibshirani. Regularization paths via coordinate descent. *JSS*, 2010.
- J. Platt. Sequential minimal optimization. Microsoft Research TR, 1998; S. Shalev-Shwartz et al. Pegasos. *ICML*, 2007.
- D. Arthur, S. Vassilvitskii. k-means++. *SODA*, 2007; P. Rousseeuw. Silhouettes. *J. Comp. Appl. Math.*, 1987; L. Hubert, P. Arabie. Comparing partitions (adjusted Rand). *J. Classification*, 1985.
- M. Ester, H.-P. Kriegel, J. Sander, X. Xu. DBSCAN. *KDD*, 1996.
- L. van der Maaten, G. Hinton. Visualizing data using t-SNE. *JMLR*, 2008.
- Y. Koren, R. Bell, C. Volinsky. Matrix factorization for recommender systems. *IEEE Computer*, 2009.
- G. Golub, C. Van Loan. *Matrix Computations*, 4th ed., 2013 (Jacobi eigenvalue method).
