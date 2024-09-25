"""
"""

import numpy as np
import numpy.typing as npt


def principle_components(X: npt.NDArray[np.float64], n_components: int | None = None):
    """
    See: sklearn.decomposition.PCA._fit_full()
    https://github.com/scikit-learn/scikit-learn/blob/baf828ca1/sklearn/decomposition/_pca.py#L465
    """
    X = np.array(X, dtype=float)
    _mean = np.mean(X, axis=0)
    X -= _mean
    U, _, Vt = np.linalg.svd(X, full_matrices=False)
    """See sklearn.utils.extmath.svd_flip"""
    max_abs_cols = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_abs_cols, range(U.shape[1])])
    # U *= signs
    Vt *= signs[:, np.newaxis]
    return Vt[:n_components], _mean


def transform(X, components, mean=None):
    if mean is None:
        mean = np.mean(X, axis=0)
    return np.dot(X - mean, components.T)
