"""Port of the compute-bound parts of ``verde.utils``."""

import numpy as np

from ._lib import f64, lib
from .base import check_data


def variance_to_weights(variance, tol=1e-15, dtype="float64"):
    """Converts data variances to weights for gridding.

    Weights are defined as the inverse of the variance, scaled to the range
    [0, 1], i.e. ``variance.min()/variance``. Any variance smaller than *tol*
    receives a weight of 1 to avoid zero division or blown up weights.

    Parameters
    ----------
    variance : array or tuple of arrays
        The variance of each point. NaNs are converted to zero and therefore
        receive a weight of 1.
    tol : float
        The tolerance, or cutoff threshold, for small variances.
    dtype : str or numpy dtype
        The type of the output weights array.

    Returns
    -------
    weights : array or tuple of arrays
        Data weights in the range [0, 1] with the same shape as *variance*.

    """
    variance = check_data(variance)
    weights = []
    for var in variance:
        shape = np.shape(np.atleast_1d(var))
        var = f64(np.nan_to_num(np.atleast_1d(var), copy=True).ravel())
        w = np.empty(var.size, dtype="float64")
        lib().mv_variance_to_weights((var), (w), var.size, float(tol))
        weights.append(w.astype(dtype, copy=False).reshape(shape))
    if len(weights) == 1:
        return weights[0]
    return tuple(weights)
