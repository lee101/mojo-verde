"""Port of ``verde.neighbors``."""

import warnings

import numpy as np

from ._lib import WORKERS, f64, lib
from .base import BaseGridder, check_fit_input, n_1d_arrays
from .coordinates import get_region


def knn_query(coordinates, data_coordinates, k):
    """The distances and indices of the k nearest points of *data_coordinates*.

    Upstream builds a KD-tree; this is a brute-force scan, which for the
    2D point sets Verde works with needs no tree build. Ties are resolved
    by whichever index the scan reaches first.
    """
    east_q, north_q = (f64(np.ravel(i)) for i in coordinates[:2])
    east_t, north_t = (f64(np.ravel(i)) for i in data_coordinates[:2])
    nq = east_q.size
    nt = east_t.size
    if nt == 0 or nq == 0 or k < 1:
        raise ValueError("Need at least one data point and k >= 1.")
    if k > nt:
        raise ValueError(
            f"k={k} is larger than the number of data points ({nt})."
        )
    dist = np.empty(nq * k, dtype="float64")
    idx = np.empty(nq * k, dtype="int64")
    lib().mv_knn_query(
        (east_q),
        (north_q),
        (east_t),
        (north_t),
        (dist),
        (idx),
        nq,
        nt,
        int(k),
        int(WORKERS),
    )
    return dist.reshape((nq, k)), idx.reshape((nq, k))


class KNeighbors(BaseGridder):
    """Nearest neighbor interpolation.

    Interpolation based on the values of the *k* nearest neighbors of each
    interpolated point, combined with a reduction function that defaults to
    the mean.

    Parameters
    ----------
    k : int
        The number of neighbors to use for each interpolated point.
    reduction : function
        Function that combines the values of the *k* neighbors into a single
        value, called as ``reduction(values, axis=1)``.

    Attributes
    ----------
    data_ : 1D array
        A copy of the input data, used to look up values during prediction.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit.

    """

    def __init__(self, k=1, reduction=np.mean):
        self.k = k
        self.reduction = reduction

    def fit(self, coordinates, data, weights=None):
        """Fit the interpolator to the given data.

        Data weights are not supported and are ignored.
        """
        if weights is not None:
            warnings.warn(
                f"{self.__class__.__name__} does not support weights and they "
                "will be ignored.",
                stacklevel=2,
            )
        coordinates, data, weights = check_fit_input(coordinates, data, weights)
        self.region_ = get_region(coordinates[:2])
        self.data_ = np.asarray(data).ravel().copy()
        self.coords_ = tuple(
            f64(np.ravel(i)) for i in n_1d_arrays(coordinates, n=2)
        )
        return self

    def predict(self, coordinates):
        """Interpolate data on the given set of points."""
        shape = np.broadcast(*coordinates[:2]).shape
        _, indices = knn_query(coordinates, self.coords_, self.k)
        neighbor_values = self.data_[indices.ravel()].reshape(indices.shape)
        data = self.reduction(neighbor_values, axis=1)
        return data.reshape(shape)
