"""Port of ``verde.spline``."""

import warnings

import numpy as np

from ._lib import WORKERS, f64, lib
from .base import BaseGridder, check_fit_input, n_1d_arrays
from .coordinates import get_region
from .least_squares import least_squares


class Spline(BaseGridder):
    r"""Biharmonic spline interpolation using Green's functions.

    Implements the 2D splines of [Sandwell1987]_. The Green's function for
    an observation point at the origin and a force at :math:`\mathbf{x}` is
    :math:`\|\mathbf{x}\|^2 (\log \|\mathbf{x}\| - 1)`, evaluated here with
    the same two-branch form upstream uses so that coincident points do not
    produce NaNs.

    Parameters
    ----------
    mindist : float or None
        Fudge factor added to the distance before the logarithm. Upstream
        deprecated it; the default ``None`` is the future behaviour.
    damping : None or float
        The positive damping regularization parameter. If None, no
        regularization is used.
    force_coords : None or tuple of arrays
        The easting and northing coordinates of the point forces. If None
        (default), set to the data coordinates used to fit the spline.
    engine : str
        Accepted and ignored, for signature compatibility with upstream.

    Attributes
    ----------
    force_ : array
        The estimated forces that fit the observed data.
    force_coords_ : tuple of arrays
        The coordinates of the point forces.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit.

    """

    def __init__(self, mindist=None, damping=None, force_coords=None, engine="auto"):
        self.damping = damping
        self.force_coords = force_coords
        self.engine = engine
        if mindist is None:
            self.mindist = 0
        else:
            self.mindist = mindist
            warnings.warn(
                "The mindist parameter of verde.Spline is no longer required "
                "and will be removed in Verde 2.0.0. Use the default value "
                "to obtain the future behavior.",
                FutureWarning,
                stacklevel=2,
            )

    def fit(self, coordinates, data, weights=None):
        """Fit the biharmonic spline to the given data."""
        coordinates, data, weights = check_fit_input(coordinates, data, weights)
        warn_weighted_exact_solution(self, weights)
        self.region_ = get_region(coordinates[:2])
        if self.force_coords is None:
            self.force_coords_ = tuple(
                i.copy() for i in n_1d_arrays(coordinates, n=2)
            )
        else:
            self.force_coords_ = self.force_coords
        jacobian = self.jacobian(coordinates[:2], self.force_coords_)
        self.force_ = least_squares(jacobian, data, weights, self.damping)
        return self

    def predict(self, coordinates):
        """Evaluate the estimated spline on the given set of points."""
        shape = np.broadcast(*coordinates[:2]).shape
        force_east, force_north = n_1d_arrays(self.force_coords_, n=2)
        east, north = n_1d_arrays(coordinates, n=2)
        data = np.empty(east.size, dtype="float64")
        lib().mv_predict(
            (f64(east)),
            (f64(north)),
            (f64(force_east)),
            (f64(force_north)),
            (f64(self.force_)),
            (data),
            east.size,
            force_east.size,
            float(self.mindist),
            int(WORKERS),
        )
        return data.reshape(shape)

    def jacobian(self, coordinates, force_coords, dtype="float64"):
        """Make the Jacobian matrix for the 2D biharmonic spline."""
        force_east, force_north = n_1d_arrays(force_coords, n=2)
        east, north = n_1d_arrays(coordinates, n=2)
        n = east.size
        m = force_east.size
        jac = np.empty((n, m), dtype="float64")
        lib().mv_jacobian(
            (f64(east)),
            (f64(north)),
            (f64(force_east)),
            (f64(force_north)),
            (jac),
            n,
            m,
            float(self.mindist),
            int(WORKERS),
        )
        if np.dtype(dtype) != np.float64:
            jac = jac.astype(dtype)
        return jac


def warn_weighted_exact_solution(spline, weights):
    """Warn the user that weights don't work for the exact solution."""
    if weights is not None and spline.damping is None:
        warnings.warn(
            "Weights might have no effect if no regularization is used. "
            "Use damping or specify force positions that are different from the data.",
            stacklevel=2,
        )


def greens_func(east, north, mindist):
    """Calculate the Green's function for the bi-harmonic spline."""
    distance = np.sqrt(east**2 + north**2)
    distance = distance + mindist
    result = np.empty_like(distance)
    small = distance < 1
    big = ~small
    result[small] = distance[small] * (
        np.log(distance[small] ** distance[small]) - distance[small]
    )
    result[big] = distance[big] ** 2 * (np.log(distance[big]) - 1)
    return result
