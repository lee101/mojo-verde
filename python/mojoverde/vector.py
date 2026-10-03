"""Port of ``verde.vector``."""

import numpy as np

from ._lib import WORKERS, f64, lib
from .base import BaseGridder, check_fit_input, n_1d_arrays
from .coordinates import get_region
from .least_squares import least_squares
from .spline import warn_weighted_exact_solution


class Vector(BaseGridder):
    """Fit an estimator to each component of multi-component vector data.

    Each data component provided to :meth:`fit` is fitted to a separate
    estimator; :meth:`predict` operates on all of them at once.

    Parameters
    ----------
    components : tuple or list
        The estimator/gridder instances used for each component, in order.

    """

    dims = ("northing", "easting", "component")
    data_names = ["east_component", "north_component", "vertical_component"]

    def __init__(self, components):
        self.components = components

    def fit(self, coordinates, data, weights=None):
        """Fit the estimators to the given multi-component data."""
        if not isinstance(data, tuple):
            raise ValueError(
                f"Data must be a tuple of arrays. {type(data)} given."
            )
        if weights is not None and not isinstance(weights, tuple):
            raise ValueError(
                f"Weights must be a tuple of arrays. {type(weights)} given."
            )
        coordinates, data, weights = check_fit_input(coordinates, data, weights)
        self.region_ = get_region(coordinates[:2])
        for estimator, data_comp, weight_comp in zip(
            self.components, data, weights
        ):
            estimator.fit(coordinates, data_comp, weight_comp)
        return self

    def predict(self, coordinates):
        """Evaluate each data component on a set of points."""
        return tuple(comp.predict(coordinates) for comp in self.components)


class VectorSpline2D(BaseGridder):
    r"""Elastically coupled interpolation of 2-component vector data.

    Implements the elastic Green's functions of [SandwellWessel2016]_. The
    Jacobian is segmented into 4 parts, each relating a force component to a
    data component::

        | J_ee  J_ne |*|f_e| = |d_e|
        | J_ne  J_nn | |f_n|   |d_n|

    Parameters
    ----------
    poisson : float
        Poisson's ratio for the medium.
    mindist : float
        Fudge factor for the singularity when a force and a computation
        point are too close.
    damping : None or float
        The positive damping regularization parameter.
    force_coords : None or tuple of arrays
        The coordinates of the point forces. If None, set to the data
        coordinates the first time ``fit`` is called.
    engine : str
        Accepted and ignored, for signature compatibility with upstream.

    Attributes
    ----------
    force_ : array
        The estimated forces that fit the observed data.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit.

    """

    dims = ("northing", "easting", "component")
    data_names = ["east_component", "north_component"]

    def __init__(
        self, poisson=0.5, mindist=10e3, damping=None, force_coords=None, engine="auto"
    ):
        self.poisson = poisson
        self.mindist = mindist
        self.damping = damping
        self.force_coords = force_coords
        self.engine = engine

    def fit(self, coordinates, data, weights=None):
        """Fit the gridder to the given 2-component vector data."""
        coordinates, data, weights = check_fit_input(
            coordinates, data, weights, unpack=False
        )
        self.region_ = get_region(coordinates[:2])
        if any(w is not None for w in weights):
            weights = np.concatenate([i.ravel() for i in weights])
        else:
            weights = None
        warn_weighted_exact_solution(self, weights)
        data = np.concatenate([i.ravel() for i in data])
        # Upstream stores this back into force_coords, not force_coords_,
        # even though the class docstring advertises the trailing
        # underscore; predict reads the same attribute, so mirroring the
        # code rather than the docstring keeps the two interchangeable.
        if self.force_coords is None:
            self.force_coords = tuple(
                i.copy() for i in n_1d_arrays(coordinates, n=2)
            )
        jacobian = self.jacobian(coordinates[:2], self.force_coords)
        self.force_ = least_squares(jacobian, data, weights, self.damping)
        return self

    def predict(self, coordinates):
        """Evaluate the fitted gridder on the given set of points."""
        force_east, force_north = self.force_coords
        east, north = n_1d_arrays(coordinates, n=2)
        cast = np.broadcast(*coordinates[:2])
        npoints = cast.size
        nforces = force_east.size
        vec_east = np.empty(npoints, dtype="float64")
        vec_north = np.empty(npoints, dtype="float64")
        lib().mv_predict_2d(
            (f64(east)),
            (f64(north)),
            (f64(force_east)),
            (f64(force_north)),
            (f64(self.force_)),
            (vec_east),
            (vec_north),
            npoints,
            nforces,
            float(self.mindist),
            float(self.poisson),
            int(WORKERS),
        )
        return tuple(
            comp.reshape(cast.shape) for comp in (vec_east, vec_north)
        )

    def jacobian(self, coordinates, force_coords, dtype="float64"):
        """Make the Jacobian matrix for the 2D coupled elastic deformation."""
        force_east, force_north = n_1d_arrays(force_coords, n=2)
        east, north = n_1d_arrays(coordinates, n=2)


        npoints = east.size
        nforces = force_east.size
        jac = np.empty((npoints * 2, nforces * 2), dtype="float64")
        lib().mv_jacobian_2d(
            (f64(east)),
            (f64(north)),
            (f64(force_east)),
            (f64(force_north)),
            (jac),
            npoints,
            nforces,
            float(self.mindist),
            float(self.poisson),
            int(WORKERS),
        )
        if np.dtype(dtype) != np.float64:
            jac = jac.astype(dtype)
        return jac


def greens_func_2d(east, north, mindist, poisson):
    """Calculate the Green's functions for the 2D elastic case."""
    distance = np.sqrt(east**2 + north**2)
    # The mindist factor helps avoid singular matrices when the force and
    # computation point are too close.
    distance = distance + mindist
    # Pre-compute common terms for the Green's functions of each component.
    ln_r = (3 - poisson) * np.log(distance)
    over_r2 = (1 + poisson) / distance**2
    green_ee = ln_r + over_r2 * north**2
    green_nn = ln_r + over_r2 * east**2
    green_ne = -over_r2 * east * north
    return green_ee, green_nn, green_ne
