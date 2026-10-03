"""Port of ``verde.mask``."""

import numpy as np

from .base import n_1d_arrays
from .neighbors import knn_query


def _get_grid_coordinates(coordinates, grid):
    """The coordinates to mask, and the shape to give the mask back in."""
    if coordinates is None and grid is None:
        raise ValueError(
            "Either 'coordinates' or 'grid' must be provided to mask."
        )
    if grid is not None:
        shape = grid["northing"].shape
        coordinates = tuple(
            np.ravel(grid[d]) for d in ("northing", "easting")
        )
        return (coordinates[1], coordinates[0]), shape
    shape = np.broadcast(*coordinates[:2]).shape
    return n_1d_arrays(coordinates, 2), shape


def distance_mask(
    data_coordinates, maxdist, coordinates=None, grid=None, projection=None
):

    """Mask grid points that are too far from the given data points.

    Distances are Euclidean norms. The nearest data point is found with the
    same brute-force scan :class:`~mojoverde.KNeighbors` uses, then compared
    against *maxdist* exactly as upstream does.
    """
    coordinates, shape = _get_grid_coordinates(coordinates, grid)
    if projection is not None:
        data_coordinates = projection(*n_1d_arrays(data_coordinates, 2))
        coordinates = projection(*coordinates)
    distance = knn_query(coordinates, n_1d_arrays(data_coordinates, 2), 1)[0]
    mask = (distance <= maxdist).reshape(shape)
    if grid is not None:
        return grid.where(mask)
    return mask
