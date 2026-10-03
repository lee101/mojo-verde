"""Input validation helpers, ported from ``verde.base.utils``."""

import numpy as np


def check_data(data):
    """Make *data* a tuple, wrapping a single array in a 1-element tuple."""
    if not isinstance(data, tuple):
        data = (data,)
    return data


def check_coordinates(coordinates):
    """Check that the coordinate arrays are the same shape."""
    shapes = [coord.shape for coord in coordinates]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError(
            "Coordinate arrays must have the same shape. Coordinate shapes: "
            f"{shapes}"
        )
    return coordinates


def check_fit_input(coordinates, data, weights, unpack=True):
    """Validate the inputs to the fit method of gridders."""
    data = check_data(data)
    weights = check_data(weights)
    coordinates = check_coordinates(coordinates)
    if any(i.shape != coordinates[0].shape for i in data):
        raise ValueError(
            "Data arrays must have the same shape "
            f"{coordinates[0].shape} as coordinates. Data shapes: "
            f"{[i.shape for i in data]}."
        )
    if any(w is not None for w in weights):
        if len(weights) != len(data):
            raise ValueError(
                f"Number of data '{len(data)}' and weights '{len(weights)}' "
                "must be equal."
            )
        if any(i.size != j.size for i in weights for j in data):
            raise ValueError("Weights must have the same size as the data array.")
        weights = tuple(np.ravel(i) for i in weights)
    else:
        weights = tuple([None] * len(data))
    if unpack:
        if len(weights) == 1:
            weights = weights[0]
        if len(data) == 1:
            data = data[0]
    return coordinates, data, weights


def n_1d_arrays(arrays, n):
    """The first n elements, converted to arrays and raveled."""
    return tuple(np.ravel(np.atleast_1d(i)) for i in arrays[:n])
