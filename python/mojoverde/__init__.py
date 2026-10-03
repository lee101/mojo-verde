"""mojo-verde: a Mojo port of the compute-bound parts of `verde`.

The public names mirror `verde` exactly, so a caller that already knows
Verde needs to learn nothing new. See the README for the covered subset
and the documented divergences.
"""

from .base import BaseGridder
from .blockreduce import BlockMean, BlockReduce
from .chain import Chain
from .coordinates import (
    block_split,
    expanding_window,
    get_region,
    grid_coordinates,
    inside,
    line_coordinates,
    longitude_continuity,
    pad_region,
    profile_coordinates,
    rolling_window,
    scatter_points,
    shape_to_spacing,
    spacing_to_size,
)
from .least_squares import least_squares
from .mask import distance_mask
from .neighbors import KNeighbors
from .spline import Spline
from .synthetic import CheckerBoard
from .utils import variance_to_weights
from .vector import Vector, VectorSpline2D

__version__ = "0.1.0"

__all__ = [
    "BaseGridder",
    "BlockMean",
    "BlockReduce",
    "Chain",
    "CheckerBoard",
    "KNeighbors",
    "Spline",
    "Vector",
    "VectorSpline2D",
    "block_split",
    "distance_mask",
    "expanding_window",
    "get_region",
    "grid_coordinates",
    "inside",
    "least_squares",
    "line_coordinates",
    "longitude_continuity",
    "pad_region",
    "profile_coordinates",
    "rolling_window",
    "scatter_points",
    "shape_to_spacing",
    "spacing_to_size",
    "variance_to_weights",
]
