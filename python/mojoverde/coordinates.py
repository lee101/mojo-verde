"""Port of ``verde.coordinates``."""

import warnings

import numpy as np
from sklearn.utils import check_random_state

from ._lib import WORKERS, f64, lib
from .base import check_coordinates, n_1d_arrays


def check_region(region):
    """Check that the given region dimensions are valid."""
    if len(region) != 4:
        raise ValueError(f"Invalid region '{region}' (W, E, S, N).")
    w, e, s, n = region
    if not w < e:
        raise ValueError(
            f"Invalid region '{region}' (W, E, S, N). W must be less than E."
        )
    if not s < n:
        raise ValueError(
            f"Invalid region '{region}' (W, E, S, N). S must be less than N."
        )


def get_region(coordinates):
    """The bounding region of the given coordinates."""
    easting = f64(np.ravel(np.atleast_1d(coordinates[0])))
    northing = f64(np.ravel(np.atleast_1d(coordinates[1])))
    out = np.empty(4)
    lib().mv_get_region((easting), (northing), (out), easting.size)
    return (out[0], out[1], out[2], out[3])


def pad_region(region, pad):
    """Extend the borders of a region by the given amount."""
    if np.isscalar(pad):
        pad = (pad, pad)
    w, e, s, n = region
    return (w - pad[1], e + pad[1], s - pad[0], n + pad[0])


def scatter_points(region, size, random_state=None, extra_coords=None):
    """Generate the coordinates for a random scatter of points."""
    check_region(region)
    random = check_random_state(random_state)
    coordinates = []
    for lower, upper in np.array(region).reshape((len(region) // 2, 2)):
        coordinates.append(random.uniform(lower, upper, size))
    if extra_coords is not None:
        for value in np.atleast_1d(extra_coords):
            coordinates.append(np.ones_like(coordinates[0]) * value)
    return tuple(coordinates)


def spacing_to_size(start, stop, spacing, adjust):
    """Convert a spacing to the number of points between start and stop."""
    if adjust not in ["spacing", "region"]:
        raise ValueError(
            f"Invalid value for *adjust* '{adjust}'. Should be 'spacing' or 'region'"
        )
    size = int(round((stop - start) / spacing)) + 1
    if size == 1:
        size += 1
    if adjust == "region":
        stop = start + (size - 1) * spacing
    return size, stop


def shape_to_spacing(region, shape, pixel_register=False):
    """Calculate the spacing of a grid given region and shape."""
    spacing = []
    for i, n_points in enumerate(reversed(shape)):
        if not pixel_register:
            n_points -= 1
        spacing.append((region[2 * i + 1] - region[2 * i]) / n_points)
    return tuple(reversed(spacing))


def line_coordinates(
    start, stop, size=None, spacing=None, adjust="spacing", pixel_register=False
):
    """Generate evenly spaced points between two values."""
    if size is not None and spacing is not None:
        raise ValueError("Both size and spacing provided. Only one is allowed.")
    if spacing is not None:
        size, stop = spacing_to_size(start, stop, spacing, adjust)
    elif pixel_register:
        size = size + 1
    values = np.empty(size, dtype="float64")
    lib().mv_line_coordinates(
        float(start), float(stop), int(size), (values), int(pixel_register)
    )
    if pixel_register:
        # The kernel shifts the node lines in place onto their pixel centres
        # and leaves the last node behind, exactly like the numpy expression
        # upstream uses; the shifted sequence is one element shorter.
        return values[: size - 1]
    return values


def grid_coordinates(
    region,
    shape=None,
    spacing=None,
    adjust="spacing",
    pixel_register=False,
    extra_coords=None,
    meshgrid=True,
):
    """Generate the coordinates for each point on a regular grid."""
    check_region(region)
    if shape is not None and spacing is not None:
        raise ValueError("Both grid shape and spacing provided. Only one is allowed.")
    if shape is None and spacing is None:
        raise ValueError("Either a grid shape or a spacing must be provided.")
    if shape is None:
        shape = (None, None)
        spacing = np.atleast_1d(spacing)
        if len(spacing) == 1:
            spacing = (spacing[0], spacing[0])
        elif len(spacing) > 2:
            raise ValueError(f"Only two values allowed for grid spacing: {spacing}")
    else:
        spacing = (None, None)

    east = line_coordinates(
        region[0],
        region[1],
        size=shape[1],
        spacing=spacing[1],
        adjust=adjust,
        pixel_register=pixel_register,
    )
    north = line_coordinates(
        region[2],
        region[3],
        size=shape[0],
        spacing=spacing[0],
        adjust=adjust,
        pixel_register=pixel_register,
    )
    coordinates = [east, north]
    if meshgrid:
        ne = east.size
        nn = north.size
        east_2d = np.empty(nn * ne, dtype="float64")
        north_2d = np.empty(nn * ne, dtype="float64")
        lib().mv_meshgrid_2d(
            (east), (north), (east_2d), (north_2d), ne, nn, WORKERS
        )
        coordinates = [east_2d.reshape((nn, ne)), north_2d.reshape((nn, ne))]
    if extra_coords is not None:
        if not meshgrid:
            raise ValueError(
                "Using 'meshgrid=False' and 'extra_coords' is "
                "'verde.grid_coordinates' is not allowed."
            )
        for value in np.atleast_1d(extra_coords):
            coordinates.append(np.ones_like(coordinates[0]) * value)
    return tuple(coordinates)


def profile_coordinates(point1, point2, size, extra_coords=None):
    """Coordinates for a profile along a straight line between two points."""
    if size <= 0:
        raise ValueError(f"Invalid profile size '{size}'. Must be > 0.")
    diffs = [i - j for i, j in zip(point2, point1)]
    separation = np.hypot(*diffs)
    distances = np.linspace(0, separation, size)
    angle = np.arctan2(*reversed(diffs))
    coordinates = [
        point1[0] + distances * np.cos(angle),
        point1[1] + distances * np.sin(angle),
    ]
    if extra_coords is not None:
        for value in np.atleast_1d(extra_coords):
            coordinates.append(np.ones_like(coordinates[0]) * value)
    return tuple(coordinates), distances


def inside(coordinates, region):
    """Determine which points fall inside a given region."""
    check_region(region)
    w, e, s, n = region
    easting, northing = (f64(np.ravel(i)) for i in coordinates[:2])
    out = np.empty(easting.size, dtype="uint8")
    lib().mv_inside(
        (easting),
        (northing),
        (out),
        easting.size,
        float(w),
        float(e),
        float(s),
        float(n),
        int(WORKERS),
    )
    return out.view(bool).reshape(np.broadcast(*coordinates[:2]).shape)


def _block_lattice(line_coords):
    """The first centre, the pitch and the count of a block centre lattice.

    Upstream asks a KD-tree for the nearest block centre. Block centres form
    a uniform lattice, so the nearest centre of a point is the same
    round-to-nearest arithmetic, with the two ends clamped.
    """
    east, north = n_1d_arrays(line_coords, 2)
    nb_e = east.size
    nb_n = north.size
    step_e = (east[1] - east[0]) if nb_e > 1 else 0.0
    step_n = (north[1] - north[0]) if nb_n > 1 else 0.0
    return east[0], step_e, nb_e, north[0], step_n, nb_n


def block_split(coordinates, spacing=None, adjust="spacing", region=None, shape=None):
    """Split a region into blocks and label points according to where they fall."""
    coordinates = tuple(np.ravel(np.atleast_1d(i)) for i in coordinates[:2])
    if region is None:
        region = get_region(coordinates)
    # Passing both through is deliberate: grid_coordinates raises the same
    # way upstream does when a caller supplies spacing and shape together.
    grid_kwargs = {
        "spacing": spacing,
        "shape": shape,
        "adjust": adjust,
        "pixel_register": True,
    }
    line_coords = grid_coordinates(region, meshgrid=False, **grid_kwargs)
    c0e, s_east, nb_e, c0n, s_north, nb_n = _block_lattice(line_coords)
    easting, northing = (f64(i) for i in coordinates)
    n = easting.size
    labels = np.empty(n, dtype="int64")
    lib().mv_block_split(
        (easting),
        (northing),
        (labels),
        n,
        float(c0e),
        float(c0n),
        float(s_east),
        float(s_north),
        int(nb_e),
        int(nb_n),
    )
    block_coords = n_1d_arrays(grid_coordinates(region, **grid_kwargs), 2)
    return block_coords, labels


def _chunks(n, per_chunk_slots):
    """How many chunks to split *n* points into.

    One chunk per worker, but never so many that the per-chunk window table
    grows past the cap: that table is nchunks * nwindows entries.
    """
    budget = max(1, (1 << 23) // max(1, per_chunk_slots))
    return max(1, min(4 * WORKERS, n, budget))


def _check_rolling_window_overlap(region, size, shape, spacing):
    """Warn the user if there is no overlap between neighboring windows."""
    if shape is not None:
        ndims = len(shape)
        dimensions = [region[i * ndims + 1] - region[i * ndims] for i in range(ndims)]
        # The - 1 is because we need to divide by the number of intervals, not
        # the number of nodes.
        spacing = tuple(dim / (n - 1) for dim, n in zip(dimensions, shape))
    spacing = np.atleast_1d(spacing)
    if np.any(spacing > size):
        warnings.warn(
            f"Rolling windows do not overlap (size '{size}' and spacing "
            f"'{spacing}'). Some data points may not be included in any "
            "window. Increase size or decrease spacing to avoid this.",
            stacklevel=2,
        )


def rolling_window(
    coordinates, size, spacing=None, shape=None, region=None, adjust="spacing"
):
    """Select points on a rolling (moving) window.

    A window of the given size is moved across the region at a given step
    (specified by *spacing* or *shape*). Returns the indices of points falling
    inside each window step.
    """
    # Check if shape or spacing were passed
    if shape is None and spacing is None:
        raise ValueError("Either a shape or a spacing must be provided.")
    # Select the coordinates after checking to make sure indexing will still
    # work on the ignored coordinates.
    coordinates = check_coordinates(coordinates)[:2]
    shape_ = coordinates[0].shape
    if region is None:
        region = get_region(coordinates)
    # Check if window size is bigger than the minimum dimension of the region
    region_min_width = min(region[1] - region[0], region[3] - region[2])
    if region_min_width < size:
        raise ValueError(
            f"Window size '{size}' is larger "
            + f"than dimensions of the region '{region}'."
        )
    # Calculate the region spanning the centers of the rolling windows
    window_region = [
        dimension + (-1) ** (i % 2) * size / 2 for i, dimension in enumerate(region)
    ]
    _check_rolling_window_overlap(window_region, size, shape, spacing)
    centers = grid_coordinates(
        window_region, spacing=spacing, shape=shape, adjust=adjust
    )
    # The lattice is needed one axis at a time, and the meshgrid carries it
    # in its first row and its first column.
    east_line = np.ascontiguousarray(centers[0][0])
    north_line = np.ascontiguousarray(centers[1][:, 0])
    nw_n, nw_e = centers[0].shape
    # Upstream asks a KD-tree, with p=inf, for the points of each window.
    # Window centres are a uniform lattice, so a point can only be in the
    # windows whose centres are within size/2 of it: that is an index range
    # per axis, and the same distance test decides membership.
    easting, northing = (f64(np.ravel(i)) for i in coordinates)
    n = easting.size
    nwindows = nw_e * nw_n
    half = size / 2
    nchunks = _chunks(n, nwindows)
    chunk_counts = np.zeros(nchunks * nwindows, dtype="int64")
    lib().mv_rolling_window_count(
        (easting),
        (northing),
        (east_line),
        (north_line),
        (chunk_counts),
        n,
        int(nw_e),
        int(nw_n),
        float(half),
        int(nchunks),
        int(WORKERS),
    )
    counts = chunk_counts.reshape(nchunks, nwindows)
    offsets = np.zeros(nwindows + 1, dtype="int64")
    offsets[1:] = np.cumsum(counts.sum(axis=0))
    # Where each chunk starts writing inside each window. The exclusive scan
    # along the chunks is what keeps the points of a window in increasing
    # index order, which is the order query_ball_point returns them in.
    positions = np.zeros((nchunks, nwindows), dtype="int64")
    if nchunks > 1:
        positions[1:] = np.cumsum(counts[:-1], axis=0)
    positions += offsets[:-1]
    flat = np.empty(int(offsets[-1]), dtype="int64")
    lib().mv_rolling_window_fill(
        (easting),
        (northing),
        (east_line),
        (north_line),
        (positions.ravel()),
        (flat),
        n,
        int(nw_e),
        int(nw_n),
        float(half),
        int(nchunks),
        int(WORKERS),
    )
    # Make the indices array the same shape as the center coordinates array.
    # That preserves the information of the number of windows in each
    # dimension. Need to first create an empty array of object type because
    # otherwise numpy tries to use the index tuples as dimensions (even if
    # given ndim=1 explicitly). Can't make it 1D and then reshape because the
    # reshape is ignored for some reason. The workaround is to create the array
    # with the correct shape and assign the values to a raveled view of the
    # array.
    indices = np.empty(centers[0].shape, dtype="object")
    indices.ravel()[:] = [
        np.unravel_index(flat[start:stop], shape=shape_)
        for start, stop in zip(offsets[:-1], offsets[1:])
    ]
    return centers, indices


def expanding_window(coordinates, center, sizes):
    """Select points on windows of changing size around a center point.

    Returns the indices of points falling inside each window, in the order
    given by *sizes*.
    """
    # Select the coordinates after checking to make sure indexing will still
    # work on the ignored coordinates.
    coordinates = check_coordinates(coordinates)[:2]
    shape = coordinates[0].shape
    center = np.atleast_2d(center)
    east, north = (f64(np.ravel(i)) for i in coordinates)
    # Upstream queries a KD-tree with p=inf for a square window. One window
    # means one output list, so a linear scan finds the same points in the
    # same order without building a tree first.
    indices = []
    for size in sizes:
        flat = np.empty(east.size, dtype="int64")
        found = lib().mv_expanding_window_select(
            (east),
            (north),
            (flat),
            east.size,
            float(center[0, 0]),
            float(center[0, 1]),
            float(size) / 2,
        )
        # Convert indices to an array to avoid errors when the index is empty
        # (no points in the window). unravel_index doesn't like empty lists.
        indices.append(np.unravel_index(flat[:found], shape=shape))
    return indices


def _check_geographic_coordinates(coordinates):
    """Check if geographic coordinates are within accepted degrees intervals."""
    longitude, latitude = coordinates[:2]
    if np.any(longitude > 360) or np.any(longitude < -180):
        raise ValueError(
            "Invalid longitude coordinates. They should be < 360 and > -180 degrees."
        )
    if np.any(latitude > 90) or np.any(latitude < -90):
        raise ValueError(
            "Invalid latitude coordinates. They should be < 90 and > -90 degrees."
        )


def _check_geographic_region(region):
    """Check if the region is within accepted geographic intervals."""
    w, e, s, n = region[:4]
    if np.any(np.array([w, e]) > 360) or np.any(np.array([w, e]) < -180):
        raise ValueError(
            f"Invalid region '{region}' (W, E, S, N). "
            "Longitudinal coordinates should be < 360 and > -180 degrees."
        )
    if np.any(np.array([s, n]) > 90) or np.any(np.array([s, n]) < -90):
        raise ValueError(
            f"Invalid region '{region}' (W, E, S, N). "
            "Latitudinal coordinates should be < 90 and > -90 degrees."
        )
    if abs(e - w) > 360:
        raise ValueError(
            f"Invalid region '{region}' (W, E, S, N). "
            "East and West boundaries must not be separated by an angle greater "
            "than 360 degrees."
        )


def longitude_continuity(coordinates, region):
    """Modify coordinates and region boundaries to ensure longitude continuity."""
    w, e, s, n = region[:4]
    _check_geographic_region([w, e, s, n])
    all_globe = np.allclose(abs(e - w), 360)
    interval_360 = True
    w = w % 360
    e = e % 360
    if all_globe:
        w, e = 0, 360
    if w > e:
        interval_360 = False
        e = ((e + 180) % 360) - 180
        w = ((w + 180) % 360) - 180
    region = np.array(region)
    region[:2] = w, e
    if coordinates:
        _check_geographic_coordinates(coordinates)
        longitude = f64(np.ravel(coordinates[0]))
        moved = np.empty(longitude.size, dtype="float64")
        lib().mv_longitude_continuity(
            (longitude), (moved), longitude.size, int(interval_360),
            int(WORKERS),
        )
        coordinates = np.array(coordinates)
        coordinates[0] = moved.reshape(coordinates[0].shape)
        return coordinates, region
    return region
