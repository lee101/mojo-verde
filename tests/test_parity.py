"""Every ported function is checked against upstream verde on the same data.

The tolerances are not guesses: the Mojo toolchain contracts multiply-add
into a fused multiply-add where numpy rounds twice, which is worth about
one ULP, and the biharmonic Jacobian has a condition number around 3e7, so
an undamped fit amplifies that into roughly 1e-7 on the estimated forces
and 1e-9 on the predictions. The tests pin those bounds.
"""

import numpy as np
import pytest

import mojoverde as mv

vd = pytest.importorskip("verde")

from verde.base.least_squares import least_squares as verde_least_squares  # noqa: E402

REGION = (0, 10, -5, 5)


@pytest.fixture(scope="module")
def scatter():
    return mv.scatter_points(REGION, 500, random_state=0)


@pytest.fixture(scope="module")
def data(scatter):
    east, north = scatter
    return np.sin(east / 3) * np.cos(north / 4) * 10


@pytest.fixture(scope="module")
def grid():
    return mv.grid_coordinates(REGION, spacing=1)


def rel(ours, theirs, floor=1e-9):
    return np.max(np.abs(ours - theirs) / np.maximum(np.abs(theirs), floor))


# ------------------------------------------------------------- coordinates
def test_get_region(scatter):
    assert mv.get_region(scatter) == pytest.approx(vd.get_region(scatter), abs=0)


def test_line_coordinates_documented_values():
    # The values are the ones printed in verde's own docstring.
    for (start, stop), kwargs, expected in [
        ((0, 5), {"spacing": 2.5}, [0, 2.5, 5]),
        ((0, 10), {"size": 5}, [0, 2.5, 5, 7.5, 10]),
        ((0, 10), {"spacing": 2.4}, [0, 2.5, 5, 7.5, 10]),
        ((0, 10), {"spacing": 2.4, "adjust": "region"}, [0, 2.4, 4.8, 7.2, 9.6]),
        ((0, 10), {"spacing": 2.6}, [0, 2.5, 5, 7.5, 10]),
        ((0, 10), {"spacing": 2.6, "adjust": "region"}, [0, 2.6, 5.2, 7.8, 10.4]),
        ((0, 10), {"spacing": 2.5, "pixel_register": True}, [1.25, 3.75, 6.25, 8.75]),
        ((0, 10), {"size": 5, "pixel_register": True}, [1, 3, 5, 7, 9]),
    ]:
        ours = mv.line_coordinates(start, stop, **kwargs)
        theirs = vd.line_coordinates(start, stop, **kwargs)
        np.testing.assert_allclose(ours, theirs, rtol=1e-14, atol=0)
        np.testing.assert_allclose(ours, expected, rtol=1e-14, atol=0)


def test_grid_coordinates_documented_values():
    east, north = mv.grid_coordinates((0, 5, 0, 10), shape=(5, 3))
    np.testing.assert_allclose(east, np.tile([0, 2.5, 5], (5, 1)), atol=0)
    np.testing.assert_allclose(north[:, 0], [0, 2.5, 5, 7.5, 10], atol=0)
    east, north = mv.grid_coordinates((0, 5, 0, 10), spacing=2.5, pixel_register=True)
    np.testing.assert_allclose(east, np.tile([1.25, 3.75], (4, 1)), atol=0)
    np.testing.assert_allclose(north[:, 0], [1.25, 3.75, 6.25, 8.75], atol=0)
    east, north = mv.grid_coordinates((-5, 1, 0, 10), spacing=(2.5, 1))
    np.testing.assert_allclose(east[0], [-5, -4, -3, -2, -1, 0, 1], atol=0)
    np.testing.assert_allclose(north[:, 0], [0, 2.5, 5, 7.5, 10], atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spacing": 2.5},
        {"spacing": 2.4},
        {"spacing": 2.4, "adjust": "region"},
        {"shape": (5, 3)},
        {"spacing": 2.5, "pixel_register": True},
        {"spacing": 2.5, "meshgrid": False},
        {"spacing": 2.5, "extra_coords": [57, 0.1]},
        {"spacing": (2.5, 1)},
        {"shape": (4, 2), "pixel_register": True},
    ],
)
def test_grid_coordinates_matches_upstream(kwargs):
    ours = mv.grid_coordinates((0, 5, 0, 10), **kwargs)
    theirs = vd.grid_coordinates((0, 5, 0, 10), **kwargs)
    for a, b in zip(ours, theirs):
        np.testing.assert_allclose(a, b, rtol=1e-14, atol=0)


def test_grid_coordinates_rejects_both_shape_and_spacing():
    with pytest.raises(ValueError, match="Both grid shape and spacing"):
        mv.grid_coordinates((0, 5, 0, 10), shape=(5, 3), spacing=2.5)


def test_shape_to_spacing_matches_upstream():
    for region, shape, pixel in (
        ([0, 10, -5, 1], (7, 11), False),
        ([0, 10, -5, 1], (14, 11), False),
        ([0, 10, -5, 1], (7, 21), False),
        ([-0.5, 10.5, -5.5, 1.5], (7, 11), True),
        ([-0.25, 10.25, -5.5, 1.5], (7, 21), True),
    ):
        assert mv.shape_to_spacing(region, shape, pixel) == pytest.approx(
            vd.coordinates.shape_to_spacing(region, shape, pixel)
        )


def test_inside_matches_upstream(scatter):
    east = np.array([1, 2, 3, 4, 5, 6])
    north = np.array([10, 11, 12, 13, 14, 15])
    np.testing.assert_array_equal(
        mv.inside((east, north), [2.5, 5.5, 12, 15]),
        vd.inside((east, north), [2.5, 5.5, 12, 15]),
    )
    east2 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    north2 = np.array([[5, 7, 9], [5, 7, 9], [5, 7, 9]])
    np.testing.assert_array_equal(
        mv.inside((east2, north2), [0.5, 2.5, 6, 9]),
        vd.inside((east2, north2), [0.5, 2.5, 6, 9]),
    )
    np.testing.assert_array_equal(
        mv.inside(scatter, [1, 9, -4, 4]), vd.inside(scatter, [1, 9, -4, 4])
    )


def test_scatter_points_matches_upstream():
    for seed in (0, 3, 17):
        ours = mv.scatter_points(REGION, 50, random_state=seed)
        theirs = vd.scatter_points(REGION, 50, random_state=seed)
        for a, b in zip(ours, theirs):
            np.testing.assert_array_equal(a, b)


def test_pad_region_matches_upstream():
    for pad in (1, (3, 2)):
        assert mv.pad_region((0, 1, -5, -3), pad) == vd.pad_region((0, 1, -5, -3), pad)


def test_profile_coordinates_matches_upstream():
    ours, dist_ours = mv.profile_coordinates((1, 10), (1, 20), size=11)
    theirs, dist_theirs = vd.profile_coordinates((1, 10), (1, 20), size=11)
    for a, b in zip(ours, theirs):
        np.testing.assert_allclose(a, b, rtol=0, atol=0)
    np.testing.assert_array_equal(dist_ours, dist_theirs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spacing": 1.0},
        {"spacing": (2.0, 3.0)},
        {"spacing": 0.7},
        {"spacing": 1.0, "adjust": "region"},
        {"spacing": (2.0, 3.0), "adjust": "region"},
        {"shape": (4, 3)},
        {"shape": (2, 2)},
        {"shape": (4, 3), "adjust": "region"},
    ],
)
def test_block_split_matches_upstream(scatter, kwargs):
    ours_coords, ours_labels = mv.block_split(scatter, **kwargs)
    theirs_coords, theirs_labels = vd.block_split(scatter, **kwargs)
    np.testing.assert_array_equal(ours_labels, theirs_labels)
    for a, b in zip(ours_coords, theirs_coords):
        np.testing.assert_allclose(a, b, rtol=1e-14, atol=0)


def test_block_split_documented_labels():
    coords = mv.grid_coordinates((-5, 0, 5, 10), spacing=1)
    _, labels = mv.block_split(coords, spacing=2.5)
    expected = np.array(
        [
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [2, 2, 2, 3, 3, 3],
            [2, 2, 2, 3, 3, 3],
            [2, 2, 2, 3, 3, 3],
        ]
    )
    np.testing.assert_array_equal(labels.reshape(coords[0].shape), expected)
    block_coords, _ = mv.block_split(coords, shape=(4, 2))
    np.testing.assert_allclose(
        block_coords[0][:4], [-3.75, -1.25, -3.75, -1.25], rtol=1e-14, atol=0
    )


def test_block_split_rejects_both_shape_and_spacing(scatter):
    with pytest.raises(ValueError):
        mv.block_split(scatter, spacing=1.0, shape=(4, 3))


@pytest.mark.parametrize(
    "region,grid_region",
    [
        ([-20, 20, -20, 20], [0, 350, -90, 90]),
        ([-70, -60, -40, -30], [270, 320, -50, -20]),
    ],
)
def test_longitude_continuity_matches_upstream(region, grid_region):
    coords = vd.grid_coordinates(grid_region, spacing=5)
    ours = mv.longitude_continuity((coords[0].copy(), coords[1].copy()), region)
    theirs = vd.longitude_continuity((coords[0].copy(), coords[1].copy()), region)
    np.testing.assert_array_equal(ours[0][0], theirs[0][0])
    np.testing.assert_array_equal(ours[0][1], theirs[0][1])
    np.testing.assert_array_equal(ours[1], theirs[1])


def test_longitude_continuity_region_only():
    np.testing.assert_array_equal(
        mv.longitude_continuity(None, [350, 10, -10, 10]),
        vd.longitude_continuity(None, [350, 10, -10, 10]),
    )


# ------------------------------------------------------------------- mask
def test_distance_mask_documented_values():
    coords = mv.grid_coordinates((0, 5, -10, -4), spacing=1)
    mask = mv.distance_mask((2.5, -7.5), maxdist=2, coordinates=coords)
    expected = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask, expected)
    theirs = vd.distance_mask(
        (2.5, -7.5), maxdist=2, coordinates=vd.grid_coordinates((0, 5, -10, -4), spacing=1)
    )
    np.testing.assert_array_equal(mask, theirs)


def test_distance_mask_with_projection(scatter):
    def projection(easting, northing):
        # A fake geographic-to-Cartesian round trip: the two are compared
        # through the same callable, so only the mask logic is under test.
        return easting * 1000.0, northing * 1000.0

    ours = mv.distance_mask(scatter, 2000.0, coordinates=scatter, projection=projection)
    theirs = vd.distance_mask(
        scatter, 2000.0, coordinates=scatter, projection=projection
    )
    np.testing.assert_array_equal(ours, theirs)


# ------------------------------------------------------------- neighbors
@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_knn_query_matches_kdtree(scatter, k):
    from scipy.spatial import cKDTree

    query = mv.grid_coordinates(REGION, shape=(13, 11))
    points = np.column_stack(
        [np.ravel(query[0]), np.ravel(query[1])]
    )
    ours_dist, ours_idx = mv_neighbors_query(query, scatter, k)
    tree = cKDTree(np.column_stack(scatter))
    their_dist, their_idx = tree.query(points, k=k)
    np.testing.assert_allclose(
        np.ravel(ours_dist), np.ravel(their_dist), rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(np.ravel(ours_idx), np.ravel(their_idx))


def mv_neighbors_query(query, training, k):
    from mojoverde.neighbors import knn_query

    return knn_query((np.ravel(query[0]), np.ravel(query[1])), training, k)


@pytest.mark.parametrize("k", [1, 4, 8])
def test_kneighbors_predict_matches_upstream(scatter, data, k):
    ours = mv.KNeighbors(k=k).fit(scatter, data)
    theirs = vd.KNeighbors(k=k).fit(scatter, data)
    query = mv.grid_coordinates(REGION, shape=(13, 11))
    assert rel(ours.predict(query), theirs.predict(query)) == 0.0


def test_kneighbors_rejects_k_larger_than_data():
    coords = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    model = mv.KNeighbors(k=5).fit(coords, np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="larger than the number of data points"):
        model.predict(coords)


# ------------------------------------------------------------------ utils
def test_variance_to_weights_documented_values():
    np.testing.assert_allclose(
        mv.variance_to_weights([0, 2, 0.2, 1e-16]), [1, 0.1, 1, 1], rtol=1e-15
    )
    np.testing.assert_allclose(mv.variance_to_weights([0, 0, 0, 0]), [1, 1, 1, 1])
    for ours, theirs in zip(
        mv.variance_to_weights(([0, 1, 10], [2, 4.0, 8])),
        vd.variance_to_weights(([0, 1, 10], [2, 4.0, 8])),
    ):
        np.testing.assert_allclose(ours, theirs, rtol=1e-15)


def test_variance_to_weights_handles_nan():
    ours = mv.variance_to_weights([np.nan, 1.0, 4.0])
    theirs = vd.variance_to_weights([np.nan, 1.0, 4.0])
    np.testing.assert_allclose(ours, theirs, rtol=1e-15)


# ------------------------------------------------------------- blockreduce
@pytest.mark.parametrize("spacing", [1.0, 2.0, 2.5])
@pytest.mark.parametrize("center", [False, True])
def test_blockmean_unweighted(scatter, data, spacing, center):
    kwargs = {"spacing": spacing, "center_coordinates": center}
    ours = mv.BlockMean(**kwargs).filter(scatter, data)
    theirs = vd.BlockMean(**kwargs).filter(scatter, data)
    for a, b in zip(ours, theirs):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("spacing", [1.0, 2.5])
def test_blockmean_weighted(scatter, data, spacing):
    rng = np.random.default_rng(7)
    weights = rng.uniform(0.1, 2.0, data.size)
    kwargs = {"spacing": spacing}
    ours = mv.BlockMean(**kwargs).filter(scatter, data, (weights,))
    theirs = vd.BlockMean(**kwargs).filter(scatter, data, (weights,))
    for a, b in zip(ours, theirs):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)


def test_blockmean_uncertainty(scatter, data):
    rng = np.random.default_rng(7)
    weights = rng.uniform(0.1, 2.0, data.size)
    ours = mv.BlockMean(spacing=2.0, uncertainty=True).filter(
        scatter, data, (weights,)
    )
    theirs = vd.BlockMean(spacing=2.0, uncertainty=True).filter(
        scatter, data, (weights,)
    )
    for a, b in zip(ours, theirs):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)


def test_blockmean_two_components(scatter, data):
    rng = np.random.default_rng(7)
    components = (data, data * 2 + 1)
    weights = (rng.uniform(0.1, 2.0, data.size), rng.uniform(0.1, 2.0, data.size))
    for theirs_weights in (None, weights):
        kwargs = {"spacing": 1.5}
        ours = mv.BlockMean(**kwargs).filter(scatter, components, theirs_weights)
        their = vd.BlockMean(**kwargs).filter(scatter, components, theirs_weights)
        for a, b in zip(ours, their):
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("reduction", [np.mean, np.median, np.max, np.min])
def test_blockreduce_reductions(scatter, data, reduction):
    kwargs = {"reduction": reduction, "spacing": 2.0}
    ours = mv.BlockReduce(**kwargs).filter(scatter, data)
    theirs = vd.BlockReduce(**kwargs).filter(scatter, data)
    for a, b in zip(ours, theirs):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)


def test_blockreduce_weighted_average(scatter, data):
    rng = np.random.default_rng(7)
    weights = (rng.uniform(0.1, 2.0, data.size),)
    kwargs = {"reduction": np.average, "spacing": 2.0}
    ours = mv.BlockReduce(**kwargs).filter(scatter, data, weights)
    theirs = vd.BlockReduce(**kwargs).filter(scatter, data, weights)
    for a, b in zip(ours, theirs):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)


def test_blockreduce_rejects_unknown_reduction():
    with pytest.raises(ValueError, match="Unsupported reduction"):
        mv.BlockReduce(reduction=np.ptp).filter((np.array([0.0]), np.array([0.0])), 1.0)


def test_blockmean_requires_weights_for_uncertainty():
    coords = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="Weights are required"):
        mv.BlockMean(uncertainty=True).filter(coords, np.array([1.0, 2.0]))


# ---------------------------------------------------------- least_squares
def test_least_squares_undamped_matches_upstream(scatter, data):
    ours_spline = mv.Spline()
    their_spline = vd.Spline()
    jac_ours = ours_spline.jacobian(scatter, scatter)
    jac_theirs = their_spline.jacobian(scatter, scatter)
    np.testing.assert_allclose(jac_ours, jac_theirs, rtol=1e-12, atol=1e-12)
    ours = mv.least_squares(jac_ours.copy(), data, None, None)
    theirs = verde_least_squares(jac_theirs.copy(), data, None, None)
    # The rank cutoff drops the near-null space, so only the significant
    # components are comparable; the predictions are compared below.
    significant = np.abs(theirs) > 1e-6 * np.abs(theirs).max()
    np.testing.assert_allclose(ours[significant], theirs[significant], rtol=1e-5)


def test_least_squares_damped_matches_upstream(scatter, data):
    rng = np.random.default_rng(2)
    weights = rng.uniform(0.5, 2.0, data.size)
    spline = mv.Spline()
    jacobian = spline.jacobian(scatter, scatter)
    for damping in (1e-3, 1.0, 1e3):
        ours = mv.least_squares(jacobian.copy(), data, weights, damping)
        theirs = verde_least_squares(jacobian.copy(), data, weights, damping)
        # The biharmonic Jacobian has a condition number around 3e7, so the
        # Cholesky solve of the normal equations can only be compared on
        # the scale of the solution, not element by element.
        scale = np.abs(theirs).max()
        assert np.abs(ours - theirs).max() < 1e-6 * scale


def test_least_squares_column_scaling_matches_standard_scaler(scatter, data):
    from sklearn.preprocessing import StandardScaler

    from mojoverde._lib import lib

    jacobian = mv.Spline().jacobian(scatter, scatter)
    nrows, ncols = jacobian.shape
    ours = jacobian.copy()
    scale = np.empty(ncols)
    mean = np.empty(ncols)
    lib().mv_ls_scale_fit(ours, scale, mean, nrows, ncols, 1)
    lib().mv_ls_scale_transform(ours, scale, nrows, ncols, 1)
    scaler = StandardScaler(copy=True, with_mean=False, with_std=True)
    theirs = scaler.fit_transform(jacobian)
    np.testing.assert_allclose(scale, scaler.scale_, rtol=1e-13)
    np.testing.assert_allclose(ours, theirs, rtol=1e-13, atol=1e-14)


# ----------------------------------------------------------------- spline
def test_greens_func_matches_upstream():
    rng = np.random.default_rng(11)
    east = (rng.random(20000) - 0.5) * 20
    north = (rng.random(20000) - 0.5) * 20
    np.testing.assert_allclose(
        mv.spline.greens_func(east, north, 0.0),
        vd.spline.greens_func_numpy(east, north, 0.0),
        rtol=1e-13,
    )


def test_greens_func_is_zero_at_a_coincident_force():
    # The small-distance branch exists precisely so this is finite.
    value = mv.spline.greens_func(np.array([0.0]), np.array([0.0]), 0.0)
    assert np.isfinite(value).all()
    assert value[0] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("damping", [None, 1e-3, 1.0, 1e3])
def test_spline_predict_matches_upstream(scatter, data, grid, damping):
    ours = mv.Spline(damping=damping).fit(scatter, data)
    theirs = vd.Spline(damping=damping).fit(scatter, data)
    # Undamped and lightly damped fits are dominated by the rank cutoff on
    # an ill-conditioned Jacobian; heavy damping is well conditioned.
    tolerance = 1e-6 if damping is None or damping < 1 else 1e-10
    assert rel(ours.predict(grid), theirs.predict(grid)) < tolerance


def test_spline_fits_the_data_as_well_as_upstream(scatter, data):
    """The undamped fit is a truncated least-squares solve, so it only has
    to reproduce upstream's misfit, not drive it to zero."""
    ours = mv.Spline().fit(scatter, data)
    theirs = vd.Spline().fit(scatter, data)
    ours_misfit = np.linalg.norm(ours.predict(scatter) - data)
    their_misfit = np.linalg.norm(theirs.predict(scatter) - data)
    assert ours_misfit <= 1.01 * their_misfit


def test_spline_jacobian_is_symmetric_enough(scatter):
    spline = mv.Spline()
    jacobian = spline.jacobian(scatter, scatter)
    np.testing.assert_allclose(jacobian, jacobian.T, rtol=1e-12, atol=1e-12)


def test_spline_grid_returns_xarray(scatter, data):
    ours = mv.Spline().fit(scatter, data).grid(REGION, spacing=1)
    theirs = vd.Spline().fit(scatter, data).grid(REGION, spacing=1)
    assert list(ours.data_vars) == list(theirs.data_vars)
    np.testing.assert_allclose(
        ours.scalars.values, theirs.scalars.values, rtol=1e-8, atol=1e-8
    )


def test_spline_scatter_and_profile(scatter, data):
    with pytest.warns(FutureWarning):
        ours = mv.Spline().fit(scatter, data).scatter(REGION, size=200, random_state=1)
    theirs = vd.Spline().fit(scatter, data).scatter(REGION, size=200, random_state=1)
    assert list(ours.columns) == list(theirs.columns)
    np.testing.assert_allclose(
        ours.scalars.values, theirs.scalars.values, rtol=1e-8, atol=1e-8
    )
    ours = mv.Spline().fit(scatter, data).profile((0, 0), (9, 4), size=20)
    theirs = vd.Spline().fit(scatter, data).profile((0, 0), (9, 4), size=20)
    assert list(ours.columns) == list(theirs.columns)
    np.testing.assert_allclose(
        ours.scalars.values, theirs.scalars.values, rtol=1e-8, atol=1e-8
    )


# ----------------------------------------------------------------- vector
@pytest.mark.parametrize("damping", [None, 1e-2, 1e3])
def test_vector_spline_2d_matches_upstream(scatter, grid, damping):
    rng = np.random.default_rng(5)
    east_component = np.cos(scatter[0] / 4) * 5
    north_component = np.sin(scatter[1] / 3) * 5
    weights = (rng.uniform(0.5, 2.0, 500), rng.uniform(0.5, 2.0, 500))
    kwargs = {"poisson": 0.25, "damping": damping}
    ours = mv.VectorSpline2D(**kwargs).fit(
        scatter, (east_component, north_component), weights
    )
    theirs = vd.VectorSpline2D(**kwargs).fit(
        scatter, (east_component, north_component), weights
    )
    jac_ours = ours.jacobian(scatter, ours.force_coords)
    jac_theirs = theirs.jacobian(scatter, theirs.force_coords)
    np.testing.assert_allclose(jac_ours, jac_theirs, rtol=1e-12, atol=1e-12)
    # The undamped and lightly damped elastic Jacobian is ill conditioned
    # enough that the two truncated solves differ well before the last bit.
    tolerance = 1e-6 if damping is None or damping < 1 else 1e-9
    for a, b in zip(ours.predict(grid), theirs.predict(grid)):
        assert rel(a, b) < tolerance


def test_greens_func_2d_matches_upstream():
    rng = np.random.default_rng(13)
    east = (rng.random(20000) - 0.5) * 20
    north = (rng.random(20000) - 0.5) * 20
    ours = mv.vector.greens_func_2d(east, north, 1e3, 0.25)
    theirs = vd.vector.greens_func_2d(east, north, 1e3, 0.25)
    for a, b in zip(ours, theirs):
        np.testing.assert_allclose(a, b, rtol=1e-12)


# ------------------------------------------------------- chain and vector
def test_chain_matches_upstream(scatter, data, grid):
    ours = mv.Chain(
        [("a", mv.Spline(damping=1e-3)), ("b", mv.Spline(damping=1e-1))]
    ).fit(scatter, data)
    theirs = vd.Chain(
        [("a", vd.Spline(damping=1e-3)), ("b", vd.Spline(damping=1e-1))]
    ).fit(scatter, data)
    assert rel(ours.predict(grid), theirs.predict(grid)) < 1e-6
    assert list(ours.named_steps) == list(theirs.named_steps)


def test_vector_matches_upstream(scatter, data, grid):
    components = (data, data * 2)
    ours = mv.Vector(
        [mv.Spline(damping=1e-3), mv.Spline(damping=1e-3)]
    ).fit(scatter, components)
    theirs = vd.Vector(
        [vd.Spline(damping=1e-3), vd.Spline(damping=1e-3)]
    ).fit(scatter, components)
    for a, b in zip(ours.predict(grid), theirs.predict(grid)):
        assert rel(a, b) < 1e-6


# --------------------------------------------------------- output dtypes
def test_variance_to_weights_honours_dtype():
    """A non-float64 output dtype must not change the values."""
    variance = np.array([1.0, 2.0, 4.0, np.nan, 0.0])
    reference = mv.variance_to_weights(variance)
    weights = mv.variance_to_weights(variance, dtype="float32")
    assert weights.dtype == np.float32
    np.testing.assert_allclose(weights, reference, rtol=1e-6)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_jacobian_honours_dtype(scatter, dtype):
    """A non-float64 dtype must not change the Jacobian the kernel writes."""
    ours = mv.Spline().jacobian(scatter, scatter, dtype=dtype)
    assert ours.dtype == np.dtype(dtype)
    np.testing.assert_allclose(
        ours, mv.Spline().jacobian(scatter, scatter), rtol=1e-6
    )


def test_vector_jacobian_honours_dtype(scatter):
    ours = mv.VectorSpline2D(mindist=1e3).jacobian(
        scatter, (np.ravel(scatter[0]), np.ravel(scatter[1])), dtype="float32"
    )
    assert ours.dtype == np.float32
    assert np.isfinite(ours).all()


def test_spline_predict_accepts_float32_coordinates(scatter, data, grid):
    """The prediction buffer is float64 whatever the coordinates are."""
    ours = mv.Spline(damping=1e-3).fit(scatter, data)
    single = tuple(i.astype("float32") for i in grid)
    predicted = ours.predict(single)
    assert predicted.dtype == np.float64
    assert np.isfinite(predicted).all()
    np.testing.assert_allclose(predicted, ours.predict(grid), rtol=1e-5)


def test_spacing_to_size_matches_upstream():
    cases = [
        (0.0, 10.0, 2.5, "spacing"),
        (0.0, 10.0, 2.4, "spacing"),
        (0.0, 10.0, 2.4, "region"),
        (0.0, 0.6, 1.0, "spacing"),
    ]
    for start, stop, spacing, adjust in cases:
        assert mv.spacing_to_size(start, stop, spacing, adjust) == pytest.approx(
            vd.coordinates.spacing_to_size(start, stop, spacing, adjust)
        )


def test_block_split_single_block_column(scatter):
    """A block set one column wide must label, not divide by a zero pitch."""
    _, labels = mv.block_split(scatter, shape=(200, 1))
    _, theirs = vd.block_split(scatter, shape=(200, 1))
    np.testing.assert_array_equal(labels, theirs)
    assert labels.min() >= 0 and labels.max() < 200 * 2
