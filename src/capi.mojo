"""The C ABI the Python bindings call through.

Every buffer crosses as an `Int` address: `@export` rejects parametric
functions and a pointer with an inferred origin is parametric. Callers own
all memory, including scratch, so nothing here allocates and nothing here
can leak. The only logic in this file is the pointer reconstruction.
"""

from ported import (
    binned_mean,
    binned_variance,
    block_compact,
    block_split,
    checkerboard,
    expanding_window_select,
    get_region,
    greens_func_array,
    inside,
    jacobian,
    jacobian_2d,
    knn_query,
    least_squares,
    line_coordinates,
    longitude_continuity,
    meshgrid_2d,
    predict,
    predict_2d,
    rolling_window_count,
    rolling_window_fill,
    standard_scale_fit,
    standard_scale_transform,
    variance_to_weights,
)


comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime BPtr = Pointer[UInt8, AnyOrigin[mut=True]]


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def bp(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


# ======================================================= base/least_squares.py
@export("mv_ls_scale_fit")
def mv_ls_scale_fit(
    jac: Int, scale: Int, mean: Int, nrows: Int, ncols: Int, workers: Int
) abi("C"):
    standard_scale_fit(p(jac), p(scale), p(mean), nrows, ncols, workers)


@export("mv_ls_scale_transform")
def mv_ls_scale_transform(
    jac: Int, scale: Int, nrows: Int, ncols: Int, workers: Int
) abi("C"):
    standard_scale_transform(p(jac), p(scale), nrows, ncols, workers)


@export("mv_ls_least_squares")
def mv_ls_least_squares(
    jac: Int,
    data: Int,
    weights: Int,
    params: Int,
    scale: Int,
    mean: Int,
    work: Int,
    nrows: Int,
    ncols: Int,
    damping: Float64,
    weighted: Int,
    workers: Int,
) abi("C") -> Int:
    var ok = least_squares(
        p(jac), p(data), p(weights), p(params), p(scale), p(mean), p(work),
        nrows, ncols, damping, weighted, workers,
    )
    return 1 if ok else 0


# ================================================================ blockreduce.py
@export("mv_block_compact")
def mv_block_compact(
    count: Int, block_ids: Int, block_index: Int, nblocks: Int
) abi("C") -> Int:
    return block_compact(p(count), ip(block_ids), ip(block_index), nblocks)


@export("mv_binned_mean")
def mv_binned_mean(
    labels: Int,
    values: Int,
    weights: Int,
    mean: Int,
    sum_w: Int,
    count: Int,
    work: Int,
    n: Int,
    ncomps: Int,
    nblocks: Int,
    nchunks: Int,
    weighted: Int,
    workers: Int,
) abi("C"):
    binned_mean(
        ip(labels), p(values), p(weights), p(mean), p(sum_w), p(count), p(work),
        n, ncomps, nblocks, nchunks, weighted, workers,
    )


@export("mv_binned_variance")
def mv_binned_variance(
    labels: Int,
    values: Int,
    weights: Int,
    mean: Int,
    variance: Int,
    work: Int,
    n: Int,
    ncomps: Int,
    nblocks: Int,
    nchunks: Int,
    weighted: Int,
    ddof: Float64,
    workers: Int,
) abi("C"):
    binned_variance(
        ip(labels), p(values), p(weights), p(mean), p(variance), p(work), n, ncomps,
        nblocks, nchunks, weighted, ddof, workers,
    )


# ================================================================ coordinates.py
@export("mv_line_coordinates")
def mv_line_coordinates(
    start: Float64, stop: Float64, size: Int, dst: Int, pixel_register: Int
) abi("C"):
    line_coordinates(start, stop, size, p(dst), pixel_register)


@export("mv_meshgrid_2d")
def mv_meshgrid_2d(
    east: Int, north: Int, east_2d: Int, north_2d: Int, ne: Int, nn: Int,
    workers: Int,
) abi("C"):
    meshgrid_2d(p(east), p(north), p(east_2d), p(north_2d), ne, nn, workers)


@export("mv_get_region")
def mv_get_region(east: Int, north: Int, dst: Int, n: Int) abi("C"):
    get_region(p(east), p(north), p(dst), n)


@export("mv_inside")
def mv_inside(
    east: Int,
    north: Int,
    dst: Int,
    n: Int,
    w: Float64,
    e: Float64,
    s: Float64,
    nn: Float64,
    workers: Int,
) abi("C"):
    inside(p(east), p(north), bp(dst), n, w, e, s, nn, workers)


@export("mv_longitude_continuity")
def mv_longitude_continuity(
    east: Int, dst_east: Int, n: Int, interval_360: Int, workers: Int
) abi("C"):
    longitude_continuity(p(east), p(dst_east), n, interval_360, workers)


@export("mv_block_split")
def mv_block_split(
    east: Int,
    north: Int,
    labels: Int,
    n: Int,
    w: Float64,
    s: Float64,
    spacing_e: Float64,
    spacing_n: Float64,
    nb_e: Int,
    nb_n: Int,
) abi("C"):
    block_split(
        p(east), p(north), ip(labels), n, w, s, spacing_e, spacing_n, nb_e, nb_n
    )


@export("mv_rolling_window_count")
def mv_rolling_window_count(
    east: Int,
    north: Int,
    centre_e: Int,
    centre_n: Int,
    chunk_counts: Int,
    n: Int,
    nw_e: Int,
    nw_n: Int,
    half: Float64,
    nchunks: Int,
    workers: Int,
) abi("C"):
    rolling_window_count(
        p(east), p(north), p(centre_e), p(centre_n), ip(chunk_counts), n, nw_e,
        nw_n, half, nchunks, workers,
    )


@export("mv_rolling_window_fill")
def mv_rolling_window_fill(
    east: Int,
    north: Int,
    centre_e: Int,
    centre_n: Int,
    positions: Int,
    dst: Int,
    n: Int,
    nw_e: Int,
    nw_n: Int,
    half: Float64,
    nchunks: Int,
    workers: Int,
) abi("C"):
    rolling_window_fill(
        p(east), p(north), p(centre_e), p(centre_n), ip(positions), ip(dst), n,
        nw_e, nw_n, half, nchunks, workers,
    )


@export("mv_expanding_window_select")
def mv_expanding_window_select(
    east: Int,
    north: Int,
    dst: Int,
    n: Int,
    centre_e: Float64,
    centre_n: Float64,
    half: Float64,
) abi("C") -> Int:
    return expanding_window_select(
        p(east), p(north), ip(dst), n, centre_e, centre_n, half
    )


# ================================================================ neighbors.py
@export("mv_knn_query")
def mv_knn_query(
    east_q: Int,
    north_q: Int,
    east_t: Int,
    north_t: Int,
    dist: Int,
    idx: Int,
    nq: Int,
    nt: Int,
    k: Int,
    workers: Int,
) abi("C"):
    knn_query(
        p(east_q), p(north_q), p(east_t), p(north_t), p(dist), ip(idx), nq, nt,
        k, workers,
    )


# ==================================================================== spline.py
@export("mv_greens_func")
def mv_greens_func(
    east: Int, north: Int, dst: Int, n: Int, mindist: Float64, workers: Int
) abi("C"):
    greens_func_array(p(east), p(north), p(dst), n, mindist, workers)


@export("mv_jacobian")
def mv_jacobian(
    east: Int,
    north: Int,
    force_east: Int,
    force_north: Int,
    jac: Int,
    n: Int,
    m: Int,
    mindist: Float64,
    workers: Int,
) abi("C"):
    jacobian(
        p(east), p(north), p(force_east), p(force_north), p(jac), n, m, mindist,
        workers,
    )


@export("mv_predict")
def mv_predict(
    east: Int,
    north: Int,
    force_east: Int,
    force_north: Int,
    forces: Int,
    result: Int,
    n: Int,
    m: Int,
    mindist: Float64,
    workers: Int,
) abi("C"):
    predict(
        p(east), p(north), p(force_east), p(force_north), p(forces), p(result),
        n, m, mindist, workers,
    )


# ===================================================================== utils.py
@export("mv_variance_to_weights")
def mv_variance_to_weights(
    variance: Int, dst: Int, n: Int, tol: Float64
) abi("C"):
    variance_to_weights(p(variance), p(dst), n, tol)


# ===================================================================== vector.py
@export("mv_jacobian_2d")
def mv_jacobian_2d(
    east: Int,
    north: Int,
    force_east: Int,
    force_north: Int,
    jac: Int,
    npoints: Int,
    nforces: Int,
    mindist: Float64,
    poisson: Float64,
    workers: Int,
) abi("C"):
    jacobian_2d(
        p(east), p(north), p(force_east), p(force_north), p(jac), npoints,
        nforces, mindist, poisson, workers,
    )


@export("mv_predict_2d")
def mv_predict_2d(
    east: Int,
    north: Int,
    force_east: Int,
    force_north: Int,
    forces: Int,
    vec_east: Int,
    vec_north: Int,
    npoints: Int,
    nforces: Int,
    mindist: Float64,
    poisson: Float64,
    workers: Int,
) abi("C"):
    predict_2d(
        p(east), p(north), p(force_east), p(force_north), p(forces),
        p(vec_east), p(vec_north), npoints, nforces, mindist, poisson, workers,
    )


# ================================================================= synthetic.py
@export("mv_checkerboard")
def mv_checkerboard(
    east: Int,
    north: Int,
    dst: Int,
    n: Int,
    amplitude: Float64,
    w_east: Float64,
    w_north: Float64,
    workers: Int,
) abi("C"):
    checkerboard(
        p(east), p(north), p(dst), n, amplitude, w_east, w_north, workers
    )
