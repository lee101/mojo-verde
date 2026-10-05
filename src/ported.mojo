"""Kernels ported from Verde (https://github.com/fatiando/verde).

Emitted in upstream's own module order so this file reads top to bottom
next to the Python it came from. Every function keeps the upstream name,
argument order and branch structure; the divergences the Mojo dialect
forces (no parametric `@export`, buffers crossing as `Int`, no callables
across the ABI) are absorbed by the ctypes layer, not by these kernels.
"""

from std.math import (
    abs,
    cos,
    exp as fast_exp,
    floor,
    frexp,
    ldexp,
    log as fast_log,
    pow as fast_pow,
    round,
    sin,
    sqrt,
)
from max.algorithm import parallelize
from std.sys import simd_width_of

# The width of every vector body below. The kernels that use it all walk
# a contiguous axis with a scalar tail: the force axis of the Green's
# function matrices, the column axis of the ridge Gram, the inner index of
# the Cholesky, and the training set of the kNN scan.
comptime VW: Int = simd_width_of[DType.float64]()
comptime V = SIMD[DType.float64, VW]

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime BPtr = Pointer[UInt8, AnyOrigin[mut=True]]

# Row counts below which the serial loop is kept: handing a handful of rows
# to the thread pool costs more than the rows save. The worker count is an
# argument because this dialect has no mutable globals.
comptime PARALLEL_MIN: Int = 256
comptime PARALLEL_MIN_COLS: Int = 8

comptime LN2 = 0.6931471805599453
comptime INV_LN2 = 1.4426950408889634
comptime M_SQRT2 = 1.4142135623730951
comptime M_INV_SQRT2 = 0.7071067811865476
comptime TINY = 2.2250738585072014e-308


# The toolchain's std.math.log and its `**` operator both carry ~3e-9
# relative error, which the ill-conditioned biharmonic Jacobian amplifies
# into visible force differences. frexp and ldexp are exact and sqrt is
# exact, so a range reduction plus the odd series in
# s = (m - 1) / (m + 1) restores libm accuracy with pure arithmetic.
def log(x: Float64) -> Float64:
    if x <= TINY:
        return fast_log(x)
    var t = frexp(x)
    var m = t[0]
    var e = t[1]
    if m > M_SQRT2:
        m = m * 0.5
        e += 1
    elif m < M_INV_SQRT2:
        m = m * 2.0
        e -= 1
    var s = (m - 1.0) / (m + 1.0)
    var s2 = s * s
    var p = 1.0 / 25.0
    p = p * s2 + 1.0 / 23.0
    p = p * s2 + 1.0 / 21.0
    p = p * s2 + 1.0 / 19.0
    p = p * s2 + 1.0 / 17.0
    p = p * s2 + 1.0 / 15.0
    p = p * s2 + 1.0 / 13.0
    p = p * s2 + 1.0 / 11.0
    p = p * s2 + 1.0 / 9.0
    p = p * s2 + 1.0 / 7.0
    p = p * s2 + 1.0 / 5.0
    p = p * s2 + 1.0 / 3.0
    p = p * s2 + 1.0
    return 2.0 * s * p + Float64(e) * LN2


def power(x: Float64, y: Float64) -> Float64:
    if x <= 0.0:
        return fast_pow(x, y)
    return exp(y * log(x))


# std.math.exp is accurate to about 2e-12 here, which is not enough once the
# ill-conditioned biharmonic Jacobian multiplies it. Reducing the argument
# modulo ln2 and running the Taylor series in Horner form is pure arithmetic
# and matches libm to <1e-15.
def exp(x: Float64) -> Float64:
    if x > 709.0 or x < -745.0:
        return fast_exp(x)
    var n = Int32(round(x * INV_LN2))
    var r = x - Float64(n) * LN2
    var p = 1.0 / 2432902008176640000.0
    p = p * r + 1.0 / 121645100408832000.0
    p = p * r + 1.0 / 6402373705728000.0
    p = p * r + 1.0 / 355687428096000.0
    p = p * r + 1.0 / 20922789888000.0
    p = p * r + 1.0 / 1307674368000.0
    p = p * r + 1.0 / 87178291200.0
    p = p * r + 1.0 / 6227020800.0
    p = p * r + 1.0 / 479001600.0
    p = p * r + 1.0 / 39916800.0
    p = p * r + 1.0 / 3628800.0
    p = p * r + 1.0 / 362880.0
    p = p * r + 1.0 / 40320.0
    p = p * r + 1.0 / 5040.0
    p = p * r + 1.0 / 720.0
    p = p * r + 1.0 / 120.0
    p = p * r + 1.0 / 24.0
    p = p * r + 1.0 / 6.0
    p = p * r + 1.0 / 2.0
    p = p * r + 1.0 / 1.0
    p = p * r + 1.0 / 1.0
    return ldexp(p, n)


def vlog(x: V) -> V:
    """The scalar log above, on a vector, with the same accuracy.

    Callers must guarantee every lane is positive and not subnormal; the
    two branches of the range reduction become selects, so there is no
    control flow left to stop the loop from vectorising.
    """
    var t = frexp(x)
    var m: V = t[0]
    var e: V = t[1].cast[DType.float64]()
    var big: SIMD[DType.bool, VW] = SIMD.gt(m, V(M_SQRT2))
    var low: SIMD[DType.bool, VW] = SIMD.lt(m, V(M_INV_SQRT2))
    m = big.select(m * 0.5, m)
    e = big.select(e + 1.0, e)
    m = low.select(m * 2.0, m)
    e = low.select(e - 1.0, e)
    var s = (m - 1.0) / (m + 1.0)
    var s2 = s * s
    var p = V(1.0 / 25.0)
    p = p * s2 + V(1.0 / 23.0)
    p = p * s2 + V(1.0 / 21.0)
    p = p * s2 + V(1.0 / 19.0)
    p = p * s2 + V(1.0 / 17.0)
    p = p * s2 + V(1.0 / 15.0)
    p = p * s2 + V(1.0 / 13.0)
    p = p * s2 + V(1.0 / 11.0)
    p = p * s2 + V(1.0 / 9.0)
    p = p * s2 + V(1.0 / 7.0)
    p = p * s2 + V(1.0 / 5.0)
    p = p * s2 + V(1.0 / 3.0)
    p = p * s2 + V(1.0)
    return 2.0 * s * p + e * LN2


# ======================================================= base/least_squares.py
def standard_scale_fit(
    jacobian: Ptr, scale: Ptr, mean: Ptr, nrows: Int, ncols: Int, workers: Int
):
    """The StandardScaler fit with ``with_mean=False``: column scale only.

    Two passes (mean, then mean squared deviation) instead of sklearn's
    single incremental pass: the same quantity, better conditioned.
    """
    if nrows < 1:
        # 0/0 would be NaN and would then divide every scaled column. The
        # Python layer refuses an empty system, so this is the back stop for
        # a direct call and it leaves the Jacobian untouched.
        return
    def one(c: Int) {`imm`}:
        var total = 0.0
        var i = 0
        while i < nrows:
            total += jacobian.unsafe_load(i * ncols + c)
            i += 1
        var m = total / Float64(nrows)
        mean.unsafe_store(c, m)
        var acc = 0.0
        i = 0
        while i < nrows:
            var d = jacobian.unsafe_load(i * ncols + c) - m
            acc += d * d
            i += 1
        var s = sqrt(acc / Float64(nrows))
        if s == 0.0:
            s = 1.0
        scale.unsafe_store(c, s)

    if ncols >= PARALLEL_MIN:
        parallelize(one, ncols, workers)
    else:
        var c = 0
        while c < ncols:
            one(c)
            c += 1


def standard_scale_transform(
    jacobian: Ptr, scale: Ptr, nrows: Int, ncols: Int, workers: Int
):
    """The StandardScaler transform: with_mean=False, so a bare divide.

    The fit writes 1.0 for a constant column, so the scale is never zero.
    This one is a walk down each row rather than down each column: the column
    form strides a cache line per element and this form does not.

    The vector body is bounded by a full width, never by ``c < VW and c <
    ncols``. Two conditions in the loop test let the compiler peel the last
    iteration into a lane-masked store, and a masked SIMD store is written
    back whole on this toolchain, which walks off the end of the row. The
    scalar tail below is then only reached when the width does not divide the
    row, which is the case it is there for.
    """
    if nrows < 1:
        return
    var i = 0
    while i < nrows:
        var row = jacobian.unsafe_offset(i * ncols)
        var c = 0
        while c + VW <= ncols:
            var s = scale.unsafe_load[width=VW](c)
            row.unsafe_store(c, row.unsafe_load[width=VW](c) / s)
            c += VW
        while c < ncols:
            var s = scale.unsafe_load(c)
            row.unsafe_store(c, row.unsafe_load(c) / s)
            c += 1
        i += 1


def cholesky(a: Ptr, n: Int) -> Bool:
    """In-place lower Cholesky of a symmetric positive definite matrix.

    Both inner products are a serial subtract chain, so a scalar version
    runs at the latency of one multiply-add per term. Reading four terms
    at a time into a vector accumulator and reducing once at the end
    shortens that chain by four and keeps the horizontal reduction out of
    the loop. This is the whole cost of the damped solve, which is why
    the column scaling around it barely shows.
    """
    var j = 0
    while j < n:
        var brow = a.unsafe_offset(j * n)
        var sacc = V(0.0)
        var k = 0
        while k + VW <= j:
            var v = brow.unsafe_load[width=VW](k)
            sacc += v * v
            k += VW
        var s = brow.unsafe_load(j) - sacc.reduce_add()
        while k < j:
            var v = brow.unsafe_load(k)
            s -= v * v
            k += 1
        if s <= 0.0:
            return False
        var d = sqrt(s)
        brow.unsafe_store(j, d)
        var i = j + 1
        while i < n:
            var arow = a.unsafe_offset(i * n)
            var acc = V(0.0)
            k = 0
            while k + VW <= j:
                acc += arow.unsafe_load[width=VW](k) * brow.unsafe_load[width=VW](k)
                k += VW
            var t = arow.unsafe_load(j) - acc.reduce_add()
            while k < j:
                t -= arow.unsafe_load(k) * brow.unsafe_load(k)
                k += 1
            arow.unsafe_store(j, t / d)
            i += 1
        j += 1
    return True


def cholesky_solve(a: Ptr, b: Ptr, n: Int):
    """Solve A x = b for lower-triangular A, in place on b.

    The forward sweep reads a row and so runs on vectors; the backward
    sweep reads a column, which is strided, and stays scalar. It is two
    triangular sweeps, so it is not where the solve spends its time.
    """
    var i = 0
    while i < n:
        var arow = a.unsafe_offset(i * n)
        var acc = V(0.0)
        var k = 0
        while k + VW <= i:
            acc += arow.unsafe_load[width=VW](k) * b.unsafe_load[width=VW](k)
            k += VW
        var t = b.unsafe_load(i) - acc.reduce_add()
        while k < i:
            t -= arow.unsafe_load(k) * b.unsafe_load(k)
            k += 1
        b.unsafe_store(i, t / arow.unsafe_load(i))
        i += 1
    i = n - 1
    while i >= 0:
        var t = b.unsafe_load(i)
        var k = i + 1
        while k < n:
            t -= a.unsafe_load(k * n + i) * b.unsafe_load(k)
            k += 1
        b.unsafe_store(i, t / a.unsafe_load(i * n + i))
        i -= 1


def ridge_gram(
    jacobian: Ptr, data: Ptr, weights: Ptr, gram: Ptr, rhs: Ptr,
    nrows: Int, ncols: Int, weighted: Int, workers: Int,
):
    """The lower triangle of X'WX and the right-hand side X'Wy, in place.

    Column c of the Gram only reads column c and the ones before it, so the
    columns are independent and the outer loop parallelises. gram must
    arrive zeroed.
    """
    def one(c: Int) {`imm`}:
        var total = 0.0
        var grow = gram.unsafe_offset(c * ncols)
        var j = 0
        while j < nrows:
            var w = 1.0
            if weighted:
                w = weights.unsafe_load(j)
            var row = jacobian.unsafe_offset(j * ncols)
            var x = row.unsafe_load(c) * w
            # The weight belongs in the product once, not once per
            # factor: x * xk is Xw'Xw with Xw = X * sqrt(w). The k axis
            # is contiguous in both operands, so it runs a vector body.
            var k = 0
            while k + VW <= c + 1:
                grow.unsafe_store(
                    k,
                    grow.unsafe_load[width=VW](k)
                    + V(x) * row.unsafe_load[width=VW](k),
                )
                k += VW
            while k <= c:
                grow.unsafe_store(k, grow.unsafe_load(k) + x * row.unsafe_load(k))
                k += 1
            total += x * data.unsafe_load(j)
            j += 1
        rhs.unsafe_store(c, total)

    if ncols >= PARALLEL_MIN:
        parallelize(one, ncols, workers)
    else:
        var c = 0
        while c < ncols:
            one(c)
            c += 1


def ridge_regression(
    jacobian: Ptr, data: Ptr, weights: Ptr, dst: Ptr, work: Ptr,
    nrows: Int, ncols: Int, damping: Float64, weighted: Int, workers: Int,
) -> Bool:
    """Ridge(fit_intercept=False) solved the way scikit-learn solves it.

    The system is rescaled by sqrt(sample_weight), the normal equations
    are formed, damping is added to the diagonal, and a Cholesky solve
    finishes. work must hold ncols*ncols + 2*ncols zeros.
    """
    var gram = work
    var rhs = work.unsafe_offset(ncols * ncols)
    ridge_gram(jacobian, data, weights, gram, rhs, nrows, ncols, weighted, workers)
    # Only the lower triangle was accumulated, so mirror it; sklearn forms
    # the full symmetric product and hands LAPACK a complete matrix.
    var c = 0
    while c < ncols:
        var d = 0
        while d < c:
            gram.unsafe_store(d * ncols + c, gram.unsafe_load(c * ncols + d))
            d += 1
        c += 1
    c = 0
    while c < ncols:
        gram.unsafe_store(c * ncols + c, gram.unsafe_load(c * ncols + c) + damping)
        c += 1
    if not cholesky(gram, ncols):
        return False
    cholesky_solve(gram, rhs, ncols)
    c = 0
    while c < ncols:
        dst.unsafe_store(c, rhs.unsafe_load(c))
        c += 1
    return True


def least_squares(
    jacobian: Ptr, data: Ptr, weights: Ptr, params: Ptr, scale: Ptr, mean: Ptr,
    work: Ptr, nrows: Int, ncols: Int, damping: Float64, weighted: Int,
    workers: Int,
) -> Bool:
    """verde.least_squares.least_squares for the regularized branch.

    Scales the Jacobian columns to unit variance in place, solves the
    damped system, then undoes the scaling on the parameters.
    """
    standard_scale_fit(jacobian, scale, mean, nrows, ncols, workers)
    standard_scale_transform(jacobian, scale, nrows, ncols, workers)
    if not ridge_regression(
        jacobian, data, weights, params, work, nrows, ncols, damping, weighted,
        workers,
    ):
        return False
    var c = 0
    while c < ncols:
        params.unsafe_store(c, params.unsafe_load(c) / scale.unsafe_load(c))
        c += 1
    return True


# ================================================================ blockreduce.py
def block_compact(count: Ptr, block_ids: IPtr, block_index: IPtr, nblocks: Int) -> Int:
    """List the blocks that hold data, in increasing label order.

    block_index maps a block label to its position in that list, or -1 when
    the block is empty, which is how upstream drops empty blocks.
    """
    var found = 0
    var b = 0
    while b < nblocks:
        if count.unsafe_load(b) > 0:
            block_ids.unsafe_store(found, Int64(b))
            block_index.unsafe_store(b, Int64(found))
            found += 1
        else:
            block_index.unsafe_store(b, Int64(-1))
        b += 1
    return found


def binned_mean(
    labels: IPtr, values: Ptr, weights: Ptr,
    mean: Ptr, sum_w: Ptr, count: Ptr, work: Ptr,
    n: Int, ncomps: Int, nblocks: Int, nchunks: Int, weighted: Int, workers: Int,
):
    """First pass of the blocked reduction: per-block count, sum(w) and sum(w*v).

    Points are split into contiguous chunks so each worker writes into its
    own slice of work without any atomics; the chunks are then added up.
    work must hold nchunks * nblocks * 3 * ncomps doubles, laid out as
    [chunk][sum_w | sum_wv | count][component][block]. Weights are laid
    out component-major, one full-length array per component, the same way
    upstream keeps one weight array per data component.
    """
    var span = 3 * ncomps * nblocks
    var plane = ncomps * nblocks

    def one(c: Int) {`imm`}:
        var start = c * n // nchunks
        var stop = (c + 1) * n // nchunks
        var base = work.unsafe_offset(c * span)
        var i = start
        while i < stop:
            var b = Int(labels.unsafe_load(i))
            if b >= 0 and b < nblocks:
                var k = 0
                while k < ncomps:
                    var w = 1.0
                    if weighted:
                        w = weights.unsafe_load(k * n + i)
                    var v = values.unsafe_load(i * ncomps + k)
                    base.unsafe_store(
                        k * nblocks + b, base.unsafe_load(k * nblocks + b) + w
                    )
                    base.unsafe_store(
                        plane + k * nblocks + b,
                        base.unsafe_load(plane + k * nblocks + b) + w * v,
                    )
                    k += 1
                base.unsafe_store(
                    2 * plane + b, base.unsafe_load(2 * plane + b) + 1.0
                )
            i += 1

    if nchunks > 1:
        parallelize(one, nchunks, workers)
    else:
        one(0)
    var k = 0
    while k < ncomps:
        var b = 0
        while b < nblocks:
            var sw = 0.0
            var swv = 0.0
            var cnt = 0.0
            var c = 0
            while c < nchunks:
                var base = c * span
                sw += work.unsafe_load(base + k * nblocks + b)
                swv += work.unsafe_load(base + plane + k * nblocks + b)
                c += 1
            var cc = 0
            while cc < nchunks:
                cnt += work.unsafe_load(cc * span + 2 * plane + b)
                cc += 1
            sum_w.unsafe_store(k * nblocks + b, sw)
            count.unsafe_store(b, cnt)
            var denom = sw if weighted else cnt
            mean.unsafe_store(
                k * nblocks + b, (swv / denom) if denom != 0.0 else 0.0
            )
            b += 1
        k += 1


def binned_variance(
    labels: IPtr, values: Ptr, weights: Ptr, mean: Ptr, variance: Ptr, work: Ptr,
    n: Int, ncomps: Int, nblocks: Int, nchunks: Int, weighted: Int, ddof: Float64,
    workers: Int,
):
    """Second pass: sum(w * (v - mean)**2), divided by sum(w) or count - ddof.

    That is verbatim what pandas' .var (ddof=1) and numpy.average with
    weights do, so the two branches land on the same numbers. work must hold
    nchunks * nblocks * 3 * ncomps doubles, as in binned_mean.
    """
    var span = 3 * ncomps * nblocks

    def one(c: Int) {`imm`}:
        var start = c * n // nchunks
        var stop = (c + 1) * n // nchunks
        var base = work.unsafe_offset(c * span)
        var i = start
        while i < stop:
            var b = Int(labels.unsafe_load(i))
            if b >= 0 and b < nblocks:
                var k = 0
                while k < ncomps:
                    var w = 1.0
                    if weighted:
                        w = weights.unsafe_load(k * n + i)
                    var d = values.unsafe_load(i * ncomps + k) - mean.unsafe_load(k * nblocks + b)
                    base.unsafe_store(k * nblocks + b, base.unsafe_load(k * nblocks + b) + w * d * d)
                    k += 1
            i += 1

    if nchunks > 1:
        parallelize(one, nchunks, workers)
    else:
        one(0)
    var k = 0
    while k < ncomps:
        var b = 0
        while b < nblocks:
            var total = 0.0
            var c = 0
            while c < nchunks:
                total += work.unsafe_load(c * span + k * nblocks + b)
                c += 1
            variance.unsafe_store(k * nblocks + b, total)
            b += 1
        k += 1


# ================================================================ coordinates.py
def line_coordinates(
    start: Float64, stop: Float64, size: Int, dst: Ptr, pixel_register: Int
):
    """numpy.linspace, then the shift to pixel centres.

    The last node is pinned to stop exactly, which is what numpy.linspace
    does. The nodes themselves can differ from numpy by one ULP, see the
    note on the multiply below.
    """
    var values = dst
    if size < 1:
        return
    var div = size - 1
    if div <= 0:
        values.unsafe_store(0, start)
        return
    var step = (stop - start) / Float64(div)
    var i = 0
    while i < size:
        # numpy.linspace rounds the product and the sum separately while
        # this toolchain contracts them into one fused multiply-add, so a
        # node can land one ULP away from numpy's. Everything downstream is
        # a uniform lattice, so that stays far below any meaningful grid
        # resolution, and the tests pin the bound.
        values.unsafe_store(i, Float64(i) * step + start)
        i += 1
    values.unsafe_store(size - 1, stop)
    if pixel_register:
        # Upstream shifts every node by the same half-interval, the one
        # between the first two nodes, not by its own half-interval.
        var half = (values.unsafe_load(1) - values.unsafe_load(0)) / 2.0
        var i = 0
        while i < size - 1:
            values.unsafe_store(i, values.unsafe_load(i) + half)
            i += 1


def meshgrid_2d(
    east: Ptr, north: Ptr, east_2d: Ptr, north_2d: Ptr, ne: Int, nn: Int,
    workers: Int,
):
    """numpy.meshgrid with the default xy indexing."""
    def one(i: Int) {`imm`}:
        var y = north.unsafe_load(i)
        var j = 0
        while j < ne:
            east_2d.unsafe_store(i * ne + j, east.unsafe_load(j))
            north_2d.unsafe_store(i * ne + j, y)
            j += 1

    if nn >= PARALLEL_MIN:
        parallelize(one, nn, workers)
    else:
        var i = 0
        while i < nn:
            one(i)
            i += 1


def get_region(east: Ptr, north: Ptr, dst: Ptr, n: Int):
    """The bounding region of the given coordinates."""
    if n == 0:
        dst.unsafe_store(0, 0.0)
        dst.unsafe_store(1, 0.0)
        dst.unsafe_store(2, 0.0)
        dst.unsafe_store(3, 0.0)
        return
    var w = east.unsafe_load(0)
    var e = w
    var s = north.unsafe_load(0)
    var n_ = s
    var i = 1
    while i < n:
        var x = east.unsafe_load(i)
        if x < w:
            w = x
        if x > e:
            e = x
        var y = north.unsafe_load(i)
        if y < s:
            s = y
        if y > n_:
            n_ = y
        i += 1
    dst.unsafe_store(0, w)
    dst.unsafe_store(1, e)
    dst.unsafe_store(2, s)
    dst.unsafe_store(3, n_)


def inside(
    east: Ptr, north: Ptr, dst: BPtr, n: Int,
    w: Float64, e: Float64, s: Float64, nn: Float64, workers: Int,
):
    """Points inside [w, e] x [s, n]; boundary points count as inside."""
    def one(i: Int) {`imm`}:
        var x = east.unsafe_load(i)
        var y = north.unsafe_load(i)
        var ok = 1
        if x < w or x > e or y < s or y > nn:
            ok = 0
        dst.unsafe_store(i, UInt8(ok))

    if n >= PARALLEL_MIN:
        parallelize(one, n, workers)
    else:
        var i = 0
        while i < n:
            one(i)
            i += 1


def longitude_continuity(
    east: Ptr, dst_east: Ptr, n: Int, interval_360: Int, workers: Int
):
    """Move the longitudes into the same degrees interval as the region.

    The region arithmetic lives in the Python layer, which also runs
    upstream's sanity checks; this is the per-point remainder.
    """
    def one(i: Int) {`imm`}:
        var v = east.unsafe_load(i)
        if interval_360:
            dst_east.unsafe_store(i, v % 360.0)
        else:
            dst_east.unsafe_store(i, ((v + 180.0) % 360.0) - 180.0)

    if n >= PARALLEL_MIN:
        parallelize(one, n, workers)
    else:
        var i = 0
        while i < n:
            one(i)
            i += 1


def block_split(
    east: Ptr, north: Ptr, labels: IPtr, n: Int,
    centre_e: Float64, centre_n: Float64, spacing_e: Float64, spacing_n: Float64,
    nb_e: Int, nb_n: Int,
):
    """Label each point with the index of the block that holds it.

    Upstream asks a KD-tree for the nearest block centre. Block centres form
    a uniform lattice starting at *centre_e*/*centre_n* with the given
    pitches, so the nearest centre of a point is the same round-to-nearest
    arithmetic, with the two out-of-range ends clamped.
    """
    var i = 0
    while i < n:
        var ie = 0
        var inn = 0
        # A block set with a single row or column has no pitch, and the
        # division would be a divide by zero; the point then belongs to
        # the one block along that axis by definition.
        if spacing_e > 0.0:
            ie = Int(floor((east.unsafe_load(i) - centre_e) / spacing_e + 0.5))
        if spacing_n > 0.0:
            inn = Int(floor((north.unsafe_load(i) - centre_n) / spacing_n + 0.5))
        if ie < 0:
            ie = 0
        if ie > nb_e - 1:
            ie = nb_e - 1
        if inn < 0:
            inn = 0
        if inn > nb_n - 1:
            inn = nb_n - 1
        labels.unsafe_store(i, Int64(inn * nb_e + ie))
        i += 1

# ================================================================ coordinates.py

def window_span(
    x: Float64, centres: Ptr, count: Int, half: Float64
) -> Tuple[Int, Int]:
    """The range of window indices whose centre may lie within *half* of x.

    One index of slack on each side absorbs the rounding of the division.
    The caller still measures the distance, so the membership test is the
    same `abs(x - centre) <= half` that the KD-tree query makes, and a
    centre that falls outside the range cannot satisfy it.
    """
    if count < 1:
        return (0, -1)
    var first = centres.unsafe_load(0)
    if count == 1:
        if abs(x - first) <= half:
            return (0, 0)
        return (0, -1)
    var step = centres.unsafe_load(1) - first
    var lo = Int(floor((x - half - first) / step)) - 1
    var hi = Int(floor((x + half - first) / step)) + 1
    if lo < 0:
        lo = 0
    if hi > count - 1:
        hi = count - 1
    return (lo, hi)


def rolling_window_count(
    east: Ptr, north: Ptr, centre_e: Ptr, centre_n: Ptr, chunk_counts: IPtr,
    n: Int, nw_e: Int, nw_n: Int, half: Float64, nchunks: Int, workers: Int,
):
    """How many points fall in each window, counted separately per chunk.

    The window is the square of half-width *half*, which is what
    `query_ball_point(..., p=np.inf)` asks the KD-tree for: a point is in a
    window when both coordinates are within *half* of the window centre.
    Only the candidate centres from `window_span` are visited, and the
    distance is still measured, so the boundary counts the same way the
    tree counts it.
    """
    var span = nw_e * nw_n

    def one(c: Int) {`imm`}:
        var base = chunk_counts.unsafe_offset(c * span)
        var i = c * n // nchunks
        var stop = (c + 1) * n // nchunks
        while i < stop:
            var x = east.unsafe_load(i)
            var y = north.unsafe_load(i)
            var se = window_span(x, centre_e, nw_e, half)
            var sn = window_span(y, centre_n, nw_n, half)
            var jn = sn[0]
            while jn <= sn[1]:
                var yc = centre_n.unsafe_load(jn)
                var je = se[0]
                while je <= se[1]:
                    if abs(x - centre_e.unsafe_load(je)) <= half and abs(
                        y - yc
                    ) <= half:
                        var w = jn * nw_e + je
                        base.unsafe_store(w, base.unsafe_load(w) + 1)
                    je += 1
                jn += 1
            i += 1

    if nchunks > 1:
        parallelize(one, nchunks, workers)
    else:
        one(0)


def rolling_window_fill(
    east: Ptr, north: Ptr, centre_e: Ptr, centre_n: Ptr, positions: IPtr,
    dst: IPtr, n: Int, nw_e: Int, nw_n: Int, half: Float64, nchunks: Int,
    workers: Int,
):
    """The point indices inside every window, window by window.

    positions holds the start of each window's stretch of dst for each
    chunk, laid out as [chunk][window]; every chunk appends into its own
    stretch, so the points of a window come out in increasing index order,
    which is the order query_ball_point returns them in.
    """
    var span = nw_e * nw_n

    def one(c: Int) {`imm`}:
        var i = c * n // nchunks
        var stop = (c + 1) * n // nchunks
        while i < stop:
            var x = east.unsafe_load(i)
            var y = north.unsafe_load(i)
            var se = window_span(x, centre_e, nw_e, half)
            var sn = window_span(y, centre_n, nw_n, half)
            var jn = sn[0]
            while jn <= sn[1]:
                var yc = centre_n.unsafe_load(jn)
                var je = se[0]
                while je <= se[1]:
                    if abs(x - centre_e.unsafe_load(je)) <= half and abs(
                        y - yc
                    ) <= half:
                        var slot = c * span + jn * nw_e + je
                        var at = positions.unsafe_load(slot)
                        dst.unsafe_store(at, Int64(i))
                        positions.unsafe_store(slot, at + 1)
                    je += 1
                jn += 1
            i += 1

    if nchunks > 1:
        parallelize(one, nchunks, workers)
    else:
        one(0)


def expanding_window_select(
    east: Ptr, north: Ptr, dst: IPtr, n: Int,
    centre_e: Float64, centre_n: Float64, half: Float64,
) -> Int:
    """The indices of the points inside one square window, in order.

    The same square neighbourhood of half-width *half* that
    `query_ball_point(..., p=np.inf)` asks the KD-tree for. One window means
    one output list, so there is nothing to group and the scan is the whole
    cost; the points come out in increasing index order.
    """
    var found = 0
    var i = 0
    while i < n:
        var de = east.unsafe_load(i) - centre_e
        if de < 0.0:
            de = -de
        var dn = north.unsafe_load(i) - centre_n
        if dn < 0.0:
            dn = -dn
        if de <= half and dn <= half:
            dst.unsafe_store(found, Int64(i))
            found += 1
        i += 1
    return found


# ================================================================ synthetic.py
comptime PI: Float64 = 3.141592653589793


def checkerboard(
    east: Ptr, north: Ptr, dst: Ptr, n: Int,
    amplitude: Float64, w_east: Float64, w_north: Float64, workers: Int,
):
    """amplitude * sin(2 pi east / w_east) * cos(2 pi north / w_north)."""
    def one(i: Int) {`imm`}:
        dst.unsafe_store(
            i,
            amplitude
            * sin((2.0 * PI / w_east) * east.unsafe_load(i))
            * cos((2.0 * PI / w_north) * north.unsafe_load(i)),
        )

    if n >= 256:
        parallelize(one, n, workers)
    else:
        var i = 0
        while i < n:
            one(i)
            i += 1


# ========================================================================= mask.py
# distance_mask needs nothing of its own: it is the k=1 nearest-neighbour
# query from neighbors.py followed by upstream's `distance <= maxdist`.

# Training points folded into one guarded block of the kNN scan. Bigger
# blocks amortise the horizontal reduction further, but each hit redoes
# the whole block, and the hits are the scarce resource.
comptime KNN_BLOCK: Int = 8


# ================================================================ neighbors.py
def knn_query(
    east_q: Ptr, north_q: Ptr, east_t: Ptr, north_t: Ptr,
    dist: Ptr, idx: IPtr, nq: Int, nt: Int, k: Int, workers: Int,
):
    """The k nearest training points for every query point.

    Upstream goes through a KD-tree; this is the brute-force scan, which for
    the 2D point sets Verde works with is competitive and has no tree build
    cost. Ties are resolved by whichever index the scan meets first.
    """
    def one(i: Int) {`imm`}:
        var x = east_q.unsafe_load(i)
        var y = north_q.unsafe_load(i)
        var row = dist.unsafe_offset(i * k)
        var rowi = idx.unsafe_offset(i * k)
        var t = 0
        while t < k:
            row.unsafe_store(t, 1.0e300)
            rowi.unsafe_store(t, Int64(-1))
            t += 1
        # The list holds exactly k sorted entries and a candidate is
        # accepted only if it beats the current worst, so `worst` is the
        # same number the scalar scan reads out of the last slot. That
        # lets the scan run on whole vectors: a block is reduced to one
        # number, and only a block that beats the worst is redone lane by
        # lane. Every hit raises the bar for the next one, so a query
        # redoes the k * ln(nt) entries a selection sort would touch and
        # nothing else. Reducing a block at a time rather than a vector
        # at a time is what makes it pay: the horizontal reduction is
        # paid once per block instead of once per vector.
        var worst = 1.0e300
        var j = 0
        while j + KNN_BLOCK * VW <= nt:
            var lo = V(1.0e300)
            for c in range(KNN_BLOCK):
                var dx = V(x) - east_t.unsafe_load[width=VW](j + c * VW)
                var dy = V(y) - north_t.unsafe_load[width=VW](j + c * VW)
                var d2 = dx * dx + dy * dy
                # A NaN lane compares false and so keeps the incumbent,
                # which is what the scalar scan does with it; the free
                # min() would instead poison the whole block.
                var closer: SIMD[DType.bool, VW] = SIMD.lt(d2, lo)
                lo = closer.select(d2, lo)
            if lo.reduce_min() < worst:
                for lane in range(KNN_BLOCK * VW):
                    var dx = x - east_t.unsafe_load(j + lane)
                    var dy = y - north_t.unsafe_load(j + lane)
                    var cand = dx * dx + dy * dy
                    if cand < worst:
                        var pos = k - 1
                        while pos > 0 and row.unsafe_load(pos - 1) > cand:
                            row.unsafe_store(pos, row.unsafe_load(pos - 1))
                            rowi.unsafe_store(pos, rowi.unsafe_load(pos - 1))
                            pos -= 1
                        row.unsafe_store(pos, cand)
                        rowi.unsafe_store(pos, Int64(j + lane))
                        worst = row.unsafe_load(k - 1)
            j += KNN_BLOCK * VW
        while j < nt:
            var dx = x - east_t.unsafe_load(j)
            var dy = y - north_t.unsafe_load(j)
            var cand = dx * dx + dy * dy
            if cand < worst:
                var pos = k - 1
                while pos > 0 and row.unsafe_load(pos - 1) > cand:
                    row.unsafe_store(pos, row.unsafe_load(pos - 1))
                    rowi.unsafe_store(pos, rowi.unsafe_load(pos - 1))
                    pos -= 1
                row.unsafe_store(pos, cand)
                rowi.unsafe_store(pos, Int64(j))
                worst = row.unsafe_load(k - 1)
            j += 1
        t = 0
        while t < k:
            row.unsafe_store(t, sqrt(row.unsafe_load(t)))
            t += 1

    if nq >= PARALLEL_MIN:
        parallelize(one, nq, workers)
    else:
        var i = 0
        while i < nq:
            one(i)
            i += 1


# ==================================================================== spline.py
def greens_func(east: Float64, north: Float64, mindist: Float64) -> Float64:
    """The Green's function for the bi-harmonic spline."""
    var distance = sqrt(east * east + north * north)
    distance += mindist
    var result: Float64
    if distance < 1:
        result = distance * (log(power(distance, distance)) - distance)
    else:
        result = distance * distance * (log(distance) - 1)
    return result


def greens_func_array(
    east: Ptr, north: Ptr, dst: Ptr, n: Int, mindist: Float64, workers: Int
):
    """The elementwise numpy form, one entry per point."""
    def one(i: Int) {`imm`}:
        dst.unsafe_store(
            i, greens_func(east.unsafe_load(i), north.unsafe_load(i), mindist)
        )

    if n >= PARALLEL_MIN:
        parallelize(one, n, workers)
    else:
        var i = 0
        while i < n:
            one(i)
            i += 1


def jacobian(
    east: Ptr, north: Ptr, force_east: Ptr, force_north: Ptr, jac: Ptr,
    n: Int, m: Int, mindist: Float64, workers: Int,
):
    """One column of the Jacobian per force: its Green's function."""
    def one(i: Int) {`imm`}:
        var ei = east.unsafe_load(i)
        var ni = north.unsafe_load(i)
        var row = jac.unsafe_offset(i * m)
        var j = 0
        while j + VW <= m:
            var de = V(ei) - force_east.unsafe_load[width=VW](j)
            var dn = V(ni) - force_north.unsafe_load[width=VW](j)
            var d = sqrt(de * de + dn * dn) + V(mindist)
            # The d < 1 branch needs an exp, which the range reduction
            # cannot vectorise, so only the vectors that touch it are
            # redone lane by lane with the scalar kernel.
            if d.reduce_min() >= 1.0:
                row.unsafe_store(j, d * d * (vlog(d) - 1.0))
            else:
                for lane in range(VW):
                    row.unsafe_store(
                        j + lane, greens_func(de[lane], dn[lane], mindist)
                    )
            j += VW
        while j < m:
            row.unsafe_store(
                j,
                greens_func(
                    ei - force_east.unsafe_load(j),
                    ni - force_north.unsafe_load(j),
                    mindist,
                ),
            )
            j += 1

    if n >= PARALLEL_MIN and m >= PARALLEL_MIN_COLS:
        parallelize(one, n, workers)
    else:
        var i = 0
        while i < n:
            one(i)
            i += 1


def predict(
    east: Ptr, north: Ptr, force_east: Ptr, force_north: Ptr, forces: Ptr,
    result: Ptr, n: Int, m: Int, mindist: Float64, workers: Int,
):
    """The predicted data, summed over the point forces."""
    def one(i: Int) {`imm`}:
        var ei = east.unsafe_load(i)
        var ni = north.unsafe_load(i)
        var total = 0.0
        var j = 0
        while j + VW <= m:
            var de = V(ei) - force_east.unsafe_load[width=VW](j)
            var dn = V(ni) - force_north.unsafe_load[width=VW](j)
            var fv = forces.unsafe_load[width=VW](j)
            var d = sqrt(de * de + dn * dn) + V(mindist)
            if d.reduce_min() >= 1.0:
                total += (d * d * (vlog(d) - 1.0) * fv).reduce_add()
            else:
                for lane in range(VW):
                    total += greens_func(de[lane], dn[lane], mindist) * fv[lane]
            j += VW
        while j < m:
            total += greens_func(
                ei - force_east.unsafe_load(j),
                ni - force_north.unsafe_load(j),
                mindist,
            ) * forces.unsafe_load(j)
            j += 1
        result.unsafe_store(i, total)

    if n >= PARALLEL_MIN and m >= PARALLEL_MIN_COLS:
        parallelize(one, n, workers)
    else:
        var i = 0
        while i < n:
            one(i)
            i += 1


# ===================================================================== utils.py
def variance_to_weights(variance: Ptr, dst: Ptr, n: Int, tol: Float64):
    """Inverse variance, scaled to [0, 1]; anything below tol gets weight 1."""
    var smallest = 0.0
    var found = False
    var i = 0
    while i < n:
        var v = variance.unsafe_load(i)
        if v != v:
            v = 0.0
        if v > tol and (not found or v < smallest):
            smallest = v
            found = True
        i += 1
    i = 0
    while i < n:
        var v = variance.unsafe_load(i)
        if v != v:
            v = 0.0
        if v > tol:
            dst.unsafe_store(i, smallest / v)
        else:
            dst.unsafe_store(i, 1.0)
        i += 1


# ===================================================================== vector.py
def greens_func_2d(
    east: Float64, north: Float64, mindist: Float64, poisson: Float64
) -> Tuple[Float64, Float64, Float64]:
    """The Green's functions for the 2D elastic case."""
    var distance = sqrt(east * east + north * north)
    distance += mindist
    var ln_r = (3.0 - poisson) * log(distance)
    var over_r2 = (1.0 + poisson) / (distance * distance)
    var green_ee = ln_r + over_r2 * north * north
    var green_nn = ln_r + over_r2 * east * east
    var green_ne = -over_r2 * east * north
    return green_ee, green_nn, green_ne


def vgreens2d(d: V, east: V, north: V, poisson: Float64) -> Tuple[V, V, V]:
    """The three Green's functions of the elastic case, on a vector.

    Every lane of *d* has to be positive: the range reduction has no
    answer for a log of zero or of a negative number, and the scalar
    kernel is what reproduces that.
    """
    var ln_r = (3.0 - poisson) * vlog(d)
    var over_r2 = (1.0 + poisson) / (d * d)
    return (
        ln_r + over_r2 * north * north,
        ln_r + over_r2 * east * east,
        -over_r2 * east * north,
    )


def jacobian_2d(
    east: Ptr, north: Ptr, force_east: Ptr, force_north: Ptr, jac: Ptr,
    npoints: Int, nforces: Int, mindist: Float64, poisson: Float64, workers: Int,
):
    """The four blocks of the coupled elastic Jacobian; J is symmetric."""
    def one(i: Int) {`imm`}:
        var ei = east.unsafe_load(i)
        var ni = north.unsafe_load(i)
        var top = jac.unsafe_offset(i * 2 * nforces)
        var bottom = jac.unsafe_offset((i + npoints) * 2 * nforces)
        var j = 0
        while j + VW <= nforces:
            var de = V(ei) - force_east.unsafe_load[width=VW](j)
            var dn = V(ni) - force_north.unsafe_load[width=VW](j)
            var d = sqrt(de * de + dn * dn) + V(mindist)
            if d.reduce_min() > 0.0:
                var g = vgreens2d(d, de, dn, poisson)
                top.unsafe_store(j, g[0])
                top.unsafe_store(j + nforces, g[2])
                bottom.unsafe_store(j, g[2])
                bottom.unsafe_store(j + nforces, g[1])
            else:
                for lane in range(VW):
                    var s = greens_func_2d(de[lane], dn[lane], mindist, poisson)
                    top.unsafe_store(j + lane, s[0])
                    top.unsafe_store(j + lane + nforces, s[2])
                    bottom.unsafe_store(j + lane, s[2])
                    bottom.unsafe_store(j + lane + nforces, s[1])
            j += VW
        while j < nforces:
            var s = greens_func_2d(
                ei - force_east.unsafe_load(j),
                ni - force_north.unsafe_load(j),
                mindist,
                poisson,
            )
            top.unsafe_store(j, s[0])
            top.unsafe_store(j + nforces, s[2])
            bottom.unsafe_store(j, s[2])
            bottom.unsafe_store(j + nforces, s[1])
            j += 1

    if npoints >= PARALLEL_MIN and nforces >= PARALLEL_MIN_COLS:
        parallelize(one, npoints, workers)
    else:
        var i = 0
        while i < npoints:
            one(i)
            i += 1


def predict_2d(
    east: Ptr, north: Ptr, force_east: Ptr, force_north: Ptr, forces: Ptr,
    vec_east: Ptr, vec_north: Ptr, npoints: Int, nforces: Int,
    mindist: Float64, poisson: Float64, workers: Int,
):
    """The predicted east and north components."""
    def one(i: Int) {`imm`}:
        var ei = east.unsafe_load(i)
        var ni = north.unsafe_load(i)
        var total_e = 0.0
        var total_n = 0.0
        var j = 0
        while j + VW <= nforces:
            var de = V(ei) - force_east.unsafe_load[width=VW](j)
            var dn = V(ni) - force_north.unsafe_load[width=VW](j)
            var fv = forces.unsafe_load[width=VW](j)
            var fv2 = forces.unsafe_load[width=VW](j + nforces)
            var d = sqrt(de * de + dn * dn) + V(mindist)
            if d.reduce_min() > 0.0:
                var g = vgreens2d(d, de, dn, poisson)
                total_e += (g[0] * fv + g[2] * fv2).reduce_add()
                total_n += (g[2] * fv + g[1] * fv2).reduce_add()
            else:
                for lane in range(VW):
                    var s = greens_func_2d(de[lane], dn[lane], mindist, poisson)
                    total_e += s[0] * fv[lane] + s[2] * fv2[lane]
                    total_n += s[2] * fv[lane] + s[1] * fv2[lane]
            j += VW
        while j < nforces:
            var s = greens_func_2d(
                ei - force_east.unsafe_load(j),
                ni - force_north.unsafe_load(j),
                mindist,
                poisson,
            )
            total_e += s[0] * forces.unsafe_load(j) + s[2] * forces.unsafe_load(j + nforces)
            total_n += s[2] * forces.unsafe_load(j) + s[1] * forces.unsafe_load(j + nforces)
            j += 1
        vec_east.unsafe_store(i, total_e)
        vec_north.unsafe_store(i, total_n)

    if npoints >= PARALLEL_MIN and nforces >= PARALLEL_MIN_COLS:
        parallelize(one, npoints, workers)
    else:
        var i = 0
        while i < npoints:
            one(i)
            i += 1
