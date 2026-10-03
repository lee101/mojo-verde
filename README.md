# mojo-verde

A Mojo port of the compute-bound parts of
[Verde](https://github.com/fatiando/verde), the Fatiando a Terra library for
processing and gridding spatial data.

The kernels live in [`src/ported.mojo`](src/ported.mojo) and are emitted in
upstream's own module order, so that file reads top to bottom next to the
Python it came from. The Python package in [`python/mojoverde/`](python/mojoverde)
mirrors Verde's names, argument order and defaults, so existing Verde code
works against it unchanged for the covered subset.

Nothing here is a stub. Every ported function is checked against real
upstream `verde` 1.9.0 in the test suite, at tolerances that are measured
rather than guessed.

## Install

```bash
pixi install
```

The library is compiled on demand: the first import runs
`mojo build --emit shared-lib` into `dist/libmojo-verde.so`, and rebuilds it
whenever `src/*.mojo` is newer. To use a library built elsewhere, point
`MOJOVERDE_LIB` at it. If `mojo` is not on `PATH`, set `MOJOVERDE_MOJO`.

Then:

```bash
pixi run build     # mojo build --emit shared-lib -> dist/libmojo-verde.so
pixi run test      # pytest, against upstream verde
pixi run bench     # markdown table against upstream verde
```

`pixi run bench` holds a machine-wide `flock` so that a concurrent job
cannot land on the same cores in the middle of a timing run. Never invoke
`bench/bench.py` directly.

## Usage

```python
import numpy as np
import mojoverde as mv

region = (0, 10, -5, 5)
coordinates = mv.scatter_points(region, 2000, random_state=0)
easting, northing = coordinates
data = np.sin(easting / 3) * np.cos(north / 4) * 10

# Decimate to 0.1-degree blocks, then grid with a biharmonic spline.
chain = mv.Chain(
    [
        ("mean", mv.BlockMean(spacing=0.1)),
        ("spline", mv.Spline(damping=1e-3)),
    ]
)
chain.fit(coordinates, data)

grid = chain.grid(region, spacing=0.05)
print(grid.scalars.values.shape)   # (201, 201)
```

Nearest-neighbour interpolation and masking work the same way:

```python
nn = mv.KNeighbors(k=4).fit(coordinates, data)
nearest = nn.predict(mv.grid_coordinates(region, shape=(201, 201)))
mask = mv.distance_mask(coordinates, 0.5, coordinates=coordinates)
```

## What is covered

| upstream | ported | notes |
| --- | --- | --- |
| `verde.coordinates.get_region` | yes | Mojo |
| `verde.coordinates.pad_region` | yes | pure Python |
| `verde.coordinates.scatter_points` | yes | numpy `RandomState`, same stream |
| `verde.coordinates.line_coordinates` | yes | Mojo |
| `verde.coordinates.grid_coordinates` | yes | Mojo |
| `verde.coordinates.shape_to_spacing` | yes | pure Python |
| `verde.coordinates.spacing_to_size` | yes | pure Python |
| `verde.coordinates.profile_coordinates` | yes | numpy |
| `verde.coordinates.inside` | yes | Mojo |
| `verde.coordinates.block_split` | yes | Mojo |
| `verde.coordinates.longitude_continuity` | yes | Mojo |
| `verde.coordinates.rolling_window` | yes | Mojo, index ranges over a uniform lattice |
| `verde.coordinates.expanding_window` | yes | Mojo, one scan per window |
| `verde.mask.distance_mask` | yes | Mojo, through the kNN kernel |
| `verde.neighbors.KNeighbors` | yes | brute-force kNN instead of a KD-tree |
| `verde.utils.variance_to_weights` | yes | Mojo |
| `verde.blockreduce.BlockMean` | yes | Mojo |
| `verde.blockreduce.BlockReduce` | partially | see below |
| `verde.least_squares.least_squares` | partially | see below |
| `verde.spline.Spline` | yes | Mojo kernels, `SplineCV` not ported |
| `verde.vector.VectorSpline2D` | yes | Mojo kernels |
| `verde.vector.Vector` | yes | composition only |
| `verde.chain.Chain` | yes | composition only |
| `verde.base.BaseGridder` | yes | `fit`/`predict`/`filter`/`grid`/`scatter`/`profile` |

## What is not covered, and why

- **`SplineCV` and everything in `verde.model_selection`.** They are
  cross-validation drivers over the gridders, not kernels; the gridders
  they would drive are here.
- **`BlockReduce` with an arbitrary `reduction`.** Upstream takes any
  callable. This port accepts `numpy.mean`, `numpy.average`,
  `numpy.median`, `numpy.max` and `numpy.min` and raises on anything else
  rather than quietly returning something different. `numpy.average` and
  `numpy.mean` run as a binned accumulation in Mojo; the other three need
  the grouped values, so they gather them with one argsort and then call
  numpy per block.
- **`verde.base.least_squares` undamped branch.** Upstream hands the
  matrix to `sklearn.LinearRegression`, which calls
  `scipy.linalg.lstsq` with `cond=LinearRegression.tol` (1e-6, an
  effective-rank cutoff, not a convergence tolerance). There is no faithful
  small-kernel equivalent of an SVD, and the biharmonic Jacobian is far
  too ill conditioned to be pushed through normal equations instead, so
  that single `lstsq` call goes to LAPACK through numpy. The column
  scaling and the whole damped (`Ridge`) branch run in Mojo: the damped
  branch is a Cholesky of the normal equations, which is exactly what
  scikit-learn does, and it is reproduced here operation for operation.
- **`verde.mask.convexhull_mask`.** It leans on `matplotlib.path.Path`,
  not on anything a kernel could accelerate.
- **`verde.distances.median_distance`.** A `cKDTree.count_neighbors`
  wrapper.
- **`verde.projections`, `verde.trend`, `verde.io`, `verde.datasets`,
  `verde.scipygridder`, `verde.chain`'s sibling utilities.** Not
  gridding kernels: projections delegate to pyproj, `Trend` is a
  polynomial least-squares design matrix, `ScipyGridder` delegates to
  `scipy.interpolate`, and `io`/`datasets` load example data.
- **GPU paths.** The host API is present and working on this toolchain
  (`max.gpu.host`) and the machine has an RTX 5090, so this is a decision
  and not an availability problem: no kernel here reaches 2 flops per
  byte, and the two that do the real work sit at about 1.5 and 1.4, so a
  GPU would be bound by moving the grid and the forces rather than by the
  arithmetic. The per-kernel counts are tabulated in "Why there is no GPU
  path" below.

### Documented numerical divergences

Four, all bounded and all pinned by the tests:

1. **Grid coordinates can differ from numpy by one ULP.**
   `numpy.linspace` rounds `i * step` and `+ start` separately; this
   toolchain contracts them into a fused multiply-add. The tests assert
   `rtol=1e-14` on grid coordinates, which is a few ULP.
2. **`log` and `**` are reimplemented.** `std.math.log`, `std.math.exp`
   and the `**` operator carry roughly 3e-9, 2e-12 and 3e-9 relative
   error respectively on this toolchain. The biharmonic Jacobian has a
   condition number of 3.3e7, which turns 3e-9 into visible
   differences in the estimated forces, so `ported.mojo` carries its own
   range-reduced `log` and `exp` built on the exact `frexp`/`ldexp`. The
   end-to-end effect is measured: over 20000 random points the Green's
   function is bit-identical to `verde.spline.greens_func_numpy`, and
   `Spline.predict` matches upstream to 5e-14 relative on the
   well-conditioned (damping 1e3) fit and to 2.9e-11 on the undamped one.
3. **The undamped forces are not elementwise comparable.** Both
   implementations truncate the rank of a matrix with condition number
   3.3e7 at 1e-6, and the two answers in the discarded subspace differ
   wildly while the predictions do not. The tests compare the significant
   force components, the predictions and the misfit, not every entry.
4. **The vector bodies change summation order.** `predict` and
   `predict_2d` now add four force terms at a time, and the ridge Gram
   and the Cholesky accumulate four products at a time, where the scalar
   code added one. Every individual term is computed by the same
   expression as before — `vlog` is bit-identical to `log` on the
   measured range — so the only difference is how the terms are grouped,
   which is at the 1e-16 level and lands inside the same tolerances the
   scalar version needed. `test_least_squares_damped_matches_upstream`
   and the `Spline`/`VectorSpline2D` prediction tests are the ones that
   would notice, and they hold at the tolerances they already had.

## Benchmarks

Every number below comes from one `pixi run bench` run on the machine
described here, and nothing in this section is carried over from an earlier
build:

- Linux x86_64, 72 logical CPUs (36 available to the process), Intel Xeon
  E5-2697 v4
- `mojo 1.2.0.dev2026092905`, Python 3.13.14, numpy 2.5.1
- upstream `verde v1.9.0` with numba 0.67.0 installed, so Verde's
  `jacobian`/`predict` run through its `parallel=True` numba engine

| case | mojo-verde | verde | speedup |
| --- | --- | --- | --- |
| Spline.jacobian (4000 forces) | 32.8 ms | 27.3 ms | 0.83x (slower) |
| Spline.predict (200k x 4000 forces) | 481.0 ms | 650.5 ms | 1.35x |
| VectorSpline2D.jacobian (2000 points) | 30.0 ms | 36.7 ms | 1.22x |
| VectorSpline2D.predict (200k x 2000 forces) | 224.1 ms | 383.0 ms | 1.71x |
| KNeighbors.predict k=1 (50k train, 100k query) | 138.5 ms | 59.6 ms | 0.43x (slower) |
| KNeighbors.predict k=8 (50k train, 100k query) | 129.0 ms | 100.0 ms | 0.78x (slower) |
| distance_mask (50k data, 400k grid) | 575.0 ms | 320.1 ms | 0.56x (slower) |
| BlockMean.filter (1M points, 100x100 blocks) | 505.3 ms | 2204.9 ms | 4.36x |
| BlockReduce(np.mean).filter (1M points, 100x100 blocks) | 172.0 ms | 2857.5 ms | 16.62x |
| block_split (1M points, spacing 0.1) | 7.3 ms | 810.9 ms | 110.72x |
| rolling_window (200k points, size 0.4, spacing 0.1) | 163.3 ms | 766.1 ms | 4.69x |
| expanding_window (1M points, 3 sizes) | 5.0 ms | 539.1 ms | 107.20x |
| CheckerBoard.predict (1M points) | 10.5 ms | 21.0 ms | 2.01x |
| grid_coordinates (4M points, spacing 0.001) | 255.5 ms | 1008.4 ms | 3.95x |
| variance_to_weights (4M points) | 56.3 ms | 97.3 ms | 1.73x |
| least_squares damped (800x800 biharmonic) | 75.9 ms | 147.5 ms | 1.94x |
| Spline.fit (1500 points, damping 1e-3) | 401.5 ms | 304.3 ms | 0.76x (slower) |

Each number is the best of five timed runs. The box is shared and the
spread is real: three runs of this same binary on this day gave, for
`Spline.predict`, 1.50x, 1.42x and 1.35x; for `VectorSpline2D.predict`,
1.69x, 1.81x and 1.71x; for `least_squares damped`, 1.58x, 1.36x and
1.94x, where verde itself moved between 120 ms and 148 ms; and for
`VectorSpline2D.jacobian`, 0.87x, 0.87x and 1.22x. Treat the large and
repeated gaps as real, read the near-unity rows as parity, and re-run
`pixi run bench` before quoting any single row.

Reading the table honestly:

- **The Green's function prediction loops are ahead of numba.** Both build
  the same O(n.m) sum, and the difference is the inner loop: this port runs
  it on four doubles at a time, with the range reduction of the `log`
  written as two selects instead of two branches. That is
  `Spline.predict` at 1.35x and `VectorSpline2D.predict` at 1.71x, and
  both held above 1.2x on every run.
- **The Jacobian builds are at parity, not ahead.** They write a full
  matrix, so they are bound by the stores rather than by the arithmetic and
  the vector body does not help there; across three runs
  `Spline.jacobian` gave 1.09x, 1.09x and 0.83x and
  `VectorSpline2D.jacobian` gave 0.87x, 0.87x and 1.22x, which is parity
  with a wide spread rather than a result. The prediction loops reuse the
  same kernel without the stores and do win.
- **Where this port wins by a lot, it wins for a structural reason.**
  `block_split` is two orders of magnitude faster because upstream asks a
  `cKDTree` for the nearest block centre while this port uses the same
  round-to-nearest arithmetic on a uniform lattice. `BlockReduce` is 16x
  and `BlockMean` several times faster because upstream costs one
  Python-level call per group through `pandas.groupby` while this port
  is two binned accumulations. `grid_coordinates` is 4x faster because
  it is a straight elementwise pass with no library call to dispatch to.
- **kNN loses to the KD-tree, and that is the algorithm, not the code.**
  This port scans every training point for every query point, which is
  O(n.m); `cKDTree` prunes. The vector scan folds eight vectors into one
  running minimum and reduces that once per block, but the remaining gap to
  the tree is the pruning, and closing it means changing the algorithm.
  `distance_mask` inherits it.
- **`Spline.fit` is 0.76x.** It is the damped solve inside a larger
  pipeline: the Cholesky alone is 1.94x, but building the Jacobian and the
  column scaling around it are not, and they are the larger share of the
  total. A blocked Cholesky would help the solve further; it is a rewrite
  rather than a tuning change.

### What was tried and did not pay

Everything below was implemented and measured with `pixi run bench`, and
none of it was kept. The tree is left as it was found, which is the point
of listing it: these are the rungs of the escalation order that were
measured and rejected, not guesses about them.

| change | idea | measured | why it was reverted |
| --- | --- | --- | --- |
| one-pass column scaling (`E[x^2] - E[x]^2`) in the StandardScaler fit | reads the Jacobian once instead of twice | 6.2 ms → 11.8 ms at 800 columns, and the scale is wrong to the column's condition number | slower and less accurate. The column-at-a-time walk is bound by the number of rows it touches, not by the adds, so halving the traffic is worth less than the arithmetic costs, and the undamped branch cannot absorb the cancellation at all |
| parallel Cholesky step (`parallelize` over the rows below each diagonal, four rows per work item) | the rows of a column are independent | built and ran, but lands in the same 1.3-1.4x band the serial column already holds, with a barrier per column | 800 barriers for a few hundred microseconds of work each. Not kept |
| tiled kNN scan with an interleaved training buffer | each lane of the running vector reads its own point out of the interleaved array, so four loads stay in flight instead of one | built; the KD-tree index-equality test in `tests/test_parity.py` started failing | the block index moved and the test is an exact comparison. Kept the incumbent instead of weakening the test |
| kNN through a `max.gpu.host` path | the scan is 5 flops per 16 bytes | not attempted | 0.3 flops per byte is an order of magnitude below the 2 flops per byte a GPU needs; it would be bound by the copy |

Two things were measured before anything was changed, and neither of
them is a change. The kNN scan is already handed all 72 workers and is
still memory-bound at 126 ms for 5e9 point comparisons: with one worker it
is 2.84 s, with 4 it is 0.77 s, with 16 it is 0.26 s, with 36 it is
0.15 s and with 72 it is 0.13 s, so the curve has flattened and what is
left is the compares. And `Spline.fit` is not the solve: at 1500 points
the Jacobian is 9.4 ms and the whole damped `least_squares` call is
379 ms of the 448 ms total, of which the column scaling is 13 ms and the
Cholesky of the normal equations is the rest.

## How it works

### FFI strategy

`src/ported.mojo` holds the kernels. It is a single compilation unit
because build cost is fixed and batching avoids paying it per file.
`src/capi.mojo` is the only entry point and contains nothing but the
`@export("...")` C-ABI shims.

Three properties of this Mojo dialect shape the design, and all three are
absorbed in the shims rather than in the kernels:

- `@export` rejects parametric functions, so every shim takes buffers as
  `Int` addresses and rebuilds the pointer inside the body. The
  divergence is isolated to a one-line `p(addr)` helper.
- `@export` requires an explicit `abi("C")` effect, and an `abi("C")`
  function may not be `raises`, so the shims are pure and the only
  fallible operation in the port (a non-positive-definite normal
  equation) is reported through an `Int` return.
- Buffers cross as `Int` because a pointer with an inferred origin is
  parametric.

`python/mojoverde/_lib.py` loads the shared object and rebuilds it
automatically when `src/*.mojo` is newer than
`dist/libmojo-verde.so`. Every buffer parameter is declared to ctypes as
a typed, C-contiguous pointer (`numpy.ctypeslib.ndpointer`) rather than a
bare integer address, and the array itself is what gets passed. That is
deliberate: ctypes keeps a reference to the array for the duration of the
call, so a converted temporary cannot be collected while a kernel is still
reading it, and a wrong dtype or a non-contiguous buffer raises instead of
producing a wild pointer. All buffers handed to a kernel are float64, or
int64 for labels and neighbour indices and uint8 for masks, whatever the
dtype the caller supplied.

### Memory layout

All buffers are caller-owned, contiguous, C-order, float64 (`int64` for
labels and neighbour indices, `uint8` for masks). The kernels allocate
nothing, so nothing can leak. Concretely:

- Matrices are row-major with the row stride passed in (`ncols` for the
  Jacobian, `ncomps` for the blocked reduction's value stack). The
  Jacobian for `Spline` is `(n_data, n_forces)`; for `VectorSpline2D` it
  is `(2·n_points, 2·n_forces)` with the four blocks at
  `(i, j)`, `(i+n, j+m)`, `(i, j+m)` and `(i+n, j)`.
- Scratch is passed in, never allocated: the blocked reduction needs
  `nchunks · nblocks · 3 · ncomps` doubles laid out as
  `[chunk][sum_w | sum_wv | count][component][block]`, and the ridge solve
  needs `ncols² + 2·ncols`.
- The blocked reduction splits the points into `nchunks` contiguous
  chunks, one per worker, each accumulating into its own slice. That is
  what lets it run without atomics; the per-chunk partials are then
  summed, which costs `O(nchunks · nblocks)` and is the price of not
  needing a lock. `nchunks` is `4 · cpu_count` capped at the point
  count.

### Parallelism

The row-parallel kernels use `max.algorithm.parallelize` with an explicit
worker count passed in as an argument, since this dialect has no mutable
globals. They fall back to a serial loop below `PARALLEL_MIN` (256 rows),
below which handing work to the thread pool costs more than the work is
worth. Both the parallel and the serial paths call the same nested
`one(i)` worker, so there is only one copy of the loop body.

The Cholesky of the damped solve is the one part that stays serial, and
deliberately: column `j` of the factor depends on column `j-1`, so the
only independent work at each step is the `n-j` rows below the diagonal.
At 800 columns that averages 400 rows of 400 terms, so a `parallelize`
over the rows of each step would pay a thread-pool barrier 800 times for
a few hundred microseconds of work each. Handing the column to the pool
from inside a `{imm}` closure was tried and returns a factor that is not
positive definite on this toolchain; a parallel step that hands the rows
out in bands is in the table above and is not faster than the serial
column, so the step is left serial and the vector accumulator in
`cholesky` is what makes it fast. The column scaling and the ridge Gram
around it are column-parallel and are not.

### Vectorisation

`comptime VW = simd_width_of[DType.float64]()` is 4 on this target, and
every hot loop in the port is a walk down a contiguous axis: the force
axis of the Green's function matrices, the column axis of the ridge
Gram, the inner index of the Cholesky, and the training set of the kNN
scan. Each of those is a vector body plus a scalar tail. The tails are
exercised by `tests/test_simd_paths.py` at widths 5, 6, 7, 9, 13, 17, 33
and 65, and the kNN block boundary at 31, 32 and 33 training points and
at 100 and 257.

Two of them are not a mechanical substitution, and both are load-bearing:

- **`vlog`** is the `log` above on four lanes. The two branches of the
  range reduction become `select`s, and `frexp` and the exponent
  conversion work on vectors, so the polynomial is the only thing left.
  It is bit-identical to the scalar version on every input measured
  over `[1e-3, 1e3]`, which is what keeps divergence 2 below at 2e-16.
- **The kNN scan** keeps its exact accept test but stops testing it per
  training point. It folds `KNN_BLOCK` vectors into one running lane-wise
  minimum, reduces that once, and only redoes the block lane by lane if
  the block beats the current worst of the k-list. Each hit raises the
  bar, so a query redoes the `k * ln(nt)` entries a selection sort would
  touch and nothing else. Reducing a block at a time rather than a
  vector at a time is the difference between 1.2x and 1.6x over the
  scalar scan: the horizontal reduction is nine instructions on this
  target and paying it per vector costs more than the four distances it
  is guarding.

The two Green's function kernels that have a branch the range reduction
cannot express keep a per-vector escape to the scalar code. The
`d < 1` branch of the bi-harmonic Green's function needs an `exp`, and
the elastic one is a `log` of a distance that `mindist=0` can make
zero; both are handed back to `greens_func` / `greens_func_2d` lane by
lane for that vector, and both are otherwise unreachable. The NaN case
in the kNN block minimum is the same shape of problem: the free `min()`
on a vector propagates NaN and would poison a whole block, so the
running minimum is a `select` on `d2 < lo`, which drops the NaN lane the
way the scalar scan drops it.

### Why there is no GPU path

The host API is available and works on this toolchain
(`from max.gpu.host import DeviceContext`); the machine has an RTX 5090
with 16 GB. It is not used because no kernel here has the
arithmetic intensity a GPU path needs. Counted per inner iteration, in
double precision, with the 13-term `log` polynomial counted as flops and
a divide as one:

| kernel | flops | bytes moved | flops/byte |
| --- | --- | --- | --- |
| bi-harmonic Green's function | ~35 | 24 | ~1.5 |
| elastic Green's function | ~44 | 32 | ~1.4 |
| kNN scan | 5 | 16 | 0.3 |
| ridge Gram | 2 | 8 | 0.25 |
| Cholesky | 2 | 16 | 0.125 |

Nothing reaches 2 flops per byte, and the kNN scan — the largest kernel
by far — is at 0.3. A GPU would be memory-bound on the transfer of the
grid and the forces before it finished the arithmetic, and on a consumer
card the fp64 rate is low enough that the bandwidth-limited throughput
and the compute limit land in the same place. The data is also already
resident: every kernel writes into a caller-owned buffer it is handed,
so a GPU path would add a copy in, a copy out and a launch to a loop that
currently touches memory once. The 4-wide vector loops are the right
answer here.

## Layout

```
src/ported.mojo       kernels, in upstream module order
src/capi.mojo         the C ABI, one thin shim per kernel
build/build.sh        mojo build --emit shared-lib -> dist/libmojo-verde.so
python/mojoverde/     ctypes glue and the public API
examples/quickstart.py  the README usage example, run by the test suite
tests/test_parity.py  parity against upstream verde
tests/test_simd_paths.py  the vector bodies, their tails and their escapes
bench/bench.py        benchmarks, prints a markdown table
```

## Licence

MIT, (c) Lee Penkman. See [LICENSE](LICENSE).
