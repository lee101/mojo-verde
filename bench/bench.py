"""mojo-verde against upstream verde on the same data.

Run it with ``pixi run bench``: that task holds a machine-wide flock so a
concurrent factory job cannot distort the numbers. The table is markdown so
it can be pasted straight into the README.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import mojoverde as mv  # noqa: E402
import verde as vd  # noqa: E402
from verde.base.least_squares import least_squares as verde_least_squares  # noqa: E402

REPEAT = 5


def timeit(fn, repeat: int = REPEAT) -> float:
    """Best of *repeat* wall-clock seconds, in seconds."""
    fn()
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def scatter(n: int, region=(0, 10, -5, 5), seed: int = 0):
    return mv.scatter_points(region, n, random_state=seed)


def field(coordinates, seed: int = 0):
    east, north = coordinates[0], coordinates[1]
    return np.sin(east / 3) * np.cos(north / 4) * 10 + seed


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn

    return deco


@case("Spline.jacobian (4000 forces)")
def _():
    coords = scatter(4000)
    ours = mv.Spline()
    theirs = vd.Spline()
    forces = coords
    return (
        lambda: ours.jacobian(coords, forces),
        lambda: theirs.jacobian(coords, forces),
    )


@case("Spline.predict (200k x 4000 forces)")
def _():
    coords = scatter(4000)
    grid = mv.grid_coordinates((0, 10, -5, 5), shape=(200_000, 1))
    forces = np.linspace(-1.0, 1.0, 4000)
    ours = mv.Spline()
    ours.force_coords_ = (np.linspace(0, 10, 4000), np.linspace(-5, 5, 4000))
    ours.force_ = forces
    theirs = vd.Spline()
    theirs.force_coords_ = ours.force_coords_
    theirs.force_ = forces
    return lambda: ours.predict(grid), lambda: theirs.predict(grid)


@case("VectorSpline2D.jacobian (2000 points)")
def _():
    coords = scatter(2000)
    ours = mv.VectorSpline2D(poisson=0.25, mindist=1e3)
    theirs = vd.VectorSpline2D(poisson=0.25, mindist=1e3)
    forces = (np.ravel(coords[0]), np.ravel(coords[1]))
    return (
        lambda: ours.jacobian(coords, forces),
        lambda: theirs.jacobian(coords, forces),
    )


@case("VectorSpline2D.predict (200k x 2000 forces)")
def _():
    coords = scatter(2000)
    grid = mv.grid_coordinates((0, 10, -5, 5), shape=(200_000, 1))
    ours = mv.VectorSpline2D(poisson=0.25, mindist=1e3)
    theirs = vd.VectorSpline2D(poisson=0.25, mindist=1e3)
    ours.force_coords = theirs.force_coords = (
        np.linspace(0, 10, 2000),
        np.linspace(-5, 5, 2000),
    )
    ours.force_ = theirs.force_ = np.linspace(-1, 1, 4000)
    return lambda: ours.predict(grid), lambda: theirs.predict(grid)


@case("KNeighbors.predict k=1 (50k train, 100k query)")
def _():
    coords = scatter(50_000)
    data = field(coords)
    query = mv.grid_coordinates((0, 10, -5, 5), shape=(100_000, 1))
    ours = mv.KNeighbors(k=1).fit(coords, data)
    theirs = vd.KNeighbors(k=1).fit(coords, data)
    return lambda: ours.predict(query), lambda: theirs.predict(query)


@case("KNeighbors.predict k=8 (50k train, 100k query)")
def _():
    coords = scatter(50_000)
    data = field(coords)
    query = mv.grid_coordinates((0, 10, -5, 5), shape=(100_000, 1))
    ours = mv.KNeighbors(k=8).fit(coords, data)
    theirs = vd.KNeighbors(k=8).fit(coords, data)
    return lambda: ours.predict(query), lambda: theirs.predict(query)


@case("distance_mask (50k data, 400k grid)")
def _():
    coords = scatter(50_000)
    grid = mv.grid_coordinates((0, 10, -5, 5), shape=(632, 632))
    return (
        lambda: mv.distance_mask(coords, 0.5, coordinates=grid),
        lambda: vd.distance_mask(coords, 0.5, coordinates=grid),
    )


@case("BlockMean.filter (1M points, 100x100 blocks)")
def _():
    coords = scatter(1_000_000)
    data = field(coords)
    kwargs = {"spacing": 0.1}
    ours = mv.BlockMean(**kwargs)
    theirs = vd.BlockMean(**kwargs)
    return lambda: ours.filter(coords, data), lambda: theirs.filter(coords, data)


@case("BlockReduce(np.mean).filter (1M points, 100x100 blocks)")
def _():
    coords = scatter(1_000_000)
    data = field(coords)
    kwargs = {"reduction": np.mean, "spacing": 0.1}
    ours = mv.BlockReduce(**kwargs)
    theirs = vd.BlockReduce(**kwargs)
    return lambda: ours.filter(coords, data), lambda: theirs.filter(coords, data)


@case("block_split (1M points, spacing 0.1)")
def _():
    coords = scatter(1_000_000)
    return (
        lambda: mv.block_split(coords, spacing=0.1),
        lambda: vd.block_split(coords, spacing=0.1),
    )


@case("rolling_window (200k points, size 0.4, spacing 0.1)")
def _():
    coords = scatter(200_000)
    return (
        lambda: mv.rolling_window(coords, size=0.4, spacing=0.1),
        lambda: vd.rolling_window(coords, size=0.4, spacing=0.1),
    )


@case("expanding_window (1M points, 3 sizes)")
def _():
    coords = scatter(1_000_000)
    return (
        lambda: mv.expanding_window(coords, center=(5, 0), sizes=[0.2, 0.5, 1.0]),
        lambda: vd.expanding_window(coords, center=(5, 0), sizes=[0.2, 0.5, 1.0]),
    )


@case("CheckerBoard.predict (1M points)")
def _():
    coords = scatter(1_000_000)
    ours = mv.CheckerBoard()
    theirs = vd.synthetic.CheckerBoard()
    return lambda: ours.predict(coords), lambda: theirs.predict(coords)


@case("grid_coordinates (1.00M points, spacing 0.01)")
def _():
    region = (0, 10, -5, 5)
    return (
        lambda: mv.grid_coordinates(region, spacing=0.01),
        lambda: vd.grid_coordinates(region, spacing=0.01),
    )


@case("variance_to_weights (4M points)")
def _():
    rng = np.random.default_rng(0)
    variance = rng.uniform(0.0, 1.0, 4_000_000)
    return (
        lambda: mv.variance_to_weights(variance),
        lambda: vd.variance_to_weights(variance),
    )


@case("least_squares damped (800x800 biharmonic)")
def _():
    coords = scatter(800)
    data = field(coords)
    jac = mv.Spline().jacobian(coords, coords)
    rng = np.random.default_rng(1)
    weights = rng.uniform(0.5, 2.0, 800)
    return (
        lambda: mv.least_squares(jac.copy(), data, weights, 1e-2),
        lambda: verde_least_squares(jac.copy(), data, weights, 1e-2),
    )


@case("Spline.fit (1500 points, damping 1e-3)")
def _():
    coords = scatter(1500)
    data = field(coords)
    grid = mv.grid_coordinates((0, 10, -5, 5), spacing=0.05)
    return (
        lambda: mv.Spline(damping=1e-3).fit(coords, data).predict(grid),
        lambda: vd.Spline(damping=1e-3).fit(coords, data).predict(grid),
    )


def main() -> None:
    print(f"Python {platform.python_version()} - numpy {np.__version__} - "
          f"verde {vd.__version__}")
    print(f"{platform.system()} {platform.machine()}, "
          f"{os.cpu_count()} logical CPUs\n")
    print(f"| {'case':<48} | {'mojo-verde':>12} | {'verde':>12} | {'speedup':>8} |")
    print(f"| {'-' * 48} | {'-' * 12} | {'-' * 12} | {'-' * 8} |")
    for name, fn in CASES:
        ours, theirs = fn()
        a = timeit(ours)
        b = timeit(theirs)
        flag = " (slower)" if a > b else ""
        print(
            f"| {name:<48} | {a * 1e3:>9.1f} ms | {b * 1e3:>9.1f} ms | "
            f"{b / a:>7.2f}x{flag} |"
        )


if __name__ == "__main__":
    main()
