"""The README usage example, so the two cannot drift apart.

Run it with ``pixi run python examples/quickstart.py``. It also runs the
same pipeline through upstream verde and prints the difference.
"""

import numpy as np

import mojoverde as mv
import verde as vd

REGION = (0, 10, -5, 5)


def main() -> None:
    coordinates = mv.scatter_points(REGION, 2000, random_state=0)
    easting, northing = coordinates[0], coordinates[1]
    data = np.sin(easting / 3) * np.cos(northing / 4) * 10

    # Decimate to 0.1-degree blocks, then grid with a biharmonic spline.
    chain = mv.Chain(
        [
            ("mean", mv.BlockMean(spacing=0.1)),
            ("spline", mv.Spline(damping=1e-3)),
        ]
    )
    chain.fit(coordinates, data)
    grid = chain.grid(REGION, spacing=0.05)
    print("mojo-verde grid", grid.scalars.values.shape)

    their_chain = vd.Chain(
        [
            ("mean", vd.BlockMean(spacing=0.1)),
            ("spline", vd.Spline(damping=1e-3)),
        ]
    )
    their_chain.fit(coordinates, data)
    their_grid = their_chain.grid(REGION, spacing=0.05)
    print("verde grid", their_grid.scalars.values.shape)
    print(
        "max abs diff",
        float(np.abs(grid.scalars.values - their_grid.scalars.values).max()),
    )

    nn = mv.KNeighbors(k=4).fit(coordinates, data)
    nearest = nn.predict(mv.grid_coordinates(REGION, shape=(201, 201)))
    print("nearest", nearest.shape)

    mask = mv.distance_mask(coordinates, 0.5, coordinates=coordinates)
    print("mask", mask.shape, int(mask.sum()))


if __name__ == "__main__":
    main()
