"""Port of ``verde.blockreduce``.

Upstream reduces in blocks through a ``pandas`` ``groupby``, which costs a
Python-level call per group. The same arithmetic here is a pair of binned
accumulations split across worker-local chunks, so no atomics are needed,
and the run of occupied block labels is compacted the way ``groupby`` sorts
keys and drops the empty ones.
"""

import numpy as np

from ._lib import WORKERS, f64, lib
from .base import check_fit_input
from .coordinates import block_split
from .utils import variance_to_weights

# Upstream's BlockReduce takes an arbitrary callable as *reduction*. Only
# the reductions that reduce to a binned mean, or to a single grouped pass,
# are ported; the rest raise rather than silently returning something else.
_REDUCTIONS = {
    np.mean: "mean",
    np.average: "average",
    np.median: "median",
    np.max: "max",
    np.min: "min",
}


def _nchunks(n):
    return max(1, min(n, 4 * WORKERS))


def _binned_mean(labels, values, weights, nblocks, ncomps, weighted):
    """Per-block count, sum(w) and weighted mean."""
    n = labels.size
    nchunks = _nchunks(n)
    mean = np.empty(ncomps * nblocks, dtype="float64")
    sum_w = np.empty(ncomps * nblocks, dtype="float64")
    count = np.empty(nblocks, dtype="float64")
    work = np.zeros(nchunks * nblocks * 3 * ncomps, dtype="float64")
    lib().mv_binned_mean(
        (labels), (values), (weights), (mean), (sum_w),
        (count), (work), n, ncomps, nblocks, nchunks, int(weighted),
        int(WORKERS),
    )
    return mean, sum_w, count


def _binned_variance(labels, values, weights, mean, nblocks, ncomps, weighted, ddof):
    """Per-block sum of w * (v - mean)**2; the division happens in the caller."""
    n = labels.size
    nchunks = _nchunks(n)
    total = np.empty(ncomps * nblocks, dtype="float64")
    work = np.zeros(nchunks * nblocks * 3 * ncomps, dtype="float64")
    lib().mv_binned_variance(
        (labels), (values), (weights), (mean), (total),
        (work), n, ncomps, nblocks, nchunks, int(weighted), float(ddof),
        int(WORKERS),
    )
    return total


def _compact(count, nblocks):
    """The occupied block labels, in increasing order, and their positions."""
    block_ids = np.empty(nblocks, dtype="int64")
    block_index = np.empty(nblocks, dtype="int64")
    n = lib().mv_block_compact(
        (count), (block_ids), (block_index), nblocks
    )
    return block_ids[:n], block_index


def _grouped(values, labels, block_ids, nblocks, name, weights=None):
    """Apply *name* to the values of each occupied block.

    An unweighted mean comes from the binned accumulation; every other
    reduction, and every weighted one, needs the grouped values, which this
    gathers with one argsort, the same ordering pandas' groupby builds.
    """
    if name == "mean" and weights is None:
        mean, _, _ = _binned_mean(
            labels, f64(values).ravel(), np.ones(labels.size), nblocks, 1, 0
        )
        return mean[block_ids]
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    starts = np.searchsorted(sorted_labels, block_ids, side="left")
    stops = np.searchsorted(sorted_labels, block_ids, side="right")
    grouped = f64(values).ravel()[order]
    func = getattr(np, name)
    out = np.empty(block_ids.size, dtype="float64")
    for b in range(block_ids.size):
        chunk = grouped[starts[b] : stops[b]]
        if weights is not None:
            out[b] = func(chunk, weights=weights[order][starts[b] : stops[b]])
        else:
            out[b] = func(chunk)
    return out


class BlockReduce:
    """Apply a reduction/aggregation operation to the data in blocks/windows.

    Parameters
    ----------
    reduction : function
        A reduction function that takes an array and returns a single value.
        One of ``numpy.mean``, ``numpy.average``, ``numpy.median``,
        ``numpy.max`` or ``numpy.min``.
    shape : tuple = (n_north, n_east) or None
        The number of blocks in the South-North and West-East directions.
    spacing : float, tuple = (s_north, s_east), or None
        The block size in both directions.
    region : list = [W, E, S, N]
        The boundaries of a given region.
    adjust : {'spacing', 'region'}
        Whether to adjust the spacing or the region if required.
    center_coordinates : bool
        If True, return the center of each block instead of reducing the
        input coordinates.
    drop_coords : bool
        If True, only the reduced ``easting`` and ``northing`` are returned.

    """

    def __init__(
        self,
        reduction,
        spacing=None,
        region=None,
        adjust="spacing",
        center_coordinates=False,
        shape=None,
        drop_coords=True,
    ):
        self.reduction = reduction
        self.shape = shape
        self.spacing = spacing
        self.region = region
        self.adjust = adjust
        self.center_coordinates = center_coordinates
        self.drop_coords = drop_coords

    def _reduction_name(self):
        name = _REDUCTIONS.get(self.reduction)
        if name is None:
            raise ValueError(
                "Unsupported reduction. mojo-verde ports numpy.mean, "
                "numpy.average, numpy.median, numpy.max and numpy.min; got "
                f"{self.reduction!r}."
            )
        return name

    def _block_coordinates(self, coordinates, blocks, labels, block_ids, nblocks):
        """One coordinate per occupied block.

        Doing the coordinates separately from the data because in case of
        weights the reduction applied to them is different (no weights ever).
        """
        name = self._reduction_name()
        out = []
        for i, coord in enumerate(coordinates):
            if self.center_coordinates and i < 2:
                out.append(blocks[i][block_ids])
            else:
                out.append(_grouped(coord, labels, block_ids, nblocks, name))
        return tuple(out)

    def filter(self, coordinates, data, weights=None):  # noqa: A003
        """Apply the blocked aggregation to the given data."""
        name = self._reduction_name()
        coordinates, data, weights = check_fit_input(
            coordinates, data, weights, unpack=False
        )
        weighted = any(w is not None for w in weights)
        if weighted and name != "average":
            raise ValueError(
                f"'{name}' does not accept weights. Use BlockMean or"
                " numpy.average."
            )
        blocks, labels = block_split(
            coordinates,
            spacing=self.spacing,
            shape=self.shape,
            adjust=self.adjust,
            region=self.region,
        )
        nblocks = int(labels.max()) + 1
        count = np.bincount(labels, minlength=nblocks).astype("float64")
        block_ids, _ = _compact(count, nblocks)
        blocked_data = tuple(
            _grouped(comp, labels, block_ids, nblocks, name, weight)
            for comp, weight in zip(data, weights)
        )
        if self.drop_coords:
            coordinates = coordinates[:2]
        blocked_coords = self._block_coordinates(
            coordinates, blocks, labels, block_ids, nblocks
        )
        if len(blocked_data) == 1:
            return blocked_coords, blocked_data[0]
        return blocked_coords, blocked_data


class BlockMean(BlockReduce):
    """Apply a (weighted) mean to the data in blocks/windows.

    Returns the mean data value for each block along with the associated
    coordinates and weights. The blocked weights are the inverse variance
    of the block, propagated from the input weights, or the inverse weighted
    variance, depending on *uncertainty* and on whether weights were given.
    All output weights are scaled to the range [0, 1].

    Parameters
    ----------
    spacing : float, tuple = (s_north, s_east), or None
        The block size in both directions.
    region : list = [W, E, S, N]
        The boundaries of a given region.
    adjust : {'spacing', 'region'}
        Whether to adjust the spacing or the region if required.
    center_coordinates : bool
        If True, return the center of each block.
    uncertainty : bool
        If True, propagate the uncertainty of the data, which requires the
        input weights to be ``1/uncertainty**2``.
    shape : tuple = (n_north, n_east) or None
        The number of blocks in the South-North and West-East directions.
    drop_coords : bool
        If True, only the reduced ``easting`` and ``northing`` are returned.

    """

    def __init__(
        self,
        spacing=None,
        region=None,
        adjust="spacing",
        center_coordinates=False,
        uncertainty=False,
        shape=None,
        drop_coords=True,
    ):
        super().__init__(
            reduction=np.average,
            shape=shape,
            spacing=spacing,
            region=region,
            adjust=adjust,
            center_coordinates=center_coordinates,
            drop_coords=drop_coords,
        )
        self.uncertainty = uncertainty

    def filter(self, coordinates, data, weights=None):  # noqa: A003
        """Apply the blocked mean to the given data."""
        coordinates, data, weights = check_fit_input(
            coordinates, data, weights, unpack=False
        )
        if any(w is None for w in weights) and self.uncertainty:
            raise ValueError(
                "Weights are required for uncertainty propagation. "
                "Either provide weights (as 1/uncertainty**2) or use "
                "'uncertainty=False' to produce variance weights instead."
            )
        blocks, labels = block_split(
            coordinates,
            spacing=self.spacing,
            shape=self.shape,
            adjust=self.adjust,
            region=self.region,
        )
        ncomps = len(data)
        nblocks = int(labels.max()) + 1
        values = f64(np.stack([np.ravel(c) for c in data], axis=1).ravel())
        weighted = int(any(w is not None for w in weights))
        wvals = np.ones(labels.size, dtype="float64")
        if weighted:
            wvals = f64(np.concatenate([np.ravel(w) for w in weights]))
        mean, sum_w, count = _binned_mean(
            labels, values, wvals, nblocks, ncomps, weighted
        )
        block_ids, _ = _compact(count, nblocks)
        if self.uncertainty:
            # The reduction of the weights is the propagated uncertainty
            # squared: 1/(sum(1/uncertainty**2)) = 1/sum(weights).
            variance = np.stack(
                [1.0 / sum_w[c * nblocks + block_ids] for c in range(ncomps)]
            )
        else:
            ddof = 0.0 if weighted else 1.0
            total = _binned_variance(
                labels, values, wvals, mean, nblocks, ncomps, weighted, ddof
            )
            # A block holding a single point has a zero denominator, and
            # pandas' .var gives NaN for it too, so the NaN is the answer
            # rather than a warning.
            with np.errstate(invalid="ignore", divide="ignore"):
                if weighted:
                    # np.average of the squared deviations, no correction.
                    variance = np.stack(
                        [
                            total[c * nblocks + block_ids]
                            / sum_w[c * nblocks + block_ids]
                            for c in range(ncomps)
                        ]
                    )
                else:
                    # pandas' .var, i.e. the sample variance with ddof=1.
                    counts = count[block_ids] - ddof
                    variance = np.stack(
                        [
                            total[c * nblocks + block_ids] / counts
                            for c in range(ncomps)
                        ]
                    )
        blocked_data = tuple(mean[c * nblocks + block_ids] for c in range(ncomps))
        blocked_weights = tuple(variance_to_weights(var) for var in variance)
        if self.drop_coords:
            coordinates = coordinates[:2]
        blocked_coords = self._block_coordinates(
            coordinates, blocks, labels, block_ids, nblocks
        )
        if ncomps == 1:
            return blocked_coords, blocked_data[0], blocked_weights[0]
        return blocked_coords, blocked_data, blocked_weights
