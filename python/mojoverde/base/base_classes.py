"""``verde.base.base_classes``: the estimator/gridder interfaces."""

import numpy as np

from .utils import check_data


def _grid_coordinates(*args, **kwargs):
    from ..coordinates import grid_coordinates

    return grid_coordinates(*args, **kwargs)


def _scatter_points(*args, **kwargs):
    from ..coordinates import scatter_points

    return scatter_points(*args, **kwargs)


def _profile_coordinates(*args, **kwargs):
    from ..coordinates import profile_coordinates

    return profile_coordinates(*args, **kwargs)


def make_xarray_grid(
    coordinates,
    data,
    data_names,
    dims=("northing", "easting"),
    extra_coords_names=None,
):
    """Create an :class:`xarray.Dataset` grid from 2D numpy arrays.

    Parameters
    ----------
    coordinates : tuple of arrays
        Coordinates of each point in the grid, in the order easting,
        northing, vertical, ... All arrays must be 2d with the same shape.
    data : array, tuple of arrays or None
        The data values on each point of the grid. ``None`` produces a
        dataset with no data variables.
    data_names : str or list
        The name(s) of the data variables.
    dims : list
        The northing and easting dimension names, in that order.
    extra_coords_names : str or list
        Names for the coordinates beyond easting and northing.

    """
    import xarray

    data = check_data(data)
    if not isinstance(data_names, (list, tuple)):
        data_names = [data_names]
    data_names = list(data_names)
    if data_names == ["scalars"] and len(data) > 1:
        data_names = data_names * len(data)
    data_vars = {}
    if data[0] is not None:
        for name, component in zip(data_names, data):
            data_vars[name] = (dims, component)
    if extra_coords_names is None:
        extra_coords_names = []
    for name, extra in zip(extra_coords_names, coordinates[2:]):
        data_vars[name] = (dims, extra)
    return xarray.Dataset(
        data_vars,
        coords={dims[0]: coordinates[1][:, 0], dims[1]: coordinates[0][0, :]},
    )



class BaseGridder:
    """Base class for gridders.

    Provides the ``fit``/``predict`` pair plus the ``filter``, ``grid``,
    ``scatter`` and ``profile`` wrappers, with the same signatures and
    return types as upstream: a grid is an :class:`xarray.Dataset`, while a
    scatter and a profile are :class:`pandas.DataFrame`.

    Attributes
    ----------
    dims : tuple
        The default names of the northing and easting dimensions.
    data_names_defaults : list
        Default data names for 1, 2 and 3 data components.

    """

    dims = ("northing", "easting")
    data_names_defaults = [
        ["scalars"],
        ["east_component", "north_component"],
        ["east_component", "north_component", "vertical_component"],
    ]
    extra_coords_name = "extra_coord"

    def fit(self, coordinates, data, weights=None):  # noqa: U100
        raise NotImplementedError

    def predict(self, coordinates):  # noqa: U100
        raise NotImplementedError

    def filter(self, coordinates, data, weights=None):  # noqa: A003
        """Filter the data through the gridder and produce residuals.

        Calls ``fit`` on the data, evaluates the residuals (data minus
        predicted data), and returns the coordinates, residuals and weights.
        This is what lets gridders join a :class:`~mojoverde.Chain`.
        """
        self.fit(coordinates, data, weights)
        data = check_data(data)
        pred = check_data(self.predict(coordinates))
        residuals = tuple(
            datai - predi.reshape(datai.shape) for datai, predi in zip(data, pred)
        )
        if len(residuals) == 1:
            residuals = residuals[0]
        return coordinates, residuals, weights

    def _get_dims(self, dims):
        """Get default dimension names."""
        if dims is not None:
            return dims
        return self.dims

    def _get_data_names(self, data, data_names):
        """Get default names for data fields if none are given."""
        if data_names is not None:
            return (data_names,) if isinstance(data_names, str) else tuple(data_names)
        if len(data) > len(self.data_names_defaults):
            raise ValueError(
                "Invalid number of data components. "
                f"Only {len(self.data_names_defaults)} supported."
            )
        return tuple(self.data_names_defaults[len(data) - 1])

    def _get_extra_coords_names(self, coordinates):
        """Return names for the coordinates beyond easting and northing."""
        n = len(coordinates) - 2
        if n <= 0:
            return []
        name = self.extra_coords_name
        return [name] + [f"{name}_{i + 1}" for i in range(n - 1)]

    def _get_instance_region(self, region):
        if region is None:
            return getattr(self, "region_", None)
        return region

    def grid(
        self,
        region=None,
        shape=None,
        spacing=None,
        dims=None,
        data_names=None,
        projection=None,
        coordinates=None,
        **kwargs,
    ):
        """Interpolate the data onto a regular grid.

        Parameters
        ----------
        region : list = [W, E, S, N]
            The boundaries of the region. Defaults to the data region.
        shape : tuple = (n_north, n_east) or None
            The number of points in each direction.
        spacing : float, tuple or None
            The grid spacing in each direction.
        dims : list or None
            The names of the northing and easting data dimensions.
        data_names : str, list or None
            The name(s) of the data variables in the output grid.
        projection : callable or None
            ``projection(easting, northing) -> (proj_easting, proj_northing)``
            applied to the grid before predicting.
        coordinates : tuple of arrays
            Grid coordinates. If given, ``region``, ``shape`` and ``spacing``
            are ignored.
        **kwargs
            Extra arguments for :func:`~mojoverde.grid_coordinates`.

        Returns
        -------
        grid : xarray.Dataset

        """
        if coordinates is not None and (spacing is not None or shape is not None):
            raise ValueError(
                "Both coordinates and spacing or shape were provided. "
                "Please pass only coordinates or either the spacing or the shape."
            )
        if coordinates is not None and region is not None:
            raise ValueError(
                "Both coordinates and region were provided. "
                "Please pass region only if spacing or shape is specified."
            )
        if coordinates is None:
            region = self._get_instance_region(region)
            coordinates = _grid_coordinates(
                region, shape=shape, spacing=spacing, **kwargs
            )
        else:
            coordinates = tuple(np.ravel(np.atleast_1d(i)) for i in coordinates)
        if projection is not None:
            coordinates = projection(*coordinates[:2]) + tuple(coordinates[2:])
        data = check_data(self.predict(coordinates))
        dims = self._get_dims(dims)
        data_names = self._get_data_names(data, data_names)
        extra_coords_names = self._get_extra_coords_names(coordinates)
        dataset = make_xarray_grid(
            coordinates,
            data,
            data_names,
            dims=dims,
            extra_coords_names=extra_coords_names,
        )
        metadata = f"Generated by {self!r}"
        dataset.attrs["metadata"] = metadata
        for array in dataset:
            dataset[array].attrs["metadata"] = metadata
        return dataset


    def scatter(
        self,
        region=None,
        size=300,
        random_state=0,
        dims=None,
        data_names=None,
        projection=None,
        **kwargs,
    ):
        """Interpolate values onto a random scatter of points.

        Parameters
        ----------
        region : list = [W, E, S, N]
            The boundaries of the region. Defaults to the data region.
        size : int
            The number of points.
        random_state : int
            The seed for the random number generator.
        dims : list or None
            The names of the northing and easting columns.
        data_names : str, list or None
            The name(s) of the data columns.
        projection : callable or None
            Applied to the scatter before predicting.
        **kwargs
            Extra arguments for :func:`~mojoverde.scatter_points`.

        Returns
        -------
        table : pandas.DataFrame

        """
        import warnings

        import pandas

        warnings.warn(
            "The 'scatter' method is deprecated and will be removed in Verde "
            "2.0.0. Use 'verde.scatter_points' and the 'predict' method "
            "instead.",
            FutureWarning,
            stacklevel=2,
        )
        dims = self._get_dims(dims)
        region = self._get_instance_region(region)
        coordinates = _scatter_points(
            region, size, random_state=random_state, **kwargs
        )
        if projection is None:
            data = check_data(self.predict(coordinates))
        else:
            data = check_data(self.predict(projection(*coordinates[:2])))
        data_names = self._get_data_names(data, data_names)
        columns = [(dims[0], coordinates[1]), (dims[1], coordinates[0])]
        extra_coords_names = self._get_extra_coords_names(coordinates)
        columns.extend(zip(extra_coords_names, coordinates[2:]))
        columns.extend(zip(data_names, data))
        return pandas.DataFrame(dict(columns), columns=[c[0] for c in columns])

    def profile(
        self,
        point1,
        point2,
        size,
        dims=None,
        data_names=None,
        projection=None,
        **kwargs,
    ):
        """Interpolate data along a profile between two points.

        Parameters
        ----------
        point1, point2 : tuple
            The (easting, northing) coordinates of the profile ends.
        size : int
            The number of points to generate.
        dims : list or None
            The names of the northing and easting columns.
        data_names : str, list or None
            The name(s) of the data columns.
        projection : callable or None
            Applied to the profile ends before generating the coordinates.
        **kwargs
            Extra arguments for :func:`~mojoverde.profile_coordinates`.

        Returns
        -------
        table : pandas.DataFrame

        """
        import pandas

        dims = self._get_dims(dims)
        if projection is not None:
            point1 = projection(*np.atleast_1d(point1))
            point2 = projection(*np.atleast_1d(point2))
        coordinates, distances = _profile_coordinates(point1, point2, size, **kwargs)
        data = check_data(self.predict(coordinates))
        if projection is not None:
            coordinates = projection(*coordinates[:2], inverse=True)
        data_names = self._get_data_names(data, data_names)
        columns = [
            (dims[0], coordinates[1]),
            (dims[1], coordinates[0]),
            ("distance", distances),
        ]
        extra_coords_names = self._get_extra_coords_names(coordinates)
        columns.extend(zip(extra_coords_names, coordinates[2:]))
        columns.extend(zip(data_names, data))
        return pandas.DataFrame(dict(columns), columns=[c[0] for c in columns])
