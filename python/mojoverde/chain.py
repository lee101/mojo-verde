"""Port of ``verde.chain``."""

from .base import BaseGridder, check_data
from .coordinates import get_region


class Chain(BaseGridder):
    """Chain filtering operations to fit on each subsequent output.

    Each step is fitted to the residuals of the previous one, which is what
    makes a trend-then-gridder pipeline possible. When predicting, the
    predictions of every step that has a ``predict`` method are added
    together.

    Parameters
    ----------
    steps : list
        A list of ``('name', step)`` pairs where ``step`` is any class that
        implements a ``filter(coordinates, data, weights)`` method
        (including ``Chain`` itself).

    Attributes
    ----------
    region_ : tuple
        The boundaries of the data used to fit the chain.
    named_steps : dict
        The steps, keyed by their names.

    """

    def __init__(self, steps):
        self.steps = steps

    @property
    def named_steps(self):
        """A dictionary version of *steps*."""
        return dict(self.steps)

    def fit(self, coordinates, data, weights=None):
        """Fit the chained estimators to the given data.

        The coordinates are preserved; only the data values change.
        """
        self.region_ = get_region(coordinates[:2])
        args = coordinates, data, weights
        for _, step in self.steps:
            args = step.filter(*args)
        return self

    def predict(self, coordinates):
        """Evaluate the data predicted by the chain on the given points."""
        result = None
        for _, step in self.steps:
            if not hasattr(step, "predict"):
                continue
            predicted = check_data(step.predict(coordinates))
            if result is None:
                result = [0 for _ in predicted]
            for i, pred in enumerate(predicted):
                result[i] = result[i] + pred
        if result is not None and len(result) == 1:
            return result[0]
        return result
