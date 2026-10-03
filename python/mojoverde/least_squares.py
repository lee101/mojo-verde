"""Port of ``verde.base.least_squares``."""

import warnings

import numpy as np

from ._lib import WORKERS, f64, lib


def least_squares(jacobian, data, weights, damping=None, copy_jacobian=False):
    """Solve a weighted least-squares problem with optional damping regularization.

    Scales the Jacobian matrix so that each column has unit variance, solves
    the (optionally damped) system, then undoes the scaling on the estimated
    parameters.

    The undamped branch is ``sklearn.LinearRegression(fit_intercept=False)``,
    which hands ``scipy.linalg.lstsq`` the same matrix with
    ``cond=LinearRegression.tol``. That default ``tol`` of 1e-6 is the
    effective-rank cutoff rather than a convergence tolerance, and it is
    what keeps the near-null space of a biharmonic Jacobian from
    exploding the estimated forces. There is no faithful small-kernel
    equivalent of an SVD, so that one call goes to LAPACK through numpy;
    the column scaling and the damped branch both run in Mojo.
    """
    if jacobian.shape[0] < jacobian.shape[1]:
        warnings.warn(
            "Under-determined problem detected (ndata, nparams)={}.".format(
                jacobian.shape
            )
        )
    nrows, ncols = jacobian.shape
    if nrows < 1 or ncols < 1:
        # Upstream hands the matrix to sklearn's StandardScaler, which rejects
        # a sample with no rows. The kernels here divide by the row count to
        # form the column mean, so an empty problem has to be refused here
        # rather than turned into a silent array of NaNs.
        raise ValueError(
            f"Found array with {nrows} sample(s) (shape={jacobian.shape}) while a "
            "minimum of 1 is required to fit the column scaling."
        )
    if copy_jacobian:
        jac = f64(jacobian).copy()
    else:
        jac = f64(jacobian)
    y = f64(np.ravel(data))
    if weights is not None:
        w = f64(np.ravel(weights))
        weighted = 1
    else:
        w = np.ones(nrows, dtype="float64")
        weighted = 0
    mean = np.empty(ncols, dtype="float64")
    scale = np.empty(ncols, dtype="float64")
    if damping is None:
        lib().mv_ls_scale_fit(
            (jac), (scale), (mean), nrows, ncols, int(WORKERS)
        )
        lib().mv_ls_scale_transform(
            (jac), (scale), nrows, ncols, int(WORKERS)
        )
        sw = np.sqrt(w) if weighted else np.ones(nrows)
        coef = np.linalg.lstsq(jac * sw[:, np.newaxis], y * sw, rcond=1e-6)[0]
        return coef / scale

    params = np.empty(ncols, dtype="float64")
    work = np.zeros(ncols * ncols + 2 * ncols, dtype="float64")
    ok = lib().mv_ls_least_squares(
        (jac),
        (y),
        (w),
        (params),
        (scale),
        (mean),
        (work),
        nrows,
        ncols,
        float(damping),
        int(weighted),
        int(WORKERS),
    )
    if not ok:
        raise np.linalg.LinAlgError(
            "The damped least-squares system is not positive definite. "
            "Increase 'damping' or use undamped 'least_squares'."
        )
    return params
