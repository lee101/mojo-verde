"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np
from numpy.ctypeslib import ndpointer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJOVERDE_LIB") or os.path.join(
    ROOT, "dist", "libmojo-verde.so"
)

I = ctypes.c_int64
F = ctypes.c_double

# Buffer parameters are declared as typed pointers rather than as bare
# integers. ctypes holds a reference to the array for the duration of the
# call, which is what keeps a converted temporary alive while the kernel
# reads it, and the dtype/contiguity flags turn a wrong buffer into a
# Python-level error instead of a wild pointer.
_F64 = ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_I64 = ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
_U8 = ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")

# name -> (argtypes, restype)
_SIGNATURES = {
    "mv_ls_scale_fit": ([_F64] * 3 + [I] * 3, None),
    "mv_ls_scale_transform": ([_F64] * 2 + [I] * 3, None),
    "mv_ls_least_squares": ([_F64] * 7 + [I, I, F, I, I], I),
    "mv_block_compact": ([_F64, _I64, _I64, I], I),
    "mv_binned_mean": ([_I64] + [_F64] * 6 + [I] * 6, None),
    "mv_binned_variance": ([_I64] + [_F64] * 5 + [I] * 5 + [F, I], None),
    "mv_line_coordinates": ([F, F, I, _F64, I], None),
    "mv_meshgrid_2d": ([_F64] * 4 + [I] * 3, None),
    "mv_get_region": ([_F64] * 3 + [I], None),
    "mv_inside": ([_F64, _F64, _U8, I, F, F, F, F, I], None),
    "mv_longitude_continuity": ([_F64] * 2 + [I] * 3, None),
    "mv_block_split": ([_F64, _F64, _I64, I] + [F] * 4 + [I, I], None),
    "mv_knn_query": ([_F64] * 5 + [_I64] + [I] * 4, None),
    "mv_greens_func": ([_F64] * 3 + [I, F, I], None),
    "mv_jacobian": ([_F64] * 5 + [I, I, F, I], None),
    "mv_predict": ([_F64] * 6 + [I, I, F, I], None),
    "mv_variance_to_weights": ([_F64] * 2 + [I, F], None),
    "mv_jacobian_2d": ([_F64] * 5 + [I, I, F, F, I], None),
    "mv_predict_2d": ([_F64] * 7 + [I, I, F, F, I], None),
    "mv_expanding_window_select": ([_F64, _F64, _I64, I, F, F, F], I),
    "mv_rolling_window_count": ([_F64] * 4 + [_I64] + [I] * 3 + [F, I, I], None),
    "mv_rolling_window_fill": ([_F64] * 4 + [_I64] * 2 + [I] * 3 + [F, I, I], None),
    "mv_checkerboard": ([_F64] * 3 + [I] + [F] * 3 + [I], None),
}

# Worker count handed to every kernel that has a parallel branch. One
# worker silently serialises all of them, which is why this is never 1.
# The cap is measured, not guessed: 72 workers buy under 5% over 32 on
# the kNN scan and lose on the O(n.m) prediction. Override with
# MOJOVERDE_WORKERS. Note that this caps the work items, not the threads:
# the first parallel call opens MAX's own pool, which is sized from the
# machine and is roughly 80 threads whatever this number says, so the
# 512-pid cgroup budget of a shared CI box is the real limit.
MAX_WORKERS = 32


def _worker_count() -> int:
    env = os.environ.get("MOJOVERDE_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, min(MAX_WORKERS, len(os.sched_getaffinity(0))))


WORKERS = _worker_count()


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOVERDE_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    manifest = os.path.join(ROOT, "pixi.toml")
    if os.path.exists(pixi) and os.path.exists(manifest):
        return [pixi, "run", "--manifest-path", manifest, "mojo"]
    raise BuildError("mojo not found; set MOJOVERDE_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `dist/libmojo-verde.so` if it is stale."""
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        "-I",
        SRC,
        os.path.join(SRC, "capi.mojo"),
        "-o",
        LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            func = getattr(_lib, name)
            func.argtypes = argtypes
            func.restype = restype
    return _lib


def f64(a) -> np.ndarray:
    """A C-contiguous float64 view of `a`, copying only when it has to."""
    arr = np.asarray(a, dtype=np.float64)
    if not arr.flags["C_CONTIGUOUS"] or arr.dtype != np.float64:
        arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr


def main() -> int:
    """`python -m mojoverde._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
