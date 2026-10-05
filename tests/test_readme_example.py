"""Runs the README usage example so the documentation cannot drift.

The example runs in this process rather than in a subprocess: a second
interpreter means a second MAX thread pool, and on a shared box with a 512
pid budget the pair of pools is enough to lose a race against the other
jobs for pids.
"""

import runpy
import warnings

import pytest

ROOT = "examples/quickstart.py"


def test_readme_example_runs(capsys):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runpy.run_path(ROOT, run_name="__main__")
    out = capsys.readouterr().out
    runtime = [
        str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)
    ]
    assert not runtime, runtime
    assert "mojo-verde grid (201, 201)" in out
    assert "verde grid (201, 201)" in out
    assert "nearest (201, 201)" in out


@pytest.mark.parametrize("name", ["README.md"])
def test_readme_uses_the_example(name):
    with open(name, encoding="utf-8") as fh:
        text = fh.read()
    # The block the README tells the reader to run.
    assert "np.sin(easting / 3) * np.cos(northing / 4) * 10" in text
    assert "north / 4" not in text.replace("northing / 4", "")
