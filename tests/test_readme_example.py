"""Runs the README usage example so the documentation cannot drift."""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_readme_example_runs():
    script = os.path.join(ROOT, "examples", "quickstart.py")
    proc = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert "mojo-verde grid (201, 201)" in proc.stdout
    assert "verde grid (201, 201)" in proc.stdout
    assert "RuntimeWarning" not in proc.stderr
