#!/usr/bin/env python

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":  # pragma: no cover
    sys.argv = [sys.argv[0], "--reference", "qa", *sys.argv[1:]]
    runpy.run_path(str(Path(__file__).with_name("QH_fixed_resolution_jax.py")), run_name="__main__")