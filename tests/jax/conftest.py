"""Local JAX pytest configuration and assertion rewriting; unittest never imports it."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT_STR = str(_REPO_ROOT)
while _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Rewrite asserts in the helpers that JAX test modules import from there.
pytest.register_assert_rewrite("jax_test_support")
