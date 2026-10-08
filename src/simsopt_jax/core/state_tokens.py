"""Small state-token factories for cache invalidation and drift detection."""

from __future__ import annotations

from collections.abc import Callable
from itertools import count


def make_state_token_factory() -> Callable[[], int]:
    """Return an independent monotonic integer token generator."""
    counter = count()

    def new_state_token() -> int:
        return next(counter)

    return new_state_token
