"""Fresh-process child: the removed float32 backend mode is rejected.

Usage: ``removed_float32_mode.py {set_backend,environment}``. For
``environment`` the parent exports ``SIMSOPT_BACKEND_MODE`` with the removed
mode before this process starts.
"""

import sys
import unittest

from simsopt_jax.backend import get_backend_config, set_backend


def main(selector: str) -> None:
    """Select the removed mode explicitly or through the environment."""
    with unittest.TestCase().assertRaisesRegex(
        ValueError, "Backend mode 'jax_cpu_float32_smoke' is not valid. Accepted:"
    ) as error:
        if selector == "set_backend":
            # The mode is deliberately outside the accepted ``BackendMode`` values.
            set_backend("jax_cpu_float32_smoke")  # pyright: ignore[reportArgumentType]
        else:
            get_backend_config()
    assert "jax_cpu_float32_smoke" not in str(error.exception).split("Accepted:", 1)[1]


if __name__ == "__main__":
    main(sys.argv[1])
