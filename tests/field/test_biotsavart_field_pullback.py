"""Biot-Savart pullback pytree leaves and static coil metadata."""

from __future__ import annotations

from unittest_jax_support import JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax  # noqa: F401
    import jax.numpy as jnp
    from simsopt_jax_adapters.field import biotsavart_backend as _backend


import numpy as np


class TestBiotsavartFieldPullback(JaxTestCase):
    def test_native_pullback_preserves_leaf_order_and_static_coil_indices(self):
        """Grouped pullbacks retain leaf order and static coil indices through flattening
        and JIT."""
        arrays = tuple(jnp.asarray([value], dtype=jnp.float64) for value in range(1, 7))
        pullback = _backend.BiotSavartFieldPullback(
            d_coil_arrays=(
                (arrays[0], arrays[1], arrays[2]),
                (arrays[3], arrays[4], arrays[5]),
            ),
            coil_indices=((2, 0), (1,)),
        )
        leaves, treedef = jax.tree_util.tree_flatten(pullback)
        self.assertTrue(
            all(
                observed is expected
                for observed, expected in zip(leaves, arrays, strict=True)
            ),
            "all(observed is expected for observed, expected in zip(leaves, arrays, strict=True))",
        )
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertTrue(
            restored.coil_indices == ((2, 0), (1,)),
            "restored.coil_indices == ((2, 0), (1,))",
        )
        compiled = jax.jit(lambda value: jax.tree.map(lambda leaf: 2 * leaf, value))(
            restored
        )
        self.assertTrue(
            compiled.coil_indices == pullback.coil_indices,
            "compiled.coil_indices == pullback.coil_indices",
        )
        for actual, expected in zip(
            jax.tree_util.tree_leaves(compiled), arrays, strict=True
        ):
            np.testing.assert_array_equal(actual, 2 * expected)
