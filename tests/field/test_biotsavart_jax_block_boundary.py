"""Quadrature-block and coil-chunk tiling keep the Biot-Savart kernels' results.

``_block_sum`` sums over quadrature blocks or coil chunks along one of three
paths: a single block, an exact two-block split, or a loop over full blocks
plus a tail. The fused B and dB/dX kernel writes the same paths out. The loops
in each kernel's trace show which path each tiling row takes. Each tiled
kernel here is compared with the untiled one (quadrature_block_size=0,
coil_chunk_size=0); only the floating-point summation order may differ.
"""

from __future__ import annotations

from unittest_jax_support import JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    from jax.extend.core import jaxprs_in_params
    from simsopt_jax.core import biotsavart as biotsavart_core


import numpy as np

_COIL_COUNT = 5
_QUADRATURE_COUNT = 24
_POINT_COUNT = 11

# (name, coil_chunk_size, quadrature_block_size, loop trip counts) for 5 coils
# of 24 nodes. One or two blocks or chunks take the single or exact two-way
# path, which has no loop; three or more loop over the full blocks or chunks
# and add the tail. coil_chunk_tail loops over two full 2-coil chunks, and the
# quadrature sum of each chunk, traced once in the loop body and once for the
# 1-coil tail, loops over three full 7-node blocks.
_TILINGS = (
    ("two_quadrature_blocks", 0, 16, ()),
    ("three_quadrature_blocks_and_tail", 0, 7, (3,)),
    ("two_coil_chunks", 3, 16, ()),
    ("coil_chunk_tail", 2, 7, (2, 3, 3)),
)

_FORWARD_KERNELS = (
    ("B", "VALUE"),
    ("B", "JACOBIAN"),
    ("A", "VALUE"),
    ("A", "JACOBIAN"),
    ("B", "HESSIAN"),
    ("A", "HESSIAN"),
    ("B", "VALUE_AND_JACOBIAN"),
)

# Measured over every row below (JAX 0.12.0.1), |tiled - untiled| is at most
# 4.5e-16 * (|untiled| + max|untiled|) on CPU and 5.5e-16 * (...) on an RTX
# 5090, two to two and a half float64 epsilons (2.2e-16). Without the
# max|untiled| floor the elementwise relative difference reaches 3.6e-12, on
# entries that cancel to near zero. 1e-14 (45 epsilons) leaves room for other
# summation orders and FMA contraction. One node more or less changes each
# forward kernel's result by at least 1.7% of its largest entry.
_BLOCK_PARITY_RTOL = 1e-14

# Random coil nodes with every coordinate 0.5 to 1.5 in magnitude, so each node
# is at least 0.2 per coordinate away from the points, which lie in [-0.3, 0.3]^3.
_RNG = np.random.default_rng(20261105)
_COIL_SHAPE = (_COIL_COUNT, _QUADRATURE_COUNT, 3)
_GAMMAS = _RNG.uniform(0.5, 1.5, _COIL_SHAPE) * _RNG.choice((-1.0, 1.0), _COIL_SHAPE)
_GAMMADASHS = _RNG.standard_normal(_COIL_SHAPE)
_CURRENTS = _RNG.uniform(0.5, 1.5, _COIL_COUNT) * 1e4
_POINTS = _RNG.uniform(-0.3, 0.3, (_POINT_COUNT, 3))
_COTANGENT = _RNG.standard_normal((_POINT_COUNT, 3))
_FORWARD_INPUTS = (_POINTS, _GAMMAS, _GAMMADASHS, _CURRENTS)
_VJP_INPUTS = (_POINTS, _COTANGENT, _GAMMAS, _GAMMADASHS, _CURRENTS)


def _forward_kernel(integrand, diff_mode, coil_chunk_size, quadrature_block_size):
    return biotsavart_core._make_kernel(
        biotsavart_core._Integrand[integrand],
        biotsavart_core._DiffMode[diff_mode],
        coil_chunk_size,
        quadrature_block_size,
        0,
    )


def _loop_trip_counts(jaxpr):
    """Sorted trip counts of the scans in ``jaxpr`` and the jaxprs nested in it.

    ``lax.fori_loop`` with static bounds traces to a scan of that length.
    """
    return tuple(
        sorted(
            [eqn.params["length"] for eqn in jaxpr.eqns if eqn.primitive.name == "scan"]
            + [
                count
                for eqn in jaxpr.eqns
                for sub_jaxpr in jaxprs_in_params(eqn.params)
                for count in _loop_trip_counts(sub_jaxpr)
            ]
        )
    )


# The helpers below return NumPy values, so a test row holds no JAX object when
# its case cleans up the runtime.
def _loops_and_result(kernel, inputs):
    """Trace ``kernel`` once; return its loop trip counts and its compiled result."""
    traced = kernel.trace(*inputs)
    result = traced.lower().compile()(*inputs)
    return _loop_trip_counts(traced.jaxpr.jaxpr), jax.tree.map(np.asarray, result)


def _B_vjp(coil_chunk_size, quadrature_block_size):
    kernel = biotsavart_core._make_B_vjp_kernel(
        coil_chunk_size, quadrature_block_size, 0
    )
    return jax.tree.map(np.asarray, kernel(*_VJP_INPUTS))


def _assert_results_match(tiled, untiled):
    """Elementwise rtol with an absolute floor at rtol times the largest value."""
    jax.tree.map(
        lambda tiled_leaf, untiled_leaf: np.testing.assert_allclose(
            tiled_leaf,
            untiled_leaf,
            rtol=_BLOCK_PARITY_RTOL,
            atol=_BLOCK_PARITY_RTOL * np.max(np.abs(untiled_leaf)),
        ),
        tiled,
        untiled,
    )


class TestBiotSavartBlockBoundaryParity(JaxTestCase):
    def test_forward_kernels_take_their_paths_and_match_untiled(self):
        """B, A, their point Jacobians and Hessians and the fused B and dB/dX
        kernel run the loops each tiling row states, and match the untiled
        kernel, which runs none, for two blocks, three blocks plus a tail and
        coil chunk tails."""
        for integrand, diff_mode in _FORWARD_KERNELS:
            kernel = {"integrand": integrand, "diff_mode": diff_mode}
            # The reference is computed outside a subTest, so no row compares
            # against another kernel's reference if computing it fails.
            with self.case():
                loops, untiled = _loops_and_result(
                    _forward_kernel(integrand, diff_mode, 0, 0), _FORWARD_INPUTS
                )
            with self.subTest(**kernel, tiling="untiled"):
                self.assertEqual(loops, ())
            for name, coil_chunk_size, quadrature_block_size, trip_counts in _TILINGS:
                with self.subTest(**kernel, tiling=name), self.case():
                    loops, tiled = _loops_and_result(
                        _forward_kernel(
                            integrand, diff_mode, coil_chunk_size, quadrature_block_size
                        ),
                        _FORWARD_INPUTS,
                    )
                    self.assertEqual(loops, trip_counts)
                    _assert_results_match(tiled, untiled)

    def test_B_vjp_matches_untiled_across_block_boundaries(self):
        """The B cotangents for positions, tangents and currents match the
        untiled VJP kernel for every tiling row."""
        with self.case():
            untiled = _B_vjp(0, 0)
        for name, coil_chunk_size, quadrature_block_size, _ in _TILINGS:
            with self.subTest(tiling=name), self.case():
                _assert_results_match(
                    _B_vjp(coil_chunk_size, quadrature_block_size), untiled
                )
