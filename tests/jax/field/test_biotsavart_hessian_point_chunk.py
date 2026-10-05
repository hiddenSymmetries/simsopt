"""The ``d2B/dXdX`` kernel tiles points with a Hessian-sized chunk.

The tuning table's ``point_chunk_size`` is sized for the ``B``/``B+dB``
integrands. Reusing it verbatim for the ``jacfwd(jacfwd)`` Hessian integrand
materialized more than 20 GB at 2048 points against 18 NCSX coils and failed
on a 32 GB device. The derived tile must keep results equal to the untiled
kernel: point tiling never changes any single point's reduction.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simsopt_jax.core import biotsavart as biotsavart_core

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "tuned,expected",
    [(0, 0), (1, 1), (7, 1), (8, 1), (128, 16), (2048, 256), (4096, 512)],
)
def test_hessian_point_chunk_size_is_one_eighth_and_keeps_disabled(tuned, expected):
    assert biotsavart_core.hessian_point_chunk_size(tuned) == expected


def _coil_set(rng):
    nquad = 12
    phi = np.linspace(0.0, 2.0 * np.pi, nquad, endpoint=False)
    gammas = []
    gammadashs = []
    for center in ((0.0, 0.0, 0.0), (0.3, 0.1, -0.2)):
        gamma = np.stack(
            (np.cos(phi) + center[0], np.sin(phi) + center[1], np.zeros(nquad) + center[2]),
            axis=-1,
        )
        gammadash = np.stack((-np.sin(phi), np.cos(phi), np.zeros(nquad)), axis=-1)
        gammas.append(gamma)
        gammadashs.append(gammadash)
    currents = rng.uniform(0.5, 1.5, size=2)
    return (
        jnp.asarray(np.stack(gammas)),
        jnp.asarray(np.stack(gammadashs)),
        jnp.asarray(currents),
    )


def test_d2B_kernel_tiled_over_points_matches_untiled(monkeypatch):
    rng = np.random.default_rng(7)
    gammas, gammadashs, currents = _coil_set(rng)
    points = jnp.asarray(rng.uniform(-0.4, 0.4, size=(37, 3)))
    biotsavart_core.invalidate_kernel_cache()
    untiled = biotsavart_core._make_kernel(
        biotsavart_core._Integrand.B,
        biotsavart_core._DiffMode.HESSIAN,
        0,
        0,
        0,
        None,
    )(points, gammas, gammadashs, currents)
    # A tuned size of 64 becomes an 8-point Hessian tile: five tiles of 8 plus a
    # 5-point tail, so both loop bodies and the tail are exercised.
    tiled = biotsavart_core._make_kernel(
        biotsavart_core._Integrand.B,
        biotsavart_core._DiffMode.HESSIAN,
        0,
        0,
        64,
        None,
    )(points, gammas, gammadashs, currents)
    assert tiled.shape == (37, 3, 3, 3)
    np.testing.assert_allclose(
        np.asarray(tiled), np.asarray(untiled), rtol=1e-13, atol=1e-16
    )
    b_and_db_tiled = biotsavart_core._make_kernel(
        biotsavart_core._Integrand.B,
        biotsavart_core._DiffMode.VALUE_AND_JACOBIAN,
        0,
        0,
        64,
        None,
    )(points, gammas, gammadashs, currents)
    # The non-Hessian kernels keep the tuned size: 37 points fit one 64-point
    # tile, so their result must equal the untiled evaluation exactly.
    b_and_db_untiled = biotsavart_core._make_kernel(
        biotsavart_core._Integrand.B,
        biotsavart_core._DiffMode.VALUE_AND_JACOBIAN,
        0,
        0,
        0,
        None,
    )(points, gammas, gammadashs, currents)
    for tiled_leaf, untiled_leaf in zip(b_and_db_tiled, b_and_db_untiled, strict=True):
        np.testing.assert_array_equal(np.asarray(tiled_leaf), np.asarray(untiled_leaf))
