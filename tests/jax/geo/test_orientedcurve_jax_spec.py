"""Immutable-spec tests for oriented CurveXYZFourier JAX geometry."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import jax
import numpy as np

from simsopt_jax.core import (
    OrientedCurveXYZFourierSpec,
    curve_gamma_and_dash_from_spec,
    make_oriented_curve_xyzfourier_spec,
)


_DIRECT_KERNEL = {'rtol': 1e-10, 'atol': 1e-12, 'requires_same_state': True, 'requires_direct_cpp_oracle': True, 'vector_parity_required': True}
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]


def _oriented_curve_numpy(dofs: np.ndarray, quadpoints: np.ndarray, order: int):
    xyz = dofs[:3]
    yaw, pitch, roll = dofs[3:6]
    coeffs = np.split(dofs[6:], 3)

    yaw_matrix = np.asarray(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pitch_matrix = np.asarray(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    roll_matrix = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ],
        dtype=np.float64,
    )
    rotation = yaw_matrix @ pitch_matrix @ roll_matrix

    gamma = np.zeros((quadpoints.size, 3), dtype=np.float64)
    gammadash = np.zeros_like(gamma)
    for component_index, component_coeffs in enumerate(coeffs):
        for mode_index in range(order):
            mode = float(mode_index + 1)
            angle = 2.0 * np.pi * mode * quadpoints
            sin_coeff = component_coeffs[2 * mode_index]
            cos_coeff = component_coeffs[2 * mode_index + 1]
            gamma[:, component_index] += sin_coeff * np.sin(angle)
            gamma[:, component_index] += cos_coeff * np.cos(angle)
            gammadash[:, component_index] += (
                2.0 * np.pi * mode * sin_coeff * np.cos(angle)
            )
            gammadash[:, component_index] -= (
                2.0 * np.pi * mode * cos_coeff * np.sin(angle)
            )

    return gamma @ rotation + xyz, gammadash @ rotation


def test_oriented_curve_spec_matches_numpy_geometry_oracle():
    quadpoints = np.linspace(0.0, 1.0, 16, endpoint=False)
    dofs = np.array(
        [
            1.2,
            -0.1,
            0.3,
            0.2,
            -0.15,
            0.05,
            0.1,
            -0.04,
            0.03,
            0.2,
            -0.05,
            0.07,
            0.09,
            -0.11,
            0.08,
            0.13,
            -0.02,
            0.06,
        ],
        dtype=np.float64,
    )

    spec = make_oriented_curve_xyzfourier_spec(
        dofs=dofs,
        quadpoints=quadpoints,
        order=2,
    )
    assert isinstance(spec, OrientedCurveXYZFourierSpec)
    gamma, gammadash = jax.jit(curve_gamma_and_dash_from_spec)(spec)
    expected_gamma, expected_gammadash = _oriented_curve_numpy(dofs, quadpoints, 2)

    np.testing.assert_allclose(
        np.asarray(gamma),
        expected_gamma,
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(gammadash),
        expected_gammadash,
        rtol=_RTOL,
        atol=_ATOL,
    )
