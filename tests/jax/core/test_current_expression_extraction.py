from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import Coil, Current, coils_via_symmetries
from simsopt.geo import create_equally_spaced_curves
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX


def test_coil_specs_preserve_fixed_total_current_expression() -> None:
    curves = create_equally_spaced_curves(
        2,
        1,
        stellsym=False,
        R0=1.0,
        R1=0.25,
        order=2,
        numquadpoints=12,
    )
    for curve in curves:
        curve.fix_all()
    first = Current(3.0)
    total = Current(10.0)
    total.fix_all()
    field = BiotSavartJAX(
        [
            Coil(curves[0], first),
            Coil(curves[1], total - first),
        ]
    )

    def currents(owner_dofs: jax.Array) -> jax.Array:
        return jnp.stack(
            tuple(
                coil.current.value[0] for coil in field.coil_specs_from_dofs(owner_dofs)
            )
        )

    owner_dofs = jnp.asarray([4.0], dtype=jnp.float64)
    np.testing.assert_allclose(currents(owner_dofs), np.asarray([4.0, 6.0]))
    np.testing.assert_allclose(
        jax.jacrev(currents)(owner_dofs),
        np.asarray([[1.0], [-1.0]]),
    )


def test_coil_set_spec_preserves_scaled_current_optimizer_coordinates() -> None:
    curve = create_equally_spaced_curves(
        1,
        1,
        stellsym=False,
        R0=1.0,
        R1=0.25,
        order=2,
        numquadpoints=12,
    )[0]
    curve.fix_all()
    field = BiotSavartJAX([Coil(curve, Current(1.0) * 1.0e5)])

    def current(owner_dofs: jax.Array) -> jax.Array:
        return field.coil_set_spec_from_dofs(owner_dofs).groups[0].currents

    owner_dofs = jnp.asarray(field.x, dtype=jnp.float64)
    np.testing.assert_array_equal(owner_dofs, np.asarray([1.0]))
    np.testing.assert_allclose(current(owner_dofs), np.asarray([1.0e5]))
    np.testing.assert_allclose(
        jax.jacrev(current)(owner_dofs),
        np.asarray([[1.0e5]]),
    )


def test_coil_specs_reuse_geometry_for_symmetry_replicas() -> None:
    curve = create_equally_spaced_curves(
        1,
        2,
        stellsym=True,
        R0=1.0,
        R1=0.25,
        order=2,
        numquadpoints=12,
    )[0]
    current = Current(3.0)
    coils = coils_via_symmetries([curve], [current], 2, True)
    field = BiotSavartJAX(coils)

    coil_specs = field.coil_specs_from_dofs(jnp.asarray(field.x))

    assert len(coil_specs) == 4
    assert len({id(coil_spec.curve) for coil_spec in coil_specs}) == 1
