"""Captured extraction specs must use host-resident constants under a strict transfer guard.

Argument specs retain device-resident leaves. These tests exercise both kernel
call patterns without changing the immutable spec constructors.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import jax
import numpy as np
from simsopt_jax.core.field import coil_set_spec_from_dof_extraction_spec
from simsopt_jax.core.specs import (
    CoilDofExtractionSpec,
    host_resident_spec,
    make_coil_dof_extraction_spec,
    make_coil_set_dof_extraction_spec,
    make_curve_xyzfourier_spec,
    make_optimizable_dof_map_spec,
)

CURVE_DOF_COUNT = 9
OWNER_DOFS = np.asarray(
    [0.9, 0.1, -0.2, 0.05, 0.8, 0.15, -0.1, 0.02, 0.3, 1.5e5],
    dtype=np.float64,
)
ROTMAT = np.asarray(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def _dof_map_spec(*, size: int, start: int):
    return make_optimizable_dof_map_spec(
        template_full_dofs=np.zeros(size, dtype=np.float64),
        owner_segments=[(start, start + size, 0, size)],
        input_mode="full",
        input_start=start,
        input_end=start + size,
    )


def _extraction_spec() -> CoilDofExtractionSpec:
    return make_coil_dof_extraction_spec(
        curve=make_curve_xyzfourier_spec(
            dofs=np.zeros(CURVE_DOF_COUNT, dtype=np.float64),
            quadpoints=np.linspace(0.0, 1.0, 4, endpoint=False, dtype=np.float64),
            order=1,
        ),
        curve_map=_dof_map_spec(size=CURVE_DOF_COUNT, start=0),
        current_map=_dof_map_spec(size=1, start=CURVE_DOF_COUNT),
        rotmat=ROTMAT,
        scale=-1.0,
    )


def _coil_field_inputs(spec: CoilDofExtractionSpec):
    return coil_set_spec_from_dof_extraction_spec(
        make_coil_set_dof_extraction_spec([spec]),
        jax.device_put(OWNER_DOFS),
    ).field_inputs()


def test_the_maker_leaves_the_extraction_spec_device_resident():
    """The argument role is the default, so the maker must not host-place."""
    leaves = jax.tree.leaves(_extraction_spec())

    assert leaves
    off_device = [leaf for leaf in leaves if not isinstance(leaf, jax.Array)]
    assert not off_device, (
        f"make_coil_dof_extraction_spec left {len(off_device)} leaf/leaves off the "
        "device; passing the spec to a jitted program would then be an implicit "
        "host-to-device transfer"
    )


def test_host_resident_spec_moves_every_array_leaf_to_the_host():
    """Nothing may be left behind for lowering to read back."""
    on_host = host_resident_spec(_extraction_spec())

    still_on_device = [
        leaf for leaf in jax.tree.leaves(on_host) if isinstance(leaf, jax.Array)
    ]
    assert not still_on_device, (
        f"host_resident_spec left {len(still_on_device)} leaf/leaves on a device; "
        "a program capturing them reads every one back to the host at lowering"
    )
    assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(on_host))
    assert jax.tree.structure(on_host) == jax.tree.structure(_extraction_spec())


def test_host_placement_reconstructs_the_coil_arrays_bit_for_bit():
    """Where the frozen template lives may not change one bit of the coils."""
    device_inputs = _coil_field_inputs(_extraction_spec())
    host_inputs = _coil_field_inputs(host_resident_spec(_extraction_spec()))

    assert len(host_inputs) == len(device_inputs) == 1
    for name, host_array, device_array in zip(
        ("gammas", "gammadashs", "currents"),
        host_inputs[0],
        device_inputs[0],
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(host_array),
            np.asarray(device_array),
            err_msg=f"host-resident template changed the reconstructed {name}",
        )


def test_a_program_capturing_the_host_resident_spec_compiles_under_the_strict_guard():
    """Lowering may not read the captured template back to the host.

    On a single-memory CPU backend the guard cannot observe that read, so this
    is a GPU-effective pin; the leaf-placement test above holds the property
    everywhere.
    """
    spec = make_coil_set_dof_extraction_spec(
        [host_resident_spec(_extraction_spec())]
    )

    def total_gamma(owner_dofs):
        gammas, _gammadashs, _currents = coil_set_spec_from_dof_extraction_spec(
            spec, owner_dofs
        ).field_inputs()[0]
        return gammas.sum()

    with jax.transfer_guard("disallow"):
        total = jax.jit(total_gamma)(jax.device_put(OWNER_DOFS))

    assert np.isfinite(np.asarray(jax.device_get(total)))
