"""Public static facade for the pure JAX kernel layer."""

from .specs import (
    CurveXYZFourierSymmetriesSpec,
    make_coil_dof_extraction_spec,
    make_coil_set_dof_extraction_spec,
    make_curve_filament_spec,
    make_curve_helical_spec,
    make_curve_planarfourier_spec,
    make_curve_perturbed_spec,
    make_curve_rzfourier_spec,
    make_curve_xyzfourier_spec,
    make_curve_xyzfouriersymmetries_spec,
    make_frame_rotation_spec,
    make_optimizable_dof_map_spec,
    make_zero_rotation_spec,
)


from .curve_geometry import (
    curve_gamma_and_gammadash_from_dofs,
    curve_gamma_and_gammadash_from_spec,
    curve_geometry_from_dofs,
    curve_spec_from_curve,
)

from .field import (
    coil_specs_from_dof_extraction_spec,
)

from .biotsavart import (
    invalidate_kernel_cache,
)


__all__ = [
    "CurveXYZFourierSymmetriesSpec",
    "curve_gamma_and_gammadash_from_dofs",
    "curve_gamma_and_gammadash_from_spec",
    "curve_geometry_from_dofs",
    "curve_spec_from_curve",
    "coil_specs_from_dof_extraction_spec",
    "invalidate_kernel_cache",
    "make_coil_dof_extraction_spec",
    "make_coil_set_dof_extraction_spec",
    "make_curve_filament_spec",
    "make_curve_helical_spec",
    "make_curve_planarfourier_spec",
    "make_curve_perturbed_spec",
    "make_curve_rzfourier_spec",
    "make_curve_xyzfourier_spec",
    "make_curve_xyzfouriersymmetries_spec",
    "make_frame_rotation_spec",
    "make_optimizable_dof_map_spec",
    "make_zero_rotation_spec",
]
