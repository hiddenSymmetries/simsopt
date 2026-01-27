import warnings
from simsopt.configs.zoo import get_data

__all__ = ["get_LHD_like_data"]

def get_LHD_like_data(numquadpoints_circular=400, numquadpoints_helical=1000, numquadpoints_axis=30):
    """Return the coils and axis for an LHD-like configuration.

    This coil set is a single-filament approximation of the coils in LHD, the
    Large Helical Device in Japan. Each filament corresponds to the center of
    the winding pack of the real finite-thickness LHD coils. The coil currents
    correspond to the configuration in which the major radius of the magnetic
    axis is 3.6 m.

    This configuration has 6 circular coils and 2 helical coils. In the
    lists of curves and currents, the order is OVU, OVL, ISU, ISL, IVU, IVL,
    helical1, helical2. Here, U and L indicate upper and lower, OV and IV indicate
    the outer and inner vertical field coils, and IS indicates the inner shaping
    coils.

    These coils were generated from data generously provided by Yasuhiro Suzuki.
    They produce a configuration similar to that used in Suzuki, Y., K. Y.
    Watanabe, and S. Sakakibara. "Theoretical studies of equilibrium beta limit
    in LHD plasmas." Physics of Plasmas 27, 10 (2020).
    
    Typical usage::

        from simsopt.configs import get_LHD_like_data
        from simsopt.field import BiotSavart, Current, Coil
        
        coils, currents, axis = get_LHD_like_data()
        coils = [
            Coil(curve, Current(current)) for curve, current in zip(coils, currents)
        ]
        field = BiotSavart(coils)

    Args:
        numquadpoints_circular: number of quadrature points for the circular coils
        numquadpoints_helical: number of quadrature points for helical coils
        numquadpoints_axis: number of quadrature points for the magnetic axis

    Returns:
        3 element tuple containing the curves, currents, and the magnetic axis.
        
    .. deprecated:: 1.11.0
       Use :func:`get_data` instead:
       ``get_data('lhd_like', numquadpoints_circular=..., numquadpoints_helical=..., numquadpoints_axis=...)``.
    """
    warnings.warn(
        "get_LHD_like_data is deprecated and will be removed in the next major release; "
        "please use get_data('lhd_like', numquadpoints_circular="
        f"{numquadpoints_circular}, numquadpoints_helical="
        f"{numquadpoints_helical}, numquadpoints_axis="
        f"{numquadpoints_axis}) instead.",
        DeprecationWarning,
        stacklevel=2)

    # call the unified loader, drop the extra outputs
    base_curves, base_currents, axis, _, _ = get_data(
        "lhd_like",
        numquadpoints_circular=numquadpoints_circular,
        numquadpoints_helical=numquadpoints_helical,
        numquadpoints_axis=numquadpoints_axis
    )
    return base_curves, base_currents, axis