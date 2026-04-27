# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
This module provides a Simsopt wrapper for virtual_casing_jax.
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

try:
    import virtual_casing_jax as virtual_casing_jax_mod
except ImportError as e:
    virtual_casing_jax_mod = None
    logger.debug(str(e))

try:
    from virtual_casing_jax import functional as virtual_casing_jax_functional
except ImportError as e:
    virtual_casing_jax_functional = None
    logger.debug(str(e))

from .virtual_casing import VirtualCasing as _VirtualCasingBase
from .vmec import Vmec
from .vmec_jax import B_cartesian_jax, VmecJax
from .vmec_diagnostics import B_cartesian
from ..geo.surface import best_nphi_over_ntheta
from ..geo.surfacerzfourier import SurfaceRZFourier

__all__ = [
    "VirtualCasingJax",
    "B_external_normal_from_data",
    "B_external_normal_jvp_from_data",
    "B_external_normal_jacobian_from_surface",
]


def _soa_from_3d(arr3d):
    return np.transpose(arr3d, (2, 0, 1))


def _3d_from_soa(arr_soa):
    return np.transpose(arr_soa, (1, 2, 0))


def _validate_grid(name, arr, shape=None):
    arr = np.asarray(arr)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{name} must have shape (nphi, ntheta, 3)")
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return arr


def _require_functional_api():
    if virtual_casing_jax_functional is None:
        raise RuntimeError(
            "B_external_normal_from_data requires a virtual_casing_jax "
            "version with the functional normal-field API."
        )
    required = [
        "prepare_functional_setup",
        "compute_external_B_functional",
        "compute_external_B_normal_functional",
    ]
    for name in required:
        if not hasattr(virtual_casing_jax_functional, name):
            raise RuntimeError(
                "B_external_normal_from_data requires a virtual_casing_jax "
                "version with the functional normal-field API."
            )
    return virtual_casing_jax_functional


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as e:
        raise RuntimeError("B_external_normal_from_data requires JAX.") from e
    return jax, jnp


def _prepare_normal_field_call(
    gamma,
    B_total,
    nfp,
    stellsym,
    digits,
    trgt_nphi,
    trgt_ntheta,
    quad_nphi,
    quad_ntheta,
    patch_dim0,
):
    functional = _require_functional_api()
    _, jnp = _require_jax()

    gamma = _validate_grid("gamma", gamma)
    B_total = _validate_grid("B_total", B_total, gamma.shape)
    src_nphi, src_ntheta = gamma.shape[:2]
    if trgt_nphi is None:
        trgt_nphi = src_nphi
    if trgt_ntheta is None:
        trgt_ntheta = src_ntheta

    X = jnp.asarray(_soa_from_3d(gamma))
    B0 = jnp.asarray(_soa_from_3d(B_total))
    if quad_nphi is None or quad_ntheta is None:
        nfp_eff = int(nfp) * (2 if stellsym else 1)
        quad_nphi_select = nfp_eff * int(trgt_nphi)
        quad_ntheta_select = int(trgt_ntheta)
        if virtual_casing_jax_mod is not None and hasattr(
            virtual_casing_jax_mod, "VirtualCasingJAX"
        ):
            vc_select = virtual_casing_jax_mod.VirtualCasingJAX()
            vc_select.setup(
                digits,
                int(nfp),
                bool(stellsym),
                src_nphi,
                src_ntheta,
                X,
                src_nphi,
                src_ntheta,
                int(trgt_nphi),
                int(trgt_ntheta),
            )
            quad_nphi_select, quad_ntheta_select = vc_select._select_quad_sizes(digits)
        if quad_nphi is None:
            quad_nphi = quad_nphi_select
        if quad_ntheta is None:
            quad_ntheta = quad_ntheta_select

    setup = functional.prepare_functional_setup(
        X,
        digits=digits,
        nfp=int(nfp),
        half_period=bool(stellsym),
        surf_nt=src_nphi,
        surf_np=src_ntheta,
        src_nt=src_nphi,
        src_np=src_ntheta,
        trg_nt=int(trgt_nphi),
        trg_np=int(trgt_ntheta),
        quad_nt=int(quad_nphi),
        quad_np=int(quad_ntheta),
        patch_dim0=patch_dim0,
    )
    kwargs = dict(
        digits=digits,
        nfp=int(nfp),
        half_period=bool(stellsym),
        surf_nt=src_nphi,
        surf_np=src_ntheta,
        src_nt=src_nphi,
        src_np=src_ntheta,
        trg_nt=int(trgt_nphi),
        trg_np=int(trgt_ntheta),
        quad_nt=int(quad_nphi),
        quad_np=int(quad_ntheta),
        patch_dim0=setup.patch_dim0,
        patch_idx=setup.patch_idx,
        orient=setup.orient,
    )
    return functional, X, B0, kwargs


def B_external_normal_from_data(
    gamma,
    B_total,
    nfp,
    stellsym,
    digits=6,
    trgt_nphi=None,
    trgt_ntheta=None,
    quad_nphi=None,
    quad_ntheta=None,
    patch_dim0=None,
    chunk_size="auto",
    target_chunk_size="auto",
    pou_dtype=None,
    patch_dtype=None,
    interp_block_size="auto",
    remat=None,
    unit_normal=None,
):
    """
    Compute ``B_external_normal`` from Simsopt-shaped surface data.

    ``gamma`` and ``B_total`` must both have shape ``(nphi, ntheta, 3)``.
    If supplied, ``unit_normal`` must have shape
    ``(trgt_nphi, trgt_ntheta, 3)`` and is used for the final projection.
    Otherwise the target normal is computed inside ``virtual_casing_jax``.
    """
    functional, X, B0, kwargs = _prepare_normal_field_call(
        gamma,
        B_total,
        nfp,
        stellsym,
        digits,
        trgt_nphi,
        trgt_ntheta,
        quad_nphi,
        quad_ntheta,
        patch_dim0,
    )
    if unit_normal is None:
        Bnormal = functional.compute_external_B_normal_functional(
            X,
            B0,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
            pou_dtype=pou_dtype,
            patch_dtype=patch_dtype,
            interp_block_size=interp_block_size,
            remat=remat,
            **kwargs,
        )
        return np.asarray(Bnormal)

    unit_normal = _validate_grid(
        "unit_normal", unit_normal, (kwargs["trg_nt"], kwargs["trg_np"], 3)
    )
    Bexternal = functional.compute_external_B_functional(
        X,
        B0,
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
        pou_dtype=pou_dtype,
        patch_dtype=patch_dtype,
        interp_block_size=interp_block_size,
        remat=remat,
        **kwargs,
    )
    Bexternal = _3d_from_soa(np.asarray(Bexternal))
    return np.sum(Bexternal * unit_normal, axis=2)


def B_external_normal_jvp_from_data(
    gamma,
    B_total,
    tangent_gamma,
    tangent_B_total=None,
    nfp=1,
    stellsym=True,
    digits=6,
    trgt_nphi=None,
    trgt_ntheta=None,
    quad_nphi=None,
    quad_ntheta=None,
    patch_dim0=None,
    chunk_size="auto",
    target_chunk_size="auto",
    pou_dtype=None,
    patch_dtype=None,
    interp_block_size="auto",
    remat=None,
    unit_normal=None,
    tangent_unit_normal=None,
):
    """
    Return ``B_external_normal`` and its forward-mode directional derivative.

    The derivative is taken with respect to ``gamma`` and ``B_total`` in
    Simsopt array convention. If ``unit_normal`` is supplied, the returned
    normal field is projected onto that target normal, and
    ``tangent_unit_normal`` contributes to the directional derivative.
    """
    functional, X, B0, kwargs = _prepare_normal_field_call(
        gamma,
        B_total,
        nfp,
        stellsym,
        digits,
        trgt_nphi,
        trgt_ntheta,
        quad_nphi,
        quad_ntheta,
        patch_dim0,
    )
    jax, jnp = _require_jax()

    gamma = _validate_grid("gamma", gamma)
    tangent_gamma = _validate_grid("tangent_gamma", tangent_gamma, gamma.shape)
    if tangent_B_total is None:
        tangent_B_total = np.zeros_like(gamma)
    tangent_B_total = _validate_grid("tangent_B_total", tangent_B_total, gamma.shape)
    dX = jnp.asarray(_soa_from_3d(tangent_gamma))
    dB0 = jnp.asarray(_soa_from_3d(tangent_B_total))

    if unit_normal is None and tangent_unit_normal is not None:
        raise ValueError("tangent_unit_normal can only be used with unit_normal")

    if unit_normal is not None:
        unit_normal = _validate_grid(
            "unit_normal", unit_normal, (kwargs["trg_nt"], kwargs["trg_np"], 3)
        )
        if tangent_unit_normal is None:
            tangent_unit_normal = np.zeros_like(unit_normal)
        tangent_unit_normal = _validate_grid(
            "tangent_unit_normal", tangent_unit_normal, unit_normal.shape
        )

        def external_field(x, b0):
            return functional.compute_external_B_functional(
                x,
                b0,
                chunk_size=chunk_size,
                target_chunk_size=target_chunk_size,
                pou_dtype=pou_dtype,
                patch_dtype=patch_dtype,
                interp_block_size=interp_block_size,
                remat=remat,
                **kwargs,
            )

        Bexternal, dBexternal = jax.jvp(external_field, (X, B0), (dX, dB0))
        Bexternal = _3d_from_soa(np.asarray(Bexternal))
        dBexternal = _3d_from_soa(np.asarray(dBexternal))
        Bnormal = np.sum(Bexternal * unit_normal, axis=2)
        dBnormal = np.sum(
            dBexternal * unit_normal + Bexternal * tangent_unit_normal, axis=2
        )
        return Bnormal, dBnormal

    def normal_field(x, b0):
        return functional.compute_external_B_normal_functional(
            x,
            b0,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
            pou_dtype=pou_dtype,
            patch_dtype=patch_dtype,
            interp_block_size=interp_block_size,
            remat=remat,
            **kwargs,
        )

    Bnormal, dBnormal = jax.jvp(normal_field, (X, B0), (dX, dB0))
    return np.asarray(Bnormal), np.asarray(dBnormal)


def B_external_normal_jacobian_from_surface(
    surface,
    B_total,
    nfp=None,
    stellsym=None,
    digits=6,
    quad_nphi=None,
    quad_ntheta=None,
    patch_dim0=None,
    chunk_size="auto",
    target_chunk_size="auto",
    pou_dtype=None,
    patch_dtype=None,
    interp_block_size="auto",
    remat=None,
    B_total_tangents=None,
    free_only=True,
):
    """
    Return ``B_external_normal`` and its Jacobian with respect to a surface.

    The source and target grids are the quadrature grid of ``surface``.
    ``B_total_tangents`` may be supplied with shape
    ``(nphi, ntheta, 3, ndof)`` to include the VMEC-field contribution for
    each returned surface column.
    """
    if nfp is None:
        nfp = surface.nfp
    if stellsym is None:
        stellsym = getattr(surface, "stellsym", True)

    gamma = surface.gamma()
    unit_normal = surface.unitnormal()
    B_total = _validate_grid("B_total", B_total, gamma.shape)
    functional, X, B0, kwargs = _prepare_normal_field_call(
        gamma,
        B_total,
        nfp,
        stellsym,
        digits,
        gamma.shape[0],
        gamma.shape[1],
        quad_nphi,
        quad_ntheta,
        patch_dim0,
    )
    jax, jnp = _require_jax()

    dgamma = surface.dgamma_by_dcoeff()
    dunit_normal = surface.dunitnormal_by_dcoeff()
    if free_only:
        columns = np.where(surface.dofs_free_status)[0]
    else:
        columns = np.arange(dgamma.shape[-1])
    ncols = len(columns)

    if B_total_tangents is None:
        B_total_tangents = np.zeros(gamma.shape + (ncols,))
    else:
        B_total_tangents = np.asarray(B_total_tangents)
        expected_shape = gamma.shape + (ncols,)
        if B_total_tangents.shape != expected_shape:
            raise ValueError(f"B_total_tangents must have shape {expected_shape}")

    def external_field(x, b0):
        return functional.compute_external_B_functional(
            x,
            b0,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
            pou_dtype=pou_dtype,
            patch_dtype=patch_dtype,
            interp_block_size=interp_block_size,
            remat=remat,
            **kwargs,
        )

    Bnormal = None
    jacobian = np.zeros(gamma.shape[:2] + (ncols,))
    for j, col in enumerate(columns):
        dX = jnp.asarray(_soa_from_3d(dgamma[:, :, :, col]))
        dB0 = jnp.asarray(_soa_from_3d(B_total_tangents[:, :, :, j]))
        Bexternal, dBexternal = jax.jvp(external_field, (X, B0), (dX, dB0))
        Bexternal = _3d_from_soa(np.asarray(Bexternal))
        dBexternal = _3d_from_soa(np.asarray(dBexternal))
        if Bnormal is None:
            Bnormal = np.sum(Bexternal * unit_normal, axis=2)
        jacobian[:, :, j] = np.sum(
            dBexternal * unit_normal + Bexternal * dunit_normal[:, :, :, col],
            axis=2,
        )
    return Bnormal, jacobian


class VirtualCasingJax(_VirtualCasingBase):
    r"""
    Compute virtual-casing fields using ``virtual_casing_jax``.

    The attributes and saved-file format match
    :class:`~simsopt.mhd.virtual_casing.VirtualCasing`, so existing coil
    objectives can consume ``B_external_normal`` without changes.
    """

    @classmethod
    def from_vmec(
        cls,
        vmec,
        src_nphi,
        src_ntheta=None,
        trgt_nphi=None,
        trgt_ntheta=None,
        use_stellsym=True,
        digits=6,
        filename="auto",
    ):
        """
        Compute the external magnetic field from a VMEC equilibrium.
        """
        if virtual_casing_jax_mod is None:
            raise RuntimeError(
                "VirtualCasingJax requires the virtual_casing_jax package."
            )

        if not isinstance(vmec, (Vmec, VmecJax)):
            vmec = VmecJax(vmec)

        vmec.run()
        nfp = vmec.wout.nfp
        stellsym = (not bool(vmec.wout.lasym)) and use_stellsym
        if vmec.wout.lasym:
            raise RuntimeError("virtual casing presently only works for stellarator symmetry")

        if src_ntheta is None:
            src_ntheta = int(
                (1 + int(stellsym)) * nfp * src_nphi / best_nphi_over_ntheta(vmec.boundary)
            )
            logger.info(f"new src_ntheta: {src_ntheta}")

        ran = "half period" if stellsym else "field period"
        surf = SurfaceRZFourier.from_nphi_ntheta(
            mpol=vmec.wout.mpol,
            ntor=vmec.wout.ntor,
            nfp=nfp,
            nphi=src_nphi,
            ntheta=src_ntheta,
            range=ran,
        )
        for jmn in range(vmec.wout.mnmax):
            surf.set_rc(int(vmec.wout.xm[jmn]), int(vmec.wout.xn[jmn] / nfp), vmec.wout.rmnc[jmn, -1])
            surf.set_zs(int(vmec.wout.xm[jmn]), int(vmec.wout.xn[jmn] / nfp), vmec.wout.zmns[jmn, -1])

        if isinstance(vmec, VmecJax):
            Bxyz = B_cartesian_jax(vmec, nphi=src_nphi, ntheta=src_ntheta, range=ran)
        else:
            Bxyz = B_cartesian(vmec, nphi=src_nphi, ntheta=src_ntheta, range=ran)
        gamma = surf.gamma()

        if trgt_nphi is None:
            trgt_nphi = src_nphi
        if trgt_ntheta is None:
            trgt_ntheta = src_ntheta
        trgt_surf = SurfaceRZFourier.from_nphi_ntheta(
            mpol=vmec.wout.mpol,
            ntor=vmec.wout.ntor,
            nfp=nfp,
            nphi=trgt_nphi,
            ntheta=trgt_ntheta,
            range=ran,
        )
        trgt_surf.x = surf.x
        unit_normal = trgt_surf.unitnormal()

        gamma_soa = _soa_from_3d(gamma)
        B_total_soa = np.asarray(Bxyz)
        B3d = _3d_from_soa(B_total_soa)

        vc_jax = virtual_casing_jax_mod.VirtualCasingJAX()
        vc_jax.setup(
            digits,
            nfp,
            stellsym,
            src_nphi,
            src_ntheta,
            gamma_soa,
            src_nphi,
            src_ntheta,
            trgt_nphi,
            trgt_ntheta,
        )
        Bexternal_soa = vc_jax.compute_external_B(B_total_soa, digits=digits)
        Bexternal3d = _3d_from_soa(np.asarray(Bexternal_soa))
        Bexternal_normal = np.sum(Bexternal3d * unit_normal, axis=2)

        vc = cls()
        vc.src_ntheta = src_ntheta
        vc.src_nphi = src_nphi
        vc.src_theta = surf.quadpoints_theta
        vc.src_phi = surf.quadpoints_phi

        vc.trgt_ntheta = trgt_ntheta
        vc.trgt_nphi = trgt_nphi
        vc.trgt_theta = trgt_surf.quadpoints_theta
        vc.trgt_phi = trgt_surf.quadpoints_phi

        vc.nfp = nfp
        vc.B_total = B3d
        vc.gamma = gamma
        vc.unit_normal = unit_normal
        vc.B_external = Bexternal3d
        vc.B_external_normal = Bexternal_normal

        Bnormal_with_last_point = np.hstack((Bexternal_normal, Bexternal_normal[:, [0]]))
        Bnormal_with_last_point = np.vstack(
            (Bnormal_with_last_point, -np.flip(np.flip(Bnormal_with_last_point, axis=0), axis=1)[0])
        )
        flipped_B = -np.flip(np.flip(Bnormal_with_last_point, axis=0), axis=1)
        vc.B_external_normal_extended = np.concatenate(
            [np.concatenate((Bexternal_normal, flipped_B[:-1, :-1])) for _ in range(nfp)]
        )

        if filename is not None:
            if filename == "auto":
                directory, basefile = os.path.split(vmec.output_file)
                filename = os.path.join(directory, "vcasing" + basefile[4:])
                logger.debug(f"New filename: {filename}")
            vc.save(filename)

        return vc
