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

from .virtual_casing import VirtualCasing as _VirtualCasingBase
from .vmec import Vmec
from .vmec_jax import VmecJax
from .vmec_diagnostics import B_cartesian
from ..geo.surface import best_nphi_over_ntheta
from ..geo.surfacerzfourier import SurfaceRZFourier

__all__ = ["VirtualCasingJax"]


def _soa_from_3d(arr3d):
    return np.transpose(arr3d, (2, 0, 1))


def _3d_from_soa(arr_soa):
    return np.transpose(arr_soa, (1, 2, 0))


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
