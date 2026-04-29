# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
This module contains JAX-backed postprocessing for VMEC output.
"""

import logging
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import vmec_jax as vmec_jax_mod
except ImportError as e:
    vmec_jax_mod = None
    logger.debug(str(e))

from .._core.optimizable import Optimizable
from .._core.types import RealArray
from .._core.util import Struct

__all__ = ["QuasisymmetryRatioResidualJax"]


class QuasisymmetryRatioResidualJax(Optimizable):
    r"""
    JAX-backed quasisymmetry-ratio residual from VMEC output.

    The public interface mirrors
    :class:`~simsopt.mhd.vmec_diagnostics.QuasisymmetryRatioResidual`, while
    the calculation is delegated to ``vmec_jax``. This class is intended as
    the VMEC-side quasisymmetry objective for JAX optimization workflows.

    Args:
        vmec: Equilibrium object with a ``run()`` method and ``wout`` data.
        surfaces: Flux surfaces on which the residual is evaluated.
        helicity_m: Desired poloidal helicity.
        helicity_n: Desired toroidal helicity divided by ``nfp``.
        weights: Surface weights. If ``None``, unit weights are used.
        ntheta: Number of poloidal grid points.
        nphi: Number of toroidal grid points per field period.
    """

    def __init__(
        self,
        vmec,
        surfaces: Union[float, RealArray],
        helicity_m: int = 1,
        helicity_n: int = 0,
        weights: Optional[RealArray] = None,
        ntheta: int = 63,
        nphi: int = 64,
    ) -> None:
        if vmec_jax_mod is None:
            raise RuntimeError(
                "QuasisymmetryRatioResidualJax requires the vmec_jax package."
            )

        self.vmec = vmec
        self.ntheta = ntheta
        self.nphi = nphi
        self.helicity_m = helicity_m
        self.helicity_n = helicity_n

        try:
            self.surfaces = list(surfaces)
        except TypeError:
            self.surfaces = [surfaces]

        if weights is None:
            self.weights = np.ones(len(self.surfaces))
        else:
            self.weights = weights
        if len(self.weights) != len(self.surfaces):
            raise ValueError("weights must have the same length as surfaces")
        super().__init__(depends_on=[vmec])

    def compute_jax(self):
        """
        Return the raw ``vmec_jax`` result dictionary.
        """
        self.vmec.run()
        return vmec_jax_mod.quasisymmetry_ratio_residual_from_wout(
            self.vmec.wout,
            surfaces=self.surfaces,
            helicity_m=self.helicity_m,
            helicity_n=self.helicity_n,
            weights=self.weights,
            ntheta=self.ntheta,
            nphi=self.nphi,
        )

    def compute(self):
        """
        Compute the quasisymmetry metric and return a ``Struct``.
        """
        data = self.compute_jax()
        results = Struct()
        for key, value in data.items():
            results.__setattr__(key, np.asarray(value))
        results.ns = len(self.surfaces)
        results.ntheta = self.ntheta
        results.nphi = self.nphi
        results.nfp = self.vmec.wout.nfp
        results.total = float(np.asarray(data["total"]))
        return results

    def residuals(self):
        """
        Evaluate the quasisymmetry residual vector.
        """
        return self.compute().residuals1d

    def profile(self):
        """
        Return the radial profile of the quasisymmetry metric.
        """
        return self.compute().profile

    def total(self):
        """
        Return the scalar quasisymmetry metric.
        """
        return self.compute().total
