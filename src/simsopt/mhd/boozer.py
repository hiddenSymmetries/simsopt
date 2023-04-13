# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides a class that handles the transformation to
Boozer coordinates, and an optimization target for quasisymmetry.
"""

import logging
from typing import Union, Iterable

import numpy as np

logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI
except ImportError as e:
    MPI = None
    logger.debug(str(e))

try:
    import booz_xform
except ImportError as e:
    booz_xform = None
    logger.debug(str(e))

from .vmec import Vmec
from .._core.optimizable import Optimizable
from .._core.types import RealArray

__all__ = ['Boozer', 'Quasisymmetry']


class Boozer(Optimizable):
    """
    This class handles the transformation to Boozer coordinates.

    A Boozer instance maintains a set "s", which is a registry of the
    surfaces on which other objects want Boozer-coordinate data. When
    the run() method is called, the Boozer transformation is carried
    out on all these surfaces. The registry can be cleared at any time
    by setting the s attribute to {}.
    """

    def __init__(self,
                 equil: Vmec,
                 mpol: int = 32,
                 ntor: int = 32) -> None:
        """
        Constructor
        """
        if booz_xform is None:
            raise RuntimeError(
                "To use a Boozer object, the booz_xform package "
                "must be installed. Run 'pip install -v booz_xform'")

        self.equil = equil
        self.mpol = mpol
        self.ntor = ntor
        self.bx = booz_xform.Booz_xform()
        self.s = set()
        self.need_to_run_code = True
        self._calls = 0  # For testing, keep track of how many times we call bx.run()

        # We may at some point want to allow booz_xform to use a
        # different partitioning of the MPI processors compared to the
        # equilibrium code. But for simplicity, we'll use the same mpi
        # partition for now. For unit tests, we allow equil to be None,
        # so we have to allow for this case here.
        self.mpi = None
        if equil is not None:
            self.mpi = equil.mpi
        if equil is not None:
            super().__init__(depends_on=[equil])
        else:
            super().__init__()

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True

    def register(self, s: Union[float, Iterable[float]]) -> None:
        """
        This function is called by objects that depend on this Boozer
        object, to indicate that they will want Boozer data on the
        given set of surfaces.

        Args:
            s: 1 or more surfaces on which Boozer data will be requested.
        """
        # Force input surface data to be a set:
        try:
            ss = set(s)
        except:
            ss = {s}

        for new_s in ss:
            if new_s < 0 or new_s > 1:
                raise ValueError("Normalized toroidal flux values s must lie"
                                 "in the interval [0, 1]")
        logger.info("Adding entries to Boozer registry: {}".format(ss))
        self.s = self.s.union(ss)
        self.need_to_run_code = True

    def run(self):
        """
        Run booz_xform on all the surfaces that have been registered.
        """

        if (self.mpi is not None) and (not self.mpi.proc0_groups):
            logger.info("This proc is skipping Boozer.run since it is not a group leader.")
            return

        if not self.need_to_run_code:
            logger.info("Boozer.run() called but no need to re-run Boozer transformation.")
            return

        s = sorted(list(self.s))
        logger.info("Preparing to run Boozer transformation. Registry:{}".format(s))

        if isinstance(self.equil, Vmec):
            self.equil.run()
            wout = self.equil.wout  # Shorthand

            # Get the half-grid points that are closest to the requested values
            ns = wout.ns
            s_full = np.linspace(0, 1, ns)
            ds = s_full[1] - s_full[0]
            s_half = s_full[1:] - 0.5 * ds

            # For each float value of s at which the Boozer results
            # have been requested, we need to find the corresponding
            # radial index of the booz_xform results. The result is
            # self.s_to_index. Computing this is tricky because
            # multiple values of s may get rounded to the same
            # half-grid surface. The solution here is done in two
            # steps. First we find a map from each float value of s to
            # the corresponding radial index among all half-grid
            # surfaces (even ones where we won't compute the Boozer
            # transformation.) This resulting map is
            # s_to_index_all_surfs. In a second step,
            # s_to_index_all_surfs and the list of compute_surfs are
            # used to find s_to_index.

            compute_surfs = []
            s_to_index_all_surfs = dict()
            self.s_used = dict()
            for ss in s:
                index = np.argmin(np.abs(s_half - ss))
                compute_surfs.append(index)
                s_to_index_all_surfs[ss] = index
                self.s_used[ss] = s_half[index]

            # Eliminate any duplicates
            compute_surfs = sorted(list(set(compute_surfs)))
            logger.info("compute_surfs={}".format(compute_surfs))
            logger.info("s_to_index_all_surfs={}".format(s_to_index_all_surfs))
            self.s_to_index = dict()
            for ss in s:
                self.s_to_index[ss] = compute_surfs.index(s_to_index_all_surfs[ss])
            logger.info("s_to_index={}".format(self.s_to_index))

            # Transfer data in memory from VMEC to booz_xform
            self.bx.asym = bool(wout.lasym)
            self.bx.nfp = wout.nfp

            self.bx.mpol = wout.mpol
            self.bx.ntor = wout.ntor
            self.bx.mnmax = wout.mnmax
            self.bx.xm = wout.xm
            self.bx.xn = wout.xn
            print('mnmax:', wout.mnmax, ' len(xm):', len(wout.xm), ' len(xn):', len(wout.xn))
            print('mnmax_nyq:', wout.mnmax_nyq, ' len(xm_nyq):', len(wout.xm_nyq), ' len(xn_nyq):', len(wout.xn_nyq))
            assert len(wout.xm) == wout.mnmax
            assert len(wout.xn) == wout.mnmax
            assert len(self.bx.xm) == self.bx.mnmax
            assert len(self.bx.xn) == self.bx.mnmax

            self.bx.mpol_nyq = int(wout.xm_nyq[-1])
            self.bx.ntor_nyq = int(wout.xn_nyq[-1] / wout.nfp)
            self.bx.mnmax_nyq = wout.mnmax_nyq
            self.bx.xm_nyq = wout.xm_nyq
            self.bx.xn_nyq = wout.xn_nyq
            assert len(wout.xm_nyq) == wout.mnmax_nyq
            assert len(wout.xn_nyq) == wout.mnmax_nyq
            assert len(self.bx.xm_nyq) == self.bx.mnmax_nyq
            assert len(self.bx.xn_nyq) == self.bx.mnmax_nyq

            if wout.lasym:
                rmns = wout.rmns
                zmnc = wout.zmnc
                lmnc = wout.lmnc
                bmns = wout.bmns
                bsubumns = wout.bsubumns
                bsubvmns = wout.bsubvmns
            else:
                # For stellarator-symmetric configs, the asymmetric
                # arrays have not been initialized.
                arr = np.array([[]])
                rmns = arr
                zmnc = arr
                lmnc = arr
                bmns = arr
                bsubumns = arr
                bsubvmns = arr

            # For quantities that depend on radius, booz_xform handles
            # interpolation and discarding the rows of zeros:
            self.bx.init_from_vmec(wout.ns,
                                   wout.iotas,
                                   wout.rmnc,
                                   rmns,
                                   zmnc,
                                   wout.zmns,
                                   lmnc,
                                   wout.lmns,
                                   wout.bmnc,
                                   bmns,
                                   wout.bsubumnc,
                                   bsubumns,
                                   wout.bsubvmnc,
                                   bsubvmns)
            self.bx.compute_surfs = compute_surfs
            self.bx.mboz = self.mpol
            self.bx.nboz = self.ntor

        else:
            # Cases for SPEC, GVEC, etc could be added here.
            raise ValueError("equil is not an equilibrium type supported by"
                             "Boozer")

        logger.info("About to call booz_xform.Booz_xform.run().")
        self.bx.run()
        self._calls += 1
        logger.info("Returned from calling booz_xform.Booz_xform.run().")
        self.need_to_run_code = False


class Quasisymmetry(Optimizable):
    """
    This class is used to compute the departure from quasisymmetry on
    a given flux surface based on the Boozer spectrum.

    Args:
        boozer: A Boozer object on which the calculation will be based.
        s: The normalized toroidal magnetic flux for the flux surface to analyze. Should be in the range [0, 1].
        helicity_m: The poloidal mode number of the symmetry you want to achive.
           The departure from symmetry ``B(helicity_m * theta - nfp * helicity_n * zeta)`` will be reported.
        helicity_n: The toroidal mode number of the symmetry you want to achieve.
           The departure from symmetry ``B(helicity_m * theta - nfp * helicity_n * zeta)`` will be reported.
        normalization: A uniform normalization applied to all bmnc harmonics.
           If ``"B00"``, the symmetry-breaking modes will be divided by the m=n=0 mode amplitude
           on the same surface. If ``"symmetric"``, the symmetry-breaking modes will be
           divided by the square root of the sum of the squares of all the symmetric
           modes on the same surface. This is the normalization used in stellopt.
        weight: An option for a m- or n-dependent weight to be applied to the bmnc amplitudes.
    """

    def __init__(self,
                 boozer: Boozer,
                 s: Union[float, Iterable[float]],
                 helicity_m: int,
                 helicity_n: int,
                 normalization: str = "B00",
                 weight: str = "even") -> None:
        """
        Constructor

        """
        self.boozer = boozer
        self.helicity_m = helicity_m
        self.helicity_n = helicity_n
        self.normalization = normalization
        self.weight = weight

        # If s is not already iterable, make it so:
        try:
            iter(s)
        except:
            s = [s]
        self.s = s
        boozer.register(s)
        super().__init__(depends_on=[boozer])

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True

    def J(self) -> RealArray:
        """
        Carry out the calculation of the quasisymmetry error.

        Returns:
            1D numpy array listing all the normalized mode amplitudes of
            symmetry-breaking Fourier modes of ``|B|``.
        """

        # Only group leaders do anything:
        if (self.boozer.mpi is not None) and (not self.boozer.mpi.proc0_groups):
            logger.info("This proc is skipping Quasisymmetry.J since it is not a group leader.")
            return np.array([])

        # The next line is the expensive part of the calculation:
        self.boozer.run()

        symmetry_error = []
        for js, s in enumerate(self.s):
            index = self.boozer.s_to_index[s]
            bmnc = self.boozer.bx.bmnc_b[:, index]
            xm = self.boozer.bx.xm_b
            xn = self.boozer.bx.xn_b / self.boozer.bx.nfp

            if self.helicity_m != 0 and self.helicity_m != 1:
                raise ValueError("m for quasisymmetry should be 0 or 1.")

            # Find the indices of the symmetric modes:
            if self.helicity_n == 0:
                # Quasi-axisymmetry
                symmetric = (xn == 0)

            elif self.helicity_m == 0:
                # Quasi-poloidal symmetry
                symmetric = (xm == 0)

            else:
                # Quasi-helical symmetry
                symmetric = (xm * self.helicity_n + xn * self.helicity_m == 0)
                # Stellopt takes the "and" of this with mod(xm, self.helicity_m),
                # which does not seem necessary since self.helicity_m must be 1 to
                # get here.
            nonsymmetric = np.logical_not(symmetric)

            # Scale all bmnc modes so the average |B| is 1 or close to 1:

            if self.normalization == "B00":
                # Normalize by the (m,n) = (0,0) mode amplitude:
                assert xm[0] == 0
                assert xn[0] == 0
                bnorm = bmnc[0]

            elif self.normalization == "symmetric":
                # Normalize by sqrt(sum_{symmetric modes} B{m,n}^2)
                temp = bmnc[symmetric]
                bnorm = np.sqrt(np.dot(temp, temp))

            else:
                raise ValueError("Unrecognized value for normalization in Quasisymmetry")

            logger.info("For s={}, bnorm={}".format(s, bnorm))
            bmnc = bmnc / bnorm

            # Apply any weight that depends on m and/or n:

            if self.weight == "even":
                # Evenly weight each bmnc mode. Normalize by the m=n=0 mode on that surface.
                symmetry_error.append(bmnc[nonsymmetric])

            elif self.weight == "stellopt":
                # Stellopt appears to apply a m-dependent radial
                # weight, assuming sigma > 0. However, the m is
                # evaluated outside of any loop over m, so m ends up
                # taking the value mboz instead of the actual m for
                # each mode. As a result, there is an even weight by s_used**2.

                s_used = self.boozer.s_used[s]
                logger.info('s_used, in Quasisymmetry: {}'.format(s_used))
                """
                rad_sigma = np.full(len(xm), s_used * s_used)
                rad_sigma[xm < 3] = s_used
                rad_sigma[xm == 3] = s_used ** 1.5
                """
                rad_sigma = s_used * s_used
                temp = bmnc / rad_sigma
                symmetry_error.append(temp[nonsymmetric])

            elif self.weight == "stellopt_ornl":
                # This option is similar to "stellopt" except we
                # return a single number for the residual instead of a
                # vector of residuals.

                # For this option, stellopt applies a m-dependent
                # radial weight only when sigma < 0, the opposite of
                # when using the non-ORNL helicity! Here, we do not
                # apply such a weight.

                temp = bmnc[nonsymmetric]
                symmetry_error.append(np.array([np.sqrt(np.sum(temp * temp))]))

            else:
                raise ValueError("Unrecognized value for weight in Quasisymmetry")

        return np.array(symmetry_error).flatten()

    return_fn_map = {'J': J}
