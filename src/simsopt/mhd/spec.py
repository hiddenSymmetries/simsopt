# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides a class that handles the SPEC equilibrium code.
"""

import copy
import logging
import os.path
import traceback
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI
except ImportError as e:
    MPI = None
    logger.debug(str(e))

try:
    import spec.spec_f90wrapped as spec
except ImportError as e:
    spec = None
    logger.debug(str(e))

try:
    import py_spec
except ImportError as e:
    py_spec = None
    logger.debug(str(e))

try:
    import pyoculus
except ImportError as e:
    pyoculus = None
    logger.debug(str(e))

from .._core.optimizable import Optimizable
from .._core.util import ObjectiveFailure
from ..field.normal_field import NormalField
from ..geo.surfacerzfourier import SurfaceRZFourier
from .profiles import ProfileSpec

if MPI is not None:
    from ..util.mpi import MpiPartition
else:
    MpiPartition = None


__all__ = ['Spec', 'Residue']


class Spec(Optimizable):
    """
    This class represents the SPEC equilibrium code.

    Philosophy regarding mpol and ntor: The Spec object keeps track of
    mpol and ntor values that are independent of those for the
    boundary Surface object. If the Surface object has different
    mpol/ntor values, the Surface's rbc/zbs arrays are first copied,
    then truncated or expanded to fit the mpol/ntor values of the Spec
    object to before Spec is run. Therefore, you may sometimes need to
    manually change the mpol and ntor values for the Spec object.

    The default behavior is that all  output files will be
    deleted except for the first and most recent iteration on worker
    group 0. If you wish to keep all the output files, you can set
    ``keep_all_files = True``. If you want to save the output files
    for a certain intermediate iteration, you can set the
    ``files_to_delete`` attribute to ``[]`` after that run of SPEC.

    Args:
        filename: SPEC input file to use for initialization. It should end
          in ``.sp``. Or, if None, default values will be used.
        mpi: A :obj:`simsopt.util.mpi.MpiPartition` instance, from which
          the worker groups will be used for SPEC calculations. If ``None``,
          each MPI process will run SPEC independently.
        verbose: Whether to print SPEC output to stdout.
        keep_all_files: If ``False``, all output files will be deleted
          except for the first and most recent ones from worker group 0. If
          ``True``, all output files will be kept.
        tolerance: Max force balance residue to consider the equilibrium as
          converged; if :math:`|f|>` tolerance, raise ``ObjectiveFailure`` exception. By
          default set to 1E-12.
    """

    def __init__(self,
                 filename: Union[str, None] = None,
                 mpi: Union[MpiPartition, None] = None,
                 verbose: bool = True,
                 keep_all_files: bool = False,
                 tolerance: float = 1e-12):

        if spec is None:
            raise RuntimeError(
                "Using Spec requires spec python wrapper to be installed.")
        if py_spec is None:
            raise RuntimeError(
                "Using Spec requires py_spec to be installed.")

        self.lib = spec
        # For the most commonly accessed fortran modules, provide a
        # shorthand so ".lib" is not needed:
        modules = [
            "inputlist",
            "allglobal",
        ]
        for key in modules:
            setattr(self, key, getattr(spec, key))

        self.verbose = verbose
        # mute screen output if necessary
        # TODO: relies on /dev/null being accessible (Windows!)
        if not self.verbose:
            self.lib.fileunits.mute(1)

        # python wrapper does not need to write files along the run
        #self.lib.allglobal.skip_write = True

        # If mpi is not specified, use a single worker group:
        if mpi is None:
            self.mpi = MpiPartition(ngroups=1)
        else:
            self.mpi = mpi
        # SPEC will use the "groups" communicator from the MpiPartition:
        self.lib.allglobal.set_mpi_comm(self.mpi.comm_groups.py2f())

        if filename is None:
            # Read default input file, which should be in the same
            # directory as this file:
            filename = os.path.join(os.path.dirname(__file__), 'defaults.sp')
            logger.info(
                f"Initializing a SPEC object from defaults in {filename}")
        else:
            if not filename.endswith('.sp'):
                filename = f"{filename}.sp"
            logger.info(f"Initializing a SPEC object from file: {filename}")

        if tolerance <= 0:
            raise ValueError(
                'tolerance should be greater than zero'
            )
        self.tolerance = tolerance

        self.init(filename)
        si = spec.inputlist  # Shorthand

        # Read number of (plasma) volumes
        self.nvol = si.nvol

        # Read number of (plasma+vacuum) volumes
        if si.lfreebound:
            self.mvol = self.nvol + 1 
        else:
            self.mvol = self.nvol

        # Store initial guess data
        # The initial guess is a collection of SurfaceRZFourier instances,
        # stored in a list of size Mvol-1 (the number of inner interfaces)
        nmodes = self.allglobal.num_modes
        stellsym = bool(si.istellsym)
        if nmodes > 0 and self.nvol > 1:
            self.initial_guess = [ 
                SurfaceRZFourier(nfp=si.nfp, stellsym=stellsym, mpol=si.mpol, ntor=si.ntor) for n in range(0, self.mvol-1)
            ]
            for imode in range(0, nmodes):
                mm = self.allglobal.mmrzrz[imode]
                nn = self.allglobal.nnrzrz[imode]
                if mm > si.mpol:
                    continue
                if abs(nn) > si.ntor:
                    continue

                # Populate SurfaceRZFourier instances, except plasma boundary
                for lvol in range(0, self.nvol-1):
                    self.initial_guess[lvol].set_rc(mm, nn, self.allglobal.allrzrz[0, lvol, imode])
                    self.initial_guess[lvol].set_zs(mm, nn, self.allglobal.allrzrz[1, lvol, imode])

                    if not si.istellsym:
                        self.initial_guess[lvol].set_rs(mm, nn, self.allglobal.allrzrz[2, lvol, imode])
                        self.initial_guess[lvol].set_zc(mm, nn, self.allglobal.allrzrz[3, lvol, imode])

                if si.lfreebound:  # Populate plasma boundary as well
                    self.initial_guess[self.nvol-1].set_rc(mm, nn, si.rbc[si.mntor+nn, si.mmpol+mm])
                    self.initial_guess[self.nvol-1].set_zs(mm, nn, si.zbs[si.mntor+nn, si.mmpol+mm])

                    if not si.istellsym:
                        self.initial_guess[self.nvol-1].set_rs(mm, nn, si.rbs[si.mntor+nn, si.mmpol+mm])
                        self.initial_guess[self.nvol-1].set_zc(mm, nn, si.zbc[si.mntor+nn, si.mmpol+mm])

            # In general, initial guess is NOT a degree of freedom for the
            # optimization - we thus fix them.
            for lvol in range(0, self.mvol-1):
                self.initial_guess[lvol].fix_all()

        else: 
            # There is no initial guess - in this case, we let SPEC handle
            # the construction of the initial guess. This generally means
            # that the geometry of the inner interfaces will be constructed
            # by interpolation between the plasma (or computational) boundary
            # and the magnetic axis

            self.initial_guess = None

        # Store axis data
        self.axis = {}
        self.axis['rac'] = copy.copy(si.rac[0:si.ntor+1])
        self.axis['zas'] = copy.copy(si.zas[0:si.ntor+1])
        if si.istellsym == 0:
            self.axis['ras'] = copy.copy(si.ras[0:si.ntor+1])
            self.axis['zac'] = copy.copy(si.zac[0:si.ntor+1])

        self.extension = filename[:-3]
        self.keep_all_files = keep_all_files
        self.files_to_delete = []

        # Create a surface object for the boundary:
        logger.debug(f"In __init__, si.istellsym={si.istellsym} stellsym={stellsym}")
        self._boundary = SurfaceRZFourier(nfp=si.nfp,
                                          stellsym=stellsym,
                                          mpol=si.mpol,
                                          ntor=si.ntor)

        # Transfer the boundary shape from fortran to the boundary
        # surface object:
        for m in range(si.mpol + 1):
            for n in range(-si.ntor, si.ntor + 1):
                self._boundary.rc[m,
                                  n + si.ntor] = si.rbc[n + si.mntor,
                                                        m + si.mmpol]
                self._boundary.zs[m,
                                  n + si.ntor] = si.zbs[n + si.mntor,
                                                        m + si.mmpol]
                if not stellsym:
                    self._boundary.rs[m,
                                      n + si.ntor] = si.rbs[n + si.mntor,
                                                            m + si.mmpol]
                    self._boundary.zc[m,
                                      n + si.ntor] = si.zbc[n + si.mntor,
                                                            m + si.mmpol]
        self._boundary.local_full_x = self._boundary.get_dofs()

        self.need_to_run_code = True
        self.counter = -1

        # Set profiles as None - these have to be defined in a script if the user
        # wish to use them as degrees of freedom
        self._volume_current_profile = None
        self._interface_current_profile = None
        self._pressure_profile = None
        self._iota_profile = None
        self._oita_profile = None
        self._mu_profile = None
        self._pflux_profile = None
        self._tflux_profile = None
        self._helicity_profile = None

        # Define normal field - these are the Vns, Vnc harmonics. Can be used as
        # dofs in an optimization
        if si.lfreebound:
            self.normal_field = NormalField.from_spec(filename)
        else:
            self.normal_field = None

        # By default, all dofs owned by SPEC directly, as opposed to
        # dofs owned by the boundary surface object, are fixed.
        x0 = self.get_dofs()
        fixed = np.full(len(x0), True)
        names = ['phiedge', 'curtor']
        if si.lfreebound:
            depends_on = [self.normal_field]
        else:
            depends_on = [self._boundary]

        super().__init__(x0=x0, fixed=fixed, names=names,
                         depends_on=depends_on,
                         external_dof_setter=Spec.set_dofs)

    @property
    def boundary(self):
        """
        Getter for the plasma boundary

        Returns:
            SurfaceRZFourier instance representing the plasma boundary
        """
        return self._boundary

    @property
    def pressure_profile(self):
        """
        Getter for the pressure profile

        Returns:
            ProfileSpec instance representing the pressure profile
        """
        return self._pressure_profile

    @pressure_profile.setter
    def pressure_profile(self, pressure_profile):
        """
        Setter for the pressure profile

        Args:
            ProfileSpec instance for the pressure profile
        """

        # Check inputs
        if not isinstance(pressure_profile, ProfileSpec):
            ValueError('Input should be a ProfileSpec')

        # Check size
        if pressure_profile.dofs.full_x.size != self.mvol:
            ValueError('Invalid number of dofs. Shoudl be equal to Mvol!')

        # Update pressure profile
        if pressure_profile is not self._pressure_profile:
            logging.debug('Replacing pressure_profile in setter')
            if self._pressure_profile is not None:
                self.remove_parent(self._pressure_profile)
            self._pressure_profile = pressure_profile
            if pressure_profile is not None:
                self.append_parent(pressure_profile)
                self.need_to_run_code = True

    @property
    def volume_current_profile(self):
        """
        Getter for the volume current profile (Ivolume)

        Returns:
            ProfileSpec instance representing the volume current profile
        """
        return self._volume_current_profile

    @volume_current_profile.setter
    def volume_current_profile(self, volume_current_profile):
        """
        Setter for the volume current profile

        Args:
            ProfileSpec instance for the volume current profile
        """

        if not isinstance(volume_current_profile, ProfileSpec):
            ValueError('Input should be a ProfileSpec')

        # Check size
        if volume_current_profile.dofs.full_x.size != self.mvol:
            ValueError('Invalid number of dofs. Shoudl be equal to Mvol!')

        # Volume current is a cumulative property
        volume_current_profile.cumulative = True

        if volume_current_profile is not self._volume_current_profile:
            logging.debug('Replacing pressure_profile in setter')
            if self._volume_current_profile is not None:
                self.remove_parent(self._volume_current_profile)
            self._volume_current_profile = volume_current_profile
            if volume_current_profile is not None:
                self.append_parent(volume_current_profile)
                self.need_to_run_code = True

    @property
    def interface_current_profile(self):
        """
        Getter for the surface current profile (Isurf)

        Returns:
            ProfileSpec instance representing the surface current profile
        """
        return self._interface_current_profile

    @interface_current_profile.setter
    def interface_current_profile(self, interface_current_profile):
        """
        Setter for the surface current profile

        Args:
            ProfileSpec instance for the surface current profile
        """

        if not isinstance(interface_current_profile, ProfileSpec):
            ValueError('Input should be a ProfileSpec')

        # Check size
        if interface_current_profile.dofs.full_x.size != self.mvol:
            ValueError('Invalid number of dofs. Shoudl be equal to Mvol!')

        if interface_current_profile is not self._interface_current_profile:
            logging.debug('Replacing pressure_profile in setter')
            if self._interface_current_profile is not None:
                self.remove_parent(self._interface_current_profile)
            self._interface_current_profile = interface_current_profile
            if interface_current_profile is not None:
                self.append_parent(interface_current_profile)
                self.need_to_run_code = True

    @property
    def iota_profile(self):
        """
        Getter for the inner rotational transform profile (iota)

        Returns:
            ProfileSpec instance representing the iota profile
        """
        return self._iota_profile

    @iota_profile.setter
    def iota_profile(self, iota_profile):
        """
        Setter for the inner rotational transform profile (iota)

        Args:
            ProfileSpec instance for the inner rotational transform profile
        """

        if not isinstance(iota_profile, ProfileSpec):
            ValueError('Input should be a ProfileSpec')

        # Check size
        if iota_profile.dofs.full_x.size != self.mvol:
            ValueError('Invalid number of dofs. Shoudl be equal to Mvol!')

        if iota_profile is not self._iota_profile:
            logging.debug('Replacing pressure_profile in setter')
            if self._iota_profile is not None:
                self.remove_parent(self._iota_profile)
            self._iota_profile = iota_profile
            if iota_profile is not None:
                self.append_parent(iota_profile)
                self.need_to_run_code = True

    @property
    def oita_profile(self):
        """
        Getter for the outer rotational transform profile (oita)

        Returns:
            ProfileSpec instance representing the oita profile
        """
        return self._oita_profile

    @oita_profile.setter
    def oita_profile(self, oita_profile):
        """
        Setter for the outer rotational transform profile (oita)

        Args:
            ProfileSpec instance for the outer rotational transform profile
        """

        if not isinstance(oita_profile, ProfileSpec):
            ValueError('Input should be a ProfileSpec')

        # Check size
        if oita_profile.dofs.full_x.size != self.mvol:
            ValueError('Invalid number of dofs. Shoudl be equal to Mvol!')

        if oita_profile is not self._oita_profile:
            logging.debug('Replacing pressure_profile in setter')
            if self._oita_profile is not None:
                self.remove_parent(self._oita_profile)
            self._oita_profile = oita_profile
            if oita_profile is not None:
                self.append_parent(oita_profile)
                self.need_to_run_code = True

    @property
    def mu_profile(self):
        """
        Getter for the mu-profile

        Returns:
            ProfileSpec instance representing the mu profile
        """
        return self._mu_profile

    @mu_profile.setter
    def mu_profile(self, mu_profile):
        """
        Setter for the mu profile (oita)

        Args:
            ProfileSpec instance for the outer rotational transform profile
        """

        if not isinstance(mu_profile, ProfileSpec):
            ValueError('Input should be a ProfileSpec')

        # Check size
        if mu_profile.dofs.full_x.size != self.mvol:
            ValueError('Invalid number of dofs. Shoudl be equal to Mvol!')

        if mu_profile is not self._mu_profile:
            logging.debug('Replacing pressure_profile in setter')
            if self._mu_profile is not None:
                self.remove_parent(self._mu_profile)
            self._mu_profile = mu_profile
            if mu_profile is not None:
                self.append_parent(mu_profile)
                self.need_to_run_code = True

    @property
    def pflux_profile(self):
        """
        Getter for the poloidal flux profile (pflux)

        Returns:
            ProfileSpec instance representing the poloidal flux profile
        """
        return self._pflux_profile

    @pflux_profile.setter
    def pflux_profile(self, pflux_profile):
        """
        Setter for the poloidal flux profile (pflux)

        Args:
            ProfileSpec instance for the poloidal flux profile
        """

        if not isinstance(pflux_profile, ProfileSpec):
            ValueError('Input should be a ProfileSpec')

        # Check size
        if pflux_profile.dofs.full_x.size != self.mvol:
            ValueError('Invalid number of dofs. Shoudl be equal to Mvol!')

        # pflux is a cumulative property
        pflux_profile.cumulative = True

        if pflux_profile is not self._pflux_profile:
            logging.debug('Replacing pressure_profile in setter')
            if self._pflux_profile is not None:
                self.remove_parent(self._pflux_profile)
            self._pflux_profile = pflux_profile
            if pflux_profile is not None:
                self.append_parent(pflux_profile)
                self.need_to_run_code = True

    @property
    def tflux_profile(self):
        """
        Getter for the toroidal flux profile (tflux)

        Returns:
            ProfileSpec instance representing the toroidal flux profile
        """
        return self._tflux_profile

    @tflux_profile.setter
    def tflux_profile(self, tflux_profile):
        """
        Setter for the toroidal flux profile (tflux)

        Args:
            ProfileSpec instance for the toroidal flux profile
        """

        if not isinstance(tflux_profile, ProfileSpec):
            ValueError('Input should be a ProfileSpec')

        # Check size
        if tflux_profile.dofs.full_x.size != self.mvol:
            ValueError('Invalid number of dofs. Shoudl be equal to Mvol!')

        # pflux is a cumulative property
        tflux_profile.cumulative = True

        if tflux_profile is not self._tflux_profile:
            logging.debug('Replacing pressure_profile in setter')
            if self._tflux_profile is not None:
                self.remove_parent(self._tflux_profile)
            self._tflux_profile = tflux_profile
            if tflux_profile is not None:
                self.append_parent(tflux_profile)
                self.need_to_run_code = True

    @property
    def helicity_profile(self):
        """
        Getter for the magnetic helicity profile (helicity)

        Returns:
            ProfileSpec instance representing the magnetic helicity profile
        """
        return self._helicity_profile

    @helicity_profile.setter
    def helicity_profile(self, helicity_profile):
        """
        Setter for the toroidal flux profile (tflux)

        Args:
            ProfileSpec instance for the toroidal flux profile
        """

        if not isinstance(helicity_profile, ProfileSpec):
            ValueError('Input should be a ProfileSpec')

        # Check size
        if helicity_profile.dofs.full_x.size != self.mvol:
            ValueError('Invalid number of dofs. Shoudl be equal to Mvol!')

        if helicity_profile is not self._helicity_profile:
            logging.debug('Replacing pressure_profile in setter')
            if self._helicity_profile is not None:
                self.remove_parent(self._tflux_profile)
            self._helicity_profile = helicity_profile
            if helicity_profile is not None:
                self.append_parent(helicity_profile)
                self.need_to_run_code = True

    def set_profile(self, longname, lvol, value):
        """
        This function is used to set the pressure, currents, iota, oita,
        mu pflux and/or tflux in volume lvol

        Args:
            longname: string, either 
                - 'pressure'
                - 'volume_current'
                - 'surface_current'
                - 'iota'
                - 'oita'
                - 'mu'
                - 'pflux'
                - 'tflux'
                - 'helicity'
            lvol: integer, from 0 to Mvol-1
            value: real, new value
        """
        profile = self.__getattribute__(longname + "_profile")
        if profile is None:
            return

        # If the profile is cumulative, values in lvol to Mvol-1 are modified.
        # If it is not cumulative, only the value in lvol is modified.
        if profile.cumulative: 
            old_value = profile.f(lvol)

            profile.set(lvol, value)
            for ivol in range(lvol + 1, self.mvol):
                profile.set(ivol, profile.f(ivol) - old_value + value)
        else:
            profile.set(lvol, value)

    def get_profile(self, longname, lvol):
        """
        This function is used to get the pressure, currents, iota, oita,
        mu pflux and/or tflux profiles.

        Args:
            longname: string, either
                - 'pressure'
                - 'volume_current'
                - 'surface_current'
                - 'iota'
                - 'oita'
                - 'mu'
                - 'pflux'
                - 'tflux'
                - 'helicity'
            lvol: integer, list or np.array of volume indices, from 0 to Mvol-1
        Returns:
            np.array of length lvol, with the profiles values.
        """

        profile = self.__getattribute__(longname + "_profile")
        if profile is None:
            return

        return profile.f(lvol)

    @boundary.setter
    def boundary(self, boundary):
        """
        Setter for the geometry of the plasma boundary
        """

        if self._boundary is not boundary:
            self.remove_parent(self._boundary)
            self._boundary = boundary
            self.append_parent(boundary)

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True

    def get_dofs(self):
        return np.array([self.inputlist.phiedge,
                         self.inputlist.curtor])

    def set_dofs(self, x):
        self.need_to_run_code = True
        self.inputlist.phiedge = x[0]
        self.inputlist.curtor = x[1]

        profiles = [
            self._volume_current_profile,
            self._interface_current_profile,
            self._pressure_profile,
            self._iota_profile,
            self._oita_profile,
            self._mu_profile,
            self._pflux_profile,
            self._tflux_profile,
            self._helicity_profile
        ]
        for p in profiles:
            if p is not None:
                p.phiedge = x[0]

    def init(self, filename: str):
        """
        Initialize SPEC fortran state from an input file.

        Args:
            filename: Name of the file to load. It should end in ``.sp``.
        """
        logger.debug("Entering init")
        if self.mpi.proc0_groups:
            spec.inputlist.initialize_inputs()
            logger.debug("Done with initialize_inputs")
            self.extension = filename[:-3]  # Remove the ".sp"
            spec.allglobal.ext = self.extension
            spec.allglobal.read_inputlists_from_file()
            logger.debug("Done with read_inputlists_from_file")
            spec.allglobal.check_inputs()

        logger.debug('About to call broadcast_inputs')
        spec.allglobal.broadcast_inputs()
        logger.debug('About to call preset')
        spec.preset()
        logger.debug("Done with init")

    def run(self, update_guess: bool = True):
        """
        Run SPEC, if needed.

        Args:
            - update_guess: boolean. If True, initial guess will be updated with
              the geometry of the interfaces found at equilibrium. Default is 
              True
        """
        if not self.need_to_run_code:
            logger.info("run() called but no need to re-run SPEC.")
            return
        logger.info("Preparing to run SPEC.")
        self.counter += 1

        si = self.inputlist  # Shorthand

        # Check that number of volumes in internal memory is consistent with
        # the input file
        if self.nvol != si.nvol:
            ValueError('Inconsistent Nvol')

        # nfp must be consistent between the surface and SPEC. The surface's
        # value trumps.
        si.nfp = self.boundary.nfp
        si.istellsym = int(self.boundary.stellsym)

        # Convert boundary to RZFourier if needed:
        boundary_RZFourier = self.boundary.to_RZFourier()

        # Transfer boundary data to fortran:
        si.rbc[:, :] = 0.0
        si.zbs[:, :] = 0.0
        si.rbs[:, :] = 0.0
        si.zbc[:, :] = 0.0
        mpol_capped = np.min([boundary_RZFourier.mpol, si.mmpol])
        ntor_capped = np.min([boundary_RZFourier.ntor, si.mntor])
        stellsym = bool(si.istellsym)
        logger.debug(f"In run, si.istellsym = {si.istellsym} stellsym = {stellsym}")
        for m in range(mpol_capped + 1):
            for n in range(-ntor_capped, ntor_capped + 1):
                si.rbc[n + si.mntor, m + si.mmpol] = boundary_RZFourier.get_rc(m, n)
                si.zbs[n + si.mntor, m + si.mmpol] = boundary_RZFourier.get_zs(m, n)
                if not stellsym:
                    si.rbs[n + si.mntor, m + si.mmpol] = boundary_RZFourier.get_rs(m, n)
                    si.zbc[n + si.mntor, m + si.mmpol] = boundary_RZFourier.get_zc(m, n)

        # Set the coordinate axis using the lrzaxis=2 feature:
        si.lrzaxis = 2

        # Set axis from latest converged state
        mn = self.axis['rac'].size
        si.rac[0:mn] = self.axis['rac']
        si.zas[0:mn] = self.axis['zas']
        if si.istellsym == 0:
            si.ras[0:mn] = self.axis['ras']
            si.zac[0:mn] = self.axis['zac']

        # Set initial guess
        if not self.initial_guess is None:        
            # Set all modes to zero
            spec.allglobal.mmrzrz[:] = 0
            spec.allglobal.nnrzrz[:] = 0
            spec.allglobal.allrzrz[:] = 0

            # transform to SurfaceRZFourier if necessary
            initial_guess = [s.to_RZFourier() for s in self.initial_guess]

            # Loop on modes
            imn = -1  # counter
            for mm in range(0, si.mpol+1):
                for nn in range(-si.ntor, si.ntor+1):
                    if mm == 0 and nn < 0:
                        continue

                    imn += 1

                    spec.allglobal.mmrzrz[imn] = mm
                    spec.allglobal.nnrzrz[imn] = nn

                    # Populate inner plasma boundaries
                    for lvol in range(0, self.nvol-1):
                        spec.allglobal.allrzrz[0, lvol, imn] = initial_guess[lvol].get_rc(mm, nn)
                        spec.allglobal.allrzrz[1, lvol, imn] = initial_guess[lvol].get_zs(mm, nn)

                        if not si.istellsym:
                            spec.allglobal.allrzrz[2, lvol, imn] = initial_guess[lvol].get_rs(mm, nn)
                            spec.allglobal.allrzrz[3, lvol, imn] = initial_guess[lvol].get_zc(mm, nn)

                    # Populate plasma boundary
                    if si.lfreebound:
                        si.rbc[si.mntor+nn, si.mmpol+mm] = initial_guess[self.nvol-1].get_rc(mm, nn)
                        si.zbs[si.mntor+nn, si.mmpol+mm] = initial_guess[self.nvol-1].get_zs(mm, nn)

                        if not si.istellsym:
                            si.rbs[si.mntor+nn, si.mmpol+mm] = initial_guess[self.nvol-1].get_rs(mm, nn)
                            si.zbc[si.mntor+nn, si.mmpol+mm] = initial_guess[self.nvol-1].get_zc(mm, nn)

            spec.allglobal.num_modes = imn + 1

        # Set profiles from dofs
        if self.pressure_profile is not None:
            si.pressure[0:self.nvol] = self.pressure_profile.get(
                np.arange(0, self.nvol))
            if si.lfreebound:
                si.pressure[self.nvol] = 0

        if self.volume_current_profile is not None:
            # Volume current is a cumulative profile; special care is required
            # when a dofs is changed in order to keep fixed dofs fixed!
            old_ivolume = copy.copy(si.ivolume)
            for lvol in range(0, self.mvol):
                if self.volume_current_profile.is_fixed(lvol):
                    if lvol != 0:
                        si.ivolume[lvol] = si.ivolume[lvol] - \
                            old_ivolume[lvol - 1] + si.ivolume[lvol - 1]
                        self.set_profile(
                            'volume_current', lvol=lvol, value=si.ivolume[lvol])
                else:
                    si.ivolume[lvol] = self.get_profile('volume_current', lvol)

            if si.lfreebound:
                si.ivolume[self.nvol] = si.ivolume[self.nvol - 1]
                self.volume_current_profile.set(
                    key=self.mvol - 1, new_val=si.ivolume[self.nvol - 1])

        if self.interface_current_profile is not None:
            si.isurf[0:self.mvol - 1] = \
                self.interface_current_profile.get(np.arange(0, self.mvol-1))

        # Update total plasma toroidal current in case of freeboundary
        # calculation
        if ((self.volume_current_profile is not None) or
            (self.interface_current_profile is not None)) and \
                si.lfreebound:
            si.curtor = si.ivolume[self.nvol - 1] + np.sum(si.isurf)

        if self.iota_profile is not None:
            si.iota[0:self.nvol+1] = self.iota_profile.get(np.arange(0, self.nvol))

        if self.oita_profile is not None:
            si.oita[0:self.nvol+1] = self.oita_profile.get(np.arange(0, self.nvol))

        if self.mu_profile is not None:
            si.mu[0:self.nvol] = self.mu_profile.get(np.arange(0, self.nvol))
            if si.lfreebound:
                si.mu[self.mvol] = 0

        if self.pflux_profile is not None:
            # Pflux is a cumulative profile; special care is required
            # when a dofs is changed in order to keep fixed dofs fixed!
            old_pflux = copy.copy(si.pflux)
            for lvol in range(0, self.mvol):
                if self.pflux_profile.is_fixed(lvol):
                    if lvol != 0:
                        si.pflux[lvol] = si.pflux[lvol] - \
                            old_pflux[lvol - 1] + si.pflux[lvol - 1]
                        self.pflux_profile.set(
                            key=lvol, new_val=si.pflux[lvol])
                else:
                    si.pflux[lvol] = self.pflux_profile.get(lvol)

        if self.tflux_profile is not None:
            # tflux is a cumulative profile; special care is required
            # when a dofs is changed in order to keep fixed dofs fixed!
            old_tflux = copy.copy(si.tflux)
            for lvol in range(0, self.mvol):
                if self.tflux_profile.is_fixed(lvol):
                    if lvol != 0:
                        si.tflux[lvol] = si.tflux[lvol] - \
                            old_tflux[lvol - 1] + si.tflux[lvol - 1]
                        self.tflux_profile.set(
                            key=lvol, new_val=si.tflux[lvol])
                else:
                    si.tflux[lvol] = self.tflux_profile.get(lvol)

        if self.helicity_profile is not None:
            si.helicity[0:self.nvol] = self.helicity_profile.get(np.arange(0, self.nvol))
            if si.lfreebound:
                si.helicity[self.mvol] = 0

        # Another possible way to initialize the coordinate axis: use
        # the m=0 modes of the boundary.
        # m = 0
        # for n in range(2):
        #     si.rac[n] = si.rbc[n + si.mntor, m + si.mmpol]
        #     si.zas[n] = si.zbs[n + si.mntor, m + si.mmpol]
        filename = self.extension + \
            '_{:03}_{:06}'.format(self.mpi.group, self.counter)
        logger.info("Running SPEC using filename " + filename)
        self.allglobal.ext = filename
        try:
            # Here is where we actually run SPEC:
            if self.mpi.proc0_groups:
                logger.debug('About to call check_inputs')
                spec.allglobal.check_inputs()
            logger.debug('About to call broadcast_inputs')
            spec.allglobal.broadcast_inputs()
            logger.debug('About to call preset')
            spec.preset()
            logger.debug(f'About to call init_outfile')
            spec.sphdf5.init_outfile()
            logger.debug('About to call mirror_input_to_outfile')
            spec.sphdf5.mirror_input_to_outfile()
            if self.mpi.proc0_groups:
                logger.debug('About to call wrtend')
                spec.allglobal.wrtend()
            logger.debug('About to call init_convergence_output')
            spec.sphdf5.init_convergence_output()
            logger.debug(f'About to call spec')
            spec.spec()
            logger.debug('About to call diagnostics')
            spec.final_diagnostics()
            logger.debug('About to call write_grid')
            spec.sphdf5.write_grid()
            if self.mpi.proc0_groups:
                logger.debug('About to call wrtend')
                spec.allglobal.wrtend()
            logger.debug('About to call hdfint')
            spec.sphdf5.hdfint()
            logger.debug('About to call finish_outfile')
            spec.sphdf5.finish_outfile()
            logger.debug('About to call ending')
            spec.ending()

        except BaseException:
            if self.verbose:
                traceback.print_exc()
            raise ObjectiveFailure("SPEC did not run successfully.")

        logger.info("SPEC run complete.")
        # Barrier so workers do not try to read the .h5 file before it
        # is finished:
        self.mpi.comm_groups.Barrier()

        # Try to read SPEC output. 
        try:
            self.results = py_spec.SPECout(filename + '.sp.h5')
        except BaseException:
            if self.verbose:
                traceback.print_exc()
            raise ObjectiveFailure(
                "Unable to read results following SPEC execution")

        logger.info("Successfully loaded SPEC results.")
        self.need_to_run_code = False

        # Deal with unconverged equilibria - these are excluded by 
        # the optimizer, and the objective function is set to a large number
        if self.results.output.ForceErr > self.tolerance:
            raise ObjectiveFailure(
                'SPEC could not find force balance'
            )

        # Save geometry as initial guess for next iterations
        if update_guess:
            new_guess = None
            if self.mvol > 1:
                new_guess = [
                    SurfaceRZFourier(nfp=si.nfp, stellsym=si.istellsym, mpol=si.mpol, ntor=si.ntor) for n in range(0, self.mvol-1)
                ]

                for ii, (mm, nn) in enumerate(zip(self.results.output.im, self.results.output.in_)):
                    nnorm = (nn / si.nfp).astype('int')
                    for lvol in range(0, self.mvol-1):
                        new_guess[lvol].set_rc(mm, nnorm, self.results.output.Rbc[lvol+1, ii])
                        new_guess[lvol].set_zs(mm, nnorm, self.results.output.Zbs[lvol+1, ii])

                        if not si.istellsym:
                            new_guess[lvol].set_rs(mm, nnorm, self.results.output.Rbs[lvol+1, ii])
                            new_guess[lvol].set_zc(mm, nnorm, self.results.output.Zbc[lvol+1, ii])

                axis = {}
                axis['rac'] = self.results.output.Rbc[0, 0:si.ntor+1]
                axis['zas'] = self.results.output.Zbs[0, 0:si.ntor+1]
                self.axis = copy.copy(axis)

            # Enforce SPEC to use initial guess
            self.initial_guess = new_guess
            self.inputlist.linitialize = 0

        # Group leaders handle deletion of files:
        if self.mpi.proc0_groups:
            # If the worker group is not 0, delete all wout files, unless
            # keep_all_files is True:
            if (not self.keep_all_files) and (self.mpi.group > 0):
                os.remove(filename + '.sp.h5')
                os.remove(filename + '.sp.end')

            # Delete the previous output file, if desired:
            for file_to_delete in self.files_to_delete:
                os.remove(file_to_delete)
            self.files_to_delete = []

            # Record the latest output file to delete if we run again:
            if (self.mpi.group == 0) and (
                    self.counter > 0) and (not self.keep_all_files):
                self.files_to_delete.append(filename + '.sp.h5')
                self.files_to_delete.append(filename + '.sp.end')

    def volume(self):
        """
        Return the volume inside the boundary flux surface.
        """
        self.run()
        return self.results.output.volume * self.results.input.physics.Nfp

    def iota(self):
        """
        Return the rotational transform in the middle of the volume.
        """
        self.run()
        return self.results.transform.fiota[1, 0]


class Residue(Optimizable):
    """
    Greene's residue, evaluated from a Spec equilibrum

    Args:
        spec: a Spec object
        pp, qq: Numerator and denominator for the resonant iota = pp / qq
        vol: Index of the Spec volume to consider
        theta: Spec's theta coordinate at the periodic field line
        s_guess: Guess for the value of Spec's s coordinate at the periodic
                field line
        s_min, s_max: bounds on s for the search
        rtol: the relative tolerance of the integrator
    """

    def __init__(self, spec, pp, qq, vol=1, theta=0, s_guess=None, s_min=-1.0,
                 s_max=1.0, rtol=1e-9):
        # if not spec_found:
        if spec is None:
            raise RuntimeError(
                "Residue requires spec package to be installed.")
        # if not pyoculus_found:
        if pyoculus is None:
            raise RuntimeError(
                "Residue requires pyoculus package to be installed.")

        self.spec = spec
        self.pp = pp
        self.qq = qq
        self.vol = vol
        self.theta = theta
        self.rtol = rtol
        if s_guess is None:
            self.s_guess = 0.0
        else:
            self.s_guess = s_guess
        self.s_min = s_min
        self.s_max = s_max
        self.depends_on = ['spec']
        self.need_to_run_code = True
        self.fixed_point = None
        # We may at some point want to allow Residue to use a
        # different MpiPartition than the Spec object it is attached
        # to, but for now we'll use the same MpiPartition for
        # simplicity.
        self.mpi = spec.mpi
        super().__init__(depends_on=[spec])

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True

    def J(self):
        """
        Run Spec if needed, find the periodic field line, and return the residue
        """
        if not self.mpi.proc0_groups:
            logger.info(
                "This proc is skipping Residue.J() since it is not a group leader.")
            return

        if self.need_to_run_code:
            self.spec.run()
            specb = pyoculus.problems.SPECBfield(self.spec.results, self.vol)
            # Set nrestart=0 because otherwise the random guesses in
            # pyoculus can cause examples/tests to be
            # non-reproducible.
            fp = pyoculus.solvers.FixedPoint(
                specb, {
                    'theta': self.theta, 'nrestart': 0}, integrator_params={
                    'rtol': self.rtol})
            self.fixed_point = fp.compute(self.s_guess,
                                          sbegin=self.s_min,
                                          send=self.s_max,
                                          pp=self.pp, qq=self.qq)
            self.need_to_run_code = False

        if self.fixed_point is None:
            raise ObjectiveFailure("Residue calculation failed")

        return self.fixed_point.GreenesResidue
