# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides a class that handles the SPEC equilibrium code.
"""

import copy
from .normal_field import NormalField
from ..geo.surfacerzfourier import SurfaceRZFourier
from .._core.util import ObjectiveFailure
from .._core.optimizable import Optimizable
import logging
from typing import Union
import os.path
import traceback

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
    """

    def __init__(self,
                 filename: Union[str, None] = None,
                 initial_guess: Union[str, None] = None,
                 mpi: Union[MpiPartition, None] = None,
                 verbose: bool = True,
                 keep_all_files: bool = False):

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

        self.init(filename)
        si = spec.inputlist  # Shorthand

        nvol = si.nvol
        if si.lfreebound==0:
            mvol = nvol
        else:
            mvol = nvol + 1
        
        # Store initial guess data
        nmodes = self.allglobal.num_modes
        mn = si.ntor+1 + si.mpol*(2*si.ntor+1)
        if nmodes>0:
            # Save inner boundaries geometry
            self.initial_guess = {}
            self.initial_guess['mm'] = np.zeros((mn,), dtype='int')
            self.initial_guess['nn'] = np.zeros((mn,), dtype='int')
            self.initial_guess['rbc'] = np.zeros((mvol-1,mn))
            self.initial_guess['zbs'] = np.zeros((mvol-1,mn))
            if si.istellsym==0:
                self.initial_guess['rbs'] = np.zeros((1,mn))
                self.initial_guess['zbc'] = np.zeros((1,mn))

            ii = 0
            for mm in range(0,si.mpol+1):
                for nn in range(-si.ntor,si.ntor+1):
                    if mm==0 and nn<0: continue

                    self.initial_guess['mm'][ii] = mm
                    self.initial_guess['nn'][ii] = nn

                    indm = np.where(self.allglobal.mmrzrz[0:nmodes]==mm)
                    indn = np.where(self.allglobal.nnrzrz[0:nmodes]==nn)
                    ind = np.intersect1d( indm, indn )

                    if ind.size==1:
                        self.initial_guess['rbc'] [:,ii] = copy.copy(self.allglobal.allrzrz[0,0:mvol-1,ind])
                        self.initial_guess['zbs'] [:,ii] = copy.copy(self.allglobal.allrzrz[1,0:mvol-1,ind])

                        if si.istellsym==0:
                            self.initial_guess['rbs'] [:,ii] = copy.copy(self.allglobal.allrzrz[2,0:mvol-1,ind])
                            self.initial_guess['zbc'] [:,ii] = copy.copy(self.allglobal.allrzrz[3,0:mvol-1,ind])
                    
                    ii = ii + 1

        else:
            self.initial_guess = None

        # Store plasma boundary
        if si.lfreebound==1:
            plasma_boundary = {}
            plasma_boundary['mm'] = np.zeros((mn,))
            plasma_boundary['nn'] = np.zeros((mn,))
            plasma_boundary['rbc'] = np.zeros((1,mn))
            plasma_boundary['zbs'] = np.zeros((1,mn))
            if si.istellsym==0:
                plasma_boundary['rbs'] = np.zeros((1,mn))
                plasma_boundary['zbc'] = np.zeros((1,mn))

            ii = 0
            for mm in range(0,si.mpol+1):
                for nn in range(-si.ntor,si.ntor+1):
                    if mm==0 and nn<0: continue

                    plasma_boundary['mm'][ii] = mm
                    plasma_boundary['nn'][ii] = nn

                    plasma_boundary['rbc'][0,ii] = si.rbc[si.mntor+nn,si.mmpol+mm]
                    plasma_boundary['zbs'][0,ii] = si.zbs[si.mntor+nn,si.mmpol+mm]

                    if si.istellsym==0:
                        plasma_boundary['rbs'][0,ii] = si.rbs[si.mntor+nn,si.mmpol+mm]
                        plasma_boundary['zbc'][0,ii] = si.zbc[si.mntor+nn,si.mmpol+mm]

                    ii = ii + 1
            
            if self.initial_guess is None:
                self.initial_guess = plasma_boundary
            else:
                self.initial_guess['rbc'][nvol-1] = plasma_boundary['rbc'][0,:]
                self.initial_guess['zbs'][nvol-1] = plasma_boundary['zbs'][0,:]

                if si.istellsym==0:
                    self.initial_guess['rbs'][nvol-1] = plasma_boundary['rbs'][0,:]
                    self.initial_guess['zbc'][nvol-1] = plasma_boundary['zbc'][0,:]


        # Store axis data
        self.axis = {}
        self.axis['rac'] = copy.copy(si.rac[0:si.ntor+1])
        self.axis['zas'] = copy.copy(si.zas[0:si.ntor+1])
        if si.istellsym==0:
            self.axis['ras'] = copy.copy(si.ras[0:si.ntor+1])
            self.axis['zac'] = copy.copy(si.zac[0:si.ntor+1])


        self.extension = filename[:-3]
        self.keep_all_files = keep_all_files
        self.files_to_delete = []

        # Create a surface object for the boundary:
        stellsym = bool(si.istellsym)
        print(f"In __init__, si.istellsym={si.istellsym} stellsym={stellsym}")
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

        # self.depends_on = ["boundary"]
        self.need_to_run_code = True
        self.counter = -1

        # Set profiles as None - these have to be defined in a script if the user
        # wish to optimize them
        self._volume_current_profile = None
        self._interface_current_profile = None
        self._pressure_profile = None
        self._iota_profile = None
        self._oita_profile = None
        self._mu_profile = None
        self._pflux_profile = None

        # Define normal field
        if not si.lfreebound:
            self.normal_field = None
        else:
            self.normal_field = NormalField()
            self.normal_field.init_from_spec(filename)

        # By default, all dofs owned by SPEC directly, as opposed to
        # dofs owned by the boundary surface object, are fixed.
        x0 = self.get_dofs()
        fixed = np.full(len(x0), True)
        names = ['phiedge', 'curtor']
        if si.lfreebound == 0:
            depends_on = [self._boundary]
        else:
            depends_on = [self.normal_field]

        super().__init__(x0=x0, fixed=fixed, names=names,
                         depends_on=depends_on,
                         external_dof_setter=Spec.set_dofs)

    @property
    def boundary(self):
        return self._boundary

    @property
    def pressure_profile(self):
        return self._pressure_profile

    @pressure_profile.setter
    def pressure_profile(self, pressure_profile):
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
        return self._volume_current_profile

    @volume_current_profile.setter
    def volume_current_profile(self, volume_current_profile):

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
        return self._interface_current_profile

    @interface_current_profile.setter
    def interface_current_profile(self, interface_current_profile):
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
        return self._iota_profile

    @iota_profile.setter
    def iota_profile(self, iota_profile):
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
        return self._oita_profile

    @oita_profile.setter
    def oita_profile(self, oita_profile):
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
        return self._mu_profile

    @mu_profile.setter
    def mu_profile(self, mu_profile):
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
        return self._pflux_profile

    @pflux_profile.setter
    def pflux_profile(self, pflux_profile):

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

    def set_profile(self, longname, lvol, value):
        """
        This function is used to set the pressure, currents, iota, oita,
        mu and/or pflux profiles.

        lvol: from 0 to Mvol-1
        """
        profile = self.__getattribute__(longname + "_profile")
        if profile is None:
            return

        # define nvol, mvol
        nvol = self.inputlist.nvol
        if self.inputlist.lfreebound == 0:
            mvol = nvol
        else:
            mvol = nvol + 1

        if profile.cumulative:
            old_value = profile.f(lvol)

            profile.set(lvol, value)
            for ivol in range(lvol + 1, mvol):
                profile.set(ivol, profile.f(ivol) - old_value + value)
        else:
            profile.set(lvol, value)

    def get_profile(self, longname, lvol):
        """
        This function is used to get the pressure, currents, iota, oita,
        mu and/or pflux profiles.

        lvol: from 0 to Mvol-1
        """

        profile = self.__getattribute__(longname + "_profile")
        if profile is None:
            return

        return profile.f(lvol)

    @boundary.setter
    def boundary(self, boundary):
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

    def run(self):
        """
        Run SPEC, if needed.
        """
        if not self.need_to_run_code:
            logger.info("run() called but no need to re-run SPEC.")
            return
        logger.info("Preparing to run SPEC.")
        self.counter += 1

        si = self.inputlist  # Shorthand

        # define nvol, mvol
        nvol = si.nvol
        if si.lfreebound == 0:
            mvol = nvol
        else:
            mvol = nvol + 1

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
        print("In run, si.istellsym=", si.istellsym, " stellsym=", stellsym)
        for m in range(mpol_capped + 1):
            for n in range(-ntor_capped, ntor_capped + 1):
                si.rbc[n + si.mntor, m +
                       si.mmpol] = boundary_RZFourier.get_rc(m, n)
                si.zbs[n + si.mntor, m +
                       si.mmpol] = boundary_RZFourier.get_zs(m, n)
                if not stellsym:
                    si.rbs[n + si.mntor, m +
                           si.mmpol] = boundary_RZFourier.get_rs(m, n)
                    si.zbc[n + si.mntor, m +
                           si.mmpol] = boundary_RZFourier.get_zc(m, n)

        # Set the coordinate axis using the lrzaxis=2 feature:
        si.lrzaxis = 2
        
        # Set axis from latest converged state
        mn = self.axis['rac'].size
        si.rac[0:mn] = self.axis['rac']
        si.zas[0:mn] = self.axis['zas']
        if si.istellsym==0:
            si.ras[0:mn] = self.axis['ras']
            si.zac[0:mn] = self.axis['zac']


        # Set initial guess
        mn = si.ntor+1 + si.mpol*(2*si.ntor+1)
        if not self.initial_guess is None:

            # Set all modes to zero
            spec.allglobal.mmrzrz[:] = 0
            spec.allglobal.nnrzrz[:] = 0
            spec.allglobal.allrzrz[:] = 0

            if si.lfreebound==1:
                si.rbc[:] = 0
                si.zbs[:] = 0

                if si.istellsym==0:
                    si.rbs[:] = 0
                    si.zbc[:] = 0

            # Populate initial guess of inner boundaries
            for imn, mm in enumerate(self.initial_guess['mm']):
                
                nn = self.initial_guess['nn'][imn]
                if mm>si.mpol or np.abs(nn)>si.ntor: continue

                if not (si.lfreebound==1 and si.nvol==1):
                    spec.allglobal.mmrzrz[imn] = mm
                    spec.allglobal.nnrzrz[imn] = self.initial_guess['nn'][imn]

                    spec.allglobal.allrzrz[0,0:nvol-1,imn] = self.initial_guess['rbc'][0:nvol-1,imn]
                    spec.allglobal.allrzrz[1,0:nvol-1,imn] = self.initial_guess['zbs'][0:nvol-1,imn]

                    if si.istellsym==0:
                        spec.allglobal.allrzrz[2,0:nvol-1,imn] = self.initial_guess['rbs'][0:nvol-1,imn]
                        spec.allglobal.allrzrz[3,0:nvol-1,imn] = self.initial_guess['zbc'][0:nvol-1,imn]

                if si.lfreebound==1:
                    x = self.initial_guess['rbc'][nvol-1,imn]
                    si.rbc[si.mntor+nn,si.mmpol+mm] = x
                    si.zbs[si.mntor+nn,si.mmpol+mm] = self.initial_guess['zbs'][nvol-1,imn]

                    if si.istellsym==0:
                        si.rbs[si.mntor+nn,si.mmpol+mm] = self.initial_guess['rbs'][nvol-1,imn]
                        si.zbc[si.mntor+nn,si.mmpol+mm] = self.initial_guess['zbc'][nvol-1,imn]

        # Set profiles from dofs
        if self.pressure_profile is not None:
            si.pressure[0:si.nvol] = self.pressure_profile.get(
                np.arange(0, si.nvol))
            if si.lfreebound == 1:
                si.pressure[si.nvol] = 0

        if self.volume_current_profile is not None:
            # Volume current is a cumulative profile; special care is required
            # when a dofs is changed in order to keep fixed dofs fixed!
            old_ivolume = copy.copy(si.ivolume)
            for lvol in range(0, mvol):
                if self.volume_current_profile.is_fixed(lvol):
                    if lvol != 0:
                        si.ivolume[lvol] = si.ivolume[lvol] - \
                            old_ivolume[lvol - 1] + si.ivolume[lvol - 1]
                        self.set_profile(
                            'volume_current', lvol=lvol, value=si.ivolume[lvol])
                else:
                    si.ivolume[lvol] = self.get_profile('volume_current', lvol)

            if si.lfreebound == 1:
                si.ivolume[si.nvol] = si.ivolume[si.nvol - 1]
                self.volume_current_profile.set(
                    key=mvol - 1, new_val=si.ivolume[nvol - 1])

        if self.interface_current_profile is not None:
            si.isurf[0:nvol -
                     1] = self.interface_current_profile.get(np.arange(0, nvol))
            if si.lfreebound == 1:
                si.ivolume[mvol - 1] = si.ivolume[nvol - 1]

        # Update total plasma toroidal current in case of freeboundary
        # calculation
        if ((self.volume_current_profile is not None) or
            (self.interface_current_profile is not None)) and \
                si.lfreebound == 1:
            si.curtor = si.ivolume[nvol - 1] + np.sum(si.isurf)

        if self.iota_profile is not None:
            si.iota[0:nvol] = self.iota_profile.get(np.arange(0, nvol))

        if self.oita_profile is not None:
            si.oita[0:nvol] = self.oita_profile.get(np.arange(0, nvol))

        if self.mu_profile is not None:
            si.mu[0:nvol - 1] = self.mu_profile.get(np.arange(0, nvol))
            if si.lfreebound == 1:
                si.mu[mvol] = 0

        if self.pflux_profile is not None:
            # Pflux is a cumulative profile; special care is required
            # when a dofs is changed in order to keep fixed dofs fixed!
            old_pflux = copy.copy(si.pflux)
            for lvol in range(0, mvol):
                if self.pflux_profile.is_fixed(lvol):
                    if lvol != 0:
                        si.pflux[lvol] = si.pflux[lvol] - \
                            old_pflux[lvol - 1] + si.pflux[lvol - 1]
                        self.pflux_profile.set(
                            key=lvol, new_val=si.pflux[lvol])
                else:
                    si.pflux[lvol] = self.pflux_profile.get(lvol)

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
        # Barrier so workers do not try to read the .h5 file before it is
        # finished:
        self.mpi.comm_groups.Barrier()

        try:
            self.results = py_spec.SPECout(filename + '.sp.h5')
        except BaseException:
            if self.verbose:
                traceback.print_exc()
            raise ObjectiveFailure(
                "Unable to read results following SPEC execution")

        logger.info("Successfully loaded SPEC results.")
        self.need_to_run_code = False

        try:
            # Save geometry as initial guess for next iterations
            if self.results.output.ForceErr < 1e-12 and self.results.output.Mvol>1:
                initial_guess = {}
                initial_guess['rbc'] = self.results.output.Rbc[1:mvol+1,:]
                initial_guess['zbs'] = self.results.output.Zbs[1:mvol+1,:]
                initial_guess['mm'] = self.results.output.im
                initial_guess['nn'] = (self.results.output.in_ / si.nfp).astype('int')

                axis = {}
                axis['rac'] = self.results.output.Rbc[0,0:si.ntor+1]
                axis['zas'] = self.results.output.Zbs[0,0:si.ntor+1]

                if si.istellsym == 0:
                    initial_guess['rbs'] = self.results.output.Rbs[1:mvol+1,:]
                    initial_guess['zbc'] = self.results.output.Zbc[1:mvol+1,:]

                    axis['ras'] = self.results.output.Rbs[0,0:si.ntor+1]
                    axis['zac'] = self.results.output.Zbc[0,0:si.ntor+1]

                self.initial_guess = copy.copy(initial_guess)
                self.axis = copy.copy(axis)

                self.inputlist.linitialize = 0
        except:
            logger.info("Failed to read initial guess.")


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
