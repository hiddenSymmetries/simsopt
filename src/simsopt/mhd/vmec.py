# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides a class that handles the VMEC equilibrium code.
"""

import logging
import os.path
from typing import Union

import numpy as np
from scipy.io import netcdf
from ..util.dev import SimsoptRequires

logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI
except ImportError as e:
    MPI = None 
    logger.warning(str(e))

# vmec_found = True
try:
    import vmec
except ImportError as e:
    # vmec_found = False
    vmec = None
    logger.warning(str(e))

from .._core.optimizable import Optimizable
from .._core.util import Struct, ObjectiveFailure
from ..geo.surfacerzfourier import SurfaceRZFourier

if MPI is not None:
    from ..util.mpi import MpiPartition
else:
    MpiPartition = None


# Flags used by runvmec():
restart_flag = 1
readin_flag = 2
timestep_flag = 4
output_flag = 8
cleanup_flag = 16
reset_jacdt_flag = 32

# Documentation of flags for runvmec() from the VMEC source code:
#
#value flag-name         calls routines to...
#----- ---------         ---------------------
#  1   restart_flag      reset internal run-control parameters
#                        (for example, if jacobian was bad, to try a smaller 
#                        time-step)
#  2   readin_flag       read in data from input_file and initialize parameters
#                        or arrays which do not dependent on radial grid size
#                        allocate internal grid-dependent arrays used by vmec;
#                        initialize internal grid-dependent vmec profiles (xc,
#                        iota, etc);
#                        setup loop for radial multi-grid meshes or, if
#                        ns_index = ictrl_array(4) is > 0, use radial grid
#                        points specified by ns_array[ns_index]
#  4   timestep_flag     iterate vmec either by "niter" time steps or until ftol
#                        satisfied, whichever comes first.
#                        If numsteps (see below) > 0, vmec will return
#                        to caller after numsteps, rather than niter, steps.
#  8   output_flag       write out output files (wout, jxbout)
# 16   cleanup_flag      cleanup (deallocate arrays) - this terminates present
#                        run of the sequence
#                        This flag will be ignored if the run might be continued.
#                        For example, if ier_flag (see below) returns the value
#                        more_iter_flag, the cleanup code will be skipped even if
#                        cleanup_flag is set, so that the run could be continued
#                        on the next call to runvmec.
# 32   reset_jacdt_flag  Resets ijacobian flag and time step to delt0
#                        thus, setting ictrl_flag = 1+2+4+8+16 will perform ALL
#                        the tasks thru cleanup_flag in addition,
#                        if ns_index = 0 and numsteps = 0 (see below), vmec will
#                        control its own run history


# Temporarily commenting out the decorator till __instancecheck__ method is made working
#@SimsoptRequires(MPI is not None, "mpi4py needs to be installed for running VMEC")
class Vmec(Optimizable):
    """
    This class represents the VMEC equilibrium code.

    The input parameters to VMEC are all accessible as attributes of
    the ``indata`` attribute. For example, if ``vmec`` is an instance
    of ``Vmec``, then you can read or write the input resolution
    parameters using ``vmec.indata.mpol``, ``vmec.indata.ntor``,
    ``vmec.indata.ns_array``, etc. However, the boundary surface is
    different: ``rbc``, ``rbs``, ``zbc``, and ``zbs`` from the
    ``indata`` attribute are always ignored, and these arrays are
    instead taken from the simsopt surface object associated to the
    ``boundary`` attribute. If ``boundary`` is a surface based on some
    other representation than VMEC's Fourier representation, the
    surface will automatically be converted to VMEC's representation
    (:obj:`~simsopt.geo.surfacerzfourier.SurfaceRZFourier`) before
    each run of VMEC. You can replace ``boundary`` with a new surface
    object, of any type that implements the conversion function
    ``to_RZFourier()`.

    VMEC is run either when the :meth:`run()` function is called, or when
    any of the output functions like :meth:`aspect()` or :meth:`iota_axis()`
    are called.

    A caching mechanism is implemented, using the attribute
    ``need_to_run_code``. Whenever VMEC is run, this attribute is set
    to ``False``. Subsequent calls to :meth:`run()` or output
    functions like :meth:`aspect()` will not actually run VMEC again,
    until ``need_to_run_code`` is changed to ``True``. The attribute
    ``need_to_run_code`` is automatically set to ``True`` whenever
    :meth:`set_dofs()` is called. However, ``need_to_run_code`` is not
    automatically set to ``True`` when entries of ``indata`` are
    modified, or when ``boundary`` is modified.

    Once VMEC has run at least once, all of the quantities in the
    ``wout`` output file are available as attributes of the ``wout``
    attribute.  For example, if ``vmec`` is an instance of ``Vmec``,
    then the flux surface shapes can be obtained from
    ``vmec.wout.rmnc`` and ``vmec.wout.zmns``.

    Since the underlying fortran implementation of VMEC uses global
    module variables, it is not possible to have more than one python
    Vmec object with different parameters; changing the parameters of
    one would change the parameters of the other.

    An instance of this class owns just a few optimizable degrees of
    freedom, particularly ``phiedge`` and ``curtor``. The optimizable
    degrees of freedom associated with the boundary surface are owned
    by that surface object.

    Args:
        filename: Name of a VMEC input file to use for loading the 
          initial parameters. If ``None``, default parameters will be used.
        mpi: A :obj:`simsopt.util.mpi.MpiPartition` instance, from which 
          the worker groups will be used for VMEC calculations. If ``None``,
          each MPI process will run VMEC independently.

    Attributes:
        iter: Number of times VMEC has run.
        s_full_grid: The "full" grid in the radial coordinate s (normalized
          toroidal flux), including points at s=0 and s=1. Used for the output
          arrays and ``zmns``.
        s_half_grid: The "half" grid in the radial coordinate s, used for
          ``bmnc``, ``lmns``, and other output arrays. In contrast to
          wout files, this array has only ns-1 entries, so there is no
          leading 0.
        ds: The spacing between grid points for the radial coordinate s.
    """

    def __init__(self,
                 filename: Union[str, None] = None,
                 mpi: Union[MpiPartition, None] = None):
        if MPI is None:
            raise RuntimeError("mpi4py needs to be installed for running VMEC")
        if vmec is None:
            raise RuntimeError(
                "Running VMEC from simsopt requires VMEC python extension. "
                "Install the VMEC python extension from "
                "https://https://github.com/hiddenSymmetries/VMEC2000")

        if filename is None:
            # Read default input file, which should be in the same
            # directory as this file:
            filename = os.path.join(os.path.dirname(__file__), 'input.default')
            logger.info("Initializing a VMEC object from defaults in "
                        + filename)
        else:
            logger.info("Initializing a VMEC object from file: " + filename)
        self.input_file = filename

        # Get MPI communicator:
        if mpi is None:
            self.mpi = MpiPartition(ngroups=1)
        else:
            self.mpi = mpi
        comm = self.mpi.comm_groups
        self.fcomm = comm.py2f()

        self.ictrl = np.zeros(5, dtype=np.int32)
        self.iter = 0
        self.wout = Struct()

        self.ictrl[0] = restart_flag + readin_flag
        self.ictrl[1] = 0  # ierr
        self.ictrl[2] = 0  # numsteps
        self.ictrl[3] = 0  # ns_index
        self.ictrl[4] = 0  # iseq
        verbose = True
        reset_file = ''
        logger.info('About to call runvmec to readin')
        vmec.runvmec(self.ictrl, filename, verbose, self.fcomm, reset_file)
        ierr = self.ictrl[1]
        logger.info('Done with runvmec. ierr={}. Calling cleanup next.'.format(ierr))
        # Deallocate arrays allocated by VMEC's fixaray():
        vmec.cleanup(False)
        if ierr != 0:
            raise RuntimeError("Failed to initialize VMEC from input file {}. "
                               "error code {}".format(filename, ierr))

        objstr = " for Vmec " + str(hex(id(self)))

        # Create an attribute for each VMEC input parameter in VMEC's fortran
        # modules,
        self.indata = vmec.vmec_input  # Shorthand
        vi = vmec.vmec_input  # Shorthand
        # A vmec object has mpol and ntor attributes independent of
        # the boundary. The boundary surface object is initialized
        # with mpol and ntor values that match those of the vmec
        # object, but the mpol/ntor values of either the vmec object
        # or the boundary surface object can be changed independently
        # by the user.
        self.boundary = SurfaceRZFourier(nfp=vi.nfp,
                                         stellsym=not vi.lasym,
                                         mpol=vi.mpol,
                                         ntor=vi.ntor)
        self.free_boundary = bool(vi.lfreeb)

        # Transfer boundary shape data from fortran to the ParameterArray:
        for m in range(vi.mpol + 1):
            for n in range(-vi.ntor, vi.ntor + 1):
                self.boundary.rc[m, n + vi.ntor] = vi.rbc[101 + n, m]
                self.boundary.zs[m, n + vi.ntor] = vi.zbs[101 + n, m]
        # Handle a few variables that are not Parameters:
        self.depends_on = ["boundary"]
        self.need_to_run_code = True

        self.fixed = np.full(len(self.get_dofs()), True)
        self.names = ['delt', 'tcon0', 'phiedge', 'curtor', 'gamma']

    def get_dofs(self):
        return np.array([self.indata.delt, self.indata.tcon0,
                         self.indata.phiedge, self.indata.curtor,
                         self.indata.gamma])

    def set_dofs(self, x):
        self.need_to_run_code = True
        self.indata.delt = x[0]
        self.indata.tcon0 = x[1]
        self.indata.phiedge = x[2]
        self.indata.curtor = x[3]
        self.indata.gamma = x[4]

    def run(self):
        """
        Run VMEC, if ``need_to_run_code`` is ``True``.
        """
        if not self.need_to_run_code:
            logger.info("run() called but no need to re-run VMEC.")
            return
        logger.info("Preparing to run VMEC.")
        # Transfer values from Parameters to VMEC's fortran modules:
        vi = vmec.vmec_input  # Shorthand
        # Convert boundary to RZFourier if needed:
        boundary_RZFourier = self.boundary.to_RZFourier()
        # VMEC does not allow mpol or ntor above 101:
        if vi.mpol > 101:
            raise ValueError("VMEC does not allow mpol > 101")
        if vi.ntor > 101:
            raise ValueError("VMEC does not allow ntor > 101")
        vi.rbc[:, :] = 0
        vi.zbs[:, :] = 0
        mpol_capped = np.min([boundary_RZFourier.mpol, 101])
        ntor_capped = np.min([boundary_RZFourier.ntor, 101])
        # Transfer boundary shape data from the surface object to VMEC:
        for m in range(mpol_capped + 1):
            for n in range(-ntor_capped, ntor_capped + 1):
                vi.rbc[101 + n, m] = boundary_RZFourier.get_rc(m, n)
                vi.zbs[101 + n, m] = boundary_RZFourier.get_zs(m, n)

        # Set axis shape to something that is obviously wrong (R=0) to
        # trigger vmec's internal guess_axis.f to run. Otherwise the
        # initial axis shape for run N will be the final axis shape
        # from run N-1, which makes VMEC results depend slightly on
        # the history of previous evaluations, confusing the finite
        # differencing.
        vi.raxis_cc[:] = 0
        vi.raxis_cs[:] = 0
        vi.zaxis_cc[:] = 0
        vi.zaxis_cs[:] = 0

        self.iter += 1
        input_file = self.input_file + '_{:03d}_{:06d}'.format(
            self.mpi.group, self.iter)
        self.output_file = os.path.join(
            os.getcwd(),
            os.path.basename(input_file).replace('input.', 'wout_') + '.nc')
        mercier_file = os.path.join(
            os.getcwd(),
            os.path.basename(input_file).replace('input.', 'mercier.'))
        jxbout_file = os.path.join(
            os.getcwd(),
            os.path.basename(input_file).replace('input.', 'jxbout_') + '.nc')

        # I should write an input file here.
        logger.info("Calling VMEC reinit().")
        vmec.reinit()

        logger.info("Calling runvmec().")
        self.ictrl[0] = restart_flag + reset_jacdt_flag \
            + timestep_flag + output_flag
        self.ictrl[1] = 0  # ierr
        self.ictrl[2] = 0  # numsteps
        self.ictrl[3] = 0  # ns_index
        self.ictrl[4] = 0  # iseq
        verbose = True
        reset_file = ''
        vmec.runvmec(self.ictrl, input_file, verbose, self.fcomm, reset_file)
        ierr = self.ictrl[1]

        # Deallocate arrays, even if vmec did not converge:
        logger.info("Calling VMEC cleanup().")
        vmec.cleanup(True)

        # See VMEC2000/Sources/General/vmec_params.f for ierr codes.
        # 11 = successful_term_flag.
        # Error codes that are expected to occur due to lack of
        # convergence cause ObjectiveFailure, which the optimizer
        # handles gracefully by treating the point as bad. But the
        # user/developer should know if an error codes arises that
        # should logically never occur, so these codes raise a
        # different exception.
        if ierr in [0, 5]:
            raise RuntimeError(f"runvmec returned an error code that should " \
                               "never occur: ierr={ierr}")
        if ierr != 11:
            raise ObjectiveFailure(f"VMEC did not converge. ierr={ierr}")

        logger.info("VMEC run complete. Now loading output.")
        self.load_wout()
        logger.info("Done loading VMEC output.")

        # Delete some files produced by VMEC that we never care
        # about. For some reason the os.remove statements give a 'file
        # not found' error in the CI, hence the try-except blocks.
        try:
            os.remove(mercier_file)
        except:
            pass
        try:
            os.remove(jxbout_file)
        except:
            pass
        try:
            os.remove("fort.9")
        except:
            pass

        self.need_to_run_code = False

    def load_wout(self):
        """
        Read in the most recent ``wout`` file created, and store all the
        data in a ``wout`` attribute of this Vmec object.
        """
        ierr = 0
        logger.info("Attempting to read file " + self.output_file)
        # vmec.read_wout_mod.read_wout_file(self.output_file, ierr)
        #if ierr == 0:
        #    logger.info('Successufully load VMEC results from ' + \
        #                self.output_file)
        # else:
        #    print('Load VMEC results from {:} failed!'.format(
        #        self.output_file))
        #wout = vmec.read_wout_mod
        #print('xm:', wout.xm)
        #print('xn:', wout.xn)
        #print('xm_nyq:', wout.xm_nyq)
        #print('xn_nyq:', wout.xn_nyq)
        #print('type(xm):', type(wout.xm), ' type(xn):', type(wout.xn), ' type(xm_nyq):', type(wout.xm_nyq), ' type(xn_nyq):', type(wout.xn_nyq))
        #print('ierr:', ierr)
        #print('mnmax:', wout.mnmax, ' len(xm):', len(wout.xm), ' len(xn):', len(wout.xn))
        #print('mnmax_nyq:', wout.mnmax_nyq, ' len(xm_nyq):', len(wout.xm_nyq), ' len(xn_nyq):', len(wout.xn_nyq))
        #assert len(wout.xm) == wout.mnmax
        #assert len(wout.xn) == wout.mnmax
        #assert len(wout.xm_nyq) == wout.mnmax_nyq
        #assert len(wout.xn_nyq) == wout.mnmax_nyq

        f = netcdf.netcdf_file(self.output_file, mmap=False)
        for key, val in f.variables.items():
            # 2D arrays need to be transposed.
            val2 = val[()]  # Convert to numpy array
            val3 = val2.T if len(val2.shape) == 2 else val2
            self.wout.__setattr__(key, val3)

        #self.wout.ier_flag = f.variables['ier_flag'][()]
        if self.wout.ier_flag != 0:
            logger.info("VMEC did not succeed!")
            raise ObjectiveFailure("VMEC did not succeed")

        # Shorthand for a long variable name:
        self.wout.lasym = f.variables['lasym__logical__'][()]
        self.wout.volume = self.wout.volume_p

        #self.wout.ier_flag = f.variables['ier_flag'][()]
        #if self.wout.ier_flag != 0:
        #    logger.info("VMEC did not succeed!")
        #    raise ObjectiveFailure("VMEC did not succeed")
        #self.wout.nfp = f.variables['nfp'][()]
        #self.wout.lasym = f.variables['lasym__logical__'][()]
        #self.wout.ns = f.variables['ns'][()]
        #self.wout.mnmax = f.variables['mnmax'][()]
        #self.wout.mnmax_nyq = f.variables['mnmax_nyq'][()]
        #self.wout.xm = f.variables['xm'][()]
        #self.wout.xn = f.variables['xn'][()]
        #self.wout.xm_nyq = f.variables['xm_nyq'][()]
        #self.wout.xn_nyq = f.variables['xn_nyq'][()]
        #self.wout.mpol = f.variables['mpol'][()]
        #self.wout.ntor = f.variables['ntor'][()]
        #self.wout.bmnc = f.variables['bmnc'][()].transpose()
        #self.wout.rmnc = f.variables['rmnc'][()].transpose()
        #self.wout.zmns = f.variables['zmns'][()].transpose()
        #self.wout.lmns = f.variables['lmns'][()].transpose()
        #self.wout.bsubumnc = f.variables['bsubumnc'][()].transpose()
        #self.wout.bsubvmnc = f.variables['bsubvmnc'][()].transpose()
        #self.wout.iotas = f.variables['iotas'][()]
        #self.wout.iotaf = f.variables['iotaf'][()]
        #self.wout.aspect = f.variables['aspect'][()]
        #self.wout.volume = f.variables['volume_p'][()]
        f.close()

        self.s_full_grid = np.linspace(0, 1, self.wout.ns)
        self.ds = self.s_full_grid[1] - self.s_full_grid[0]
        self.s_half_grid = self.s_full_grid[1:] - 0.5 * self.ds

        return ierr

    def aspect(self):
        """
        Return the plasma aspect ratio.
        """
        self.run()
        return self.wout.aspect

    def volume(self):
        """
        Return the volume inside the VMEC last closed flux surface.
        """
        self.run()
        return self.wout.volume

    def iota_axis(self):
        """
        Return the rotational transform on axis
        """
        self.run()
        return self.wout.iotaf[0]

    def iota_edge(self):
        """
        Return the rotational transform at the boundary
        """
        self.run()
        return self.wout.iotaf[-1]

    def mean_iota(self):
        """
        Return the mean rotational transform. The average is taken over
        the normalized toroidal flux s.
        """
        self.run()
        return np.mean(self.wout.iotas[1:])

    def mean_shear(self):
        """
        Return an average magnetic shear, d(iota)/ds, where s is the
        normalized toroidal flux. This is computed by fitting the
        rotational transform to a linear (plus constant) function in
        s. The slope of this fit function is returned.
        """
        self.run()

        # Fit a linear polynomial:
        poly = np.polynomial.Polynomial.fit(self.s_half_grid,
                                            self.wout.iotas[1:], deg=1)
        # Return the slope:
        return poly.deriv()(0)

    def get_max_mn(self):
        """
        Look through the rbc and zbs data in fortran to determine the
        largest m and n for which rbc or zbs is nonzero.
        """
        max_m = 0
        max_n = 0
        for m in range(1, 101):
            for n in range(1, 101):
                if np.abs(vmec.vmec_input.rbc[101 + n, m]) > 0 \
                        or np.abs(vmec.vmec_input.zbs[101 + n, m]) > 0 \
                        or np.abs(vmec.vmec_input.rbs[101 + n, m]) > 0 \
                        or np.abs(vmec.vmec_input.zbc[101 + n, m]) > 0 \
                        or np.abs(vmec.vmec_input.rbc[101 - n, m]) > 0 \
                        or np.abs(vmec.vmec_input.zbs[101 - n, m]) > 0 \
                        or np.abs(vmec.vmec_input.rbs[101 - n, m]) > 0 \
                        or np.abs(vmec.vmec_input.zbc[101 - n, m]) > 0:
                    max_m = np.max((max_m, m))
                    max_n = np.max((max_n, n))
        # It may happen that mpol or ntor exceed the max_m or max_n
        # according to rbc/zbs. In this case, go with the larger
        # value.
        max_m = np.max((max_m, vmec.vmec_input.mpol))
        max_n = np.max((max_n, vmec.vmec_input.ntor))
        return (max_m, max_n)

    def __repr__(self):
        """
        Print the object in an informative way.
        """
        return "Vmec instance " + str(hex(id(self))) \
            + " (nfp=" + str(self.indata.nfp) \
            + " mpol=" + str(self.indata.mpol) \
            + " ntor=" + str(self.indata.ntor) + ")"

    def vacuum_well(self):
        """
        Compute a single number W that summarizes the vacuum magnetic well,
        given by the formula

        W = (dV/ds(s=0) - dV/ds(s=1)) / (dV/ds(s=0)

        where dVds is the derivative of the flux surface volume with
        respect to the radial coordinate s. Positive values of W are
        favorable for stability to interchange modes. This formula for
        W is motivated by the fact that

        d^2 V / d s^2 < 0

        is favorable for stability. Integrating over s from 0 to 1
        and normalizing gives the above formula for W. Notice that W
        is dimensionless, and it scales as the square of the minor
        radius. To compute dV/ds, we use

        dV/ds = 4 * pi**2 * abs(sqrt(g)_{0,0})

        where sqrt(g) is the Jacobian of (s, theta, phi) coordinates,
        computed by VMEC in the gmnc array, and _{0,0} indicates the
        m=n=0 Fourier component. Since gmnc is reported by VMEC on the
        half mesh, we extrapolate by half of a radial grid point to s
        = 0 and 1.
        """

        self.run()

        # gmnc is on the half mesh, so drop the 0th radial entry:
        dVds = 4 * np.pi * np.pi * np.abs(self.wout.gmnc[0, 1:])

        # To get from the half grid to s=0 and s=1, we must
        # extrapolate by 1/2 of a radial grid point:
        dVds_s0 = 1.5 * dVds[0] - 0.5 * dVds[1]
        dVds_s1 = 1.5 * dVds[-1] - 0.5 * dVds[-2]

        well = (dVds_s0 - dVds_s1) / dVds_s0
        return well
