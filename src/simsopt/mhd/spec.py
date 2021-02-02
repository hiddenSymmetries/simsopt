# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides a class that handles the SPEC equilibrium code.
"""

import logging
import os.path
import numpy as np

py_spec_found = True
try:
    import py_spec
except:
    py_spec_found = False

pyoculus_found = True
try:
    import pyoculus
except:
    pyoculus_found = False

from simsopt.core import Optimizable, SurfaceRZFourier

logger = logging.getLogger(__name__)

def nested_lists_to_array(ll):
    """
    Convert a ragged list of lists to a 2D numpy array.  Any entries
    that are None are replaced by 0.

    This function is applied to the RBC and ZBS arrays in the input
    namelist.
    """
    mdim = len(ll)
    ndim = np.max([len(x) for x in ll])
    arr = np.zeros((mdim, ndim))
    for jm, l in enumerate(ll):
        for jn, x in enumerate(l):
            if x is not None:
                arr[jm, jn] = x
    return arr


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
    """
    def __init__(self, filename=None, exe='xspec'):
        """
        Constructor

        filename: SPEC input file to use to initialize parameters.
        exe: Path to the xspec executable.
        """
        if filename is None:
            # Read default input file, which should be in the same
            # directory as this file:
            filename = os.path.join(os.path.dirname(__file__), 'defaults.sp')
            logger.info("Initializing a SPEC object from defaults in " \
                            + filename)
        else:
            logger.info("Initializing a SPEC object from file: " + filename)

        self.exe = exe
        self.nml = py_spec.SPECNamelist(filename)

        # Transfer the boundary shape from the namelist to a Surface object:
        nfp = self.nml['physicslist']['nfp']
        stelsym = bool(self.nml['physicslist']['istellsym'])
        """
        mpol = self.nml['physicslist']['mpol']
        ntor = self.nml['physicslist']['ntor']
        for m in range(mpol + 1):
            for n in range(-ntor, ntor + 1):
                self.boundary.set_rc(m, n) = self.nml['physicslist']['rbc'][m][n + ntor]
                self.boundary.set_zs(m, n) = self.nml['physicslist']['zbs'][m][n + ntor]
        """
        # We can assume rbc and zbs are specified in the namelist.
        # f90nml returns rbc and zbs as a list of lists where the
        # inner lists do not necessarily all have the same
        # dimension. Hence we need to be careful when converting to
        # numpy arrays.
        rc = nested_lists_to_array(self.nml['physicslist']['rbc'])
        zs = nested_lists_to_array(self.nml['physicslist']['zbs'])

        rbc_first_n = self.nml['physicslist'].start_index['rbc'][0]
        rbc_last_n = rbc_first_n + rc.shape[1] - 1
        zbs_first_n = self.nml['physicslist'].start_index['zbs'][0]
        zbs_last_n = zbs_first_n + rc.shape[1] - 1
        ntor_boundary = np.max(np.abs(np.array([rbc_first_n, rbc_last_n, zbs_first_n, zbs_last_n], dtype='i')))

        rbc_first_m = self.nml['physicslist'].start_index['rbc'][1]
        rbc_last_m = rbc_first_m + rc.shape[0] - 1
        zbs_first_m = self.nml['physicslist'].start_index['zbs'][1]
        zbs_last_m = zbs_first_m + rc.shape[0] - 1
        mpol_boundary = np.max((rbc_last_m, zbs_last_m))
        logger.debug('Input file has ntor_boundary={} mpol_boundary={}'.format(ntor_boundary, mpol_boundary))
        self.boundary = SurfaceRZFourier(nfp=nfp, stelsym=stelsym,
                                         mpol=mpol_boundary, ntor=ntor_boundary)
        
        # Transfer boundary shape data from the namelist to the surface object:
        for jm in range(rc.shape[0]):
            m = jm + self.nml['physicslist'].start_index['rbc'][1]
            for jn in range(rc.shape[1]):
                n = jn + self.nml['physicslist'].start_index['rbc'][0]
                self.boundary.set_rc(m, n, rc[jm, jn])
                
        for jm in range(zs.shape[0]):
            m = jm + self.nml['physicslist'].start_index['zbs'][1]
            for jn in range(zs.shape[1]):
                n = jn + self.nml['physicslist'].start_index['zbs'][0]
                self.boundary.set_zs(m, n, zs[jm, jn])

        # Done transferring boundary shape.

        # py_spec does not allow mpol / ntor to be changed if rbc or
        # zbs are not the expected dimensions. These next few lines
        # are a hack to avoid this issue, allowing us to have tests
        # that involve changing mpol/ntor.
        del self.nml['physicslist']['rbc']
        del self.nml['physicslist']['zbs']
        self.nml._rectify_namelist()
        
        self.depends_on = ["boundary"]
        self.need_to_run_code = True
        self.counter = 0

        # By default, all dofs owned by SPEC directly, as opposed to
        # dofs owned by the boundary surface object, are fixed.
        self.fixed = np.full(len(self.get_dofs()), True)
        self.names = ['phiedge', 'curtor']
        
    def get_dofs(self):
        return np.array([self.nml['physicslist']['phiedge'],
                         self.nml['physicslist']['curtor']])

    def set_dofs(self, x):
        self.need_to_run_code = True
        self.nml['physicslist']['phiedge'] = x[0]
        self.nml['physicslist']['curtor'] = x[1]

    def update_resolution(self, mpol, ntor):
        """ For convenience, to save ".nml" """
        logger.info('Calling update_resolution(mpol={}, ntor={})'.format(mpol, ntor))
        self.nml.update_resolution(mpol, ntor)
        
    def run(self):
        """
        Run SPEC, if needed.
        """
        if not self.need_to_run_code:
            logger.info("run() called but no need to re-run SPEC.")
            return
        logger.info("Preparing to run SPEC.")

        # nfp must be consistent between the surface and SPEC. The surface's value trumps.
        self.nml['physicslist']['nfp'] = self.boundary.nfp

        # Convert boundary to RZFourier if needed:
        boundary_RZFourier = self.boundary.to_RZFourier()

        mpol = self.nml['physicslist']['mpol']
        ntor = self.nml['physicslist']['ntor']
        mpol_b = boundary_RZFourier.mpol
        ntor_b = boundary_RZFourier.ntor
        rc = np.zeros((mpol + 1, 2 * ntor + 1))
        zs = np.zeros((mpol + 1, 2 * ntor + 1))
        mpol_loop = np.min((mpol, mpol_b))
        ntor_loop = np.min((ntor, ntor_b))
        # Transfer boundary shape data from the surface object to SPEC:
        rc[:mpol_loop + 1, ntor - ntor_loop:ntor + ntor_loop + 1] \
            = boundary_RZFourier.rc[:mpol_loop + 1, ntor_b - ntor_loop:ntor_b + ntor_loop + 1]
        zs[:mpol_loop + 1, ntor - ntor_loop:ntor + ntor_loop + 1] \
            = boundary_RZFourier.zs[:mpol_loop + 1, ntor_b - ntor_loop:ntor_b + ntor_loop + 1]

        self.nml['physicslist']['rbc'] = rc.tolist()
        self.nml['physicslist']['zbs'] = zs.tolist()
        self.nml['physicslist'].start_index['rbc'] = [-ntor, 0]
        self.nml['physicslist'].start_index['zbs'] = [-ntor, 0]
        
        ## For now, set the coordinate axis equal to the m=0 modes of the boundary:
        #self.nml['physicslist']['rac'] = rc[0, ntor:].tolist()
        #self.nml['physicslist']['zas'] = zs[0, ntor:].tolist()

        # Set the coordinate axis using the lrzaxis=2 feature:
        self.nml['numericlist']['lrzaxis'] = 2
        # lrzaxis=2 only seems to work if the axis is not already set
        self.nml['physicslist']['rac'] = []
        self.nml['physicslist']['zas'] = []

        filename = 'spec{:05}.sp'.format(self.counter)
        logger.info("Running SPEC using filename " + filename)
        self.results = self.nml.run(spec_command=self.exe,
                                    filename=filename, force=True)
        logger.info("SPEC run complete.")
        self.counter += 1
        self.need_to_run_code = False

    def volume(self):
        """
        Return the volume inside the VMEC last closed flux surface.
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
    """
    def __init__(self, spec, pp, qq, vol=1, theta=0, s_guess=None, s_min=-1.0, s_max=1.0, rtol=1e-9):
        """
        spec: a Spec object
        pp, qq: Numerator and denominator for the resonant iota = pp / qq
        vol: Index of the Spec volume to consider
        theta: Spec's theta coordinate at the periodic field line
        s_guess: Guess for the value of Spec's s coordinate at the periodic field line
        s_min, s_max: bounds on s for the search
        rtol: the relative tolerance of the integrator
        """
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

    def J(self):
        """
        Run Spec if needed, find the periodic field line, and return the residue
        """
        if self.need_to_run_code:
            self.spec.run()
            specb = pyoculus.problems.SPECBfield(self.spec.results, self.vol)
            fp = pyoculus.solvers.FixedPoint(specb, {'theta':self.theta}, integrator_params={'rtol':self.rtol})
            self.fixed_point = fp.compute(self.s_guess, sbegin=self.s_min, send=self.s_max, pp=self.pp, qq=self.qq)
            self.need_to_run_code = False

        return self.fixed_point.GreenesResidue
    
    def get_dofs(self):
        return np.array([])

    def set_dofs(self, x):
        self.need_to_run_code = True
