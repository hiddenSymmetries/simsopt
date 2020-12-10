# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides a class that handles the SPEC equilibrium code.
"""

import logging
import os.path
import numpy as np
import py_spec

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
        
        # For now, set the coordinate axis equal to the m=0 modes of the boundary:
        self.nml['physicslist']['rac'] = rc[0, ntor:].tolist()
        self.nml['physicslist']['zas'] = zs[0, ntor:].tolist()

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
