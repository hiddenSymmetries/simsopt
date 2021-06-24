import unittest
import os
import numpy as np
try:
    from mpi4py import MPI
except:
    MPI = None
try:
    import vmec
    vmec_found = True
except:
    vmec_found = False
from . import TEST_DIR

run_modes = {'all': 63,
             'input': 35,  # STELLOPT uses 35; V3FIT uses 7                                                                  
             'output': 8,
             'main': 45}   # STELLOPT uses 61; V3FIT uses 45                                                                 

success_codes = [0, 11]
reset_file = ''

#@unittest.skipIf(not vmec_found, "Valid Python interface to VMEC not found")


@unittest.skip("These tests are obsolete due to the refactoring of vmec.py and vmec_core.py")
class F90wrapVmecTests(unittest.TestCase):
    def setUp(self):
        """
        Set up the test fixture. This subroutine is automatically run
        by python's unittest module before every test.
        """
        self.fcomm = MPI.COMM_WORLD.py2f()
        rank = MPI.COMM_WORLD.Get_rank()
        self.verbose = (rank == 0)
        self.filename = os.path.join(TEST_DIR, 'input.li383_low_res')

        self.ictrl = np.zeros(5, dtype=np.int32)

        ier = 0
        numsteps = 0
        ns_index = -1
        iseq = rank
        self.ictrl[0] = 0
        self.ictrl[1] = ier
        self.ictrl[2] = numsteps
        self.ictrl[3] = ns_index
        self.ictrl[4] = iseq

        # Change the working directory to match the directory of this
        # file. Otherwise the vmec output files may be put wherever
        # the unit tests are run from.
        this_dir = os.path.dirname(__file__)
        if this_dir != '':
            os.chdir(os.path.dirname(__file__))

    def tearDown(self):
        """
        Tear down the test fixture. This subroutine is automatically
        run by python's unittest module before every test. Here we
        call runvmec with the cleanup flag to deallocate arrays, such
        that runvmec can be called again later.
        """
        self.ictrl[0] = 16  # cleanup
        vmec.runvmec(self.ictrl, self.filename, self.verbose, \
                     self.fcomm, reset_file)

    def test_read_input(self):
        """
        Try reading a VMEC input file.
        """
        self.ictrl[0] = run_modes['input']
        vmec.runvmec(self.ictrl, self.filename, self.verbose, \
                     self.fcomm, reset_file)

        self.assertTrue(self.ictrl[1] in success_codes)

        self.assertEqual(vmec.vmec_input.nfp, 3)
        self.assertEqual(vmec.vmec_input.mpol, 4)
        self.assertEqual(vmec.vmec_input.ntor, 3)
        print('rbc.shape:', vmec.vmec_input.rbc.shape)
        print('rbc:', vmec.vmec_input.rbc[101:103, 0:4])

        # n = 0, m = 0:
        self.assertAlmostEqual(vmec.vmec_input.rbc[101, 0], 1.3782)

        # n = 0, m = 1:
        self.assertAlmostEqual(vmec.vmec_input.zbs[101, 1], 4.6465E-01)

        # n = 1, m = 1:
        self.assertAlmostEqual(vmec.vmec_input.zbs[102, 1], 1.6516E-01)

    #@unittest.skip("This test won't work until a low-level issue with VMEC is fixed to allow multiple readins.")

    def test_run_read(self):
        """
        Try running VMEC, then reading in results from the wout file.
        """

        self.ictrl[0] = 1 + 2 + 4 + 8
        vmec.runvmec(self.ictrl, self.filename, self.verbose, \
                     self.fcomm, reset_file)

        self.assertTrue(self.ictrl[1] in success_codes)

        self.assertEqual(vmec.vmec_input.nfp, 3)
        self.assertEqual(vmec.vmec_input.mpol, 4)
        self.assertEqual(vmec.vmec_input.ntor, 3)
        print('rbc.shape:', vmec.vmec_input.rbc.shape)
        print('rbc:', vmec.vmec_input.rbc[101:103, 0:4])

        # n = 0, m = 0:
        self.assertAlmostEqual(vmec.vmec_input.rbc[101, 0], 1.3782)

        # n = 0, m = 1:
        self.assertAlmostEqual(vmec.vmec_input.zbs[101, 1], 4.6465E-01)

        # n = 1, m = 1:
        self.assertAlmostEqual(vmec.vmec_input.zbs[102, 1], 1.6516E-01)

        # Now try reading in the output
        wout_file = os.path.join(os.path.dirname(__file__), 'wout_li383_low_res.nc')
        ierr = 0
        vmec.read_wout_mod.read_wout_file(wout_file, ierr)
        self.assertEqual(ierr, 0)
        self.assertAlmostEqual(vmec.read_wout_mod.betatot, \
                               0.0426211525919469, places=4)

        print('iotaf.shape:', vmec.read_wout_mod.iotaf.shape)
        print('rmnc.shape:', vmec.read_wout_mod.rmnc.shape)

        self.assertAlmostEqual(vmec.read_wout_mod.iotaf[-1], \
                               0.6556508142482989, places=4)

        self.assertAlmostEqual(vmec.read_wout_mod.rmnc[0, 0], \
                               1.4760749266902973, places=4)


if __name__ == "__main__":
    unittest.main()
