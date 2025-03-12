import os
import unittest
import numpy as np
from monty.tempfile import ScratchDir
from simsopt.geo import PortSet, CircularPort, RectangularPort

try:
    import pyevtk
except ImportError:
    pyevtk = None

class PortTests(unittest.TestCase):

    def test_circular_ports(self):
        """
        Tests for the CircularPort class and for PortSets with circular ports
        """

        # Location and symmetry properties
        Rmaj = 10 
        phiPort = 20*np.pi/180.
        nfp = 3

        # Port parameters
        [ox, oy, oz] = [Rmaj*np.cos(phiPort), Rmaj*np.sin(phiPort), 0]
        [ax, ay, az] = [0, 0, 1]
        [l0, l1] = [0, 1]
        ir = 0.5
        thk = 0.1
        gap = 0.1

        # Generate sets of points inside and outside the port and gap
        nr = 3
        ntheta = 6
        theta = np.linspace(0, 2*np.pi, ntheta)
        r_in = np.linspace(0, 0.999*ir, nr)
        r_thk = np.linspace(1.001*ir, 0.999*(ir+thk), nr)
        r_gap = np.linspace(1.001*(ir+thk), 0.999*(ir+thk+gap), nr)
        r_out = np.linspace(1.001*(ir+thk+gap), 0.999*(ir+thk+gap+1), nr)
        [Theta, R_in] = np.meshgrid(theta, r_in)
        [Theta, R_thk] = np.meshgrid(theta, r_thk)
        [Theta, R_gap] = np.meshgrid(theta, r_gap)
        [Theta, R_out] = np.meshgrid(theta, r_out)
        [X_in, Y_in] = [R_in*np.cos(Theta)+ox, R_in*np.sin(Theta)+oy]
        [X_thk, Y_thk] = [R_thk*np.cos(Theta)+ox, R_thk*np.sin(Theta)+oy]
        [X_gap, Y_gap] = [R_gap*np.cos(Theta)+ox, R_gap*np.sin(Theta)+oy]
        [X_out, Y_out] = [R_out*np.cos(Theta)+ox, R_out*np.sin(Theta)+oy]
        Z_mid = (0.5*l1)*np.ones((nr, ntheta))
        Z_gap = (l1+0.5*gap)*np.ones((nr, ntheta))
        Z_out = (l1+gap+1)*np.ones((nr, ntheta))

        # Make sure error is raised for zero-length axis
        with self.assertRaises(ValueError):
            p = CircularPort(ox=ox, oy=oy, oz=oz, ax=0, ay=0, az=0, ir=ir, \
                             thick=thk, l0=l0, l1=l1)

        # Construct the port
        p = CircularPort(ox=ox, oy=oy, oz=oz, ax=ax, ay=ay, az=az, ir=ir, \
                         thick=thk, l0=l0, l1=l1)

        # Check the points for a single port
        self.check_points(p, gap, X_in, Y_in, X_thk, Y_thk, X_gap, Y_gap, \
                          X_out, Y_out, Z_mid, Z_gap, Z_out)

        # Re-initialize port with input axis vector of non-unit length
        p = CircularPort(ox=ox, oy=oy, oz=oz, ax=5*ax, ay=5*ay, az=5*az, \
                         ir=ir, thick=thk, l0=l0, l1=l1)

        # Results should be the same irrespective of axis length
        self.check_points(p, gap, X_in, Y_in, X_thk, Y_thk, X_gap, Y_gap, \
                          X_out, Y_out, Z_mid, Z_gap, Z_out)

        # Verify that the above properties are upheld under toroidal and
        # stellarator-symmetric repetitions
        pStl = p.repeat_via_symmetries(nfp, True)
        pTor = p.repeat_via_symmetries(nfp, False)

        for i in range(nfp):

            # Rotate test points to a new field period
            X_in_i = X_in*np.cos(i*2*np.pi/nfp) - Y_in*np.sin(i*2*np.pi/nfp)
            Y_in_i = X_in*np.sin(i*2*np.pi/nfp) + Y_in*np.cos(i*2*np.pi/nfp)
            X_thk_i = X_thk*np.cos(i*2*np.pi/nfp) - Y_thk*np.sin(i*2*np.pi/nfp)
            Y_thk_i = X_thk*np.sin(i*2*np.pi/nfp) + Y_thk*np.cos(i*2*np.pi/nfp)
            X_gap_i = X_gap*np.cos(i*2*np.pi/nfp) - Y_gap*np.sin(i*2*np.pi/nfp)
            Y_gap_i = X_gap*np.sin(i*2*np.pi/nfp) + Y_gap*np.cos(i*2*np.pi/nfp)
            X_out_i = X_out*np.cos(i*2*np.pi/nfp) - Y_out*np.sin(i*2*np.pi/nfp)
            Y_out_i = X_out*np.sin(i*2*np.pi/nfp) + Y_out*np.cos(i*2*np.pi/nfp)

            # Verify that points collide as expected for toroidal symmetries
            self.check_points(pTor, gap, X_in_i, Y_in_i, X_thk_i, Y_thk_i, \
                              X_gap_i, Y_gap_i, X_out_i, Y_out_i, \
                              Z_mid, Z_gap, Z_out)

            # Verify that points collide as expected for stellarator symmetries
            self.check_points(pStl, gap, X_in_i, Y_in_i, X_thk_i, Y_thk_i, \
                              X_gap_i, Y_gap_i, X_out_i, Y_out_i, \
                              Z_mid, Z_gap, Z_out)
            self.check_points(pStl, gap, X_in_i, -Y_in_i, X_thk_i, -Y_thk_i, \
                              X_gap_i, -Y_gap_i, X_out_i, -Y_out_i, \
                              -Z_mid, -Z_gap, -Z_out)
 
    def test_rectangular_ports(self):
        """
        Tests for the RectangularPort class and for PortSets with 
        rectangular ports
        """

        # Location and symmetry properties
        Rmaj = 10 
        phiPort = 20*np.pi/180.
        nfp = 3

        # Port parameters
        [ox, oy, oz] = [Rmaj*np.cos(phiPort), Rmaj*np.sin(phiPort), 0]
        [ax, ay, az] = [0, 0, 1]
        [wx, wy, wz] = [1, 0, 0]
        [hx, hy] = [0, 1]
        [l0, l1] = [0, 1]
        iw = 0.25
        ih = 0.5
        thk = 0.1
        gap = 0.1

        # Generate sets of points inside and outside the port and gap
        nr = 3
        rrel = np.linspace(0.001, 0.999, nr)
        wdim = [1, 1, 0, -1, -1, -1, 0, 1]
        hdim = [0, 1, 1, 1, 0, -1, -1, -1]
        X_in, X_thk, X_gap, X_out = np.zeros((nr,8)), np.zeros((nr,8)), \
            np.zeros((nr,8)), np.zeros((nr,8))
        Y_in, Y_thk, Y_gap, Y_out = np.zeros((nr,8)), np.zeros((nr,8)), \
            np.zeros((nr,8)), np.zeros((nr,8))
        for i in range(8):
            X_in[:,i] = ox + 0.5*iw*rrel*wx*wdim[i] + 0.5*ih*rrel*hx*hdim[i]
            Y_in[:,i] = oy + 0.5*iw*rrel*wy*wdim[i] + 0.5*ih*rrel*hy*hdim[i]
            X_thk[:,i] = ox + (0.5*iw + thk*rrel)*wx*wdim[i] \
                            + (0.5*ih + thk*rrel)*hx*hdim[i]
            Y_thk[:,i] = oy + (0.5*iw + thk*rrel)*wy*wdim[i] \
                            + (0.5*ih + thk*rrel)*hy*hdim[i]
            X_gap[:,i] = ox + (0.5*iw + thk + gap*rrel)*wx*wdim[i] \
                            + (0.5*ih + thk + gap*rrel)*hx*hdim[i]
            Y_gap[:,i] = oy + (0.5*iw + thk + gap*rrel)*wy*wdim[i] \
                            + (0.5*ih + thk + gap*rrel)*hy*hdim[i]
            X_out[:,i] = ox + (0.5*iw + thk + gap + rrel)*wx*wdim[i] \
                            + (0.5*ih + thk + gap + rrel)*hx*hdim[i]
            Y_out[:,i] = oy + (0.5*iw + thk + gap + rrel)*wy*wdim[i] \
                            + (0.5*ih + thk + gap + rrel)*hy*hdim[i]
        Z_mid = (0.5*l1)*np.ones((nr,8))
        Z_gap = (l1+0.5*gap)*np.ones((nr,8))
        Z_out = (l1+gap+1)*np.ones((nr,8))

        # Make sure error is raised for zero or non-perpendicular axis vectors
        with self.assertRaises(ValueError):
            p = RectangularPort(ox=ox, oy=oy, oz=oz, ax=ax, ay=ay, az=az, \
                                wx=1, wy=0, wz=0.2, iw=iw, ih=ih, thick=thk, \
                                l0=l0, l1=l1)

        # Initialize the port
        p = RectangularPort(ox=ox, oy=oy, oz=oz, ax=ax, ay=ay, az=az, \
                            wx=wx, wy=wy, wz=wz, iw=iw, ih=ih, thick=thk, \
                            l0=l0, l1=l1)

        # Check the points for a single port
        self.check_points(p, gap, X_in, Y_in, X_thk, Y_thk, X_gap, Y_gap, \
                          X_out, Y_out, Z_mid, Z_gap, Z_out)

        # Re-initialize port with axis vectors of non-unit length
        # TODO: check this!
        p = RectangularPort(ox=ox, oy=oy, oz=oz, ax=5*ax, ay=5*ay, az=5*az, \
                            wx=6*wx, wy=6*wy, wz=6*wz, iw=iw, ih=ih, \
                            thick=thk, l0=l0, l1=l1)

        # Results should be the same irrespective of axis length
        self.check_points(p, gap, X_in, Y_in, X_thk, Y_thk, X_gap, Y_gap, \
                          X_out, Y_out, Z_mid, Z_gap, Z_out)

        # Verify that the above properties are upheld under toroidal and
        # stellarator-symmetric repetitions
        pStl = p.repeat_via_symmetries(nfp, True)
        pTor = p.repeat_via_symmetries(nfp, False)

        for i in range(nfp):

            # Rotate test points to a new field period
            X_in_i = X_in*np.cos(i*2*np.pi/nfp) - Y_in*np.sin(i*2*np.pi/nfp)
            Y_in_i = X_in*np.sin(i*2*np.pi/nfp) + Y_in*np.cos(i*2*np.pi/nfp)
            X_thk_i = X_thk*np.cos(i*2*np.pi/nfp) - Y_thk*np.sin(i*2*np.pi/nfp)
            Y_thk_i = X_thk*np.sin(i*2*np.pi/nfp) + Y_thk*np.cos(i*2*np.pi/nfp)
            X_gap_i = X_gap*np.cos(i*2*np.pi/nfp) - Y_gap*np.sin(i*2*np.pi/nfp)
            Y_gap_i = X_gap*np.sin(i*2*np.pi/nfp) + Y_gap*np.cos(i*2*np.pi/nfp)
            X_out_i = X_out*np.cos(i*2*np.pi/nfp) - Y_out*np.sin(i*2*np.pi/nfp)
            Y_out_i = X_out*np.sin(i*2*np.pi/nfp) + Y_out*np.cos(i*2*np.pi/nfp)

            # Verify that points collide as expected for toroidal symmetries
            self.check_points(pTor, gap, X_in_i, Y_in_i, X_thk_i, Y_thk_i, \
                              X_gap_i, Y_gap_i, X_out_i, Y_out_i, \
                              Z_mid, Z_gap, Z_out)

            # Verify that points collide as expected for stellarator symmetries
            self.check_points(pStl, gap, X_in_i, Y_in_i, X_thk_i, Y_thk_i, \
                              X_gap_i, Y_gap_i, X_out_i, Y_out_i, \
                              Z_mid, Z_gap, Z_out)
            self.check_points(pStl, gap, X_in_i, -Y_in_i, X_thk_i, -Y_thk_i, \
                              X_gap_i, -Y_gap_i, X_out_i, -Y_out_i, \
                              -Z_mid, -Z_gap, -Z_out)

    def test_port_sets(self):
        """
        Consistency checks for methods related to port set creation
        """

        # General parameters
        nfp = 3
        Rmaj = 10 
        [ax, ay, az] = [0, 0, 1]
        [l0, l1] = [0, 1]
        gap = 0.1
        thk = 0.1

        # Circular port parameters
        phiPort_c = 20*np.pi/180.
        [oxc, oyc, ozc] = [Rmaj*np.cos(phiPort_c), Rmaj*np.sin(phiPort_c), 0]
        ir = 1

        # Baseline circular port
        pCirc = CircularPort(ox=oxc, oy=oyc, oz=ozc, ax=ax, ay=ay, az=az, \
                             ir=ir, thick=thk, l0=l0, l1=l1)

        # Rectangular port parameters
        phiPort_r = 40*np.pi/180.
        [oxr, oyr, ozr] = [Rmaj*np.cos(phiPort_r), Rmaj*np.sin(phiPort_r), 0]
        [wx, wy, wz] = [1, 0, 0]
        iw = ir
        ih = 2*ir

        # Baseline rectangular port
        pRect = RectangularPort(ox=oxr, oy=oyr, oz=ozr, ax=ax, ay=ay, az=az, \
                                wx=wx, wy=wy, wz=wz, iw=iw, ih=ih, \
                                thick=thk, l0=l0, l1=l1)

        # Test points in a torus that intersects the ports
        phiAx = np.linspace(0, 2*np.pi, 72, endpoint=False)
        rAx = np.linspace(Rmaj-ir, Rmaj+ir, 8)
        zAx = np.linspace(-0.5*l1, 0.5*l1, 6)
        [Phi_test, R_test, Z_test] =  np.meshgrid(phiAx, rAx, zAx)
        X_test = R_test*np.cos(Phi_test)
        Y_test = R_test*np.sin(Phi_test)

        # Verify that colliding points calculated for ports individually
        # are the same as the colliding points for a PortSet of the two ports
        coll_circ = pCirc.collides(X_test, Y_test, Z_test, gap=gap)
        coll_rect = pRect.collides(X_test, Y_test, Z_test, gap=gap)
        coll_CR = np.logical_or(coll_circ, coll_rect)
        pBoth = pCirc + pRect
        self.assertEqual(pBoth.nPorts, 2)
        coll_both = pBoth.collides(X_test, Y_test, Z_test, gap=gap)
        self.assertTrue(np.all(coll_CR == coll_both))

        # Adding the same point twice should produce the same collisions
        pDoubleCirc = pCirc + pCirc
        coll_doub_circ = pDoubleCirc.collides(X_test, Y_test, Z_test, gap=gap)
        self.assertEqual(pDoubleCirc.nPorts, 2)
        self.assertEqual(len(pDoubleCirc.ports), 2)
        self.assertTrue(np.all(coll_doub_circ == coll_circ))

        # Compare different approaches to symmetry repetition
        pCircSym = pCirc.repeat_via_symmetries(nfp, True)
        pRectSym = pRect.repeat_via_symmetries(nfp, True)
        pCombSym = pCircSym + pRectSym
        pBothSym = pBoth.repeat_via_symmetries(nfp, True)
        self.assertEqual(len(pCircSym.ports), 6)
        self.assertEqual(pCircSym.nPorts, 6)
        self.assertEqual(len(pRectSym.ports), 6)
        self.assertEqual(pRectSym.nPorts, 6)
        self.assertEqual(len(pCombSym.ports), 12)
        self.assertEqual(pCombSym.nPorts, 12)
        self.assertEqual(len(pBothSym.ports), 12)
        self.assertEqual(pBothSym.nPorts, 12)
        coll_circ_sym = pCircSym.collides(X_test, Y_test, Z_test, gap=gap)
        coll_rect_sym = pRectSym.collides(X_test, Y_test, Z_test, gap=gap)
        coll_cr_sym = np.logical_or(coll_circ_sym, coll_rect_sym)
        coll_comb_sym = pCombSym.collides(X_test, Y_test, Z_test, gap=gap)
        coll_both_sym = pBothSym.collides(X_test, Y_test, Z_test, gap=gap)
        self.assertTrue(np.all(coll_comb_sym == coll_both_sym))
        self.assertTrue(np.all(coll_cr_sym == coll_both_sym))

    def test_port_file_io(self):
        """
        Tests methods for creating files with port parameters and loading 
        ports from files
        """
        # General parameters
        nfp = 3
        Rmaj = 10 
        [ax, ay, az] = [0, 0, 1]
        [l0, l1] = [0, 1]
        thk = 0.1

        # Circular port parameters
        phiPort_c = 20*np.pi/180.
        [oxc, oyc, ozc] = [Rmaj*np.cos(phiPort_c), Rmaj*np.sin(phiPort_c), 0]
        ir = 1

        # Baseline circular port
        pCirc = CircularPort(ox=oxc, oy=oyc, oz=ozc, ax=ax, ay=ay, az=az, \
                             ir=ir, thick=thk, l0=l0, l1=l1)

        # Rectangular port parameters
        phiPort_r = 40*np.pi/180.
        [oxr, oyr, ozr] = [Rmaj*np.cos(phiPort_r), Rmaj*np.sin(phiPort_r), 0]
        [wx, wy, wz] = [1, 0, 0]
        iw = ir
        ih = 2*ir

        # Baseline rectangular port
        pRect = RectangularPort(ox=oxr, oy=oyr, oz=ozr, ax=ax, ay=ay, az=az, \
                                wx=wx, wy=wy, wz=wz, iw=iw, ih=ih, \
                                thick=thk, l0=l0, l1=l1)

        # PortSets with circular, rectangular, and all ports with repetitions
        ports_circ = pCirc.repeat_via_symmetries(nfp, True)
        ports_rect = pRect.repeat_via_symmetries(nfp, True)
        ports_all = pCirc + pRect
        ports_all = ports_all.repeat_via_symmetries(nfp, True)

        # Save the circular ports to files and try reloading them
        with ScratchDir("."):

            ports_circ.save_ports_to_file("test")
            self.assertTrue(os.path.exists("test_circ.csv"))
            self.assertFalse(os.path.exists("test_rect.csv"))
            
            with self.assertRaises(ValueError):
                PortSet(file="test_circ.csv")

            ports_reloaded = PortSet(file="test_circ.csv", port_type='circular')
            self.assertEqual(ports_circ.nPorts, ports_reloaded.nPorts)

            # Ensure consistency in the parameters
            all_ax_orig = [p.ax for p in ports_circ]
            all_ax_reloaded = [p.ax for p in ports_reloaded]
            self.assertTrue(np.allclose(np.sort(all_ax_orig), 
                                        np.sort(all_ax_reloaded)))
 
        # Save the rectangular ports to files and try reloading them
        with ScratchDir("."):

            ports_rect.save_ports_to_file("test")
            self.assertFalse(os.path.exists("test_circ.csv"))
            self.assertTrue(os.path.exists("test_rect.csv"))
            
            with self.assertRaises(ValueError):
                PortSet(file="test_rect.csv")

            ports_reloaded = PortSet(file="test_rect.csv", 
                                     port_type='rectangular')
            self.assertEqual(ports_rect.nPorts, ports_reloaded.nPorts)

            # Ensure consistency in the parameters
            all_ax_orig = [p.ax for p in ports_rect]
            all_ax_reloaded = [p.ax for p in ports_reloaded]
            self.assertTrue(np.allclose(np.sort(all_ax_orig), 
                                        np.sort(all_ax_reloaded)))
 
        # Save the combined set of circular and rectangular ports; reload
        with ScratchDir("."):

            ports_all.save_ports_to_file("test")
            self.assertTrue(os.path.exists("test_circ.csv"))
            self.assertTrue(os.path.exists("test_rect.csv"))
            
            ports_reloaded = PortSet()
            with self.assertRaises(ValueError):
                ports_reloaded.load_circular_ports_from_file("test_rect.csv")
            with self.assertRaises(ValueError):
                ports_reloaded.load_rectangular_ports_from_file("test_circ.csv")
            ports_reloaded.load_rectangular_ports_from_file("test_rect.csv")
            ports_reloaded.load_circular_ports_from_file("test_circ.csv")
            self.assertEqual(ports_all.nPorts, ports_reloaded.nPorts)

            # Ensure consistency in the parameters
            all_ax_orig = [p.ax for p in ports_all]
            all_ax_reloaded = [p.ax for p in ports_reloaded]
            self.assertTrue(np.allclose(np.sort(all_ax_orig), 
                                        np.sort(all_ax_reloaded)))

            # Check parameter exclusive to circular ports
            all_ir_orig = [p.ir for p in ports_all 
                           if isinstance(p, CircularPort)]
            all_ir_reloaded = [p.ir for p in ports_reloaded
                               if isinstance(p, CircularPort)]
            self.assertTrue(np.allclose(np.sort(all_ir_orig), 
                                        np.sort(all_ir_reloaded)))

            # Check parameter exclusive to rectangular ports
            all_ih_orig = [p.ih for p in ports_all 
                           if isinstance(p, RectangularPort)]
            all_ih_reloaded = [p.ih for p in ports_reloaded
                               if isinstance(p, RectangularPort)]
            self.assertTrue(np.allclose(np.sort(all_ih_orig), 
                                        np.sort(all_ih_reloaded)))

    @unittest.skipIf(pyevtk is None, "pyevtk not found")
    def test_port_to_vtk(self):
        """
        Tests functions that generate VTK files from ports and port sets
        """

        # General parameters
        Rmaj = 10 
        [ax, ay, az] = [0, 0, 1]
        [l0, l1] = [0, 1]
        thk = 0.1

        # Circular port parameters
        phiPort_c = 20*np.pi/180.
        [oxc, oyc, ozc] = [Rmaj*np.cos(phiPort_c), Rmaj*np.sin(phiPort_c), 0]
        ir = 1

        # Baseline circular port
        pCirc = CircularPort(ox=oxc, oy=oyc, oz=ozc, ax=ax, ay=ay, az=az, \
                             ir=ir, thick=thk, l0=l0, l1=l1)

        # Rectangular port parameters
        phiPort_r = 40*np.pi/180.
        [oxr, oyr, ozr] = [Rmaj*np.cos(phiPort_r), Rmaj*np.sin(phiPort_r), 0]
        [wx, wy, wz] = [1, 0, 0]
        iw = ir
        ih = 2*ir

        # Baseline rectangular port
        pRect = RectangularPort(ox=oxr, oy=oyr, oz=ozr, ax=ax, ay=ay, az=az, \
                                wx=wx, wy=wy, wz=wz, iw=iw, ih=ih, \
                                thick=thk, l0=l0, l1=l1)

        bothPorts = pCirc + pRect

        with ScratchDir("."):
            pCirc.to_vtk("circular_ports")
            pRect.to_vtk("rectangular_ports")
            bothPorts.to_vtk("all_ports")
            self.assertTrue(os.path.exists("circular_ports.vtu"))
            self.assertTrue(os.path.exists("rectangular_ports.vtu"))
            self.assertTrue(os.path.exists("all_ports.vtu"))


    def check_points(self, p, gap, X_in, Y_in, X_thk, Y_thk, X_gap, Y_gap, \
                     X_out, Y_out, Z_mid, Z_gap, Z_out):
        """
        Helper function that checks multiple sets of user-input points for 
        collisions.

        NOTE: all arrays of coordinates for test points must have the same 
        dimensions.

        Parameters
        ----------
            p: PortSet, CircularPort, or RectangularPort class instance
                The port (or port set) to be tested
            gap: double
                Gap spacing to be enforced in some of the tests
            X_in, Y_in: double array
                X and Y coordinates of a set of points in the interior of the
                port(s) contained within `p`
            X_thk, Y_thk: double array
                x and y coordinates of a set of points within the finite 
                thickness of wall(s) of the port(s) in `p`
            X_gap, Y_gap: double array
                x and y coordinates of a set of points that lie within the
                gap spacing defined by `gap`
            X_out, Y_out: double array
                x and y coordinates of a set of points external to the port(s)
                in `p` and are also outside of the gap spacing
            Z_mid: double array
                z coordinates that, when used with any of the x, y pairs above,
                would locate the test points within the port in the axial
                dimension
            Z_gap: double array
                z coordinates that, when used with any of the x, y pairs above,
                would locate the test outside the nominal port length, but
                within the gap spacing, in the axial dimension
            Z_out: double array
                z coordinates that, when used with any of the x, y pairs above,
                would locate the test outside the nominal port length and the
                gap spacing in the axial dimension
        """
                     
        # Tests for points in the interior of the port
        self.assertTrue(np.all(p.collides(X_in, Y_in, Z_mid)))
        self.assertFalse(np.any(p.collides(X_in, Y_in, Z_gap)))
        self.assertTrue(np.all(p.collides(X_in, Y_in, Z_gap, gap=gap)))
        self.assertFalse(np.any(p.collides(X_in, Y_in, Z_out)))
        self.assertFalse(np.any(p.collides(X_in, Y_in, Z_out, gap=gap)))

        # Tests for points within the port's finite thkness
        self.assertTrue(np.all(p.collides(X_thk, Y_thk, Z_mid)))
        self.assertFalse(np.any(p.collides(X_thk, Y_thk, Z_gap)))
        self.assertTrue(np.all(p.collides(X_thk, Y_thk, Z_gap, gap=gap)))
        self.assertFalse(np.any(p.collides(X_thk, Y_thk, Z_out)))
        self.assertFalse(np.any(p.collides(X_thk, Y_thk, Z_out, gap=gap)))

        # Tests for points within the gap (with & without the gap enforced)
        self.assertFalse(np.any(p.collides(X_gap, Y_gap, Z_mid)))
        self.assertTrue(np.all(p.collides(X_gap, Y_gap, Z_mid, gap=gap)))
        self.assertFalse(np.any(p.collides(X_gap, Y_gap, Z_gap)))
        self.assertTrue(np.all(p.collides(X_gap, Y_gap, Z_gap, gap=gap)))
        self.assertFalse(np.any(p.collides(X_gap, Y_gap, Z_out)))
        self.assertFalse(np.any(p.collides(X_gap, Y_gap, Z_out, gap=gap)))

        # Tests for points external to the port (and the gap)
        self.assertFalse(np.any(p.collides(X_out, Y_out, Z_mid)))
        self.assertFalse(np.any(p.collides(X_out, Y_out, Z_mid, gap=gap)))
        self.assertFalse(np.any(p.collides(X_out, Y_out, Z_gap)))
        self.assertFalse(np.any(p.collides(X_out, Y_out, Z_gap, gap=gap)))
        self.assertFalse(np.any(p.collides(X_out, Y_out, Z_out)))
        self.assertFalse(np.any(p.collides(X_out, Y_out, Z_out, gap=gap)))

if __name__ == "__main__":
    unittest.main()

