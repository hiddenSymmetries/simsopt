import os
import unittest
import numpy as np
from monty.tempfile import ScratchDir
from simsopt.geo import PortSet, CircularPort, RectangularPort

try:
    import pyevtk
except ImportError:
    pyevtk = None

try:
    from mayavi import mlab
except ImportError:
    mlab = None


class PortTests(unittest.TestCase):

    def test_circular_ports(self):
        """
        Tests for the CircularPort class and for PortSets with circular ports
        """

        # Location and symmetry properties
        rmaj = 10
        phi_port = 20*np.pi/180.
        nfp = 3

        # Port parameters
        [ox, oy, oz] = [rmaj*np.cos(phi_port), rmaj*np.sin(phi_port), 0]
        [ax, ay, az] = [0, 0, 1]
        [l0, l1] = [0, 1]
        ir = 0.5
        thk = 0.1
        gap = 0.1

        # Generate sets of points inside and outside the port and gap
        nr = 3
        ntheta = 6
        theta_vec = np.linspace(0, 2*np.pi, ntheta)
        r_in = np.linspace(0, 0.999*ir, nr)
        r_thk = np.linspace(1.001*ir, 0.999*(ir+thk), nr)
        r_gap = np.linspace(1.001*(ir+thk), 0.999*(ir+thk+gap), nr)
        r_out = np.linspace(1.001*(ir+thk+gap), 0.999*(ir+thk+gap+1), nr)
        [theta, r_in] = np.meshgrid(theta_vec, r_in)
        [theta, r_thk] = np.meshgrid(theta_vec, r_thk)
        [theta, r_gap] = np.meshgrid(theta_vec, r_gap)
        [theta, r_out] = np.meshgrid(theta_vec, r_out)
        [x_in, y_in] = [r_in*np.cos(theta)+ox, r_in*np.sin(theta)+oy]
        [x_thk, y_thk] = [r_thk*np.cos(theta)+ox, r_thk*np.sin(theta)+oy]
        [x_gap, y_gap] = [r_gap*np.cos(theta)+ox, r_gap*np.sin(theta)+oy]
        [x_out, y_out] = [r_out*np.cos(theta)+ox, r_out*np.sin(theta)+oy]
        z_mid = (0.5*l1)*np.ones((nr, ntheta))
        z_gap = (l1+0.5*gap)*np.ones((nr, ntheta))
        z_out = (l1+gap+1)*np.ones((nr, ntheta))

        # Make sure error is raised for zero-length axis
        with self.assertRaises(ValueError):
            p = CircularPort(ox=ox, oy=oy, oz=oz, ax=0, ay=0, az=0, ir=ir,
                             thick=thk, l0=l0, l1=l1)

        # Construct the port
        p = CircularPort(ox=ox, oy=oy, oz=oz, ax=ax, ay=ay, az=az, ir=ir,
                         thick=thk, l0=l0, l1=l1)

        # Check the points for a single port
        self.check_points(p, gap, x_in, y_in, x_thk, y_thk, x_gap, y_gap,
                          x_out, y_out, z_mid, z_gap, z_out)

        # Re-initialize port with input axis vector of non-unit length
        p = CircularPort(ox=ox, oy=oy, oz=oz, ax=5*ax, ay=5*ay, az=5*az,
                         ir=ir, thick=thk, l0=l0, l1=l1)

        # Results should be the same irrespective of axis length
        self.check_points(p, gap, x_in, y_in, x_thk, y_thk, x_gap, y_gap,
                          x_out, y_out, z_mid, z_gap, z_out)

        # Verify that the above properties are upheld under toroidal and
        # stellarator-symmetric repetitions
        p_stl = p.repeat_via_symmetries(nfp, True)
        p_tor = p.repeat_via_symmetries(nfp, False)

        for i in range(nfp):

            # Rotate test points to a new field period
            x_in_i = x_in*np.cos(i*2*np.pi/nfp) - y_in*np.sin(i*2*np.pi/nfp)
            y_in_i = x_in*np.sin(i*2*np.pi/nfp) + y_in*np.cos(i*2*np.pi/nfp)
            x_thk_i = x_thk*np.cos(i*2*np.pi/nfp) - y_thk*np.sin(i*2*np.pi/nfp)
            y_thk_i = x_thk*np.sin(i*2*np.pi/nfp) + y_thk*np.cos(i*2*np.pi/nfp)
            x_gap_i = x_gap*np.cos(i*2*np.pi/nfp) - y_gap*np.sin(i*2*np.pi/nfp)
            y_gap_i = x_gap*np.sin(i*2*np.pi/nfp) + y_gap*np.cos(i*2*np.pi/nfp)
            x_out_i = x_out*np.cos(i*2*np.pi/nfp) - y_out*np.sin(i*2*np.pi/nfp)
            y_out_i = x_out*np.sin(i*2*np.pi/nfp) + y_out*np.cos(i*2*np.pi/nfp)

            # Verify that points collide as expected for toroidal symmetries
            self.check_points(p_tor, gap, x_in_i, y_in_i, x_thk_i, y_thk_i,
                              x_gap_i, y_gap_i, x_out_i, y_out_i,
                              z_mid, z_gap, z_out)

            # Verify that points collide as expected for stellarator symmetries
            self.check_points(p_stl, gap, x_in_i, y_in_i, x_thk_i, y_thk_i,
                              x_gap_i, y_gap_i, x_out_i, y_out_i,
                              z_mid, z_gap, z_out)
            self.check_points(p_stl, gap, x_in_i, -y_in_i, x_thk_i, -y_thk_i,
                              x_gap_i, -y_gap_i, x_out_i, -y_out_i,
                              -z_mid, -z_gap, -z_out)

    def test_rectangular_ports(self):
        """
        Tests for the RectangularPort class and for PortSets with 
        rectangular ports
        """

        # Location and symmetry properties
        rmaj = 10
        phi_port = 20*np.pi/180.
        nfp = 3

        # Port parameters
        [ox, oy, oz] = [rmaj*np.cos(phi_port), rmaj*np.sin(phi_port), 0]
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
        x_in, x_thk, x_gap, x_out = np.zeros((nr, 8)), np.zeros((nr, 8)), \
            np.zeros((nr, 8)), np.zeros((nr, 8))
        y_in, y_thk, y_gap, y_out = np.zeros((nr, 8)), np.zeros((nr, 8)), \
            np.zeros((nr, 8)), np.zeros((nr, 8))
        for i in range(8):
            x_in[:, i] = ox + 0.5*iw*rrel*wx*wdim[i] + 0.5*ih*rrel*hx*hdim[i]
            y_in[:, i] = oy + 0.5*iw*rrel*wy*wdim[i] + 0.5*ih*rrel*hy*hdim[i]
            x_thk[:, i] = ox + (0.5*iw + thk*rrel)*wx*wdim[i] \
                + (0.5*ih + thk*rrel)*hx*hdim[i]
            y_thk[:, i] = oy + (0.5*iw + thk*rrel)*wy*wdim[i] \
                + (0.5*ih + thk*rrel)*hy*hdim[i]
            x_gap[:, i] = ox + (0.5*iw + thk + gap*rrel)*wx*wdim[i] \
                + (0.5*ih + thk + gap*rrel)*hx*hdim[i]
            y_gap[:, i] = oy + (0.5*iw + thk + gap*rrel)*wy*wdim[i] \
                + (0.5*ih + thk + gap*rrel)*hy*hdim[i]
            x_out[:, i] = ox + (0.5*iw + thk + gap + rrel)*wx*wdim[i] \
                + (0.5*ih + thk + gap + rrel)*hx*hdim[i]
            y_out[:, i] = oy + (0.5*iw + thk + gap + rrel)*wy*wdim[i] \
                + (0.5*ih + thk + gap + rrel)*hy*hdim[i]
        z_mid = (0.5*l1)*np.ones((nr, 8))
        z_gap = (l1+0.5*gap)*np.ones((nr, 8))
        z_out = (l1+gap+1)*np.ones((nr, 8))

        # Make sure error is raised for zero or non-perpendicular axis vectors
        with self.assertRaises(ValueError):
            p = RectangularPort(ox=ox, oy=oy, oz=oz, ax=ax, ay=ay, az=az,
                                wx=1, wy=0, wz=0.2, iw=iw, ih=ih, thick=thk,
                                l0=l0, l1=l1)

        # Initialize the port
        p = RectangularPort(ox=ox, oy=oy, oz=oz, ax=ax, ay=ay, az=az,
                            wx=wx, wy=wy, wz=wz, iw=iw, ih=ih, thick=thk,
                            l0=l0, l1=l1)

        # Check the points for a single port
        self.check_points(p, gap, x_in, y_in, x_thk, y_thk, x_gap, y_gap,
                          x_out, y_out, z_mid, z_gap, z_out)

        # Re-initialize port with axis vectors of non-unit length
        # TODO: check this!
        p = RectangularPort(ox=ox, oy=oy, oz=oz, ax=5*ax, ay=5*ay, az=5*az,
                            wx=6*wx, wy=6*wy, wz=6*wz, iw=iw, ih=ih,
                            thick=thk, l0=l0, l1=l1)

        # Results should be the same irrespective of axis length
        self.check_points(p, gap, x_in, y_in, x_thk, y_thk, x_gap, y_gap,
                          x_out, y_out, z_mid, z_gap, z_out)

        # Verify that the above properties are upheld under toroidal and
        # stellarator-symmetric repetitions
        p_stl = p.repeat_via_symmetries(nfp, True)
        p_tor = p.repeat_via_symmetries(nfp, False)

        for i in range(nfp):

            # Rotate test points to a new field period
            x_in_i = x_in*np.cos(i*2*np.pi/nfp) - y_in*np.sin(i*2*np.pi/nfp)
            y_in_i = x_in*np.sin(i*2*np.pi/nfp) + y_in*np.cos(i*2*np.pi/nfp)
            x_thk_i = x_thk*np.cos(i*2*np.pi/nfp) - y_thk*np.sin(i*2*np.pi/nfp)
            y_thk_i = x_thk*np.sin(i*2*np.pi/nfp) + y_thk*np.cos(i*2*np.pi/nfp)
            x_gap_i = x_gap*np.cos(i*2*np.pi/nfp) - y_gap*np.sin(i*2*np.pi/nfp)
            y_gap_i = x_gap*np.sin(i*2*np.pi/nfp) + y_gap*np.cos(i*2*np.pi/nfp)
            x_out_i = x_out*np.cos(i*2*np.pi/nfp) - y_out*np.sin(i*2*np.pi/nfp)
            y_out_i = x_out*np.sin(i*2*np.pi/nfp) + y_out*np.cos(i*2*np.pi/nfp)

            # Verify that points collide as expected for toroidal symmetries
            self.check_points(p_tor, gap, x_in_i, y_in_i, x_thk_i, y_thk_i,
                              x_gap_i, y_gap_i, x_out_i, y_out_i,
                              z_mid, z_gap, z_out)

            # Verify that points collide as expected for stellarator symmetries
            self.check_points(p_stl, gap, x_in_i, y_in_i, x_thk_i, y_thk_i,
                              x_gap_i, y_gap_i, x_out_i, y_out_i,
                              z_mid, z_gap, z_out)
            self.check_points(p_stl, gap, x_in_i, -y_in_i, x_thk_i, -y_thk_i,
                              x_gap_i, -y_gap_i, x_out_i, -y_out_i,
                              -z_mid, -z_gap, -z_out)

    def test_port_sets(self):
        """
        Consistency checks for methods related to port set creation
        """

        # General parameters
        nfp = 3
        rmaj = 10
        [ax, ay, az] = [0, 0, 1]
        [l0, l1] = [0, 1]
        gap = 0.1
        thk = 0.1

        # Circular port parameters
        phi_port_c = 20*np.pi/180.
        [oxc, oyc, ozc] = [rmaj*np.cos(phi_port_c), rmaj*np.sin(phi_port_c), 0]
        ir = 1

        # Baseline circular port
        p_circ = CircularPort(ox=oxc, oy=oyc, oz=ozc, ax=ax, ay=ay, az=az,
                              ir=ir, thick=thk, l0=l0, l1=l1)

        # Rectangular port parameters
        phi_port_r = 40*np.pi/180.
        [oxr, oyr, ozr] = [rmaj*np.cos(phi_port_r), rmaj*np.sin(phi_port_r), 0]
        [wx, wy, wz] = [1, 0, 0]
        iw = ir
        ih = 2*ir

        # Baseline rectangular port
        p_rect = RectangularPort(ox=oxr, oy=oyr, oz=ozr, ax=ax, ay=ay, az=az,
                                 wx=wx, wy=wy, wz=wz, iw=iw, ih=ih,
                                 thick=thk, l0=l0, l1=l1)

        # Test points in a torus that intersects the ports
        phi_ax = np.linspace(0, 2*np.pi, 72, endpoint=False)
        r_ax = np.linspace(rmaj-ir, rmaj+ir, 8)
        z_ax = np.linspace(-0.5*l1, 0.5*l1, 6)
        [phi_test, r_test, z_test] = np.meshgrid(phi_ax, r_ax, z_ax)
        x_test = r_test*np.cos(phi_test)
        y_test = r_test*np.sin(phi_test)

        # Verify that colliding points calculated for ports individually
        # are the same as the colliding points for a PortSet of the two ports
        coll_circ = p_circ.collides(x_test, y_test, z_test, gap=gap)
        coll_rect = p_rect.collides(x_test, y_test, z_test, gap=gap)
        coll_cr = np.logical_or(coll_circ, coll_rect)
        p_both = p_circ + p_rect
        self.assertEqual(p_both.n_ports, 2)
        coll_both = p_both.collides(x_test, y_test, z_test, gap=gap)
        self.assertTrue(np.all(coll_cr == coll_both))

        # Adding the same point twice should produce the same collisions
        p_double_circ = p_circ + p_circ
        coll_doub_circ = p_double_circ.collides(x_test, y_test, z_test, gap=gap)
        self.assertEqual(p_double_circ.n_ports, 2)
        self.assertEqual(len(p_double_circ.ports), 2)
        self.assertTrue(np.all(coll_doub_circ == coll_circ))

        # Compare different approaches to symmetry repetition
        p_circ_sym = p_circ.repeat_via_symmetries(nfp, True)
        p_rect_sym = p_rect.repeat_via_symmetries(nfp, True)
        p_comb_sym = p_circ_sym + p_rect_sym
        p_both_sym = p_both.repeat_via_symmetries(nfp, True)
        self.assertEqual(len(p_circ_sym.ports), 6)
        self.assertEqual(p_circ_sym.n_ports, 6)
        self.assertEqual(len(p_rect_sym.ports), 6)
        self.assertEqual(p_rect_sym.n_ports, 6)
        self.assertEqual(len(p_comb_sym.ports), 12)
        self.assertEqual(p_comb_sym.n_ports, 12)
        self.assertEqual(len(p_both_sym.ports), 12)
        self.assertEqual(p_both_sym.n_ports, 12)
        coll_circ_sym = p_circ_sym.collides(x_test, y_test, z_test, gap=gap)
        coll_rect_sym = p_rect_sym.collides(x_test, y_test, z_test, gap=gap)
        coll_cr_sym = np.logical_or(coll_circ_sym, coll_rect_sym)
        coll_comb_sym = p_comb_sym.collides(x_test, y_test, z_test, gap=gap)
        coll_both_sym = p_both_sym.collides(x_test, y_test, z_test, gap=gap)
        self.assertTrue(np.all(coll_comb_sym == coll_both_sym))
        self.assertTrue(np.all(coll_cr_sym == coll_both_sym))

    def test_port_file_io(self):
        """
        Tests methods for creating files with port parameters and loading 
        ports from files
        """
        # General parameters
        nfp = 3
        rmaj = 10
        [ax, ay, az] = [0, 0, 1]
        [l0, l1] = [0, 1]
        thk = 0.1

        # Circular port parameters
        phi_port_c = 20*np.pi/180.
        [oxc, oyc, ozc] = [rmaj*np.cos(phi_port_c), rmaj*np.sin(phi_port_c), 0]
        ir = 1

        # Baseline circular port
        p_circ = CircularPort(ox=oxc, oy=oyc, oz=ozc, ax=ax, ay=ay, az=az,
                              ir=ir, thick=thk, l0=l0, l1=l1)

        # Rectangular port parameters
        phi_port_r = 40*np.pi/180.
        [oxr, oyr, ozr] = [rmaj*np.cos(phi_port_r), rmaj*np.sin(phi_port_r), 0]
        [wx, wy, wz] = [1, 0, 0]
        iw = ir
        ih = 2*ir

        # Baseline rectangular port
        p_rect = RectangularPort(ox=oxr, oy=oyr, oz=ozr, ax=ax, ay=ay, az=az,
                                 wx=wx, wy=wy, wz=wz, iw=iw, ih=ih,
                                 thick=thk, l0=l0, l1=l1)

        # PortSets with circular, rectangular, and all ports with repetitions
        ports_circ = p_circ.repeat_via_symmetries(nfp, True)
        ports_rect = p_rect.repeat_via_symmetries(nfp, True)
        ports_all = p_circ + p_rect
        ports_all = ports_all.repeat_via_symmetries(nfp, True)

        # Save the circular ports to files and try reloading them
        with ScratchDir("."):

            ports_circ.save_ports_to_file("test")
            self.assertTrue(os.path.exists("test_circ.csv"))
            self.assertFalse(os.path.exists("test_rect.csv"))

            with self.assertRaises(ValueError):
                PortSet(file="test_circ.csv")

            ports_reloaded = PortSet(file="test_circ.csv", port_type='circular')
            self.assertEqual(ports_circ.n_ports, ports_reloaded.n_ports)

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
            self.assertEqual(ports_rect.n_ports, ports_reloaded.n_ports)

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
            self.assertEqual(ports_all.n_ports, ports_reloaded.n_ports)

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
        rmaj = 10
        [ax, ay, az] = [0, 0, 1]
        [l0, l1] = [0, 1]
        thk = 0.1

        # Circular port parameters
        phi_port_c = 20*np.pi/180.
        [oxc, oyc, ozc] = [rmaj*np.cos(phi_port_c), rmaj*np.sin(phi_port_c), 0]
        ir = 1

        # Baseline circular port
        p_circ = CircularPort(ox=oxc, oy=oyc, oz=ozc, ax=ax, ay=ay, az=az,
                              ir=ir, thick=thk, l0=l0, l1=l1)

        # Rectangular port parameters
        phi_port_r = 40*np.pi/180.
        [oxr, oyr, ozr] = [rmaj*np.cos(phi_port_r), rmaj*np.sin(phi_port_r), 0]
        [wx, wy, wz] = [1, 0, 0]
        iw = ir
        ih = 2*ir

        # Baseline rectangular port
        p_rect = RectangularPort(ox=oxr, oy=oyr, oz=ozr, ax=ax, ay=ay, az=az,
                                 wx=wx, wy=wy, wz=wz, iw=iw, ih=ih,
                                 thick=thk, l0=l0, l1=l1)

        both_ports = p_circ + p_rect

        with ScratchDir("."):
            p_circ.to_vtk("circular_ports")
            p_rect.to_vtk("rectangular_ports")
            both_ports.to_vtk("all_ports")
            self.assertTrue(os.path.exists("circular_ports.vtu"))
            self.assertTrue(os.path.exists("rectangular_ports.vtu"))
            self.assertTrue(os.path.exists("all_ports.vtu"))

    @unittest.skipIf(mlab is None, "mayavi.mlab not found")
    def test_port_plotting(self):
        """
        Tests functions that generate plots of ports and port sets
        """

        # General parameters
        rmaj = 10
        [ax, ay, az] = [0, 0, 1]
        [l0, l1] = [0, 1]
        thk = 0.1

        # Circular port parameters
        phi_port_c = 20*np.pi/180.
        [oxc, oyc, ozc] = [rmaj*np.cos(phi_port_c), rmaj*np.sin(phi_port_c), 0]
        ir = 1

        # Baseline circular port
        p_circ = CircularPort(ox=oxc, oy=oyc, oz=ozc, ax=ax, ay=ay, az=az,
                              ir=ir, thick=thk, l0=l0, l1=l1)

        # Rectangular port parameters
        phi_port_r = 40*np.pi/180.
        [oxr, oyr, ozr] = [rmaj*np.cos(phi_port_r), rmaj*np.sin(phi_port_r), 0]
        [wx, wy, wz] = [1, 0, 0]
        iw = ir
        ih = 2*ir

        # Baseline rectangular port
        p_rect = RectangularPort(ox=oxr, oy=oyr, oz=ozr, ax=ax, ay=ay, az=az,
                                 wx=wx, wy=wy, wz=wz, iw=iw, ih=ih,
                                 thick=thk, l0=l0, l1=l1)

        both_ports = p_circ + p_rect

        # Generate the plots
        p_circ.plot(n_edges=50)
        p_rect.plot()
        both_ports.plot()
        mlab.close(all=True)

    def check_points(self, p, gap, x_in, y_in, x_thk, y_thk, x_gap, y_gap,
                     x_out, y_out, z_mid, z_gap, z_out):
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
            x_in, y_in: double array
                X and Y coordinates of a set of points in the interior of the
                port(s) contained within `p`
            x_thk, y_thk: double array
                x and y coordinates of a set of points within the finite 
                thickness of wall(s) of the port(s) in `p`
            x_gap, y_gap: double array
                x and y coordinates of a set of points that lie within the
                gap spacing defined by `gap`
            x_out, y_out: double array
                x and y coordinates of a set of points external to the port(s)
                in `p` and are also outside of the gap spacing
            z_mid: double array
                z coordinates that, when used with any of the x, y pairs above,
                would locate the test points within the port in the axial
                dimension
            z_gap: double array
                z coordinates that, when used with any of the x, y pairs above,
                would locate the test outside the nominal port length, but
                within the gap spacing, in the axial dimension
            z_out: double array
                z coordinates that, when used with any of the x, y pairs above,
                would locate the test outside the nominal port length and the
                gap spacing in the axial dimension
        """

        # Tests for points in the interior of the port
        self.assertTrue(np.all(p.collides(x_in, y_in, z_mid)))
        self.assertFalse(np.any(p.collides(x_in, y_in, z_gap)))
        self.assertTrue(np.all(p.collides(x_in, y_in, z_gap, gap=gap)))
        self.assertFalse(np.any(p.collides(x_in, y_in, z_out)))
        self.assertFalse(np.any(p.collides(x_in, y_in, z_out, gap=gap)))

        # Tests for points within the port's finite thkness
        self.assertTrue(np.all(p.collides(x_thk, y_thk, z_mid)))
        self.assertFalse(np.any(p.collides(x_thk, y_thk, z_gap)))
        self.assertTrue(np.all(p.collides(x_thk, y_thk, z_gap, gap=gap)))
        self.assertFalse(np.any(p.collides(x_thk, y_thk, z_out)))
        self.assertFalse(np.any(p.collides(x_thk, y_thk, z_out, gap=gap)))

        # Tests for points within the gap (with & without the gap enforced)
        self.assertFalse(np.any(p.collides(x_gap, y_gap, z_mid)))
        self.assertTrue(np.all(p.collides(x_gap, y_gap, z_mid, gap=gap)))
        self.assertFalse(np.any(p.collides(x_gap, y_gap, z_gap)))
        self.assertTrue(np.all(p.collides(x_gap, y_gap, z_gap, gap=gap)))
        self.assertFalse(np.any(p.collides(x_gap, y_gap, z_out)))
        self.assertFalse(np.any(p.collides(x_gap, y_gap, z_out, gap=gap)))

        # Tests for points external to the port (and the gap)
        self.assertFalse(np.any(p.collides(x_out, y_out, z_mid)))
        self.assertFalse(np.any(p.collides(x_out, y_out, z_mid, gap=gap)))
        self.assertFalse(np.any(p.collides(x_out, y_out, z_gap)))
        self.assertFalse(np.any(p.collides(x_out, y_out, z_gap, gap=gap)))
        self.assertFalse(np.any(p.collides(x_out, y_out, z_out)))
        self.assertFalse(np.any(p.collides(x_out, y_out, z_out, gap=gap)))


if __name__ == "__main__":
    unittest.main()
