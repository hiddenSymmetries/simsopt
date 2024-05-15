import unittest
import numpy as np
from simsopt.geo import CircularPort, RectangularPort, PortSet

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
        pStl = PortSet(ports=[p])
        pStl.repeat_via_symmetries(nfp, True)
        pTor = PortSet(ports=[p])
        pTor.repeat_via_symmetries(nfp, False)


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
        [hx, hy, hz] = [0, 1, 0]
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
                                wx=wx, wy=wy, wz=wy, hx=0, hy=0, hz=0, \
                                iw=iw, ih=ih, thick=thk, l0=l0, l1=l1)
        with self.assertRaises(ValueError):
            p = RectangularPort(ox=ox, oy=oy, oz=oz, ax=ax, ay=ay, az=az, \
                                wx=1, wy=0.2, wz=0, hx=hx, hy=hy, hz=hz, \
                                iw=iw, ih=ih, thick=thk, l0=l0, l1=l1)
        with self.assertRaises(ValueError):
            p = RectangularPort(ox=ox, oy=oy, oz=oz, ax=ax, ay=ay, az=az, \
                                wx=1, wy=0, wz=0.2, hx=hx, hy=hy, hz=hz, \
                                iw=iw, ih=ih, thick=thk, l0=l0, l1=l1)

        # Initialize the port
        p = RectangularPort(ox=ox, oy=oy, oz=oz, ax=ax, ay=ay, az=az, \
                            wx=wx, wy=wy, wz=wz, hx=hx, hy=hy, hz=hz, \
                            iw=iw, ih=ih, thick=thk, l0=l0, l1=l1)

        # Check the points for a single port
        self.check_points(p, gap, X_in, Y_in, X_thk, Y_thk, X_gap, Y_gap, \
                          X_out, Y_out, Z_mid, Z_gap, Z_out)

        # Re-initialize port with axis vectors of non-unit length
        p = RectangularPort(ox=ox, oy=oy, oz=oz, ax=5*ax, ay=5*ay, az=5*az, \
                            wx=6*wx, wy=6*wy, wz=6*wz, hx=7*hx, hy=7*hy, \
                            hz=7*hz, iw=iw, ih=ih, thick=thk, l0=l0, l1=l1)

        # Results should be the same irrespective of axis length
        self.check_points(p, gap, X_in, Y_in, X_thk, Y_thk, X_gap, Y_gap, \
                          X_out, Y_out, Z_mid, Z_gap, Z_out)

        # Verify that the above properties are upheld under toroidal and
        # stellarator-symmetric repetitions
        pStl = PortSet(ports=[p])
        pStl.repeat_via_symmetries(nfp, True)
        pTor = PortSet(ports=[p])
        pTor.repeat_via_symmetries(nfp, False)

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


