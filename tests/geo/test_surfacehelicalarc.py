import unittest
import numpy as np

from simsopt.geo import SurfaceHelicalArc, SurfaceRZFourier


class Tests(unittest.TestCase):
    def test_derivatives(self):

        nfp = 4

        R0 = 10.0
        d = 2.0
        a = 3.0

        nphi = 205
        ntheta = 200
        theta0 = np.pi

        # phi_range = "full torus"
        phi_range = "half period"

        surf = SurfaceHelicalArc(
            R0=R0,
            radius_d=d,
            radius_a=a,
            nfp=nfp,
            theta0=theta0,
            alpha=0,
            nphi=nphi,
            ntheta=ntheta,
            range=phi_range,
        )
        dphi = surf.quadpoints_phi[1] - surf.quadpoints_phi[0]
        dtheta = surf.quadpoints_theta[1] - surf.quadpoints_theta[0]
        gamma = surf.gamma()
        gammadash1 = surf.gammadash1()
        gammadash2 = surf.gammadash2()
        gammadash1_finite_diff = (gamma[2:, :, :] - gamma[:-2, :, :]) / (2 * dphi)
        gammadash2_finite_diff = (gamma[:, 2:, :] - gamma[:, :-2, :]) / (2 * dtheta)

        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 8))
        nrows = 3
        ncols = 3

        for xyz in range(3):

            plt.subplot(nrows, ncols, 1 + xyz)
            plt.imshow(gammadash2[:, 1:-1, xyz])
            plt.title("gammadash2")
            plt.colorbar()

            plt.subplot(nrows, ncols, 4 + xyz)
            plt.imshow(gammadash2_finite_diff[:, :, xyz])
            plt.title("Finite diff")
            plt.colorbar()

            plt.subplot(nrows, ncols, 7 + xyz)
            plt.imshow(gammadash2[:, 1:-1, xyz] - gammadash2_finite_diff[:, :, xyz])
            plt.title("difference")
            plt.colorbar()

        plt.tight_layout()
        plt.show()
        """
        print(
            "Diff in gammadash1:",
            np.max(np.abs(gammadash1[1:-1, :, :] - gammadash1_finite_diff)),
        )
        print(
            "Diff in gammadash2:",
            np.max(np.abs(gammadash2[:, 1:-1, :] - gammadash2_finite_diff)),
        )
        np.testing.assert_allclose(
            gammadash1[1:-1, :, :], gammadash1_finite_diff, atol=1.5e-3
        )
        np.testing.assert_allclose(
            gammadash2[:, 1:-1, :], gammadash2_finite_diff, atol=3.5e-3
        )

    # @unittest.skip
    def test_theta0_pi(self):
        """If theta0=pi, properties should match an equivalent SurfaceRZFourier"""
        nfp = 4

        R0 = 10.0
        d = 2.0
        a = 3.0

        nphi = 121
        ntheta = 40
        theta0 = np.pi

        phi_range = "full torus"
        # phi_range = "half period"
        # phi_range = "field period"

        surf1 = SurfaceHelicalArc(
            R0=R0,
            radius_d=d,
            radius_a=a,
            nfp=nfp * 2,  # Note the factor of 2 is needed here!
            theta0=theta0,
            alpha=0,
            nphi=nphi,
            ntheta=ntheta,
            range=phi_range,
        )
        surf2 = SurfaceRZFourier.from_nphi_ntheta(
            nphi=nphi, ntheta=ntheta, nfp=nfp, range=phi_range, mpol=1, ntor=1
        )
        surf2.set_rc(0, 0, R0)
        surf2.set_rc(1, 0, a)
        surf2.set_zs(1, 0, a)
        surf2.set_rc(0, 1, d)
        surf2.set_zs(0, 1, -d)
        surf2.x = surf2.x

        # surf1.plot(engine="mayavi", color=(1, 1, 1), show=False)
        # surf2.plot(engine="mayavi", color=(1, 0, 0))

        print(
            f"volume:  {surf1.volume()}  {surf2.volume()}  difference={surf1.volume() - surf2.volume()}"
        )
        print(
            f"area:  {surf1.area()}  {surf2.area()}  difference={surf1.area() - surf2.area()}"
        )
        np.testing.assert_allclose(surf1.volume(), surf2.volume())
        np.testing.assert_allclose(surf1.area(), surf2.area())

    def test_integral_convergence(self):

        nfp = 4

        R0 = 10.0
        d = 2.0
        a = 3.0

        nphi0 = 85
        ntheta0 = 20
        theta0 = 1.0

        phi_range = "full torus"
        # phi_range = "half period"

        print()
        previous = 0
        for j in range(5):
            nphi = nphi0 * (2**j)
            ntheta = ntheta0 * (2**j)
            surf = SurfaceHelicalArc(
                R0=R0,
                radius_d=d,
                radius_a=a,
                nfp=nfp,
                theta0=theta0,
                alpha=0,
                nphi=nphi,
                ntheta=ntheta,
                range=phi_range,
            )
            rel_change = (surf.area() - previous) / surf.area()
            previous = surf.area()
            print(f"j: {j}  area: {surf.area()}  relative change: {rel_change}")
            # np.testing.assert_allclose(surf.area(), 408.2930788361204, rtol=1e-4)
            np.testing.assert_allclose(surf.area(), 385.23150218761555, rtol=4e-5)
