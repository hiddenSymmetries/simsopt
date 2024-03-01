import numpy as np

import simsoptpp as sopp
from .surface import Surface

__all__ = ["SurfaceHelicalArc"]


class SurfaceHelicalArc(sopp.Surface, Surface):
    def __init__(self, nfp, R0, radius_d, radius_a, theta0, alpha, nphi, ntheta, range):
        self.nfp = nfp
        self.R0 = R0
        self.radius_a = radius_a
        self.radius_d = radius_d
        self.theta0 = theta0
        self.alpha = alpha

        quadpoints_phi, quadpoints_theta = Surface.get_quadpoints(
            nphi=nphi, ntheta=ntheta, range=range, nfp=nfp
        )
        self.dtheta = quadpoints_theta[1] - quadpoints_theta[0]

        sopp.Surface.__init__(self, quadpoints_phi, quadpoints_theta)
        Surface.__init__(self, x0=[])

    def gamma_impl(self, data, quadpoints_phi, quadpoints_theta):
        phi1d = quadpoints_phi * 2 * np.pi
        theta1d = -self.theta0 + 2 * self.theta0 * (
            quadpoints_theta + 0.5 * self.dtheta
        )
        theta, phi = np.meshgrid(theta1d, phi1d)
        # shapes of theta and phi are (nphi, ntheta).

        angle = self.alpha + self.nfp * phi / 2
        R = (
            self.R0
            + self.radius_d * np.cos(angle)
            - self.radius_a * np.cos(theta + angle)
        )
        Z = self.radius_d * np.sin(angle) - self.radius_a * np.sin(theta + angle)
        data[:, :, 0] = R * np.cos(phi)
        data[:, :, 1] = R * np.sin(phi)
        data[:, :, 2] = Z

    def gammadash1_impl(self, data):
        """
        Evaluate the derivative of the position vector with respect to the
        toroidal angle phi.
        """
        phi1d = self.quadpoints_phi * 2 * np.pi
        theta1d = -self.theta0 + 2 * self.theta0 * (
            self.quadpoints_theta + 0.5 * self.dtheta
        )
        theta, phi = np.meshgrid(theta1d, phi1d)
        # shapes of theta and phi are (nphi, ntheta).

        angle = self.alpha + self.nfp * phi / 2
        R = (
            self.R0
            + self.radius_d * np.cos(angle)
            - self.radius_a * np.cos(theta + angle)
        )
        temp = (
            0.5
            * self.nfp
            * (-self.radius_d * np.sin(angle) + self.radius_a * np.sin(theta + angle))
        )
        data[:, :, 0] = 2 * np.pi * (-R * np.sin(phi) + temp * np.cos(phi))
        data[:, :, 1] = 2 * np.pi * (R * np.cos(phi) + temp * np.sin(phi))
        data[:, :, 2] = (
            np.pi
            * self.nfp
            * (self.radius_d * np.cos(angle) - self.radius_a * np.cos(theta + angle))
        )

    def gammadash2_impl(self, data):
        """
        Evaluate the derivative of the position vector with respect to the
        poloidal angle theta.
        """
        phi1d = self.quadpoints_phi * 2 * np.pi
        theta1d = -self.theta0 + 2 * self.theta0 * (
            self.quadpoints_theta + 0.5 * self.dtheta
        )
        theta, phi = np.meshgrid(theta1d, phi1d)
        # shapes of theta and phi are (nphi, ntheta).

        angle = self.alpha + self.nfp * phi / 2
        temp = 2 * self.theta0 * self.radius_a * np.sin(theta + angle)
        data[:, :, 0] = temp * np.cos(phi)
        data[:, :, 1] = temp * np.sin(phi)
        data[:, :, 2] = -2 * self.theta0 * self.radius_a * np.cos(theta + angle)
