from pathlib import Path
import unittest

import numpy as np
from monty.tempfile import ScratchDir

from simsopt.geo import PSCgrid, WPgrid, SurfaceRZFourier
from simsopt.field import BiotSavart, coils_via_symmetries, Current, CircularCoil, PSC_BiotSavart
import simsoptpp as sopp

input_name = 'input.LandremanPaul2021_QA_lowres'
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
surf1 = SurfaceRZFourier.from_vmec_input(
    surface_filename, range='full torus', nphi=16, ntheta=16
)
surf1.nfp = 1
surf1.stellsym = False
surf2 = SurfaceRZFourier.from_vmec_input(
    surface_filename, range='half period', nphi=16, ntheta=16
)
input_name = 'input.LandremanPaul2021_QH'
TEST_DIR = (Path(__file__).parent / ".." / ".." / "examples" / "2_Intermediate" / "inputs").resolve()
surface_filename = TEST_DIR / input_name
surf3 = SurfaceRZFourier.from_vmec_input(
    surface_filename, range='half period', nphi=16, ntheta=16
)
input_name = 'wout_c09r00_fixedBoundary_0.5T_vacuum_ns201.nc'
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
surf4 = SurfaceRZFourier.from_wout(
    surface_filename, range='half period', nphi=16, ntheta=16
)
surfs = [surf1, surf2, surf3, surf4]
ncoils = 8
np.random.seed(29)
R = 0.05
a = 1e-5
points = (np.random.rand(ncoils, 3) - 0.5) * 3
points[:, -1] = 0.4
alphas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
deltas = (np.random.rand(ncoils) - 0.5) * np.pi
epsilon = 1e-7  # smaller epsilon and numerical accumulation starts to be an issue

class Testing(unittest.TestCase):
    
    # def test_coil_forces_derivatives(self):
    #     ncoils = 7
    #     np.random.seed(1)
    #     R = 1.0
    #     a = 1e-5
    #     points = (np.random.rand(ncoils, 3) - 0.5) * 20
    #     alphas = (np.random.rand(ncoils) - 0.5) * np.pi
    #     deltas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
    #     epsilon = 1e-4  # smaller epsilon and numerical accumulation starts to be an issue
    #     for surf in surfs:
    #         print('Surf = ', surf)
    #         kwargs_manual = {"plasma_boundary": surf}
    #         wp_array = WPgrid.geo_setup_manual(
    #             points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
    #         )
    #         b_F, A_F = wp_array.coil_forces()
    #         I = wp_array.I
    #         b_F *= I 
    #         A_F *= I
    #         force_objective = 0.5 * (A_F @ I + b_F).T @ (A_F @ I + b_F)
    #         deltas_new = np.copy(deltas)
    #         deltas_new[0] += epsilon
    #         alphas_new = np.copy(alphas)
    #         psc_array_new = WPgrid.geo_setup_manual(
    #             points, R=R, a=a, alphas=alphas_new, deltas=deltas_new, **kwargs_manual
    #         )
    #         b_F, A_F_new = wp_array.coil_forces()
    #         I = wp_array.I
    #         b_F *= I 
    #         A_F *= I
    #         force_objective_new = 0.5 * (A_F_new @ I + b_F).T @ (A_F_new @ I + b_F)
    #         df_objective = (force_objective_new - force_objective) / epsilon
    #         # print(A @ Linv @ psi_new, A @ Linv @ psi, b)
    #         dAF_ddelta = (A_F_new - A_F) / epsilon
    #         AF_deriv = wp_array.AF_deriv()
    #         print(dAF_ddelta[0], AF_deriv[ncoils])
    #         assert(np.allclose(dAF_ddelta[0], AF_deriv[ncoils], rtol=1e-3))
    #         df_analytic = (A_F @ I + b_F).T @ (A_F @ I)
    #         print(df_objective, df_analytic[0])
    #         assert np.isclose(df_objective, df_analytic[0], rtol=1e-2)
    
    # def test_coil_forces(self):
    #     from scipy.special import ellipk, ellipe
        
    #     ncoils = 2
    #     I1 = 5.0
    #     I2 = -3.0
    #     Z1 = 1.0
    #     Z2 = 4.0
    #     R1 = 0.5
    #     R2 = R1
    #     a = 1e-5
    #     points = np.array([[0.0, 0.0, Z1], [0.0, 0.0, Z2]])
    #     alphas = np.zeros(ncoils)  # (np.random.rand(ncoils) - 0.5) * np.pi
    #     deltas = np.zeros(ncoils)  # (np.random.rand(ncoils) - 0.5) * 2 * np.pi
    #     k = np.sqrt(4.0 * R1 * R2 / ((R1 + R2) ** 2 + (Z2 - Z1) ** 2))
    #     mu0 = 4 * np.pi * 1e-7
    #     # Jackson uses K(k) and E(k) but this corresponds to
    #     # K(k^2) and E(k^2) in scipy library
    #     F_analytic = mu0 * I1 * I2 * k * (Z2 - Z1) * (
    #         (2  - k ** 2) * ellipe(k ** 2) / (1 - k ** 2) - 2.0 * ellipk(k ** 2)
    #     ) / (4.0 * np.sqrt(R1 * R2))
        
    #     wp_array = WPgrid.geo_setup_manual(
    #         points, R=R1, a=a, alphas=alphas, deltas=deltas, I=np.array([I1, I2])
    #     )
    #     b_F, A_F = wp_array.coil_forces()
    #     F_TF = wp_array.I[:, None] * b_F
    #     # print('A = ', A_F[:, :, 2])
    #     F_WP = wp_array.I[:, None] * np.tensordot(A_F, wp_array.I, axes=([1, 0])) * wp_array.fac
    #     print(F_TF, F_WP, F_WP.shape, F_analytic)
    #     assert np.allclose(F_WP[:, 0], 0.0)
    #     assert np.allclose(F_WP[:, 1], 0.0)
    #     assert np.allclose(np.abs(F_WP[:, 2]), abs(F_analytic))

    def test_quaternion_derivs(self):
        dofs = (np.random.rand(4) - 0.5) * 0.1   # quaternion dofs
        normalization = np.sqrt(np.sum(dofs ** 2))
        w = dofs[0]
        x = dofs[1]
        y = dofs[2]
        z = dofs[3]
        alphas = np.arctan2(2 * (w * x + y * z), 
                            1 - 2.0 * (x ** 2 + y ** 2))
        deltas = -np.pi / 2.0 + 2.0 * np.arctan2(
            np.sqrt(1.0 + 2 * (w * y - x * z)), 
            np.sqrt(1.0 - 2 * (w * y - x * z)))
        adiff = []
        ddiff = []
        for eps in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
            print(eps)
            for i in range(1):
                print(i)
                epsilon = np.zeros(4)  # (np.random.rand(4) - 0.5)
                epsilon[i] = eps
                dofs2 = dofs + epsilon
                # dofs = dofs / normalization  # normalize the quaternion
                normalization2 = np.sqrt(np.sum(dofs2 ** 2))
                # dofs2 = dofs2 / normalization2
                # print(normalization, normalization2, dofs, dofs2, np.sqrt(np.sum(dofs ** 2)), np.sqrt(np.sum(dofs2 ** 2)), epsilon)
                
                # sis = np.arctan2(2 * (w * z + y * x), 
                #                     1 - 2.0 * (z ** 2 + y ** 2))
                alphas2 = np.arctan2(2 * (dofs2[0] * dofs2[1] + dofs2[2] * dofs2[3]), 
                                    1 - 2.0 * (dofs2[1] ** 2 + dofs2[2] ** 2))
                deltas2 = -np.pi / 2.0 + 2.0 * np.arctan2(
                    np.sqrt(1.0 + 2 * (dofs2[0] * dofs2[2] - dofs2[1] * dofs2[3])), 
                    np.sqrt(1.0 - 2 * (dofs2[0] * dofs2[2] - dofs2[1] * dofs2[3])))
                # ca = np.cos(alphas / 2.0)
                # sa = np.sin(alphas / 2.0)
                # cd = np.cos(deltas / 2.0)
                # sd = np.sin(deltas / 2.0)
                # cp = np.cos(sis / 2.0)
                # sp = np.sin(sis / 2.0)

                # q = [ca * cd * cp + sa * sd * sp, 
                #     sa * cd * cp - ca * sd * sp, 
                #     ca * sd * cp + sa * cd * sp, 
                #     ca * cd * sp - sa * sd * cp]
                # print(q)
                dalpha_fd = (alphas2 - alphas) / (dofs2 - dofs)[i]
                ddelta_fd = (deltas2 - deltas) / (dofs2 - dofs)[i]
                print((dofs2 - dofs)[i])
                print(alphas, deltas)
                print(alphas2, deltas2)
                # print('dofs = ', dofs, normalization)

                def dalpha_func():
                    d1 = (4 * (w * x + y * z) ** 2 + (1 - 2 * ( x ** 2 + y ** 2)) ** 2)
                    d2 = (x ** 2 * (w ** 2 + 2 * y ** 2 - 1) + 2 * w * x * y * z + x ** 4 + y ** 4 + y ** 2 * (z ** 2 - 1) + 0.25)
                    n1 = 2 * x * (-2 * (x ** 2 + y ** 2) + 1)
                    n2 = -(w * (-0.5 - x ** 2 + y ** 2) - 2 * x * y * z)
                    n3 = -(z * (-0.5 + x ** 2 - y ** 2) - 2 * x * y * w)
                    n4 = 2 * y * (-2 * (x ** 2 + y ** 2) + 1)
                    print('here = ', d1, d2, n1, n2, n3, n4, x ** 4, y ** 4, x, y, z, w)
                    dalpha_dw = n1 / d1
                    dalpha_dx = n2 / d2
                    dalpha_dy = n3 / d2
                    dalpha_dz = n4 / d1
                    return dalpha_dw, dalpha_dx, dalpha_dy, dalpha_dz 
            
                dalpha_dw, dalpha_dx, dalpha_dy, dalpha_dz = dalpha_func()

                # dalpha_dw = (2 * y * (-2 * (x ** 2 + y ** 2) + 1)) / \
                #     (4 * (w * x + y * z) ** 2 + (1 - 2 * ( x ** 2 + y ** 2)) ** 2)
                # dalpha_dx = -(w * (-0.5 - x ** 2 + y ** 2) - 2 * x * y * z) / \
                #     (x ** 2 * (w ** 2 + 2 * y ** 2 - 1) + 2 * w * x * y * z + x ** 4 + y ** 4 + y ** 2 * (z ** 2 - 1) + 0.25)
                # dalpha_dy = -(z * (-0.5 + x ** 2 - y ** 2) - 2 * x * y * w) / \
                #     (x ** 2 * (w ** 2 + 2 * y ** 2 - 1) + 2 * w * x * y * z + x ** 4 + y ** 4 + y ** 2 * (z ** 2 - 1) + 0.25)
                # dalpha_dz = (2 * y * (1 - 2 * (x ** 2 + y ** 2))) / \
                #     (4 * (w * x + y * z) ** 2 + (1 - 2 * ( x ** 2 + y ** 2)) ** 2)
                ddelta_dw = y / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
                ddelta_dx = -z / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
                ddelta_dy = w / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
                ddelta_dz = -x / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
                dalpha = np.array([dalpha_dw, dalpha_dx, dalpha_dy, dalpha_dz])
                ddelta = np.array([ddelta_dw, ddelta_dx, ddelta_dy, ddelta_dz])
                print('da, dd = ', dalpha[i], ddelta[i])
                print('da_fd, dd_fd = ', dalpha_fd, ddelta_fd)
                adiff.append(np.abs(dalpha_fd - dalpha[i]))
                ddiff.append(np.abs(ddelta_fd - ddelta[i]))
                # assert np.isclose(dalpha[i], dalpha_fd)
                # assert np.isclose(ddelta[i], ddelta_fd)

        adiff = np.array(adiff)
        ddiff = np.array(ddiff)
        from matplotlib import pyplot as plt 
        print(adiff)
        t = np.array([1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12])
        plt.figure(1000)
        plt.grid()
        plt.loglog(t, adiff, 'ro')
        plt.loglog(t, ddiff, 'bo')
        plt.show()

        dofs = (np.random.rand(4) - 0.5) * 10   # quaternion dofs
        normalization = np.sqrt(np.sum(dofs ** 2))
        dofs_unnormalized = np.copy(dofs)
        dofs = dofs / normalization  # normalize the quaternion
        w = dofs[0]
        x = dofs[1]
        y = dofs[2]
        z = dofs[3]
        print('here2 = ', x ** 4, y ** 4, x, y, z, w)
        alphas = np.arctan2(2 * (w * x + y * z), 
                                    1 - 2.0 * (x ** 2 + y ** 2))
        deltas = -np.pi / 2.0 + 2.0 * np.arctan2(
            np.sqrt(1.0 + 2 * (w * y - x * z)), 
            np.sqrt(1.0 - 2 * (w * y - x * z)))
        # now try normalized
        for i in range(4):
            adiff = []
            ddiff = []
            print('i = ', i)
            for eps in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
                epsilon = np.zeros(4)  # (np.random.rand(4) - 0.5)
                epsilon[i] = eps
                dofs2 = dofs_unnormalized + epsilon
                normalization2 = np.sqrt(np.sum(dofs2 ** 2))
                dofs2 = dofs2 / normalization2
                # print(normalization, normalization2, dofs, dofs2, np.sqrt(np.sum(dofs ** 2)), np.sqrt(np.sum(dofs2 ** 2)), epsilon)
                # sis = np.arctan2(2 * (w * z + y * x), 
                #                     1 - 2.0 * (z ** 2 + y ** 2))
                alphas2 = np.arctan2(2 * (dofs2[0] * dofs2[1] + dofs2[2] * dofs2[3]), 
                                    1 - 2.0 * (dofs2[1] ** 2 + dofs2[2] ** 2))
                deltas2 = -np.pi / 2.0 + 2.0 * np.arctan2(
                    np.sqrt(1.0 + 2 * (dofs2[0] * dofs2[2] - dofs2[1] * dofs2[3])), 
                    np.sqrt(1.0 - 2 * (dofs2[0] * dofs2[2] - dofs2[1] * dofs2[3])))
                # ca = np.cos(alphas / 2.0)
                # sa = np.sin(alphas / 2.0)
                # cd = np.cos(deltas / 2.0)
                # sd = np.sin(deltas / 2.0)
                # cp = np.cos(sis / 2.0)
                # sp = np.sin(sis / 2.0)

                # Normalized derivative should be
                # dalpha/dofs = dalpha / normed * dnormed/dofs
                # normed_dofs = dofs / sqrt(dofs[0] ** 2 + ... )
                # wn = dofs[0] / normalization = w / normalization
                # dalpha_dw = dalpha_dwn * (dwn/dw) + dalpha_dxn * (dxn/dw) + ...
                # Changing one dof changes them all through the normalization!
                # print(normalization, dofs_unnormalized, dofs_unnormalized * dofs_unnormalized[i] / normalization ** 3)
                eye = np.zeros(4)
                eye[i] = 1.0
                dnormalization = eye / normalization - dofs_unnormalized * dofs_unnormalized[i] / normalization ** 3
                # print('dnorm = ', dnormalization)

                # q = [ca * cd * cp + sa * sd * sp, 
                #     sa * cd * cp - ca * sd * sp, 
                #     ca * sd * cp + sa * cd * sp, 
                #     ca * cd * sp - sa * sd * cp]
                # print('q = ', q)
                dalpha_fd = (alphas2 - alphas) / epsilon[i]
                ddelta_fd = (deltas2 - deltas) / epsilon[i]
                print((alphas2 - alphas) / (dofs2 - dofs))
                print((deltas2 - deltas) / (dofs2 - dofs))
                print(alphas, alphas2, deltas, deltas2, epsilon[i])
                dalpha_dw = (2 * x * (-2 * (x ** 2 + y ** 2) + 1)) / \
                    (4 * (w * x + y * z) ** 2 + (1 - 2 * ( x ** 2 + y ** 2)) ** 2)
                dalpha_dx = -(w * (-0.5 - x ** 2 + y ** 2) - 2 * x * y * z) / \
                    (x ** 2 * (w ** 2 + 2 * y ** 2 - 1) + 2 * w * x * y * z + x ** 4 + y ** 4 + y ** 2 * (z ** 2 - 1) + 0.25)
                dalpha_dy = -(z * (-0.5 + x ** 2 - y ** 2) - 2 * x * y * w) / \
                    (x ** 2 * (w ** 2 + 2 * y ** 2 - 1) + 2 * w * x * y * z + x ** 4 + y ** 4 + y ** 2 * (z ** 2 - 1) + 0.25)
                dalpha_dz = (2 * y * (1 - 2 * (x ** 2 + y ** 2))) / \
                    (4 * (w * x + y * z) ** 2 + (1 - 2 * ( x ** 2 + y ** 2)) ** 2)
                ddelta_dw = y / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
                ddelta_dx = -z / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
                ddelta_dy = w / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
                ddelta_dz = -x / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
                dalpha = np.array([dalpha_dw, dalpha_dx, dalpha_dy, dalpha_dz]) @ dnormalization
                ddelta = np.array([ddelta_dw, ddelta_dx, ddelta_dy, ddelta_dz]) @ dnormalization
                print('da, dd = ', dalpha, ddelta)
                # print('da, dd = ', dalpha , ddelta @ dnormalization)
                print('da_fd, dd_fd = ', dalpha_fd, ddelta_fd)
                adiff.append(np.abs(dalpha_fd - dalpha))
                ddiff.append(np.abs(ddelta_fd - ddelta))
                # assert np.isclose(dalpha, dalpha_fd, rtol=1e-3)
                # assert np.isclose(ddelta, ddelta_fd, rtol=1e-3)

            adiff = np.array(adiff)
            ddiff = np.array(ddiff)
            from matplotlib import pyplot as plt 
            print(adiff)
            t = np.array([1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12])
            plt.figure(1000)
            plt.grid()
            plt.loglog(t, adiff, 'ro')
            plt.loglog(t, ddiff, 'bo')
            plt.show()

        hh = (np.random.rand(4) - 0.5)
        dofs = (np.random.rand(4) - 0.5) * 10   # quaternion dofs
        adiff = []
        ddiff = []
        dofs = (np.random.rand(4) - 0.5) * 10   # quaternion dofs
        normalization = np.sqrt(np.sum(dofs ** 2))
        dofs_unnormalized = np.copy(dofs)
        dofs = dofs / normalization  # normalize the quaternion
        w = dofs[0]
        x = dofs[1]
        y = dofs[2]
        z = dofs[3]
        alphas = np.arctan2(2 * (w * x + y * z), 
                                    1 - 2.0 * (x ** 2 + y ** 2))
        deltas = -np.pi / 2.0 + 2.0 * np.arctan2(
            np.sqrt(1.0 + 2 * (w * y - x * z)), 
            np.sqrt(1.0 - 2 * (w * y - x * z)))
        for eps in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
            # now try normalized in a general direction
            epsilon = eps * hh
            dofs2 = dofs_unnormalized + epsilon
            normalization2 = np.sqrt(np.sum(dofs2 ** 2))
            dofs2 = dofs2 / normalization2
            # print(normalization, normalization2, dofs, dofs2, np.sqrt(np.sum(dofs ** 2)), np.sqrt(np.sum(dofs2 ** 2)), epsilon)
            # sis = np.arctan2(2 * (w * z + y * x), 
            #                     1 - 2.0 * (z ** 2 + y ** 2))
            alphas2 = np.arctan2(2 * (dofs2[0] * dofs2[1] + dofs2[2] * dofs2[3]), 
                                1 - 2.0 * (dofs2[1] ** 2 + dofs2[2] ** 2))
            deltas2 = -np.pi / 2.0 + 2.0 * np.arctan2(
                np.sqrt(1.0 + 2 * (dofs2[0] * dofs2[2] - dofs2[1] * dofs2[3])), 
                np.sqrt(1.0 - 2 * (dofs2[0] * dofs2[2] - dofs2[1] * dofs2[3])))
            # ca = np.cos(alphas / 2.0)
            # sa = np.sin(alphas / 2.0)
            # cd = np.cos(deltas / 2.0)
            # sd = np.sin(deltas / 2.0)
            # cp = np.cos(sis / 2.0)
            # sp = np.sin(sis / 2.0)

            # Normalized derivative should be
            # dalpha/dofs = dalpha / normed * dnormed/dofs
            # normed_dofs = dofs / sqrt(dofs[0] ** 2 + ... )
            # wn = dofs[0] / normalization = w / normalization
            # dalpha_dw = dalpha_dwn * (dwn/dw) + dalpha_dxn * (dxn/dw) + ...
            # Changing one dof changes them all through the normalization!
            # print(normalization, dofs_unnormalized, dofs_unnormalized * dofs_unnormalized[i] / normalization ** 3)
            dnormalization = np.zeros((4, 4))
            for i in range(4):
                eye = np.zeros(4)
                eye[i] = 1.0
                dnormalization[:, i] = eye / normalization - dofs_unnormalized * dofs_unnormalized[i] / normalization ** 3
            # print('dnorm = ', dnormalization)
            dalpha_fd = (alphas2 - alphas) / eps
            ddelta_fd = (deltas2 - deltas) / eps
            print(alphas, alphas2, deltas, deltas2, epsilon)
            dalpha_dw = (2 * x * (-2 * (x ** 2 + y ** 2) + 1)) / \
                (4 * (w * x + y * z) ** 2 + (1 - 2 * ( x ** 2 + y ** 2)) ** 2)
            dalpha_dx = -(w * (-0.5 - x ** 2 + y ** 2) - 2 * x * y * z) / \
                (x ** 2 * (w ** 2 + 2 * y ** 2 - 1) + 2 * w * x * y * z + x ** 4 + y ** 4 + y ** 2 * (z ** 2 - 1) + 0.25)
            dalpha_dy = -(z * (-0.5 + x ** 2 - y ** 2) - 2 * x * y * w) / \
                (x ** 2 * (w ** 2 + 2 * y ** 2 - 1) + 2 * w * x * y * z + x ** 4 + y ** 4 + y ** 2 * (z ** 2 - 1) + 0.25)
            dalpha_dz = (2 * y * (1 - 2 * (x ** 2 + y ** 2))) / \
                (4 * (w * x + y * z) ** 2 + (1 - 2 * ( x ** 2 + y ** 2)) ** 2)
            ddelta_dw = y / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
            ddelta_dx = -z / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
            ddelta_dy = w / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
            ddelta_dz = -x / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
            dalpha = np.array([dalpha_dw, dalpha_dx, dalpha_dy, dalpha_dz]) @ dnormalization @ epsilon / eps
            ddelta = np.array([ddelta_dw, ddelta_dx, ddelta_dy, ddelta_dz]) @ dnormalization @ epsilon / eps
            print('da, dd = ', dalpha, ddelta)
            # print('da, dd = ', dalpha , ddelta @ dnormalization)
            print('da_fd, dd_fd = ', dalpha_fd, ddelta_fd)
            # assert np.isclose(dalpha, dalpha_fd, rtol=1e-3)
            # assert np.isclose(ddelta, ddelta_fd, rtol=1e-3)
            adiff.append(np.abs(dalpha_fd - dalpha))
            ddiff.append(np.abs(ddelta_fd - ddelta))

        adiff = np.array(adiff)
        ddiff = np.array(ddiff)
        from matplotlib import pyplot as plt 
        print(adiff)
        t = np.array([1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12])
        plt.figure(1000)
        plt.grid()
        plt.loglog(t, adiff, 'ro')
        plt.loglog(t, ddiff, 'bo')
        plt.show()

    def test_dpsi_ddofs(self):
        from simsopt.field import PSCCoil, psc_coils_via_symmetries
        t = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-12])
        for surf in surfs:
            print('Surf = ', surf)
            kwargs_manual = {"plasma_boundary": surf}
            psc_array = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
            )
            ncoils = psc_array.num_psc
            hh = (np.random.rand(ncoils, 4) - 0.5)
            dofs = np.array([psc_array.curves[i].get_dofs() for i in range(len(psc_array.curves))])
            dofs = dofs[:, 2 * psc_array.curves[0].order + 1:2 * psc_array.curves[0].order + 5]
            A = psc_array.A_matrix
            psi = psc_array.psi / psc_array.fac
            Linv = psc_array.L_inv[:psc_array.num_psc, :psc_array.num_psc] # / psc_array.fac

            # psc_array.I not trustworthy!
            I = (-Linv @ psi)
            normalization = np.sqrt(np.sum(dofs ** 2, axis=-1))
            normalization3 = normalization ** 3
            dofs_unnormalized = np.copy(dofs)
            dofs = dofs / normalization[:, None]  # normalize the quaternion
            w = dofs[:, 0]
            x = dofs[:, 1]
            y = dofs[:, 2]
            z = dofs[:, 3]
            alphas1 = np.arctan2(2 * (w * x + y * z), 
                                1 - 2.0 * (x ** 2 + y ** 2))
            deltas1 = -np.pi / 2.0 + 2.0 * np.arctan2(
                np.sqrt(1.0 + 2 * (w * y - x * z)), 
                np.sqrt(1.0 - 2 * (w * y - x * z)))

            dnormalization = np.ones((4, 4, dofs.shape[0]))
            for j in range(dofs.shape[0]):
                for i in range(4):
                    eye = np.zeros(4)
                    eye[i] = 1.0
                    dnormalization[:, i, j] = eye * normalization[j] - dofs_unnormalized[j, :] * dofs_unnormalized[j, i] * normalization3[j]
            # print('dnorm = ', dnormalization)

            def deuler_dquaternion():
                dalpha_dw = (2 * x * (-2 * (x ** 2 + y ** 2) + 1)) / \
                (4 * (w * x + y * z) ** 2 + (1 - 2 * ( x ** 2 + y ** 2)) ** 2)
                dalpha_dx = -(w * (-0.5 - x ** 2 + y ** 2) - 2 * x * y * z) / \
                    (x ** 2 * (w ** 2 + 2 * y ** 2 - 1) + 2 * w * x * y * z + x ** 4 + y ** 4 + y ** 2 * (z ** 2 - 1) + 0.25)
                dalpha_dy = -(z * (-0.5 + x ** 2 - y ** 2) - 2 * x * y * w) / \
                    (x ** 2 * (w ** 2 + 2 * y ** 2 - 1) + 2 * w * x * y * z + x ** 4 + y ** 4 + y ** 2 * (z ** 2 - 1) + 0.25)
                dalpha_dz = (2 * y * (1 - 2 * (x ** 2 + y ** 2))) / \
                    (4 * (w * x + y * z) ** 2 + (1 - 2 * ( x ** 2 + y ** 2)) ** 2)
                ddelta_dw = y / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
                ddelta_dx = -z / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
                ddelta_dy = w / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
                ddelta_dz = -x / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
                return dalpha_dw, dalpha_dx, dalpha_dy, dalpha_dz, ddelta_dw, ddelta_dx, ddelta_dy, ddelta_dz
        
            dalpha_dw, dalpha_dx, dalpha_dy, dalpha_dz, ddelta_dw, ddelta_dx, ddelta_dy, ddelta_dz = deuler_dquaternion()
            # dalpha_dw = (2 * x * (-2 * (x ** 2 + y ** 2) + 1)) / \
            # (4 * (w * x + y * z) ** 2 + (1 - 2 * ( x ** 2 + y ** 2)) ** 2)
            # dalpha_dx = -(w * (-0.5 - x ** 2 + y ** 2) - 2 * x * y * z) / \
            #     (x ** 2 * (w ** 2 + 2 * y ** 2 - 1) + 2 * w * x * y * z + x ** 4 + y ** 4 + y ** 2 * (z ** 2 - 1) + 0.25)
            # dalpha_dy = -(z * (-0.5 + x ** 2 - y ** 2) - 2 * x * y * w) / \
            #     (x ** 2 * (w ** 2 + 2 * y ** 2 - 1) + 2 * w * x * y * z + x ** 4 + y ** 4 + y ** 2 * (z ** 2 - 1) + 0.25)
            # dalpha_dz = (2 * y * (1 - 2 * (x ** 2 + y ** 2))) / \
            #     (4 * (w * x + y * z) ** 2 + (1 - 2 * ( x ** 2 + y ** 2)) ** 2)
            # ddelta_dw = y / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
            # ddelta_dx = -z / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
            # ddelta_dy = w / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
            # ddelta_dz = -x / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
            dalpha = np.array([dalpha_dw, dalpha_dx, dalpha_dy, dalpha_dz]).T
            ddelta = np.array([ddelta_dw, ddelta_dx, ddelta_dy, ddelta_dz]).T
            dI_diff = []
            adiff = []
            ddiff = []
            Idiff2 = []
            psi_diff = []
            for eps in t:
                print('eps = ', eps)
                epsilon = eps * hh
                dofs2 = dofs_unnormalized + epsilon
                normalization2 = np.sqrt(np.sum(dofs2 ** 2, axis=-1))
                dofs2 = dofs2 / normalization2[:, None]
                alphas2 = np.arctan2(2 * (dofs2[:, 0] * dofs2[:, 1] + dofs2[:, 2] * dofs2[:, 3]), 
                                    1 - 2.0 * (dofs2[:, 1] ** 2 + dofs2[:, 2] ** 2))
                deltas2 = -np.pi / 2.0 + 2.0 * np.arctan2(
                    np.sqrt(1.0 + 2 * (dofs2[:, 0] * dofs2[:, 2] - dofs2[:, 1] * dofs2[:, 3])), 
                    np.sqrt(1.0 - 2 * (dofs2[:, 0] * dofs2[:, 2] - dofs2[:, 1] * dofs2[:, 3])))
                psc_array_new = PSCgrid.geo_setup_manual(
                    points, R=R, a=a, alphas=alphas2, deltas=deltas2, **kwargs_manual
                )
                #######
                psc_array_new.L = psc_array.L  # absolutely mission critical to fix L here! 
                ######

                # print('dofs1, dofs2 = ', dofs, dofs2)
                # print('diff = ', dofs2 - dofs)
                psi_new = psc_array_new.psi / psc_array.fac

                #### Careful!!!!
                I_new = -psc_array.L_inv[:psc_array.num_psc, :psc_array.num_psc] @ psi_new  #####3 psc_array_new.I
                dpsi_fd = (psi_new - psi) / eps
                print('I1, I2 = ', I, I_new)
                dI_fd = (I_new - I) / eps
                dalpha_transformed = np.zeros(dalpha.shape[0])
                ddelta_transformed = np.zeros(dalpha.shape[0])
                for i in range(ncoils):
                    dalpha_transformed[i] = dalpha[i, :] @ dnormalization[:, :, i] @ epsilon[i, :] / eps
                    ddelta_transformed[i] = ddelta[i, :] @ dnormalization[:, :, i] @ epsilon[i, :] / eps
                    
                # print(dalpha.shape, epsilon.shape)
                # dalpha_transformed = np.sum(dalpha * epsilon, axis=-1) / eps
                # ddelta_transformed = np.sum(ddelta * epsilon, axis=-1) / eps
                dalpha_fd = (alphas2 - alphas1) / eps
                ddelta_fd = (deltas2 - deltas1) / eps
                # print('a1, d1 = ', alphas1, deltas1)
                # print('a2, d2 = ', alphas2, deltas2)
                # print('diff = ', alphas2 - alphas1, deltas2 - deltas1)
                print('da, dd = ', dalpha_transformed, ddelta_transformed)
                # print('da, dd = ', dalpha , ddelta @ dnormalization)
                print('da_fd, dd_fd = ', dalpha_fd, ddelta_fd)
                print('diff = ', (dalpha_transformed - dalpha_fd), (ddelta_transformed - ddelta_fd)) # / ddelta_fd)
                adiff.append(np.sum((dalpha_fd - dalpha_transformed) ** 2))
                ddiff.append(np.sum((ddelta_fd - ddelta_transformed) ** 2))
                # assert np.allclose(dalpha_transformed, dalpha_fd, rtol=1e-2)
                # assert np.allclose(ddelta_transformed, ddelta_fd, rtol=1e-2)
                psc_array.psi_deriv()
                dpsi = (psc_array.dpsi[:psc_array.num_psc] * dalpha_transformed + \
                    psc_array.dpsi[psc_array.num_psc:] * ddelta_transformed)
                psi_diff.append(np.sum((dpsi - dpsi_fd) ** 2))
                # print('dpsi = ', dpsi, dpsi_fd)
                # assert np.allclose(dpsi, dpsi_fd, rtol=1e-2)
                # print(Linv.shape, dpsi.shape, ncoils)
                Linv = psc_array.L_inv[:psc_array.num_psc, :psc_array.num_psc] # / psc_array.fac
                dI = - Linv @ dpsi 
                print('dI = ', dI, dI_fd)
                Idiff2.append(np.sum((dI - dI_fd) ** 2))
                # dI_fd_all = (psc_array_new.I_all - psc_array.I_all) / 1e-5
                # print('dI_all = ', dI_fd_all)
                # print(psc_array.L - psc_array_new.L)
                # assert np.allclose(dI, dI_fd, rtol=1e-1, atol=1e4)

                currents = []
                for i in range(len(psc_array.I)):
                    currents.append(Current(psc_array.I[i]))
                coils = psc_coils_via_symmetries(
                                    psc_array.curves, currents, nfp=psc_array.nfp, stellsym=psc_array.stellsym
                                )
                # bpsc = BiotSavart(coils)
                # bpsc.set_points(psc_array.plasma_boundary.gamma().reshape((-1, 3)))
                ndofs = 10
                dI = np.zeros((len(coils), ndofs))
                q = 0
                if surf.stellsym:
                    stellsym = [1, -1]
                else:
                    stellsym = [1]
                for fp in range(surf.nfp):
                    for stell in stellsym:
                        for i in range(psc_array.num_psc):
                            dI[i, :] += coils[i + q * psc_array.num_psc].curve.dkappa_dcoef_vjp(
                                [1.0], psc_array.dpsi) / (surf.nfp * len(stellsym))                            
                        q += 1
                Linv = psc_array.L_inv
                dI = dI[:, 2 * coils[0].curve.order + 1: 2 * coils[0].curve.order + 5]
                dpsi = np.zeros(len(coils))
                for i in range(len(coils)):
                    dpsi[i] = dI[i, :] @ epsilon[i % ncoils, :] / eps
                # Linv[coils[0].curve.npsc:, :] = 0.0
                dI = - Linv @ dpsi
                print('dI_coils = ', dI[:ncoils], dI_fd, (dI[:ncoils] - dI_fd))
                dI_diff.append(np.sum((dI[:ncoils] - dI_fd) ** 2))
                # assert np.allclose(dI[:ncoils], dI_fd, rtol=1e-1, atol=1e4)
            
            dI_diff = np.array(dI_diff)
            psi_diff = np.array(psi_diff)
            Idiff2 = np.array(Idiff2)
            adiff = np.array(adiff)
            ddiff = np.array(ddiff)
            from matplotlib import pyplot as plt 
            plt.figure(1000)
            plt.grid()
            plt.loglog(t, dI_diff, 'ro')
            # plt.loglog(t, Idiff2, 'go')
            plt.loglog(t, psi_diff, 'mo')
            plt.show()
            plt.figure(1001)
            plt.grid()
            plt.loglog(t, adiff, 'ro')
            plt.loglog(t, ddiff, 'bo')
            plt.show()


    def test_dJ_dgamma(self):
        from simsopt.field import PSCCoil, Coil
        from simsopt.objectives import SquaredFlux

        Jf1 = None
        h = (np.random.rand(ncoils, 4) - 0.5)
        for surf in surfs:
            for eps in [1e-1, 1e-3, 1e-5, 1e-7, 1e-9]:
                kwargs_manual = {"plasma_boundary": surf}
                psc_array = PSCgrid.geo_setup_manual(
                    points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
                    )
                dofs = np.array([psc_array.curves[i].get_dofs() for i in range(len(psc_array.curves))])
                dofs = dofs[:, 2 * psc_array.curves[0].order + 1:2 * psc_array.curves[0].order + 5]
                normalization = np.sqrt(np.sum(dofs ** 2, axis=-1))
                dofs = dofs / normalization[:, None]
                print('dofs = ', dofs, dofs.shape, normalization)
                epsilon = eps * h
                print('Surf = ', surf, ', eps = ', eps)
                # now try normalized in a general direction
                alphas1 = np.arctan2(2 * (dofs[:, 0] * dofs[:, 1] + dofs[:, 2] * dofs[:, 3]), 
                                    1 - 2.0 * (dofs[:, 1] ** 2 + dofs[:, 2] ** 2))
                deltas1 = -np.pi / 2.0 + 2.0 * np.arctan2(
                    np.sqrt(1.0 + 2 * (dofs[:, 0] * dofs[:, 2] - dofs[:, 1] * dofs[:, 3])), 
                    np.sqrt(1.0 - 2 * (dofs[:, 0] * dofs[:, 2] - dofs[:, 1] * dofs[:, 3])))
                print('a1, d1 = ', alphas1, deltas1)
                psc_array.setup_orientations(alphas1, deltas1)
                psc_array.update_psi()
                psc_array.setup_currents_and_fields()
                psc_array.psi_deriv()

                # psc_array.b_opt = np.zeros(psc_array.b_opt.shape)
                coils1 = [PSCCoil(curv, curr) for (curv, curr) in zip(psc_array.all_curves, psc_array.all_currents)]
                bpsc1 = BiotSavart(coils1)
                bpsc1.set_points(psc_array.plasma_boundary.gamma().reshape((-1, 3)))
                print(coils1[1]._curve.gamma()[0, :], coils1[1]._current.get_value(), bpsc1.B().reshape(-1, 3)[0, :])
                # print(bpsc1.B())
                Jf1 = SquaredFlux(psc_array.plasma_boundary, bpsc1)
                Jf1.x = np.ravel(dofs)
                print('why = ', Jf1.x.reshape(-1, 4), np.sum(Jf1.x.reshape(-1, 4) ** 2, axis=-1))
                Jf11 = Jf1.J()
                grad1 = Jf1.dJ()

                # dofs = np.array([psc_array.curves[i].get_dofs() for i in range(len(psc_array.curves))])
                # dofs = dofs[:, 2 * psc_array.curves[0].order + 1:2 * psc_array.curves[0].order + 5]
                # print(dofs, dofs.shape)
                dofs2 = dofs * normalization[:, None] + epsilon
                Jf1.x = np.ravel(dofs2)
                # now try normalized in a general direction
                normalization2 = np.sqrt(np.sum(dofs2 ** 2, axis=-1))
                dofs2_normed = dofs2 / normalization2[:, None]
                alphas2 = np.arctan2(2 * (dofs2_normed[:, 0] * dofs2_normed[:, 1] + dofs2_normed[:, 2] * dofs2_normed[:, 3]), 
                                    1 - 2.0 * (dofs2_normed[:, 1] ** 2 + dofs2_normed[:, 2] ** 2))
                deltas2 = -np.pi / 2.0 + 2.0 * np.arctan2(
                    np.sqrt(1.0 + 2 * (dofs2_normed[:, 0] * dofs2_normed[:, 2] - dofs2_normed[:, 1] * dofs2_normed[:, 3])), 
                    np.sqrt(1.0 - 2 * (dofs2_normed[:, 0] * dofs2_normed[:, 2] - dofs2_normed[:, 1] * dofs2_normed[:, 3])))
                # psc_array = coils1[0]._curve._psc_array
                psc_array.setup_orientations(alphas2, deltas2)
                psc_array.update_psi()
                psc_array.setup_currents_and_fields()
                psc_array.psi_deriv()
                psc_array.setup_curves()
                # print(coils1[1]._curve.gamma()[0, :], coils1[1]._current.get_value())
                Jf12 = Jf1.J()
                print(Jf1.x)
                print('d1_unnormed, d2_unnormed = ', dofs * normalization[:, None], dofs2)
                print(Jf11, Jf12)
                # grad2 = Jf1.dJ()
                dJ = grad1 @ np.ravel(epsilon) / eps
                print(dJ, (Jf12 - Jf11) / eps, ', err = ', (dJ - (Jf12 - Jf11) / eps))
                print('here1 = ', Jf1.x)
                assert np.allclose(dJ, (Jf12 - Jf11) / eps, rtol=1e-1)


    def test_dJ_dgamma_basic(self):
        from simsopt.field import PSCCoil
        from simsopt.objectives import SquaredFlux
        np.random.seed(1)
        h = np.random.uniform(size=(ncoils, 4))  # (np.random.rand(ncoils, 4) - 0.5)
        print(h)
        eps = 1e-4
        epsilon = eps * h
        for surf in surfs:
            kwargs_manual = {"plasma_boundary": surf}
            psc_array = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
                )
            dofs = np.array([psc_array.curves[i].get_dofs() for i in range(len(psc_array.curves))])
            dofs = dofs[:, 2 * psc_array.curves[0].order + 1:2 * psc_array.curves[0].order + 5]
            normalization = np.sqrt(np.sum(dofs ** 2, axis=-1))
            print('dofs = ', dofs, dofs.shape, normalization)
            dofs = dofs / normalization[:, None]
            dofs2 = dofs * normalization[:, None] + epsilon
            print('Surf = ', surf, ', eps = ', eps)
            # now try normalized in a general direction
            alphas1 = np.arctan2(2 * (dofs[:, 0] * dofs[:, 1] + dofs[:, 2] * dofs[:, 3]), 
                                1 - 2.0 * (dofs[:, 1] ** 2 + dofs[:, 2] ** 2))
            deltas1 = -np.pi / 2.0 + 2.0 * np.arctan2(
                np.sqrt(1.0 + 2 * (dofs[:, 0] * dofs[:, 2] - dofs[:, 1] * dofs[:, 3])), 
                np.sqrt(1.0 - 2 * (dofs[:, 0] * dofs[:, 2] - dofs[:, 1] * dofs[:, 3])))
            psc_array.setup_orientations(alphas1, deltas1)
            psc_array.update_psi()
            psc_array.setup_currents_and_fields()
            psc_array.psi_deriv()

            # psc_array.b_opt = np.zeros(psc_array.b_opt.shape)
            coils1 = [PSCCoil(curv, curr) for (curv, curr) in zip(psc_array.all_curves, psc_array.all_currents)]
            bpsc1 = BiotSavart(coils1)
            print(coils1[1]._curve.gamma()[0, :], coils1[1]._current.get_value())
            # bpsc1.set_points(psc_array.plasma_boundary.gamma().reshape((-1, 3)))
            Jf1 = SquaredFlux(psc_array.plasma_boundary, bpsc1 + psc_array.B_TF)
            Bn1 = psc_array.least_squares(np.hstack((alphas1, deltas1))) * psc_array.normalization
            print(Bn1)
            Jf11 = Jf1.J()
            grad1 = Jf1.dJ()

            dofs = np.array([psc_array.curves[i].get_dofs() for i in range(len(psc_array.curves))])
            dofs = dofs[:, 2 * psc_array.curves[0].order + 1:2 * psc_array.curves[0].order + 5]
            print(dofs, dofs2)
            Jf1.x = np.ravel(dofs2)
            # now try normalized in a general direction
            normalization2 = np.sqrt(np.sum(dofs2 ** 2, axis=-1))
            dofs2_normed = dofs2 / normalization2[:, None]
            alphas2 = np.arctan2(2 * (dofs2_normed[:, 0] * dofs2_normed[:, 1] + dofs2_normed[:, 2] * dofs2_normed[:, 3]), 
                                1 - 2.0 * (dofs2_normed[:, 1] ** 2 + dofs2_normed[:, 2] ** 2))
            deltas2 = -np.pi / 2.0 + 2.0 * np.arctan2(
                np.sqrt(1.0 + 2 * (dofs2_normed[:, 0] * dofs2_normed[:, 2] - dofs2_normed[:, 1] * dofs2_normed[:, 3])), 
                np.sqrt(1.0 - 2 * (dofs2_normed[:, 0] * dofs2_normed[:, 2] - dofs2_normed[:, 1] * dofs2_normed[:, 3])))
            # psc_array = coils1[0]._curve._psc_array
            psc_array.setup_orientations(alphas2, deltas2)
            psc_array.update_psi()
            psc_array.setup_currents_and_fields()
            psc_array.psi_deriv()
            psc_array.setup_curves()
            print(coils1[1]._curve.gamma()[0, :], coils1[1]._current.get_value())
            # print(coils1[1]._curve.gamma()[0, :], coils1[1]._current.get_value())
            Jf12 = Jf1.J()
            Bn2 = psc_array.least_squares(np.hstack((alphas2, deltas2))) * psc_array.normalization
            print(Bn1, Bn2, (Bn2 - Bn1) / eps)
            print(Jf11, Jf12)
            # dofs3 = np.array([psc_array.curves[i].get_dofs() for i in range(len(psc_array.curves))])
            # dofs3 = dofs3[:, 2 * psc_array.curves[0].order + 1:2 * psc_array.curves[0].order + 5]
            # print(dofs3 / dofs2, dofs3 / dofs, dofs3.shape)
            # psc_array_new = PSCgrid.geo_setup_manual(
            #     points, R=R, a=a, alphas=alphas2, deltas=deltas2, **kwargs_manual
            # )
            # psc_array_new.b_opt = np.zeros(psc_array.b_opt.shape)
            # psc_array_new.psi_deriv()
            # coils2 = [PSCCoil(curv, curr) for (curv, curr) in zip(psc_array_new.all_curves, psc_array_new.all_currents)]
            # coils2 = [PSCCoil(curv, curr) for (curv, curr) in zip(psc_array.all_curves, psc_array.all_currents)]
            # bpsc2 = BiotSavart(coils2)
            # bpsc2.set_points(psc_array.plasma_boundary.gamma().reshape((-1, 3)))
            # Jf2 = SquaredFlux(psc_array.plasma_boundary, bpsc2)
            # print(alphas1, deltas1)
            # print(alphas2, deltas2)
            grad2 = Jf1.dJ()
            dJ = grad1 @ np.ravel(epsilon) / eps
            print(dJ, (Jf12 - Jf11) / eps, ', err = ', (dJ - (Jf12 - Jf11) / eps))
            assert np.allclose(dJ, (Jf12 - Jf11) / eps, rtol=1e-1)


    ### Todo: write test of the jacobians of just regular coils, varying epsilon and the number of coils
    # and the type of plasma surface. No PSC functionality at all. Make sure that works first. 
    def test_dJ_dgamma_no_psc(self):
        from simsopt.field import PSCCoil, Coil, apply_symmetries_to_psc_curves, \
            apply_symmetries_to_curves, apply_symmetries_to_currents
        from simsopt.geo import CurvePlanarFourier, PSCCurve, curves_to_vtk, create_equally_spaced_planar_curves
        from simsopt.objectives import SquaredFlux
        from matplotlib import pyplot as plt
        np.random.seed(1)
        order = 1
        R = 0.025
        # a = 1e-5
        colors = ['r', 'b', 'g', 'm']
        for ncoils in [4]:  #, 5, 6, 7, 8, 23]:
            plt.figure(ncoils)
            # points = (np.random.rand(ncoils, 3) - 0.5) * 10
            # points[:, -1] = 0.4
            h = np.random.uniform(size=(ncoils, 10))  # (np.random.rand(ncoils, 4) - 0.5)
            for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
                epsilon = (eps * h)[:, 2 * order + 1: 2 * order + 5]
                for ii, surf in enumerate(surfs):
                    surf.to_vtk(str(surf))
                    print('ncoils = ', ncoils, ', surf = ', surf, ', eps = ', eps)
                    kwargs_manual = {"plasma_boundary": surf}
                    # dofs_orig = np.zeros((ncoils, 2 * order + 8))
                    # dofs_orig[:, 2 * order + 1: 2 * order + 5] = np.random.rand(ncoils, 4)
                    # dofs_orig[:, -3:] = points
                    # dofs_orig[:, 0] = R
                    curves = create_equally_spaced_planar_curves(ncoils, surf.nfp, stellsym=surf.stellsym, R0=2.0, R1=R, order=order)
                    dofs_orig = np.array([curves[i].get_dofs() for i in range(len(curves))])
                    # curves = [CurvePlanarFourier(order*100, order, nfp=1, stellsym=False) for i in range(ncoils)]
                    for ic in range(ncoils):
                        dofs1 = dofs_orig[ic, :]
                        for j in range(2 * order + 8):
                            curves[ic].set('x' + str(j), dofs1[j])
                        names_i = curves[ic].local_dof_names
                        curves[ic].fix(names_i[0])
                        curves[ic].fix(names_i[1])
                        curves[ic].fix(names_i[2])
                        
                        # Fix the center point for now
                        curves[ic].fix(names_i[7])
                        curves[ic].fix(names_i[8])
                        curves[ic].fix(names_i[9])
                        print(ic, curves[ic].dof_names)
                    currents = [Current(1.0) * 1e6 for i in range(ncoils)]
                    [currents[i].fix_all() for i in range(ncoils)]
                    dofs = dofs_orig[:, 2 * order + 1: 2 * order + 5]
                    normalization = np.sqrt(np.sum(dofs ** 2, axis=-1))
                    dofs2 = dofs + epsilon
                    coils1 = coils_via_symmetries(curves, currents, surf.nfp, surf.stellsym)
                    curves_to_vtk([coils1[i]._curve for i in range(len(coils1))], "curves_test", 
                                  close=True, scalar_data=[coils1[i]._current.get_value() for i in range(len(coils1))])
                    bpsc1 = BiotSavart(coils1)
                    bpsc1.set_points(surf.gamma().reshape((-1, 3)))
                    Jf1 = SquaredFlux(surf, bpsc1)
                    Jf11 = Jf1.J()
                    grad1 = Jf1.dJ()
                    Jf1.x = np.ravel(dofs2)
                    Jf12 = Jf1.J()
                    grad2 = Jf1.dJ()
                    dJ = grad1 @ np.ravel(epsilon) / eps
                    print(dJ, (Jf12 - Jf11) / eps, ', err = ', (dJ - (Jf12 - Jf11) / eps))
                    plt.loglog(eps, abs(dJ - (Jf12 - Jf11) / eps), 'o', color=colors[ii])
                    # assert np.allclose(dJ, (Jf12 - Jf11) / eps, rtol=1e-1)
            plt.grid()
            # plt.show()
        R = 0.1
        for ncoils in [4]:  #, 5, 6, 7, 8, 11
            plt.figure(ncoils + 1)
            points = (np.random.rand(ncoils, 3) + 0.2) * 5
            points[:, -1] = 0.4
            alphas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
            deltas = (np.random.rand(ncoils) - 0.5) * np.pi
            h = np.random.uniform(size=(ncoils, 10))
            for ii, surf in enumerate([surf3]):
                for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-12]:
                    kwargs_manual = {"plasma_boundary": surf}
                    psc_array = PSCgrid.geo_setup_manual(
                        points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
                        )
                    L1 = psc_array.L
                    psc_array.psi_deriv()
                    # print(psc_array.I)
                    curves = [psc_array.curves[i] for i in range(len(psc_array.curves))]
                    dofs_orig = np.array([curves[i].get_dofs() for i in range(len(curves))])
                    dofs = dofs_orig[:, 2 * order + 1: 2 * order + 5]
                    bpsc1 = PSC_BiotSavart(psc_array)
                    coils = bpsc1._coils
                    bpsc1.set_points(surf.gamma().reshape((-1, 3)))
                    Jf1 = SquaredFlux(surf, bpsc1)
                    Jf11 = Jf1.J()
                    grad1 = Jf1.dJ()

                    ndofs = 10
                    dI = np.zeros((len(coils), ndofs))
                    q = 0
                    if surf.stellsym:
                        stellsym = [1, -1]
                    else:
                        stellsym = [1]
                    for fp in range(surf.nfp):
                        for stell in stellsym:
                            for i in range(psc_array.num_psc):
                                dI[i, :] += coils[i + q * psc_array.num_psc].curve.dkappa_dcoef_vjp(
                                    [1.0], bpsc1.psc_array.dpsi) / (surf.nfp * len(stellsym))                     
                            q += 1
                    Linv = psc_array.L_inv
                    dI = dI[:, 2 * coils[0].curve.order + 1: 2 * coils[0].curve.order + 5]
                    # print('dI1 = ', dI)
                    dpsi = np.zeros(len(coils))
                    for i in range(len(coils)):
                        dpsi[i] = dI[i, :] @ h[i % ncoils, 2 * coils[0].curve.order + 1: 2 * coils[0].curve.order + 5]
                    # Linv[coils[0].curve.npsc:, :] = 0.0
                    dI = - Linv @ dpsi
                    print(dI)
                    q = 0
                    dI_temp = np.zeros(dI.shape)
                    for fp in range(surf.nfp):
                        for stell in stellsym:
                            dI_temp[:ncoils] += dI[q * ncoils:ncoils * (q + 1)]
                            q += 1
                    dI = dI_temp
                    L1_inv = bpsc1.psc_array.L_inv[:psc_array.num_psc, :]
                    I1 = -L1_inv @ psc_array.psi_total / psc_array.fac
                    epsilon = (eps * h)[:, 2 * order + 1: 2 * order + 5]
                    print('ncoils = ', ncoils, ', surf = ', surf, ', eps = ', eps)
                    dI2 = bpsc1.dpsi_debug
                    # print('dI2 = ', dI2)
                    dI2_temp = np.zeros(len(coils))
                    for i in range(len(coils)):
                        dI2_temp[i] = dI2[i, :] @ h[i % ncoils, 2 * coils[0].curve.order + 1: 2 * coils[0].curve.order + 5]
                    dI2_temp = -L1_inv @ dI2_temp
                    print('dI internal = ', dI2_temp)
                    dofs2 = dofs + epsilon
                    Jf1.x = np.ravel(dofs2)
                    bpsc1.set_currents()
                    L2 = psc_array.L
                    L2_inv = bpsc1.psc_array.L_inv[:psc_array.num_psc, :]
                    I2 = -L1_inv @ bpsc1.psc_array.psi_total / psc_array.fac
                    print('dI = ', I1, I2, I2 - I1, eps, (I2 - I1))  # Possible issue is that dI is enormous from step to step?
                    dI_fd = (I2 - I1) / eps
                    print('dI_coils = ', dI[:ncoils], dI_fd, (dI[:ncoils] - dI_fd))
                    print('dL = ', np.max(np.abs(L2 - L1)), np.max(np.abs(L2_inv - L1_inv)))


                    # bpsc1 = PSC_BiotSavart(psc_array)
                    # bpsc1.set_points(surf.gamma().reshape((-1, 3)))
                    # Jf1 = SquaredFlux(surf, bpsc1)

                    Jf12 = Jf1.J()
                    grad2 = Jf1.dJ()
                    dJ = grad1 @ np.ravel(epsilon) / eps
                    print('Jf = ', Jf11, Jf12, dJ, (Jf12 - Jf11) / eps, ', err = ', (dJ - (Jf12 - Jf11) / eps))
                    plt.loglog(eps, abs(dJ - (Jf12 - Jf11) / eps), 'o', color=colors[ii])
                    plt.loglog(eps, np.sum((dI[:ncoils] - dI_fd) ** 2), 'x', color=colors[ii])
                    # plt.loglog(eps, abs(dJ - (I2 - I1) / eps), 'o', color=colors[ii])
                    # print(dofs2, Jf1.x, bpsc1.x)
                    # Jf1.x = np.ravel(dofs)
                    # bpsc1.set_currents()
                    # print(dofs, Jf1.x, bpsc1.x)
            plt.grid()
        plt.show()


    def test_dJ_ddofs(self):
        from simsopt.field import PSCCoil
        from simsopt.objectives import SquaredFlux

        for surf in surfs:
            print('Surf = ', surf)
            kwargs_manual = {"plasma_boundary": surf}
            psc_array = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
            )
            psc_array.b_opt = np.zeros(psc_array.b_opt.shape)
            coils1 = [PSCCoil(curv, curr) for (curv, curr) in zip(psc_array.all_curves, psc_array.all_currents)]
            bpsc1 = BiotSavart(coils1)
            bpsc1.set_points(psc_array.plasma_boundary.gamma().reshape((-1, 3)))
            Jf1 = SquaredFlux(psc_array.plasma_boundary, bpsc1)

            dofs = np.array([psc_array.curves[i].get_dofs() for i in range(len(psc_array.curves))])
            dofs = dofs[:, 2 * psc_array.curves[0].order + 1:2 * psc_array.curves[0].order + 5]
            # print(dofs, dofs.shape)
            fB1 = psc_array.least_squares(psc_array.kappas) * psc_array.normalization
            A = psc_array.A_matrix
            psi = psc_array.psi / psc_array.fac
            Linv = psc_array.L_inv[:psc_array.num_psc, :psc_array.num_psc] # / psc_array.fac
            I = psc_array.I  #(-Linv @ psi)
            epsilon = 1e-5 * (np.random.rand(ncoils, 4) - 0.5)
            # epsilon = np.hstack((np.zeros(3), epsilon))
            # epsilon = np.hstack((epsilon, np.zeros(3)))
            dofs2 = dofs + epsilon
            # now try normalized in a general direction
            normalization2 = np.sqrt(np.sum(dofs2 ** 2, axis=-1))
            dofs2 = dofs2 / normalization2[:, None]
            alphas2 = np.arctan2(2 * (dofs2[:, 0] * dofs2[:, 1] + dofs2[:, 2] * dofs2[:, 3]), 
                                1 - 2.0 * (dofs2[:, 1] ** 2 + dofs2[:, 2] ** 2))
            deltas2 = -np.pi / 2.0 + 2.0 * np.arctan2(
                np.sqrt(1.0 + 2 * (dofs2[:, 0] * dofs2[:, 2] - dofs2[:, 1] * dofs2[:, 3])), 
                np.sqrt(1.0 - 2 * (dofs2[:, 0] * dofs2[:, 2] - dofs2[:, 1] * dofs2[:, 3])))
            psc_array_new = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas2, deltas=deltas2, **kwargs_manual
            )
            psc_array_new.b_opt = np.zeros(psc_array.b_opt.shape)
            coils2 = [PSCCoil(curv, curr) for (curv, curr) in zip(psc_array_new.all_curves, psc_array_new.all_currents)]
            # coils2 = [PSCCoil(curv, curr) for (curv, curr) in zip(psc_array_new.all_curves, psc_array.all_currents)]
            bpsc2 = BiotSavart(coils2)
            bpsc2.set_points(psc_array_new.plasma_boundary.gamma().reshape((-1, 3)))
            Jf2 = SquaredFlux(psc_array.plasma_boundary, bpsc2)
            fB2 = psc_array_new.least_squares(psc_array_new.kappas) * psc_array.normalization
            print('dfB = ', (fB2 - fB1) / 1e-5)
            psi_new = psc_array_new.psi / psc_array.fac
            I_new = psc_array_new.I
            dpsi_fd = (psi_new - psi) / 1e-5
            dI_fd = (I_new - I) / 1e-5
            dI1 = np.array([coils1[i].current.get_value() for i in range(len(coils1))])
            dI2 = np.array([coils2[i].current.get_value() for i in range(len(coils2))])
            print('dI2 - dI1 = ', (dI2 - dI1) / 1e-5)
            # print('dpsi_fd = ', dpsi_fd)
            normalization = np.sqrt(np.sum(dofs ** 2, axis=-1))
            dofs_unnormalized = np.copy(dofs)
            dofs = dofs / normalization[:, None]  # normalize the quaternion
            w = dofs[:, 0]
            x = dofs[:, 1]
            y = dofs[:, 2]
            z = dofs[:, 3]
            alphas1 = np.arctan2(2 * (w * x + y * z), 
                                1 - 2.0 * (x ** 2 + y ** 2))
            deltas1 = -np.pi / 2.0 + 2.0 * np.arctan2(
                np.sqrt(1.0 + 2 * (w * y - x * z)), 
                np.sqrt(1.0 - 2 * (w * y - x * z)))

            dnormalization = np.zeros((4, 4, dofs.shape[0]))
            for j in range(dofs.shape[0]):
                for i in range(4):
                    eye = np.zeros(4)
                    eye[i] = 1.0
                    dnormalization[:, i, j] = eye / normalization[j] - dofs_unnormalized[j, :] * dofs_unnormalized[j, i] / normalization[j] ** 3
            # print('dnorm = ', dnormalization)
            dalpha_fd = (alphas2 - alphas1) / 1e-5
            ddelta_fd = (deltas2 - deltas1) / 1e-5
            # print(alphas1, alphas2, deltas1, deltas2, epsilon)
            dalpha_dw = (2 * x * (-2 * (x ** 2 + y ** 2) + 1)) / \
                (4 * (w * x + y * z) ** 2 + (1 - 2 * ( x ** 2 + y ** 2)) ** 2)
            dalpha_dx = -(w * (-0.5 - x ** 2 + y ** 2) - 2 * x * y * z) / \
                (x ** 2 * (w ** 2 + 2 * y ** 2 - 1) + 2 * w * x * y * z + x ** 4 + y ** 4 + y ** 2 * (z ** 2 - 1) + 0.25)
            dalpha_dy = -(z * (-0.5 + x ** 2 - y ** 2) - 2 * x * y * w) / \
                (x ** 2 * (w ** 2 + 2 * y ** 2 - 1) + 2 * w * x * y * z + x ** 4 + y ** 4 + y ** 2 * (z ** 2 - 1) + 0.25)
            dalpha_dz = (2 * y * (1 - 2 * (x ** 2 + y ** 2))) / \
                (4 * (w * x + y * z) ** 2 + (1 - 2 * ( x ** 2 + y ** 2)) ** 2)
            ddelta_dw = y / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
            ddelta_dx = -z / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
            ddelta_dy = w / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
            ddelta_dz = -x / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
            
            # dalpha array is shape (8, 4) -- previously (4)
            # dnormalization is shape (4, 4, 8) -- previously (4, 4)
            # epsilon is shape (8, 4)  -- previously (4)
            dalpha = np.array([dalpha_dw, dalpha_dx, dalpha_dy, dalpha_dz]).T
            ddelta = np.array([ddelta_dw, ddelta_dx, ddelta_dy, ddelta_dz]).T
            dalpha_transformed = np.zeros(dalpha.shape[0])
            ddelta_transformed = np.zeros(dalpha.shape[0])
            for i in range(ncoils):
                dalpha_transformed[i] = dalpha[i, :] @ dnormalization[:, :, i] @ epsilon[i, :] / 1e-5
                ddelta_transformed[i] = ddelta[i, :] @ dnormalization[:, :, i] @ epsilon[i, :] / 1e-5
            # print('da, dd = ', dalpha_transformed, ddelta_transformed)
            # print('da, dd = ', dalpha , ddelta @ dnormalization)
            # print('da_fd, dd_fd = ', dalpha_fd, ddelta_fd)
            assert np.allclose(dalpha_transformed, dalpha_fd, rtol=1e-2)
            assert np.allclose(ddelta_transformed, ddelta_fd, rtol=1e-2)
            psc_array.psi_deriv()
            psc_array_new.psi_deriv()
            dpsi = (psc_array.dpsi[:psc_array.num_psc] * dalpha_transformed + \
                   psc_array.dpsi[psc_array.num_psc:] * ddelta_transformed)
            print('dpsi = ', dpsi, dpsi_fd)
            assert np.allclose(dpsi, dpsi_fd, rtol=1e-2)
            dI = - Linv @ dpsi 
            print('dI = ', dI, dI_fd)
            assert np.allclose(dI, dI_fd, rtol=1e-1, atol=1e4)

            coils = coils1
            order = coils[0].curve.order 
            ndofs = 2 * order + 8
            dI = np.zeros((len(coils), ndofs))
            # dI_deriv = []
            for i in range(coils[0].curve.npsc):
                dI[i, :] = coils[i].curve.dkappa_dcoef_vjp([1.0], coils[i].curve._index)
            Linv = coils[0].curve._psc_array.L_inv
            # Linv[coils[0].curve.npsc:, :] = 0.0
            dI = (- Linv @ dI)[:coils[0].curve.npsc, 2 * coils[0].curve.order + 1: 2 * coils[0].curve.order + 5]
            dI = np.sum(dI * epsilon, axis=-1) / 1e-5
            print('dI = ', dI, dI_fd)
            assert np.allclose(dI, dI_fd, rtol=1e-1, atol=1e4)

            dI = np.zeros((len(coils), ndofs))
            for i in range(len(coils)):
                dI[i, :] = coils[i].curve.dkappa_dcoef_vjp([1.0], coils[i].curve._index)
            # Linv[coils[0].curve.npsc:, :] = 0.0
            dI = (- Linv @ dI)[:coils[0].curve.npsc, 2 * coils[0].curve.order + 1: 2 * coils[0].curve.order + 5]
            dI = np.sum(dI * epsilon, axis=-1) / 1e-5
            print('dI = ', dI, dI_fd)
            assert np.allclose(dI, dI_fd, rtol=1e-1, atol=1e4)

            dI1 = np.array([bpsc1._coils[i].current.get_value() for i in range(len(coils))])
            dI2 = np.array([bpsc2._coils[i].current.get_value() for i in range(len(coils))])
            print('dI_12 = ', (dI2 - dI1) / 1e-5)

            # coils1 = [PSCCoil(curv, curr) for (curv, curr) in zip(psc_array.all_curves, psc_array.all_currents)]
            # bpsc1 = BiotSavart(coils1)
            # bpsc1.set_points(psc_array.plasma_boundary.gamma().reshape((-1, 3)))
            # coils2 = [PSCCoil(curv, curr) for (curv, curr) in zip(psc_array_new.all_curves, psc_array_new.all_currents)]
            # bpsc2 = BiotSavart(coils2)
            # bpsc2.set_points(psc_array_new.plasma_boundary.gamma().reshape((-1, 3)))
            # dI = np.zeros((len(coils), ndofs))
            # for i in range(len(coils)):
            #     dI[i, :] = coils[i].curve.dkappa_dcoef_vjp([1.0], coils[i].curve._index)
            # # Linv[coils[0].curve.npsc:, :] = 0.0
            # dI = (- Linv @ dI)[:coils[0].curve.npsc, 2 * coils[0].curve.order + 1: 2 * coils[0].curve.order + 5]
            # dB_dgamma = np.array([bpsc1._coils[i].curve.dgamma_by_dcoeff_vjp_impl([1.0]) for i in range(len(coils))]
            #                      )[:coils[0].curve.npsc, 2 * coils[0].curve.order + 1: 2 * coils[0].curve.order + 5]
            # dB_dgammadash = np.array([bpsc1._coils[i].curve.dgammadash_by_dcoeff_vjp_impl([1.0]) for i in range(len(coils))]
            #                          )[:coils[0].curve.npsc, 2 * coils[0].curve.order + 1: 2 * coils[0].curve.order + 5]
            # dB_full = np.sum((dI + dB_dgamma + dB_dgammadash) * epsilon, axis=-1) / 1e-5
            # print(dB_full.shape, bpsc1._coils[0].curve.dgammadash_by_dcoeff_vjp_impl([1.0]))
            # print(dB_dgamma.shape, dB_dgammadash.shape, bpsc1._coils[0].curve.gamma().shape)
            # exit()
            # dB_no_dI = dB_dgamma * ( )

            # dI = np.zeros((len(coils), ndofs))
            # for i in range(len(coils)):
            #     dI[i, :] = coils[i].curve.dkappa_dcoef_vjp([1.0], coils[i].curve._index)
            dI = np.zeros((coils[0].curve.npsc, ndofs))
            for i in range(coils[0].curve.npsc):
                dI[i, :] = coils[i].curve.dkappa_dcoef_vjp([1.0], coils[i].curve._index)
            dI_all = np.zeros((len(coils), ndofs))
            eps_all = np.zeros((len(coils), 4))
            q = 0
            for fp in range(psc_array.nfp):
                for stell in psc_array.stell_list:
                    eps_all[q * coils[0].curve.npsc: (q + 1) * coils[0].curve.npsc, :] = epsilon
                    dI_all[q * coils[0].curve.npsc: (q + 1) * coils[0].curve.npsc, :] = dI * stell
                    q += 1
            dI = dI_all
            # print('dI_check1 = ', dI[:, 2 * coils[0].curve.order + 1: 2 * coils[0].curve.order + 5])
            # Linv[coils[0].curve.npsc:, :] = 0.0
            dI = (- Linv @ dI)[:, 2 * coils[0].curve.order + 1: 2 * coils[0].curve.order + 5]
            # print('dI_check2 = ', dI)
           
                # print(q, epsilon.shape)
            dI = np.sum(dI * eps_all, axis=-1) / 1e-5
            print('dI_check3 = ', dI)
            # assert np.allclose(dI, (dI2 - dI1) / 1e-5, rtol=1e-1)
            dB_dI = np.array(bpsc1.dB_by_dcoilcurrents())
            # print('dB = ', dB_dI.shape, dI.shape)
            dB = np.tensordot(dI, dB_dI, axes=([0, 0]))
            coils3 = [PSCCoil(curv, curr) for (curv, curr) in zip(psc_array.all_curves, psc_array_new.all_currents)]
            bpsc3 = BiotSavart(coils3)
            bpsc3.set_points(psc_array_new.plasma_boundary.gamma().reshape((-1, 3)))
            dB_fd = (bpsc3.B() - bpsc1.B()) / 1e-5
            # for q in range(1, psc_array.symmetry):
            #     dI = np.vstack((dI, dI))
            #     epsilon = np.vstack((epsilon, epsilon))
            # print(dI.shape, epsilon.shape)
            dB_fd2 = np.tensordot(dI2 - dI1, dB_dI, axes=([0, 0])) / 1e-5
            print('dB_diff = ', np.max(np.abs(dB / dB_fd)), dB[0, :], dB_fd2[0, :])
            # assert np.allclose(dB, dB_fd, rtol=1)
            # exit()

            # dB_fd = (bpsc2.B() - bpsc1.B()).reshape(-1, 3) / 1e-5
            # print(np.array(bpsc2.dB_by_dcoilcurrents()).shape, dI.shape)
            # dB = np.sum(dI[:, None, None] * np.array(bpsc2.dB_by_dcoilcurrents()), axis=0)
            # print(dB)
            # all_curves = apply_symmetries_to_psc_curves(psc_array.curves, )
            # bpsc = BiotSavart(coils)
            # bpsc.set_points(psc_array.plasma_boundary.gamma().reshape((-1, 3)))
            # Jf = SquaredFlux(psc_array.plasma_boundary, psc_array.B_TF + bpsc)
            # dB = bpsc.B_vjp(psc_array.plasma_boundary.gamma().reshape((-1, 3)))  #(bpsc)

            # dB_dgamma = [bpsc._coils[i].curve.dgamma_by_dcoeff_vjp_impl(np.ones(3)) + bpsc._coils[i].curve.dgammadash_by_dcoeff_vjp_impl(np.ones(3)) for i in range(len(coils))]
            # dB_dgamma = np.sum(np.array(dB_dgamma) * epsilon, axis=-1)
            # dB_dgamma = np.sum(dB_dgamma[:, None, None] * np.array(bpsc2.dB_by_dcoilcurrents()), axis=0)
            # print('dB_dgamma = ', np.shape(dB_dgamma))
            # print('dB = ', dB / dB_fd)
            # print('dB = ', np.shape(dB), dB_fd.shape)
            # print(psc_array.A_matrix.shape, dalpha_transformed.shape)
            # exit()
            dA_fd = ((psc_array_new.A_matrix - psc_array.A_matrix)) / 1e-5
            dA = psc_array.A_deriv()[:, :psc_array.num_psc] * dalpha_transformed + \
                psc_array.A_deriv()[:, psc_array.num_psc:] * ddelta_transformed 
            # print(dA.shape, dA_fd.shape)
            # print('dA = ', dA / dA_fd)
            print('df_direct = ', (Jf2.J() - Jf1.J()) / 1e-5)
            # Learned a few things: A_deriv calculation not looking quite right for some coils.
            # still does not explain why dgamma and dgammadash calculations do not account
            # for the full jacobian for the coils. And fact that FDs and analytic jacobian agree 
            # when either no currents are optimized, or only the currents are optimized, seems to 
            # point to a missing term in the Jacobian. Seems like before, gamma, gammdash, and I 
            # could be considered independent variables, but now gamma and gammadash certainly
            # affect the calculation of I. But the I derivative is correct when dgamma and 
            # dgammadash are removed and the A_matrix is kept the same. This is all consistent
            # in there being extra terms looking like dI / dgamma * dgamma / ddofs.
            # You can actually think about these terms as coming from a modified dB / dgamma calculation.
            # Not sure how this is consistent with the psc_array results though. But seems like in both cases
            # it is B = A * I and dB = dA * I + A * dI and the dI piece is correct, and the dA piece seems
            # like it shouldn't have changed since it is independent of I. And indeed when I is unchanged it 
            # seems to be correct.

            # dA = np.array([coils[i].curve.dgamma_by_dcoeff_vjp_impl(np.ones((256, 3))) + \
            #        coils[i].curve.dgammadash_by_dcoeff_vjp_impl(np.ones((256, 3))) for i in range(len(coils))])
            # print(dA.shape)
            # exit()
            Bn = np.sum((bpsc1.B()).reshape(-1, 3) * psc_array.plasma_unitnormals, axis=-1) * psc_array.grid_normalization
            f1 = Bn.T @ Bn * 0.5 
            Bn = np.sum((bpsc2.B()).reshape(-1, 3) * psc_array.plasma_unitnormals, axis=-1) * psc_array.grid_normalization
            f2 = Bn.T @ Bn * 0.5 
            print('df_direct2 = ', (f2 - f1) / 1e-5)
            Bn = psc_array.A_matrix @ psc_array.I * psc_array.grid_normalization
            f1 = Bn.T @ Bn * 1e-14 * 0.5
            Bn = psc_array.A_matrix @ psc_array_new.I * psc_array.grid_normalization
            f2 = Bn.T @ Bn * 1e-14 * 0.5
            print('df_direct3 = ', (f2 - f1) / 1e-5)
            jac = psc_array.least_squares_jacobian(psc_array.kappas) * psc_array.normalization
            jac = jac[:psc_array.num_psc] @ dalpha_transformed + jac[psc_array.num_psc:] @ ddelta_transformed
            print('df_from_array = ', jac)
            # Dont have the jacobian with respect to the dofs, only with respect to alpha and delta
            # so need to convert it
            grad = Jf1.dJ()
            grad2 = Jf2.dJ()
            # print(grad.shape, epsilon.shape, grad, grad2)
            # print('dI_all = ', np.sum(bpsc1.dI_psc * epsilon, axis=-1) / 1e-5)
            # print('I_internal = ', bpsc1.dI_psc, np.sum(bpsc1.dI_psc * epsilon, axis=-1) / 1e-5)
            # print(Jf2.dof_names)
            dJ = grad @ np.ravel(epsilon) / 1e-5
            print(dJ, (fB2 - fB1) / 1e-5)
            print('err = ', (dJ - (f2 - f1) / 1e-5))
            assert np.allclose(dJ, (fB2 - fB1) / 1e-5)

        
    def test_dpsi_analytic_derivatives(self):
        """
        Tests the analytic calculations of dpsi/dalpha and dpsi/ddelta
        against finite differences for tiny changes to the angles, 
        to see if the analytic calculations are correct.
        """
        for surf in surfs:
            print('Surf = ', surf)
            kwargs_manual = {"plasma_boundary": surf}
            psc_array = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
            )
            A = psc_array.A_matrix
            psi = psc_array.psi / psc_array.fac
            b = psc_array.b_opt # * psc_array.fac
            Linv = psc_array.L_inv[:psc_array.num_psc, :psc_array.num_psc] # / psc_array.fac
            I = (-Linv @ psi)
            Bn_objective = 0.5 * (A @ I + b).T @ (A @ I + b)
            deltas_new = np.copy(deltas)
            dd = epsilon * (np.random.rand(deltas.shape[0]) - 0.5)
            deltas_new += dd
            alphas_new = np.copy(alphas)
            psc_array_new = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas_new, deltas=deltas_new, **kwargs_manual
            )
            psi_new = psc_array_new.psi / psc_array.fac
            I_new = (-Linv @ psi_new)
            # print(I_new, -psc_array.L_inv @ psc_array.psi_total)
            Bn_objective_new = 0.5 * (A @ I_new + b).T @ (A @ I_new + b)
            dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
            dpsi_ddelta = (psc_array_new.psi - psc_array.psi) / epsilon  / psc_array.fac
            psi_deriv = psc_array.psi_deriv()  # * 1e-7
            psi_deriv = psi_deriv[psc_array.num_psc:] * dd / epsilon
            print(dpsi_ddelta, psi_deriv)
            # assert(np.allclose(dpsi_ddelta, psi_deriv, rtol=1e-3))
            # dBn_analytic = (A @ I + b).T @ (-A @ (Linv * psi_deriv[ncoils:]))
            # print(dBn_objective, dBn_analytic[0])
            # assert np.isclose(dBn_objective, dBn_analytic[0], rtol=1e-2)

            # dI = (I_new - I) / epsilon
            # dI_analytic = -Linv @ psi_deriv[ncoils:]
            # print('dI = ', dI, dI.shape, dI_analytic, dI_analytic.shape)

            psc_array = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
            )
            A = psc_array.A_matrix
            psi = psc_array.psi / psc_array.fac
            b = psc_array.b_opt # * psc_array.fac
            Linv = psc_array.L_inv[:psc_array.num_psc, :psc_array.num_psc] # / psc_array.fac
            I = (-Linv @ psi)
            Bn_objective = 0.5 * (A @ I + b).T @ (A @ I + b)
            deltas_new = np.copy(psc_array.deltas)
            dd = epsilon * (np.random.rand(deltas.shape[0]) - 0.5)
            alphas_new = np.copy(psc_array.alphas)
            alphas_new += dd
            psc_array_new = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas_new, deltas=deltas_new, **kwargs_manual
            )
            psi_new = psc_array_new.psi / psc_array.fac
            I_new = (-Linv @ psi_new)
            # print(I_new, -psc_array.L_inv @ psc_array.psi_total)
            Bn_objective_new = 0.5 * (A @ I_new + b).T @ (A @ I_new + b)
            dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
            dpsi_ddelta = (psc_array_new.psi - psc_array.psi) / epsilon  / psc_array.fac
            psi_deriv = psc_array_new.psi_deriv()  # * 1e-7
            psi_deriv = psi_deriv[:psc_array.num_psc] * dd / epsilon
            print('dpsi = ', dpsi_ddelta, psi_deriv)
            assert(np.allclose(dpsi_ddelta, psi_deriv, rtol=1e-3))
            # dBn_analytic = (A @ I + b).T @ (-A @ (Linv * psi_deriv[ncoils:]))
            # print(dBn_objective, dBn_analytic[0])
            # assert np.isclose(dBn_objective, dBn_analytic[0], rtol=1e-2)

            # Same arguments apply here as with the quaternions --- computing
            # dI/dalpha_j = dI/dalpha_
            
            # Repeat for changing coil 1
            psc_array = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
            )
            A = psc_array.A_matrix
            psi = psc_array.psi / psc_array.fac
            Linv = psc_array.L_inv[:psc_array.num_psc, :psc_array.num_psc]
            I = (-Linv @ psi)
            Bn_objective = 0.5 * (A @ I + b).T @ (A @ I + b)
            alphas_new = np.copy(alphas)
            alphas_new[1] += epsilon
            deltas_new = np.copy(deltas)
            psc_array_new = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas_new, deltas=deltas_new, **kwargs_manual
            )
            psi_new = psc_array_new.psi / psc_array.fac
            I_new = (-Linv @ psi_new)
            Bn_objective_new = 0.5 * (A @ I_new + b).T @ (A @ I_new + b)
            dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
            dpsi_dalpha = (psi_new - psi) / epsilon 
            psi_deriv = psc_array_new.psi_deriv() # * 1e-7
            print(dpsi_dalpha[1], psi_deriv[1])
            assert(np.allclose(dpsi_dalpha[1], psi_deriv[1], rtol=1e-3))
            dBn_analytic = (A @ I + b).T @ (-A @ (Linv * psi_deriv[:ncoils]))
            # print(dBn_objective, dBn_analytic[1])
            assert np.isclose(dBn_objective, dBn_analytic[1], rtol=1e-1)
    
    def test_L_analytic_derivatives(self):
        """
        Tests the analytic calculations of dL/dalpha and dL/ddelta
        against finite differences for tiny changes to the angles, 
        to see if the analytic calculations are correct.
        """
        for surf in surfs:
            print('Surf = ', surf)
            kwargs_manual = {"plasma_boundary": surf}
            psc_array = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
            )
            A = psc_array.A_matrix
            psi = psc_array.psi_total 
            b = psc_array.b_opt * psc_array.fac
            L = psc_array.L
            Linv = psc_array.L_inv
            I = (-Linv @ psi)[:psc_array.num_psc]
            Bn_objective = 0.5 * (A @ I + b).T @ (A @ I + b)
            L_deriv = psc_array.L_deriv()
            deltas_new = np.copy(deltas)
            deltas_new[0] += epsilon
            alphas_new = np.copy(alphas)
            psc_array_new = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas_new, deltas=deltas_new, **kwargs_manual
            )
            L_new = psc_array_new.L
            Linv_new = psc_array_new.L_inv
            I_new = (-Linv_new @ psi)[:psc_array.num_psc]
            Bn_objective_new = 0.5 * (A @ I_new + b).T @ (A @ I_new + b)
            dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
            dL_ddelta = (L_new - L) / epsilon
            dLinv_ddelta = (Linv_new - Linv) / epsilon
            ncoils_sym = L_deriv.shape[0] // 2
            dL_ddelta_analytic = L_deriv
            dLinv_ddelta_analytic = -L_deriv @ Linv @ Linv

            # Linv calculations looks much more incorrect that the L derivatives,
            # maybe because of numerical error accumulation? 
            print(np.max(np.abs(dL_ddelta - dL_ddelta_analytic[ncoils_sym, :, :])))
            assert(np.allclose(dL_ddelta, dL_ddelta_analytic[ncoils_sym, :, :], rtol=1e-1))
            dBn_analytic = (A @ I + b).T @ (-A @ (dLinv_ddelta_analytic @ psi)[:, :psc_array.num_psc].T)
            print(dBn_objective, dBn_analytic[ncoils_sym])
            assert np.isclose(dBn_objective, dBn_analytic[ncoils_sym], rtol=1e-1)
            
            # Repeat changing coil 1
            psc_array = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
            )
            L = psc_array.L
            Linv = psc_array.L_inv
            I = (-Linv @ psi)[:psc_array.num_psc]
            Bn_objective_new = 0.5 * (A @ I + b).T @ (A @ I + b)
            L_deriv = psc_array.L_deriv()
            alphas_new = np.copy(alphas)
            alphas_new[1] += epsilon
            deltas_new = np.copy(deltas)
            psc_array_new = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas_new, deltas=deltas_new, **kwargs_manual
            )
            L_new = psc_array_new.L
            Linv_new = psc_array_new.L_inv
            I_new = (-Linv_new @ psi)[:psc_array.num_psc]
            Bn_objective_new = 0.5 * (A @ I_new + b).T @ (A @ I_new + b)
            dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
            dL_ddelta = (L_new - L) / epsilon
            dLinv_ddelta = (Linv_new - Linv) / epsilon
            dL_ddelta_analytic = L_deriv
            dLinv_ddelta_analytic = -L_deriv @ Linv @ Linv
                
            # print(np.array_str(dL_ddelta * 1e7, precision=3, suppress_small=True))
            # print(np.array_str(L_deriv[1, :, :] * 1e7, precision=3, suppress_small=True))
            print(np.max(np.abs(dL_ddelta - dL_ddelta_analytic[ncoils_sym, :, :])))
            assert(np.allclose(dL_ddelta, dL_ddelta_analytic[1, :, :], rtol=1e-1))
            assert(np.allclose(dLinv_ddelta, dLinv_ddelta_analytic[1, :, :], rtol=10))
            dBn_analytic = (A @ I + b).T @ (-A @ (dLinv_ddelta_analytic @ psi)[:, :psc_array.num_psc].T)
            print(dBn_objective, dBn_analytic[1])
            assert np.isclose(dBn_objective, dBn_analytic[1], rtol=1e-1)
        
    def test_symmetrized_quantities(self):
        """
        Tests that discrete symmetries are satisfied by the various terms
        appearing in optimization. 
        """
        from simsopt.objectives import SquaredFlux
        from simsopt.field import coils_via_symmetries
        
        input_name = 'input.LandremanPaul2021_QA_lowres'
        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
        surface_filename = TEST_DIR / input_name
        nphi = 32
        surf = SurfaceRZFourier.from_vmec_input(
            surface_filename, range='half period', nphi=nphi, ntheta=nphi
        )
        surf_full = SurfaceRZFourier.from_vmec_input(
            surface_filename, range='full torus', nphi=nphi * 4, ntheta=nphi * 4
        )
        ncoils = 6
        np.random.seed(1)
        R = 0.2
        a = 1e-5
        points = (np.random.rand(ncoils, 3) - 0.5) * 20
        alphas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        deltas = (np.random.rand(ncoils) - 0.5) * np.pi
        kwargs_manual = {"plasma_boundary": surf}
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
        )
        psc_array.I = 1e4 * np.ones(len(psc_array.I))
        Ax_b = (psc_array.A_matrix @ psc_array.I + psc_array.b_opt) * psc_array.grid_normalization
        Bn = 0.5 * Ax_b.T @ Ax_b / psc_array.normalization * psc_array.fac ** 2
        
        currents = []
        for i in range(psc_array.num_psc):
            currents.append(Current(psc_array.I[i]))
        all_coils = coils_via_symmetries(
            psc_array.curves, currents, nfp=surf.nfp, stellsym=surf.stellsym
        )
        B_WP = BiotSavart(all_coils)

        fB = SquaredFlux(surf, B_WP + psc_array.B_TF, np.zeros((nphi, nphi))).J() / psc_array.normalization
        # print(Bn, fB)
        assert np.isclose(Bn, fB, rtol=1e-3)
        fB_full = SquaredFlux(surf_full, B_WP + psc_array.B_TF, np.zeros((nphi * 4, nphi * 4))).J() / psc_array.normalization
        assert np.isclose(Bn, fB_full, rtol=1e-3)
        
    def test_dA_analytic_derivatives(self):
        """
        Tests the analytic calculations of dA/dalpha and dA/ddelta
        against finite differences for tiny changes to the angles, 
        to see if the analytic calculations are correct.
        """
        for surf in surfs:
            print('Surf = ', surf)
            kwargs_manual = {"plasma_boundary": surf}
            psc_array = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
            )
            A = psc_array.A_matrix
            Linv = psc_array.L_inv
            psi = psc_array.psi_total
            I = (-Linv @ psi)[:psc_array.num_psc]
            b = psc_array.b_opt * psc_array.fac
            Bn_objective = 0.5 * (A @ I + b).T @ (A @ I + b)
            alphas_new = np.copy(alphas)
            deltas_new = np.copy(deltas)
            deltas_new[0] += epsilon
            psc_array_new = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas_new, deltas=deltas_new, **kwargs_manual
            )
            A_new = psc_array_new.A_matrix
            Bn_objective_new = 0.5 * (A_new @ I + b).T @ (A_new @ I + b)
            dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
            A_deriv = psc_array.A_deriv()
            dBn_analytic = (A @ I + b).T @ (A_deriv[:, ncoils:] * I)
            print(dBn_objective, dBn_analytic[0])
            dA_dalpha = (A_new - A) / epsilon
            dA_dkappa_analytic = A_deriv
            print('dA0 = ', dA_dalpha[0, :], dA_dkappa_analytic[0, :])
            # print('dA0 = ', dA_dalpha[:, 0] / dA_dkappa_analytic[:, ncoils])
            assert np.isclose(dBn_objective, dBn_analytic[0], rtol=1e-1)
            assert np.allclose(dA_dalpha[:, 0], dA_dkappa_analytic[:, ncoils], rtol=1)
            
            # Repeat but change coil 3
            A = psc_array.A_matrix
            Linv = psc_array.L_inv
            psi = psc_array.psi_total
            I = (-Linv @ psi)[:psc_array.num_psc]
            Bn_objective = 0.5 * (A @ I + b).T @ (A @ I + b)
            alphas_new = np.copy(alphas)
            alphas_new[1] += epsilon
            deltas_new = np.copy(deltas)
            psc_array_new = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas_new, deltas=deltas_new, **kwargs_manual
            )
            A_new = psc_array_new.A_matrix
            Bn_objective_new = 0.5 * (A_new @ I + b).T @ (A_new @ I + b)
            dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
            A_deriv = psc_array.A_deriv()
            dBn_analytic = (A @ I + b).T @ (A_deriv[:, :ncoils] * I)
            print(dBn_objective, dBn_analytic[1])
            assert np.isclose(dBn_objective, dBn_analytic[1], rtol=1)
            dA_dalpha = (A_new - A) / epsilon
            dA_dkappa_analytic = A_deriv
            print('dA0 = ', dA_dalpha[0, 1] / dA_dkappa_analytic[0, 1])
            assert np.allclose(dA_dalpha[:, 1], dA_dkappa_analytic[:, 1], rtol=10)

    def test_L(self):
        """
        Tests the inductance calculation for some limiting cases:
            1. Identical, coaxial coils 
            (solution in Jacksons textbook, problem 5.28)
        and tests that the inductance and flux calculations agree,
        when we use the "TF field" as one of the coil B fields
        and compute the flux through the other coil, for coils with random 
        separations, random orientations, etc.
        """
    
        from scipy.special import ellipk, ellipe
        
        np.random.seed(1)
        
        # initialize coaxial coils
        R0 = 4
        R = 0.1
        a = 1e-5
        mu0 = 4 * np.pi * 1e-7
        points = np.array([[R0, 0.5 * R0, 0], [R0, 0.5 * R0, R0]])
        alphas = np.zeros(2)
        deltas = np.zeros(2)
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas
        )
        L = psc_array.L * 1e-7
        L_self_analytic = mu0 * R * (np.log(8.0 * R / a) - 2.0)
        
        # Jackson problem 5.28
        k = np.sqrt(4.0 * R ** 2 / (4.0 * R ** 2 + R0 ** 2))
        # Jackson uses K(k) and E(k) but this corresponds to
        # K(k^2) and E(k^2) in scipy library
        L_mutual_analytic = mu0 * R * (
            (2 / k - k) * ellipk(k ** 2) - (2 / k) * ellipe(k ** 2)
        )
        assert(np.allclose(L_self_analytic, np.diag(L)))
        print(L_mutual_analytic * 1e10, L[0, 1] * 1e10, L)
        assert(np.isclose(L_mutual_analytic * 1e10, L[0, 1] * 1e10, rtol=1e-3))
        assert(np.isclose(L_mutual_analytic * 1e10, L[1, 0] * 1e10, rtol=1e-3))
        
        # Another simple test of the analytic formula   
        # Formulas only valid for R/a >> 1 so otherwise it will fail
        # R = 1
        # a = 1e-6
        # R0 = 1
        # mu0 = 4 * np.pi * 1e-7
        # points = np.array([[0, 0, 0], [0, 0, R0]])
        # alphas = np.zeros(2)
        # deltas = np.zeros(2)
        # psc_array = PSCgrid.geo_setup_manual(
        #     points, R=R, a=a, alphas=alphas, deltas=deltas,
        # )
        # L = psc_array.L * 1e-7
        # k = np.sqrt(4.0 * R ** 2 / (4.0 * R ** 2 + R0 ** 2))
        # L_mutual_analytic = mu0 * R * (
        #     (2 / k - k) * ellipk(k ** 2) - (2 / k) * ellipe(k ** 2)
        # )
        # L_self_analytic = mu0 * R * (np.log(8.0 * R / a) - 2.0)
        # assert(np.allclose(L_self_analytic, np.diag(L)))
        # print(L_mutual_analytic * 1e10, L[0, 1] * 1e10)
        # assert(np.isclose(L_mutual_analytic * 1e10, L[0, 1] * 1e10, rtol=1e-1))
        # assert(np.isclose(L_mutual_analytic * 1e10, L[1, 0] * 1e10, rtol=1e-1))
        I = 1e10
        # coils = coils_via_symmetries([psc_array.curves[1]], [Current(I)], nfp=1, stellsym=False)
        # bs = BiotSavart(coils)
        # center = np.array([0, 0, 0]).reshape(1, 3)
        # bs.set_points(center)
        # B_center = bs.B()
        # Bz_center = B_center[:, 2]
        # kwargs = {"B_TF": bs, "ppp": 1000}
        # psc_array = PSCgrid.geo_setup_manual(
        #     points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs
        # )
        # # This is not a robust check but it only converges when N >> 1
        # points are used to do the integrations and there
        # are no discrete symmetries. Can easily check that 
        # can decrease rtol as you increase number of integration points
        # assert(np.isclose(psc_array.psi[0] / I, L[1, 0], rtol=1e-1))
        # Only true if R << 1, assert(np.isclose(psc_array.psi[0], np.pi * psc_array.R ** 2 * Bz_center))
        
        # Test that inductance and flux calculations for wide set of
        # scenarios
        a = 1e-4
        R0 = 10
        print('starting loop')
        for R in [0.1]:
            for points in [np.array([(np.random.rand(3) - 0.5) * 40, 
                                      (np.random.rand(3) - 0.5) * 40])]:
                for alphas in [np.zeros(2), (np.random.rand(2) - 0.5) * 2 * np.pi]:
                    for deltas in [np.zeros(2), (np.random.rand(2) - 0.5) * np.pi]:
                        for surf in [surfs[0]]:
                            psc_array = PSCgrid.geo_setup_manual(
                                points, R=R, a=a, alphas=alphas, deltas=deltas,
                            )
                            coils = coils_via_symmetries([psc_array.curves[1]], [Current(I)], nfp=1, stellsym=False)
                            bs = BiotSavart(coils)
                            kwargs = {"B_TF": bs, "ppp": 2000, "plasma_boundary": surf}
                            psc_array = PSCgrid.geo_setup_manual(
                                points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs
                            )
                            L = psc_array.L
                            # print(L[0, 1] * 1e10)
                            
                            contig = np.ascontiguousarray
                            # Calculate B fields like psc_array function does
                            B_PSC = np.zeros((psc_array.nphi * psc_array.ntheta, 3))
                            Bn_PSC = np.zeros(psc_array.nphi * psc_array.ntheta)
                            nn = psc_array.num_psc
                            q = 0
                            for fp in range(psc_array.nfp):
                                for stell in psc_array.stell_list:
                                    phi0 = (2 * np.pi / psc_array.nfp) * fp
                                    # get new locations by flipping the y and z components, then rotating by phi0
                                    ox = psc_array.grid_xyz_all[q * nn: (q + 1) * nn, 0]
                                    oy = psc_array.grid_xyz_all[q * nn: (q + 1) * nn, 1]
                                    oz = psc_array.grid_xyz_all[q * nn: (q + 1) * nn, 2]
                                    xyz = np.array([ox, oy, oz]).T
                                    Bn_PSC += sopp.A_matrix(
                                        contig(xyz),
                                        contig(psc_array.plasma_boundary.gamma().reshape(-1, 3)),
                                        contig(psc_array.alphas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.deltas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.plasma_boundary.unitnormal().reshape(-1, 3)),
                                        psc_array.R,
                                    ) @ psc_array.I * 2e-7
                                    B_PSC += sopp.B_PSC(
                                        contig(xyz),
                                        contig(psc_array.plasma_boundary.gamma().reshape(-1, 3)),
                                        contig(psc_array.alphas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.deltas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.I * stell),
                                        psc_array.R,
                                    )
                                    q = q + 1
                            # Calculate Bfields from CircularCoil class
                            B_circular_coils = np.zeros(B_PSC.shape)
                            Bn_circular_coils = np.zeros(psc_array.Bn_PSC.shape)
                            for i in range(len(psc_array.alphas)):
                                PSC = CircularCoil(
                                    psc_array.R, 
                                    psc_array.grid_xyz_all[i, :], 
                                    psc_array.I_all[i], 
                                    psc_array.coil_normals_all[i, :]
                                )
                                PSC.set_points(psc_array.plasma_boundary.gamma().reshape(-1, 3))
                                B_circular_coils += PSC.B().reshape(-1, 3)
                                Bn_circular_coils += np.sum(PSC.B().reshape(
                                    -1, 3) * psc_array.plasma_boundary.unitnormal().reshape(-1, 3), axis=-1)
                            # Calculate Bfields from direct BiotSavart
                            currents = []
                            for i in range(len(psc_array.I)):
                                currents.append(Current(psc_array.I[i]))
                            coils = coils_via_symmetries(
                                psc_array.curves, currents, nfp=1, stellsym=False
                            )
                            B_direct = BiotSavart(coils)
                            B_direct.set_points(psc_array.plasma_points)
                            Bn_direct = np.sum(B_direct.B().reshape(
                                -1, 3) * psc_array.plasma_boundary.unitnormal().reshape(-1, 3), axis=-1)
                            # Calculate Bfields from direct BiotSavart, using all the coils manually defined
                            currents = []
                            for i in range(psc_array.num_psc * psc_array.symmetry):
                                currents.append(Current(psc_array.I_all[i]))
                            all_coils = coils_via_symmetries(
                                psc_array.all_curves, currents, nfp=1, stellsym=False
                            )
                            B_direct_all = BiotSavart(all_coils)
                            B_direct_all.set_points(psc_array.plasma_points)
                            Bn_direct_all = np.sum(B_direct_all.B().reshape(
                                -1, 3) * psc_array.plasma_boundary.unitnormal().reshape(-1, 3), axis=-1)
                            
                            
                            psc_array.Bn_PSC = psc_array.Bn_PSC * psc_array.fac
                            print(psc_array.Bn_PSC[-1] * 1e10, Bn_PSC[-1] * 1e10, 
                                  Bn_circular_coils[-1] * 1e10, Bn_direct[-1] * 1e10,
                                  Bn_direct_all[-1] * 1e10)
                            # Robust test of all the B and Bn calculations from circular coils
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_PSC * 1e10, atol=1e3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_circular_coils * 1e10, atol=1e3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_direct * 1e10, atol=1e3, rtol=1e-3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_direct_all * 1e10, atol=1e3, rtol=1e-3))
        
        # Applying discrete symmetries to the coils wont work unless they dont intersect the symmetry planes
        for R in [0.1]:
            for points in [np.array([[1, 2, -1], [-1, -1, 3]])]:
                for alphas in [np.zeros(2), (np.random.rand(2) - 0.5) * 2 * np.pi]:
                    for deltas in [np.zeros(2), (np.random.rand(2) - 0.5) * np.pi]:
                        print(R, points, alphas, deltas)
                        for surf in surfs[1:]:
                            psc_array = PSCgrid.geo_setup_manual(
                                points, R=R, a=a, alphas=alphas, deltas=deltas,
                            )
                            # coils = coils_via_symmetries([psc_array.curves[1]], [Current(I)], nfp=1, stellsym=False)
                            # bs = BiotSavart(coils)
                            # kwargs = {"B_TF": bs, "ppp": 2000, "plasma_boundary": surf}
                            kwargs = {"plasma_boundary": surf}

                            psc_array = PSCgrid.geo_setup_manual(
                                points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs
                            )
                            L = psc_array.L
                            
                            contig = np.ascontiguousarray
                            # Calculate B fields like psc_array function does
                            B_PSC = np.zeros((psc_array.nphi * psc_array.ntheta, 3))
                            Bn_PSC = np.zeros(psc_array.nphi * psc_array.ntheta)
                            nn = psc_array.num_psc
                            q = 0
                            for fp in range(psc_array.nfp):
                                for stell in psc_array.stell_list:
                                    phi0 = (2 * np.pi / psc_array.nfp) * fp
                                    # get new locations by flipping the y and z components, then rotating by phi0
                                    ox = psc_array.grid_xyz_all[q * nn: (q + 1) * nn, 0]
                                    oy = psc_array.grid_xyz_all[q * nn: (q + 1) * nn, 1]
                                    oz = psc_array.grid_xyz_all[q * nn: (q + 1) * nn, 2]
                                    xyz = np.array([ox, oy, oz]).T
                                    Bn_PSC += sopp.A_matrix(
                                        contig(xyz),
                                        contig(psc_array.plasma_points),
                                        contig(psc_array.alphas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.deltas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.plasma_boundary.unitnormal().reshape(-1, 3)),
                                        psc_array.R,
                                    ) @ (psc_array.I) * 2e-7
                                    B_PSC += sopp.B_PSC(
                                        contig(xyz),
                                        contig(psc_array.plasma_points),
                                        contig(psc_array.alphas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.deltas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.I),
                                        psc_array.R,
                                    )
                                    q = q + 1
                            B_PSC_all = sopp.B_PSC(
                                contig(psc_array.grid_xyz_all),
                                contig(psc_array.plasma_points),
                                contig(psc_array.alphas_total),
                                contig(psc_array.deltas_total),
                                contig(psc_array.I),
                                psc_array.R,
                            )
                            # Calculate Bfields from CircularCoil class
                            B_circular_coils = np.zeros(B_PSC.shape)
                            Bn_circular_coils = np.zeros(psc_array.Bn_PSC.shape)
                            print(surf)
                            for i in range(psc_array.alphas_total.shape[0]):
                                PSC = CircularCoil(
                                    psc_array.R, 
                                    psc_array.grid_xyz_all[i, :], 
                                    psc_array.I_all[i], # * np.sign(psc_array.I_all[i]), 
                                    psc_array.coil_normals_all[i, :]
                                )
                                PSC.set_points(psc_array.plasma_points)
                                B_circular_coils += PSC.B().reshape(-1, 3)
                                Bn_circular_coils += np.sum(PSC.B().reshape(
                                    -1, 3) * psc_array.plasma_boundary.unitnormal().reshape(-1, 3), axis=-1)
                            # Calculate Bfields from direct BiotSavart
                            currents = []
                            for i in range(len(psc_array.I)):
                                currents.append(Current(psc_array.I[i]))
                            coils = coils_via_symmetries(
                                psc_array.curves, currents, nfp=psc_array.nfp, stellsym=psc_array.stellsym
                            )
                            currents_total = [coil.current.get_value() for coil in coils]
                            # print(currents_total)
                            B_direct = BiotSavart(coils)
                            B_direct.set_points(psc_array.plasma_points)
                            Bn_direct = np.sum(B_direct.B().reshape(
                                -1, 3) * psc_array.plasma_boundary.unitnormal().reshape(-1, 3), axis=-1)
                            # Calculate Bfields from direct BiotSavart, using all the coils manually defined
                            currents = []
                            for i in range(psc_array.num_psc * psc_array.symmetry):
                                currents.append(Current(psc_array.I_all_with_sign_flips[i]))
                            all_coils = coils_via_symmetries(
                                psc_array.all_curves, currents, nfp=1, stellsym=False
                            )
                            currents_total = [coil.current.get_value() for coil in all_coils]
                            curves_total = [coil.curve for coil in all_coils]
                            
                            from simsopt.geo import curves_to_vtk

                            curves_to_vtk(curves_total, "direct_curves", close=True, scalar_data=currents_total)
                            # print(currents_total)
                            B_direct_all = BiotSavart(all_coils)
                            B_direct_all.set_points(psc_array.plasma_points)
                            Bn_direct_all = np.sum(B_direct_all.B().reshape(
                                -1, 3) * psc_array.plasma_boundary.unitnormal().reshape(-1, 3), axis=-1)
                            
                            psc_array.Bn_PSC = psc_array.Bn_PSC * psc_array.fac
                            # Robust test of all the B and Bn calculations from circular coils
                            print(psc_array.Bn_PSC[-1] * 1e10, Bn_PSC[-1] * 1e10, 
                                  Bn_circular_coils[-1] * 1e10, Bn_direct[-1] * 1e10,
                                  Bn_direct_all[-1] * 1e10)
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_PSC * 1e10, rtol=1e-2, atol=1e3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_circular_coils * 1e10, rtol=1e-2, atol=1e3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_direct * 1e10, rtol=1e-2, atol=1e3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_direct_all * 1e10, rtol=1e-2, atol=1e3))

if __name__ == "__main__":
    unittest.main()
