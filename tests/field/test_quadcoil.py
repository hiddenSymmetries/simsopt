import unittest
import numpy as np
import sys
sys.path.insert(1,'./build')
import time
try:
    from simsoptpp import test_double, test_single, integrate_multi
except ImportError:
    print('Could not import BIEST package (requires openmp built with simsopt), exiting.')
    exit()

class CurrentPotentialTests(unittest.TestCase):
    def test_run_biest(self):
        a = np.random.random()+1 # np.random.random() * 0.2 + 0.8
        b_single = np.random.random(5)*10 # np.random.random() * 0.2 + 0.8
        b_double = np.random.random(6)*20 # np.random.random() * 0.2 + 0.8

        # Running control cases
        test_example_single = []
        func_in_single_control = []
        for i in range(len(b_single)):
            func_in = np.zeros((70, 20), dtype=np.float64)
            gamma_example = np.zeros((70, 20, 3), dtype=np.float64)
            out = np.zeros((70, 20), dtype=np.float64)
            test_single(a, b_single[i], gamma_example, func_in, out)
            test_example_single.append(out)
            func_in_single_control.append(func_in)

        test_example_double = []
        func_in_double_control = []
        for i in range(len(b_double)):
            func_in = np.zeros((70, 20), dtype=np.float64)
            gamma_example = np.zeros((70, 20, 3), dtype=np.float64)
            out = np.zeros((70, 20), dtype=np.float64)
            test_double(a, b_double[i], gamma_example, func_in, out)
            test_example_double.append(out)
            func_in_double_control.append(func_in)
        print('The BIEST single-layer example evaluates to', test_example_single)
        print('The BIEST double-layer example evaluates to', test_example_double)

        # Constructing test surfaces
        _nfp = 1
        Nt = 70
        Np = 20
        gamma = np.zeros((Nt, Np, 3), dtype=np.float64)
        func_in_single = np.zeros((Nt, Np, len(b_single)), dtype=np.float64)
        func_in_double = np.zeros((Nt, Np, len(b_double)), dtype=np.float64)
        for k in range(len(b_single)):
            for i in np.arange(Nt):
                for j in np.arange(Np):
                    b = b_single[k]
                    phi = 2 * np.pi * i / Nt
                    theta = 2 * np.pi * j / Np
                    R = 1 + 0.25 * np.cos(theta)
                    x = R * np.cos(phi)
                    y = R * a * np.sin(phi)
                    z = 0.25 * np.sin(theta)
                    gamma[i, j, 0] = x
                    gamma[i, j, 1] = y
                    gamma[i, j, 2] = z
                    func_in_single[i, j, k] = x + y + b * z
                    
        for k in range(len(b_double)):
            for i in np.arange(Nt):
                for j in np.arange(Np):
                    b = b_double[k]
                    phi = 2 * np.pi * i / Nt
                    theta = 2 * np.pi * j / Np
                    R = 1 + 0.25 * np.cos(theta)
                    x = R * np.cos(phi)
                    y = R * a * np.sin(phi)
                    z = 0.25 * np.sin(theta)
                    gamma[i, j, 0] = x
                    gamma[i, j, 1] = y
                    gamma[i, j, 2] = z
                    print(x + y + b * z)
                    func_in_double[i, j, k] = x + y + b * z

        # Evaluating the integrals using the BIEST binding
        result_single = np.zeros_like(func_in_single, dtype=np.float64)
        result_double = np.zeros_like(func_in_double, dtype=np.float64)
        time1 = time.time()
        integrate_multi(
            gamma, # xt::pyarray<double> &gamma,
            func_in_single, # xt::pyarray<double> &func_in_single,
            result_single, # xt::pyarray<double> &result,
            True,
            10, # int digits,
            1, # int nfp
            False
        )
        time2 = time.time()
        print('Both eval time:', time2 - time1)

        time1 = time.time()
        integrate_multi(
            gamma, # xt::pyarray<double> &gamma,
            func_in_double, # xt::pyarray<double> &func_in_single,
            result_double, # xt::pyarray<double> &result,
            False,
            10, # int digits,
            1, # int nfp
            False
        )
        time2 = time.time()
        print('Neither eval time:', time2 - time1)

        # Assertions
        print('Testing BIEST binding.')
        print('Testing single-layer integration.')
        for i in range(len(b_single)):
            print(np.all(np.isclose(result_single[:, :, i], test_example_single[i])))
            assert(np.all(np.isclose(result_single[:, :, i], test_example_single[i])))
        print('Testing double-layer integration.')
        for i in range(len(b_double)):
            print(np.all(np.isclose(result_double[:, :, i], test_example_double[i])))
            assert(np.all(np.isclose(result_double[:, :, i], test_example_double[i])))

        print('gamma', gamma.shape)
        print('result_single[:, :, 0]', result_single[:, :, 0].shape)
        # integrate_single(
        #     gamma, # xt::pyarray<double> &gamma,
        #     result_single[:, :, 0], # xt::pyarray<double> &func_in_single,
        #     10, # int digits,
        #     nfp, # int nfp
        # )

if __name__ == "__main__":
    unittest.main()