#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
#include <Eigen/Core>

typedef xt::pyarray<double> PyArray;
#include "xtensor-python/pytensor.hpp"     // Numpy bindings
typedef xt::pytensor<double, 2, xt::layout_type::row_major> PyTensor;
#include <math.h>
#include <chrono>


#include "biot_savart_py.h"
#include "biot_savart_vjp_py.h"
#include "boozerradialinterpolant.h"
#include "dipole_field.h"
#include "dommaschk.h"
#include "integral_BdotN.h"
#include "permanent_magnet_optimization.h"
#include "wireframe_optimization.h"
#include "reiman.h"
#include "simdhelpers.h"
#include "boozerresidual_py.h"

namespace py = pybind11;

using std::vector;
using std::shared_ptr;

void init_surfaces(py::module_ &);
void init_curves(py::module_ &);
void init_magneticfields(py::module_ &);
void init_boozermagneticfields(py::module_ &);
void init_tracing(py::module_ &);
void init_distance(py::module_ &);



PYBIND11_MODULE(simsoptpp, m) {
    xt::import_numpy();

    init_curves(m);
    init_surfaces(m);
    init_magneticfields(m);
    init_boozermagneticfields(m);
    init_tracing(m);
    init_distance(m);

#if defined(USE_XSIMD)
    m.attr("using_xsimd") = true;
#else
    m.attr("using_xsimd") = false;
#endif

    m.def("biot_savart", &biot_savart);
    m.def("biot_savart_B", &biot_savart_B);
    m.def("biot_savart_vjp", &biot_savart_vjp);
    m.def("biot_savart_vjp_graph", &biot_savart_vjp_graph);
    m.def("biot_savart_vector_potential_vjp_graph", &biot_savart_vector_potential_vjp_graph);

    // Functions below are implemented for permanent magnet optimization
    m.def("dipole_field_B" , &dipole_field_B);
    m.def("dipole_field_A" , &dipole_field_A);
    m.def("dipole_field_dB", &dipole_field_dB);
    m.def("dipole_field_dA" , &dipole_field_dA);
    m.def("dipole_field_Bn" , &dipole_field_Bn, py::arg("points"), py::arg("m_points"), py::arg("unitnormal"), py::arg("nfp"), py::arg("stellsym"), py::arg("b"), py::arg("coordinate_flag") = "cartesian", py::arg("R0") = 0.0);
    m.def("define_a_uniform_cartesian_grid_between_two_toroidal_surfaces" , &define_a_uniform_cartesian_grid_between_two_toroidal_surfaces);

    // Permanent magnet optimization algorithms have many default arguments
    m.def("MwPGP_algorithm", &MwPGP_algorithm, py::arg("A_obj"), py::arg("b_obj"), py::arg("ATb"), py::arg("m_proxy"), py::arg("m0"), py::arg("m_maxima"), py::arg("alpha"), py::arg("nu") = 1.0e100, py::arg("epsilon") = 1.0e-3, py::arg("reg_l0") = 0.0, py::arg("reg_l1") = 0.0, py::arg("reg_l2") = 0.0, py::arg("max_iter") = 500, py::arg("min_fb") = 1.0e-20, py::arg("verbose") = false);
    // variants of GPMO algorithm
    m.def("GPMO_backtracking", &GPMO_backtracking, py::arg("A_obj"), py::arg("b_obj"), py::arg("mmax"), py::arg("normal_norms"), py::arg("K") = 1000, py::arg("verbose") = false, py::arg("nhistory") = 100, py::arg("backtracking") = 100, py::arg("dipole_grid_xyz"), py::arg("single_direction") = -1, py::arg("Nadjacent") = 7, py::arg("max_nMagnets"));
    m.def("GPMO_multi", &GPMO_multi, py::arg("A_obj"), py::arg("b_obj"), py::arg("mmax"), py::arg("normal_norms"), py::arg("K") = 1000, py::arg("verbose") = false, py::arg("nhistory") = 100, py::arg("dipole_grid_xyz"), py::arg("single_direction") = -1, py::arg("Nadjacent") = 7);
    m.def("GPMO_ArbVec", &GPMO_ArbVec, py::arg("A_obj"), py::arg("b_obj"), py::arg("mmax"), py::arg("normal_norms"), py::arg("pol_vectors"), py::arg("K") = 1000, py::arg("verbose") = false, py::arg("nhistory") = 100);
    m.def("GPMO_ArbVec_backtracking", &GPMO_ArbVec_backtracking, py::arg("A_obj"), py::arg("b_obj"), py::arg("mmax"), py::arg("normal_norms"), py::arg("pol_vectors"), py::arg("K") = 1000, py::arg("verbose") = false, py::arg("nhistory") = 100, py::arg("backtracking") = 100, py::arg("dipole_grid_xyz"), py::arg("Nadjacent") = 7, py::arg("thresh_angle") = 3.1415926535897931, py::arg("max_nMagnets"), py::arg("x_init"));
    m.def("GPMO_baseline", &GPMO_baseline, py::arg("A_obj"), py::arg("b_obj"), py::arg("mmax"), py::arg("normal_norms"), py::arg("K") = 1000, py::arg("verbose") = false, py::arg("nhistory") = 100, py::arg("single_direction") = -1);

    // Greedy stellarator coil optimization (GSCO) solver
    m.def("GSCO", &GSCO, py::arg("no_crossing"), py::arg("no_new_coils"), 
          py::arg("match_current"), py::arg("A_obj"), py::arg("b_obj"), 
          py::arg("current"), py::arg("max_current"), py::arg("max_loop_count"),
          py::arg("loops"), py::arg("free_loops"), py::arg("segments"), 
          py::arg("connections"), py::arg("lambda_P"), py::arg("nIter"), 
          py::arg("x_init"), py::arg("cell_count_init"), py::arg("print_freq"));

    m.def("DommaschkB" , &DommaschkB);
    m.def("DommaschkdB", &DommaschkdB);

    m.def("integral_BdotN", &integral_BdotN);

    m.def("ReimanB" , &ReimanB);
    m.def("ReimandB", &ReimandB);

    m.def("fourier_transform_even", &fourier_transform_even);
    m.def("fourier_transform_odd", &fourier_transform_odd);
    m.def("inverse_fourier_transform_even", &inverse_fourier_transform_even);
    m.def("inverse_fourier_transform_odd", &inverse_fourier_transform_odd);
    m.def("compute_kmns",&compute_kmns);
    m.def("compute_kmnc_kmns",&compute_kmnc_kmns);

    // the computation below is used in boozer_surface_residual.
    //
    // G*dB_dc - 2*np.sum(B[..., None]*dB_dc, axis=2)[:, :, None, :] * tang[..., None] - B2[..., None, None] * (dxphi_dc + iota * dxtheta_dc)
    m.def("boozer_dresidual_dc", [](double G, PyArray& dB_dc, PyArray& B, PyArray& tang, PyArray& B2, PyArray& dxphi_dc, double iota, PyArray& dxtheta_dc) {
            int nphi = dB_dc.shape(0);
            int ntheta = dB_dc.shape(1);
            int ndofs = dB_dc.shape(3);
            PyArray res = xt::zeros<double>({nphi, ntheta, 3, ndofs});
            double* B_dB_dc = new double[ndofs];
            for(int i=0; i<nphi; i++){
                for(int j=0; j<ntheta; j++){
                    for (int m = 0; m < ndofs; ++m) {
                        B_dB_dc[m] = B(i, j, 0)*dB_dc(i, j, 0, m) + B(i, j, 1)*dB_dc(i, j, 1, m) + B(i, j, 2)*dB_dc(i, j, 2, m);
                    }
                    double B2ij = B2(i, j);
                    for (int d = 0; d < 3; ++d) {
                        auto dB_dc_ptr = &(dB_dc(i, j, d, 0));
                        auto res_ptr = &(res(i, j, d, 0));
                        auto dxphi_dc_ptr = &(dxphi_dc(i, j, d, 0));
                        auto dxtheta_dc_ptr = &(dxtheta_dc(i, j, d, 0));
                        auto tangijd = tang(i, j, d);
                        for (int m = 0; m < ndofs; ++m) {
                            res_ptr[m] = G*dB_dc_ptr[m]
                            - 2*B_dB_dc[m]*tangijd
                            - B2ij * (dxphi_dc_ptr[m] + iota*dxtheta_dc_ptr[m]);
                        }
                    }
                }
            }
            delete[] B_dB_dc;
            return res;
        });

    m.def("boozer_residual", &boozer_residual);
    m.def("boozer_residual_ds", &boozer_residual_ds);
    m.def("boozer_residual_ds2", &boozer_residual_ds2);

    m.def("matmult", [](PyArray& A, PyArray&B) {
            // Product of an lxm matrix with an mxn matrix, results in an l x n matrix
            int l = A.shape(0);
            int m = A.shape(1);
            int n = B.shape(1);
            PyArray C = xt::zeros<double>({l, n});

            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigA(const_cast<double*>(A.data()), l, m);
            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigB(const_cast<double*>(B.data()), m, n);
            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigC(const_cast<double*>(C.data()), l, n);
            eigC = eigA*eigB;
            return C;
        });

    m.def("vjp", [](PyArray& v, PyArray&B) {
            // Product of v.T @ B
            int m = B.shape(0);
            int n = B.shape(1);
            PyArray C = xt::zeros<double>({n});

            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigv(const_cast<double*>(v.data()), m, 1);
            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigB(const_cast<double*>(B.data()), m, n);
            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigC(const_cast<double*>(C.data()), 1, n);
            eigC = eigv.transpose()*eigB;
            return C;
        });

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
