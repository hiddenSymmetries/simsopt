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
#include "dommaschk.h"
#include "dipole_field.h"
#include "winding_volume.h"
#include "permanent_magnet_optimization.h"
#include "reiman.h"
#include "boozerradialinterpolant.h"
#include "bounce.h"

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

    m.def("biot_savart", &biot_savart);
    m.def("biot_savart_B", &biot_savart_B);
    m.def("biot_savart_vjp", &biot_savart_vjp);
    m.def("biot_savart_vjp_graph", &biot_savart_vjp_graph);
    m.def("biot_savart_vector_potential_vjp_graph", &biot_savart_vector_potential_vjp_graph);
    
    // Functions below are implemented for winding VOLUME method
    m.def("winding_volume_geo_factors", &winding_volume_geo_factors); 

    // Functions below are implemented for permanent magnet optimization
    m.def("dipole_field_B" , &dipole_field_B);
    m.def("dipole_field_A" , &dipole_field_A);
    m.def("dipole_field_dB", &dipole_field_dB);
    m.def("dipole_field_dA" , &dipole_field_dA);
    m.def("dipole_field_Bn" , &dipole_field_Bn, py::arg("points"), py::arg("m_points"), py::arg("unitnormal"), py::arg("nfp"), py::arg("stellsym"), py::arg("phi"), py::arg("b"), py::arg("coordinate_flag") = "cartesian", py::arg("R0") = 0.0);
    m.def("make_final_surface" , &make_final_surface);

    // Permanent magnet optimization algorithms have many default arguments
    m.def("MwPGP_algorithm", &MwPGP_algorithm, py::arg("A_obj"), py::arg("b_obj"), py::arg("ATb"), py::arg("m_proxy"), py::arg("m0"), py::arg("m_maxima"), py::arg("alpha"), py::arg("nu") = 1.0e100, py::arg("epsilon") = 1.0e-3, py::arg("reg_l0") = 0.0, py::arg("reg_l1") = 0.0, py::arg("reg_l2") = 0.0, py::arg("max_iter") = 500, py::arg("min_fb") = 1.0e-20, py::arg("verbose") = false);
    // variants of GPMO algorithm
    m.def("GPMO_backtracking", &GPMO_backtracking, py::arg("A_obj"), py::arg("b_obj"), py::arg("mmax"), py::arg("normal_norms"), py::arg("K") = 1000, py::arg("verbose") = false, py::arg("nhistory") = 100, py::arg("backtracking") = 100, py::arg("dipole_grid_xyz"), py::arg("single_direction") = -1, py::arg("Nadjacent") = 7);
    m.def("GPMO_multi", &GPMO_multi, py::arg("A_obj"), py::arg("b_obj"), py::arg("mmax"), py::arg("normal_norms"), py::arg("K") = 1000, py::arg("verbose") = false, py::arg("nhistory") = 100, py::arg("dipole_grid_xyz"), py::arg("single_direction") = -1, py::arg("Nadjacent") = 7);
    m.def("GPMO_baseline", &GPMO_baseline, py::arg("A_obj"), py::arg("b_obj"), py::arg("mmax"), py::arg("normal_norms"), py::arg("K") = 1000, py::arg("verbose") = false, py::arg("nhistory") = 100, py::arg("single_direction") = -1);
    m.def("GPMO_MC", &GPMO_MC, py::arg("A_obj"), py::arg("b_obj"), py::arg("ATb"), py::arg("mmax"), py::arg("normal_norms"), py::arg("K") = 1000, py::arg("verbose") = false, py::arg("nhistory") = 100);
    //
    m.def("PQN_algorithm", &PQN_algorithm, py::arg("A_obj"), py::arg("b_obj"), py::arg("ATb"), py::arg("m_proxy"), py::arg("m0"), py::arg("m_maxima"), py::arg("nu") = 1.0e100, py::arg("epsilon") = 1.0e-3, py::arg("reg_l0") = 0.0, py::arg("reg_l1") = 0.0, py::arg("reg_l2") = 0.0, py::arg("max_iter") = 500, py::arg("verbose") = false);
    m.def("SPG", &SPG, py::arg("A_obj"), py::arg("b_obj"), py::arg("ATb"), py::arg("m_proxy"), py::arg("m0"), py::arg("m_maxima"), py::arg("alpha_min") = 1e-10, py::arg("alpha_max") = 1e10, py::arg("alpha_bb_prev") = 1, py::arg("h") = 100, py::arg("epsilon") = 1.0e-3, py::arg("reg_l2") = 0.0, py::arg("nu") = 1.0e100, py::arg("max_iter") = 500, py::arg("nu_SPG") = 1.0e-4, py::arg("verbose") = false);
    // Need to get SPG and MwPGP in same format, ideally using kwargs for the nonmatching arguments

    m.def("DommaschkB" , &DommaschkB);
    m.def("DommaschkdB", &DommaschkdB);

    m.def("ReimanB" , &ReimanB);
    m.def("ReimandB", &ReimandB);

    m.def("fourier_transform_even", &fourier_transform_even);
    m.def("fourier_transform_odd", &fourier_transform_odd);
    m.def("inverse_fourier_transform_even", &inverse_fourier_transform_even);
    m.def("inverse_fourier_transform_odd", &inverse_fourier_transform_odd);
    m.def("compute_kmns",&compute_kmns);
    m.def("compute_kmnc_kmns",&compute_kmnc_kmns);

    m.def("vprime", &vprime<xt::pytensor>,
        py::arg("field"),
        py::arg("s"),
        py::arg("theta0"),
        py::arg("nfp"),
        py::arg("nmax"),
        py::arg("step_zie")=1e-3
    );

    m.def("find_bounce_points", &find_bounce_points<xt::pytensor>,
        py::arg("field"),
        py::arg("s"),
        py::arg("theta0"),
        py::arg("zeta0"),
        py::arg("lam"),
        py::arg("nfp"),
        py::arg("option"),
        py::arg("nmax"),
        py::arg("nzeta")=1000,
        py::arg("digits")=16,
        py::arg("derivative_tol")=1e-3,
        py::arg("argmin_tol")=1e-3,
        py::arg("root_tol")=1e-8
    );

    m.def("bounce_integral", &bounce_integral<xt::pytensor>,
        py::arg("bouncel"),
        py::arg("bouncer"),
        py::arg("field"),
        py::arg("s"),
        py::arg("theta0"),
        py::arg("lam"),
        py::arg("nfp"),
        py::arg("jpar"),
        py::arg("psidot"),
        py::arg("alphadot"),
        py::arg("ihat"),
        py::arg("khat"),
        py::arg("dkhatdalpha"),
        py::arg("tau"),
        py::arg("step_size")=1e-3,
        py::arg("tol")=1e-8,
        py::arg("adjust")=false
        );

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

    m.def("integral_BdotN", [](PyArray& Bcoil, PyArray& Btarget, PyArray& n) {
        int nphi = Bcoil.shape(0);
        int ntheta = Bcoil.shape(1);
        double *Bcoil_ptr = Bcoil.data();
        double *Btarget_ptr = NULL;
        double *n_ptr = n.data();
        if(Bcoil.layout() != xt::layout_type::row_major)
              throw std::runtime_error("Bcoil needs to be in row-major storage order");
        if(Bcoil.shape(2) != 3)
            throw std::runtime_error("Bcoil has wrong shape.");
        if(Bcoil.size() != 3*nphi*ntheta)
            throw std::runtime_error("Bcoil has wrong size.");
        if(n.layout() != xt::layout_type::row_major)
              throw std::runtime_error("n needs to be in row-major storage order");
        if(n.shape(0) != nphi)
            throw std::runtime_error("n has wrong shape.");
        if(n.shape(1) != ntheta)
            throw std::runtime_error("n has wrong shape.");
        if(n.shape(2) != 3)
            throw std::runtime_error("n has wrong shape.");
        if(n.size() != 3*nphi*ntheta)
            throw std::runtime_error("n has wrong size.");
        if(Btarget.size() > 0){
            if(Btarget.layout() != xt::layout_type::row_major)
                throw std::runtime_error("Btarget needs to be in row-major storage order");
            if(Btarget.shape(0) != nphi)
                throw std::runtime_error("Btarget has wrong shape.");
            if(Btarget.shape(1) != ntheta)
                throw std::runtime_error("Btarget has wrong shape.");
            if(Btarget.size() != nphi*ntheta)
                throw std::runtime_error("Btarget has wrong size.");

            Btarget_ptr = Btarget.data();
        }
        double res = 0;
#pragma omp parallel for reduction(+:res)
        for(int i=0; i<nphi*ntheta; i++){
            double normN = std::sqrt(n_ptr[3*i+0]*n_ptr[3*i+0] + n_ptr[3*i+1]*n_ptr[3*i+1] + n_ptr[3*i+2]*n_ptr[3*i+2]);
            double Nx = n_ptr[3*i+0]/normN;
            double Ny = n_ptr[3*i+1]/normN;
            double Nz = n_ptr[3*i+2]/normN;
            double BcoildotN = Bcoil_ptr[3*i+0]*Nx + Bcoil_ptr[3*i+1]*Ny + Bcoil_ptr[3*i+2]*Nz;
            if(Btarget_ptr != NULL)
                BcoildotN -= Btarget_ptr[i];
            res += (BcoildotN * BcoildotN) * normN;
        }
        return 0.5 * res / (nphi*ntheta);
    });

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
