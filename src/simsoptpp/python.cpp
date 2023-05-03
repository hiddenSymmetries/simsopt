#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
#include <Eigen/Core>
typedef xt::pyarray<double> PyArray;
#include <math.h>
#include <chrono>




#include "biot_savart_py.h"
#include "biot_savart_vjp_py.h"
#include "dommaschk.h"
#include "reiman.h"
#include "boozerradialinterpolant.h"
#include "boozerresidual.h"

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

    // the computation below is used in boozer_surface_residual.
    m.def("boozer_residual", &boozer_residual);
    m.def("boozer_residual_ds", &boozer_residual_ds);
    m.def("boozer_residual_ds2", &boozer_residual_ds2);
    m.def("boozer_dresidual_dc", &boozer_residual_dc);

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
