#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
#include <Eigen/Core>
typedef xt::pyarray<double> PyArray;



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


PYBIND11_MODULE(simsoptpp, m) {
    xt::import_numpy();

    init_curves(m);
    init_surfaces(m);
    init_magneticfields(m);
    init_boozermagneticfields(m);
    init_tracing(m);

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

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
