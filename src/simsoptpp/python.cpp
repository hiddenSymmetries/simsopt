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

namespace py = pybind11;

using std::vector;
using std::shared_ptr;

void init_surfaces(py::module_ &);
void init_curves(py::module_ &);
void init_magneticfields(py::module_ &);
void init_boozermagneticfields(py::module_ &);
void init_tracing(py::module_ &);

template<class T>
bool empty_intersection(const std::set<T>& x, const std::set<T>& y)
{
    auto i = x.begin();
    auto j = y.begin();
    while (i != x.end() && j != y.end())
    {
        if (*i == *j)
            return false;
        else if (*i < *j)
            ++i;
        else
            ++j;
    }
    return true;
}



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

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::nanoseconds;
    m.def("get_close_candidates", [](std::vector<PyArray> pointClouds, double threshold, int mesh_factor) {
        /*
        Returns all pairings of the given pointClouds that have two points that
        are less than `threshold` away. The estimate is approximate (for
        speed), so this function may return too many (but not too few!)
        pairings.

        The basic idea of this function is the following:
        - Assume we want to compare pointcloud A and B.
        - We create a uniform grid of size threshold/mesh_factor.
        - Loop over points in cloud A, mark all cells that have a point in it (via the `set` variables below).
        - Loop over points in cloud B, mark all cells that have a point in it and also all cells in a sphere around it.
        - Check whether the intersection between the two sets is non-empty.
        */
        std::vector<std::set<std::tuple<int, int, int>>> sets;
        std::vector<std::set<std::tuple<int, int, int>>> sets_extended;
        std::vector<std::tuple<int, int>> candidates;
        for (int p = 0; p < pointClouds.size(); ++p) {
            std::set<std::tuple<int, int, int>> s;
            std::set<std::tuple<int, int, int>> s_extended;
            auto points = pointClouds[p];
            for (int l = 0; l < points.shape(0); ++l) {
                int i = std::floor(mesh_factor*points(l, 0)/threshold);
                int j = std::floor(mesh_factor*points(l, 1)/threshold);
                int k = std::floor(mesh_factor*points(l, 2)/threshold);
                s.insert({i, j, k});
                for (int ii = -mesh_factor; ii <= mesh_factor; ++ii) {
                    for (int jj = -mesh_factor; jj <= mesh_factor; ++jj) {
                        for (int kk = -mesh_factor; kk <= mesh_factor; ++kk) {
                            if(std::abs(kk) + std::abs(jj) + std::abs(kk) - 3 < std::sqrt(3)*mesh_factor)
                                s_extended.insert({i + ii, j + jj, k + kk});
                        }
                    }
                }
            }
            sets.push_back(s);
            sets_extended.push_back(s_extended);
        }

        for (int i = 0; i < pointClouds.size(); ++i) {
            for (int j = 0; j < i; ++j) {
                if(!empty_intersection(sets_extended[i], sets[j]))
                    candidates.push_back({i, j});
            }
        }
        return candidates;
    }, "Get candidates for which point clouds are closer than threshold to each other.", py::arg("pointClouds"), py::arg("threshold"), py::arg("mesh_factor")=1);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
