#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"
#include "py_shared_ptr.h"
PYBIND11_DECLARE_HOLDER_TYPE(T, py_shared_ptr<T>);
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;



#include "biot_savart_py.h"
#include "biot_savart_vjp_py.h"
#include "dommaschk.h"
#include "reiman.h"

namespace py = pybind11;

using std::vector;
using std::shared_ptr;

void init_surfaces(py::module_ &);
void init_curves(py::module_ &);
void init_magneticfields(py::module_ &);
void init_tracing(py::module_ &);


PYBIND11_MODULE(simsoptpp, m) {
    xt::import_numpy();

    init_curves(m);
    init_surfaces(m);
    init_magneticfields(m);
    init_tracing(m);

    m.def("biot_savart", &biot_savart);
    m.def("biot_savart_B", &biot_savart_B);
    m.def("biot_savart_vjp", &biot_savart_vjp);

    m.def("DommaschkB" , &DommaschkB);
    m.def("DommaschkdB", &DommaschkdB);

    m.def("ReimanB" , &ReimanB);
    m.def("ReimandB", &ReimandB);
    
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

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
