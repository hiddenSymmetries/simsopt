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

namespace py = pybind11;

using std::vector;
using std::shared_ptr;

void init_surfaces(py::module_ &);
void init_curves(py::module_ &);
void init_magneticfields(py::module_ &);


PYBIND11_MODULE(simsgeopp, m) {
    xt::import_numpy();

    init_curves(m);
    init_surfaces(m);
    init_magneticfields(m);

    m.def("biot_savart", &biot_savart);
    m.def("biot_savart_B", &biot_savart_B);
    m.def("biot_savart_vjp", &biot_savart_vjp);

    m.def("DommaschkB" , &DommaschkB);
    m.def("DommaschkdB", &DommaschkdB);



#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
