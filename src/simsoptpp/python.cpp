#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;
#include <math.h>
#include <chrono>
#include "boozerradialinterpolant.h"

namespace py = pybind11;

using std::vector;
using std::shared_ptr;

void init_boozermagneticfields(py::module_ &);
void init_tracing(py::module_ &);

PYBIND11_MODULE(simsoptpp, m) {
    xt::import_numpy();

    init_boozermagneticfields(m);
    init_tracing(m);

    m.def("fourier_transform_even", &fourier_transform_even);
    m.def("fourier_transform_odd", &fourier_transform_odd);
    m.def("inverse_fourier_transform_even", &inverse_fourier_transform_even);
    m.def("inverse_fourier_transform_odd", &inverse_fourier_transform_odd);
    m.def("compute_kmns",&compute_kmns);
    m.def("compute_kmnc_kmns",&compute_kmnc_kmns);
    m.def("simd_alignment", &simd_alignment);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
