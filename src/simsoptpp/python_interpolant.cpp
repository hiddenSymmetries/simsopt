#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"
#include "xtensor-python/pytensor.hpp"     // Numpy bindings

typedef xt::pytensor<double, 2, xt::layout_type::row_major> Array2;
using std::shared_ptr;
using std::vector;

namespace py = pybind11;
#include "regular_grid_interpolant_3d.h"

void init_interpolant(py::module_ &m){

    py::class_<InterpolationRule, shared_ptr<InterpolationRule>>(m, "InterpolationRule", "Abstract class for interpolation rules on an interval.")
        .def_readonly("degree", &InterpolationRule::degree, "The degree of the polynomial. The number of interpolation points in `degree+1`.");

    py::class_<UniformInterpolationRule, shared_ptr<UniformInterpolationRule>, InterpolationRule>(m, "UniformInterpolationRule", "Polynomial interpolation using equispaced points.")
        .def(py::init<int>())
        .def_readonly("degree", &UniformInterpolationRule::degree, "The degree of the polynomial. The number of interpolation points in `degree+1`.");
    py::class_<ChebyshevInterpolationRule, shared_ptr<ChebyshevInterpolationRule>, InterpolationRule>(m, "ChebyshevInterpolationRule", "Polynomial interpolation using chebychev points.")
        .def(py::init<int>())
        .def_readonly("degree", &ChebyshevInterpolationRule::degree, "The degree of the polynomial. The number of interpolation points in `degree+1`.");

    py::class_<RegularGridInterpolant3D<Array2>, shared_ptr<RegularGridInterpolant3D<Array2>>>(m, "RegularGridInterpolant3D",
            R"pbdoc(
            Interpolates a (vector valued) function on a uniform grid. 
            This interpolant is optimized for fast function evaluation (at the cost of memory usage). The main purpose of this class is to be used to interpolate magnetic fields and then use the interpolant for tasks such as fieldline or particle tracing for which the field needs to be evaluated many many times.
            )pbdoc")
        .def(py::init<InterpolationRule, RangeTriplet, RangeTriplet, RangeTriplet, int, bool, std::function<std::vector<bool>(Vec, Vec, Vec)>>())
        .def(py::init<InterpolationRule, RangeTriplet, RangeTriplet, RangeTriplet, int, bool>())
        .def("interpolate_batch", &RegularGridInterpolant3D<Array2>::interpolate_batch, "Interpolate a function by evaluating the function on all interpolation nodes simultanuously.")
        .def("evaluate", &RegularGridInterpolant3D<Array2>::evaluate, "Evaluate the interpolant at a point.")
        .def("evaluate_batch", &RegularGridInterpolant3D<Array2>::evaluate_batch, "Evaluate the interpolant at multiple points (faster than `evaluate` as it uses prefetching).");
}
