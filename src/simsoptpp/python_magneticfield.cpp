#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
#include "xtensor-python/pytensor.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;
typedef xt::pyarray<int> PyIntArray;
typedef xt::pytensor<double, 2, xt::layout_type::row_major> PyTensor;
using std::shared_ptr;
using std::vector;

namespace py = pybind11;
#include "magneticfield.h"
#include "magneticfield_biotsavart.h"
#include "magneticfield_wireframe.h"
#include "magneticfield_interpolated.h"
#include "pymagneticfield.h"
#include "regular_grid_interpolant_3d.h"
#include "pycurrent.h"
#include "python_deprecation.h"
typedef MagneticField<xt::pytensor> PyMagneticField;
typedef BiotSavart<xt::pytensor, PyArray> PyBiotSavart;
typedef WireframeField<xt::pytensor, PyArray, PyIntArray> PyWireframeField;
typedef InterpolatedField<xt::pytensor> PyInterpolatedField;



template <typename T, typename S> void register_common_field_methods(S &c) {
    c
     .def("B", py::overload_cast<>(&T::B), "Returns a `(npoints, 3)` array containing the magnetic field (in cartesian coordinates). Denoting the indices by `i` and `l`, the result contains  `B_l(x_i)`.")
     .def("dB_by_dX", py::overload_cast<>(&T::dB_by_dX), "Returns a `(npoints, 3, 3)` array containing the gradient of magnetic field (in cartesian coordinates). Denoting the indices by `i`, `j` and `l`, the result contains  `\\partial_j B_l(x_i)`.")
     .def("d2B_by_dXdX", py::overload_cast<>(&T::d2B_by_dXdX), "Returns a `(npoints, 3, 3, 3)` array containing the hessian of magnetic field (in cartesian coordinates). Denoting the indices by `i`, `j`, `k` and `l`, the result contains  `\\partial_k\\partial_j B_l(x_i)`.")
     .def("AbsB", py::overload_cast<>(&T::AbsB), "Returns a `(npoints, 1)` array containing the absolute value of the magnetic field (in cartesian coordinates).")
     .def("GradAbsB", py::overload_cast<>(&T::GradAbsB), "Returns a `(npoints, 3)` array containing the gradient of the absolute value of the magnetic field (in cartesian coordinates).")
     .def("GradAbsB_cyl", py::overload_cast<>(&T::GradAbsB_cyl))
     .def("B_cyl", py::overload_cast<>(&T::B_cyl), "Return a `(npoints, 3)` array containing the magnetic field (in cylindrical coordinates) (the order is :math:`(B_r, B_\\phi, B_z)`).")
     .def("A", py::overload_cast<>(&T::A), "Returns a `(npoints, 3)` array containing the magnetic potential (in cartesian coordinates). Denoting the indices by `i` and `l`, the result contains  `A_l(x_i)`.")
     .def("A_cyl", py::overload_cast<>(&T::A_cyl), "Return a `(npoints, 3)` array containing the magnetic potential (in cylindrical coordinates) (the order is :math:`(A_r, A_\\phi, A_z)`).")
     .def("dA_by_dX", py::overload_cast<>(&T::dA_by_dX), "Returns a `(npoints, 3, 3)` array containing the gradient of the magnetic potential (in cartesian coordinates). Denoting the indices by `i`, `j` and `l`, the result contains  `\\partial_j A_l(x_i)`.")
     .def("d2A_by_dXdX", py::overload_cast<>(&T::d2A_by_dXdX), "Returns a `(npoints, 3, 3)` array containing the hessian of the magnetic potential (in cartesian coordinates). Denoting the indices by `i`, `j`, `k` and `l`, the result contains  `\\partial_k\\partial_j  A_l(x_i)`.")
     .def("invalidate_cache", &T::invalidate_cache, "Clear the cache. Called automatically after each call to `set_points[...]`.")
     .def("get_points_cart", &T::get_points_cart, "Get the point where the field should be evaluated in cartesian coordinates.")
     .def("get_points_cyl", &T::get_points_cyl, "Get the point where the field should be evaluated in cylindrical coordinates (the order is :math:`(r, \\phi, z)`).")
     .def("get_points_cart_ref", &T::get_points_cart_ref, "As `get_points_cart`, but returns a reference to the array (this array should be read only). The array is overwritten in place by later calls to `set_points[...]` with the same number of points; copy it before holding on to it.")
     .def("get_points_cyl_ref", &T::get_points_cyl_ref, "As `get_points_cyl`, but returns a reference to the array (this array should be read only). The array is overwritten in place by later calls to `set_points[...]` with the same number of points; copy it before holding on to it.")
     .def("set_points_cart", &T::set_points_cart, "Set the points where to evaluate the magnetic fields, in cartesian coordinates.")
     .def("set_points_cyl", &T::set_points_cyl, "Set the points where to evaluate the magnetic fields, in cylindrical coordinates (the order is :math:`(r, \\phi, z)`).")
     .def("set_points", &T::set_points, "Shorthand for `set_points_cart`.");

    // Private cache accessors, each with a deprecated public alias kept for backward compatibility;
    // in-tree code must use the private names.
    def_cache_ref(c, "B", py::overload_cast<>(&T::B_ref));
    def_cache_ref(c, "dB_by_dX", py::overload_cast<>(&T::dB_by_dX_ref));
    def_cache_ref(c, "d2B_by_dXdX", py::overload_cast<>(&T::d2B_by_dXdX_ref));
    def_cache_ref(c, "AbsB", py::overload_cast<>(&T::AbsB_ref));
    def_cache_ref(c, "GradAbsB", py::overload_cast<>(&T::GradAbsB_ref));
    def_cache_ref(c, "B_cyl", py::overload_cast<>(&T::B_cyl_ref));
    def_cache_ref(c, "A_cyl", py::overload_cast<>(&T::A_cyl_ref));
    def_cache_ref(c, "A", py::overload_cast<>(&T::A_ref));
    def_cache_ref(c, "dA_by_dX", py::overload_cast<>(&T::dA_by_dX_ref));
    def_cache_ref(c, "d2A_by_dXdX", py::overload_cast<>(&T::d2A_by_dXdX_ref));
}

void init_magneticfields(py::module_ &m){

    py::class_<InterpolationRule, shared_ptr<InterpolationRule>>(m, "InterpolationRule", "Abstract class for interpolation rules on an interval.")
        .def_readonly("degree", &InterpolationRule::degree, "The degree of the polynomial. The number of interpolation points in `degree+1`.");

    py::class_<UniformInterpolationRule, shared_ptr<UniformInterpolationRule>, InterpolationRule>(m, "UniformInterpolationRule", "Polynomial interpolation using equispaced points.")
        .def(py::init<int>())
        .def_readonly("degree", &UniformInterpolationRule::degree, "The degree of the polynomial. The number of interpolation points in `degree+1`.");
    py::class_<ChebyshevInterpolationRule, shared_ptr<ChebyshevInterpolationRule>, InterpolationRule>(m, "ChebyshevInterpolationRule", "Polynomial interpolation using chebychev points.")
        .def(py::init<int>())
        .def_readonly("degree", &ChebyshevInterpolationRule::degree, "The degree of the polynomial. The number of interpolation points in `degree+1`.");

    py::class_<RegularGridInterpolant3D<PyTensor>, shared_ptr<RegularGridInterpolant3D<PyTensor>>>(m, "RegularGridInterpolant3D",
            R"pbdoc(
            Interpolates a (vector valued) function on a uniform grid. 
            This interpolant is optimized for fast function evaluation (at the cost of memory usage). The main purpose of this class is to be used to interpolate magnetic fields and then use the interpolant for tasks such as fieldline or particle tracing for which the field needs to be evaluated many many times.
            )pbdoc")
        .def(py::init<InterpolationRule, RangeTriplet, RangeTriplet, RangeTriplet, int, bool, std::function<std::vector<bool>(Vec, Vec, Vec)>>())
        .def(py::init<InterpolationRule, RangeTriplet, RangeTriplet, RangeTriplet, int, bool>())
        .def("interpolate_batch", &RegularGridInterpolant3D<PyTensor>::interpolate_batch, "Interpolate a function by evaluating the function on all interpolation nodes simultanuously.")
        .def("evaluate", &RegularGridInterpolant3D<PyTensor>::evaluate, "Evaluate the interpolant at a point.")
        .def("evaluate_batch", &RegularGridInterpolant3D<PyTensor>::evaluate_batch, "Evaluate the interpolant at multiple points (faster than `evaluate` as it uses prefetching).");


    py::class_<CurrentBase<PyArray>, shared_ptr<CurrentBase<PyArray>>, PyCurrentBaseTrampoline>(m, "CurrentBase")
        .def(py::init<>())
        .def("get_value", &CurrentBase<PyArray>::get_value, "Get the current.");

    py::class_<Current<PyArray>, shared_ptr<Current<PyArray>>, CurrentBase<PyArray>>(m, "Current", "Simple class that wraps around a single double representing a coil current.")
        .def(py::init<double>())
        .def("set_dofs", &Current<PyArray>::set_dofs, "Set the current.")
        .def("get_dofs", &Current<PyArray>::get_dofs, "Get the current.")
        .def("get_value", &Current<PyArray>::get_value, "Get the current.");

    py::class_<Coil<PyArray>, shared_ptr<Coil<PyArray>>>(m, "Coil", "Optimizable that represents a coil, consisting of a curve and a current.")
        .def(py::init<shared_ptr<Curve<PyArray>>, shared_ptr<CurrentBase<PyArray>>>())
        .def_readonly("curve", &Coil<PyArray>::curve, "Get the underlying curve.")
        .def_readonly("current", &Coil<PyArray>::current, "Get the underlying current.");

    auto mf = py::class_<PyMagneticField, PyMagneticFieldTrampoline<PyMagneticField>, shared_ptr<PyMagneticField>>(m, "MagneticField", "Abstract class representing magnetic fields.")
        .def(py::init<>());
    register_common_field_methods<PyMagneticField>(mf);
        //.def("B", py::overload_cast<>(&PyMagneticField::B));

    auto bs = py::class_<PyBiotSavart, PyMagneticFieldTrampoline<PyBiotSavart>, shared_ptr<PyBiotSavart>, PyMagneticField>(m, "BiotSavart")
        .def(py::init<vector<shared_ptr<Coil<PyArray>>>>())
        .def("compute", &PyBiotSavart::compute)
        .def("compute_A", &PyBiotSavart::compute_A)
        .def("_fieldcache_get_or_create", &PyBiotSavart::fieldcache_get_or_create)
        .def("_fieldcache_get_status", &PyBiotSavart::fieldcache_get_status)
        .def("fieldcache_get_or_create", deprecated_alias(&PyBiotSavart::fieldcache_get_or_create, fieldcache_deprecation_message("fieldcache_get_or_create", "dB_by_dcoilcurrents() and dA_by_dcoilcurrents()")), "Deprecated alias of `_fieldcache_get_or_create`, an internal field-cache hook. Returns a live cached array.")
        .def("fieldcache_get_status", deprecated_alias(&PyBiotSavart::fieldcache_get_status, fieldcache_deprecation_message("fieldcache_get_status", "dB_by_dcoilcurrents() and dA_by_dcoilcurrents()")), "Deprecated alias of `_fieldcache_get_status`, an internal field-cache hook.")
        .def_readonly("coils", &PyBiotSavart::coils);
    register_common_field_methods<PyBiotSavart>(bs);

    auto ifield = py::class_<PyInterpolatedField, shared_ptr<PyInterpolatedField>, PyMagneticField>(m, "InterpolatedField")
        .def(py::init<shared_ptr<PyMagneticField>, InterpolationRule, RangeTriplet, RangeTriplet, RangeTriplet, bool, int, bool, std::function<std::vector<bool>(Vec, Vec, Vec)>>())
        .def(py::init<shared_ptr<PyMagneticField>, int, RangeTriplet, RangeTriplet, RangeTriplet, bool, int, bool, std::function<std::vector<bool>(Vec, Vec, Vec)>>())
        .def("estimate_error_B", &PyInterpolatedField::estimate_error_B)
        .def("estimate_error_GradAbsB", &PyInterpolatedField::estimate_error_GradAbsB)
        .def_readonly("r_range", &PyInterpolatedField::r_range)
        .def_readonly("phi_range", &PyInterpolatedField::phi_range)
        .def_readonly("z_range", &PyInterpolatedField::z_range)
        .def_readonly("rule", &PyInterpolatedField::rule);
    //register_common_field_methods<PyInterpolatedField>(ifield);
 
    auto wf = py::class_<PyWireframeField, PyMagneticFieldTrampoline<PyWireframeField>, shared_ptr<PyWireframeField>, PyMagneticField>(m, "WireframeField")
        .def(py::init<vector<PyArray>, PyIntArray&, vector<double>, PyArray&>())
        .def("compute", &PyWireframeField::compute)
        .def("_fieldcache_get_or_create", &PyWireframeField::fieldcache_get_or_create)
        .def("_fieldcache_get_status", &PyWireframeField::fieldcache_get_status)
        .def("fieldcache_get_or_create", deprecated_alias(&PyWireframeField::fieldcache_get_or_create, fieldcache_deprecation_message("fieldcache_get_or_create", "dB_by_dsegmentcurrents()")), "Deprecated alias of `_fieldcache_get_or_create`, an internal field-cache hook. Returns a live cached array.")
        .def("fieldcache_get_status", deprecated_alias(&PyWireframeField::fieldcache_get_status, fieldcache_deprecation_message("fieldcache_get_status", "dB_by_dsegmentcurrents()")), "Deprecated alias of `_fieldcache_get_status`, an internal field-cache hook.");
    register_common_field_methods<PyWireframeField>(wf);

}
