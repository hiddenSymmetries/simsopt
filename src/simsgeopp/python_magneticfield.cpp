#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;
#include "py_shared_ptr.h"
PYBIND11_DECLARE_HOLDER_TYPE(T, py_shared_ptr<T>);
using std::shared_ptr;
using std::vector;

#include "magneticfield.h"
#include "pymagneticfield.h"
#include "regular_grid_interpolant_3d.h"
typedef MagneticField<PyArray> PyMagneticField;
typedef BiotSavart<PyArray> PyBiotSavart;
typedef InterpolatedField<PyArray> PyInterpolatedField;


template <typename T, typename S> void register_common_field_methods(S &c) {
    c.def("B", py::overload_cast<>(&T::B), py::call_guard<py::gil_scoped_release>())
     .def("dB_by_dX", py::overload_cast<>(&T::dB_by_dX), py::call_guard<py::gil_scoped_release>())
     .def("d2B_by_dXdX", py::overload_cast<>(&T::d2B_by_dXdX), py::call_guard<py::gil_scoped_release>())
     .def("B_ref", py::overload_cast<>(&T::B_ref), py::call_guard<py::gil_scoped_release>())
     .def("dB_by_dX_ref", py::overload_cast<>(&T::dB_by_dX_ref), py::call_guard<py::gil_scoped_release>())
     .def("d2B_by_dXdX_ref", py::overload_cast<>(&T::d2B_by_dXdX_ref), py::call_guard<py::gil_scoped_release>())
     .def("B_cyl_ref", py::overload_cast<>(&T::B_cyl_ref), py::call_guard<py::gil_scoped_release>())
     .def("A", py::overload_cast<>(&T::A), py::call_guard<py::gil_scoped_release>())
     .def("dA_by_dX", py::overload_cast<>(&T::dA_by_dX), py::call_guard<py::gil_scoped_release>())
     .def("d2A_by_dXdX", py::overload_cast<>(&T::d2A_by_dXdX), py::call_guard<py::gil_scoped_release>())
     .def("A_ref", py::overload_cast<>(&T::A_ref), py::call_guard<py::gil_scoped_release>())
     .def("dA_by_dX_ref", py::overload_cast<>(&T::dA_by_dX_ref), py::call_guard<py::gil_scoped_release>())
     .def("d2A_by_dXdX_ref", py::overload_cast<>(&T::d2A_by_dXdX_ref), py::call_guard<py::gil_scoped_release>())
     .def("cache_get_or_create", &T::cache_get_or_create)
     .def("cache_get_status", &T::cache_get_status)
     .def("invalidate_cache", &T::invalidate_cache)
     .def("get_points_cart_ref", &T::get_points_cart_ref)
     .def("get_points_cyl_ref", &T::get_points_cyl_ref)
     .def("get_points_cart", &T::get_points_cart)
     .def("get_points_cyl", &T::get_points_cyl)
     .def("set_points_cart", &T::set_points_cart)
     .def("set_points_cyl", &T::set_points_cyl)
     .def("set_points", &T::set_points);
     //.def_readwrite("points", &T::points);
}

void init_magneticfields(py::module_ &m){

    py::class_<InterpolationRule, shared_ptr<InterpolationRule>>(m, "InterpolationRule");
    py::class_<UniformInterpolationRule, shared_ptr<UniformInterpolationRule>>(m, "UniformInterpolationRule")
        .def(py::init<int>());
    py::class_<ChebyshevInterpolationRule, shared_ptr<ChebyshevInterpolationRule>>(m, "ChebyshevInterpolationRule")
        .def(py::init<int>());

    py::class_<Current<PyArray>, shared_ptr<Current<PyArray>>>(m, "Current")
        .def(py::init<double>())
        .def("set_dofs", &Current<PyArray>::set_dofs)
        .def("get_dofs", &Current<PyArray>::get_dofs)
        .def("set_value", &Current<PyArray>::set_value)
        .def("get_value", &Current<PyArray>::get_value);
        

    py::class_<Coil<PyArray>, shared_ptr<Coil<PyArray>>>(m, "Coil")
        .def(py::init<shared_ptr<Curve<PyArray>>, shared_ptr<Current<PyArray>>>())
        .def_readonly("curve", &Coil<PyArray>::curve)
        .def_readonly("current", &Coil<PyArray>::current);

    auto mf = py::class_<PyMagneticField, PyMagneticFieldTrampoline<PyMagneticField>, shared_ptr<PyMagneticField>>(m, "MagneticField")
        .def(py::init<>());
    register_common_field_methods<PyMagneticField>(mf);
        //.def("B", py::overload_cast<>(&PyMagneticField::B));

    auto bs = py::class_<PyBiotSavart, PyMagneticFieldTrampoline<PyBiotSavart>, shared_ptr<PyBiotSavart>, PyMagneticField>(m, "BiotSavart")
        .def(py::init<vector<shared_ptr<Coil<PyArray>>>>())
        .def("compute", &PyBiotSavart::compute);
    register_common_field_methods<PyBiotSavart>(bs);

    auto ifield = py::class_<PyInterpolatedField, PyMagneticFieldTrampoline<PyInterpolatedField>, shared_ptr<PyInterpolatedField>, PyMagneticField>(m, "InterpolatedField")
        .def(py::init<shared_ptr<PyMagneticField>, InterpolationRule, RangeTriplet, RangeTriplet, RangeTriplet>())
        .def(py::init<shared_ptr<PyMagneticField>, int, RangeTriplet, RangeTriplet, RangeTriplet>())
        .def("estimate_error", &PyInterpolatedField::estimate_error);

    register_common_field_methods<PyInterpolatedField>(ifield);
 
}
