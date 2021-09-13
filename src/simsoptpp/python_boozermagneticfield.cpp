#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
#include "xtensor-python/pytensor.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;
typedef xt::pytensor<double, 2, xt::layout_type::row_major> PyTensor;
#include "py_shared_ptr.h"
PYBIND11_DECLARE_HOLDER_TYPE(T, py_shared_ptr<T>);
using std::shared_ptr;
using std::vector;

#include "boozermagneticfield.h"
#include "boozermagneticfield_interpolated.h"
#include "pyboozermagneticfield.h"
#include "regular_grid_interpolant_3d.h"
typedef InterpolatedBoozerField<xt::pytensor> PyInterpolatedBoozerField;
typedef BoozerMagneticField<xt::pytensor> PyBoozerMagneticField;

template <typename T, typename S> void register_common_field_methods(S &c) {
    c
     .def("modB", py::overload_cast<>(&T::modB), "")
     .def("dmodBdtheta", py::overload_cast<>(&T::dmodBdtheta), "")
     .def("dmodBdzeta", py::overload_cast<>(&T::dmodBdzeta), "")
     .def("dmodBds", py::overload_cast<>(&T::dmodBds), "")
     .def("d2modBdtheta2", py::overload_cast<>(&T::d2modBdtheta2), "")
     .def("d2modBdzeta2", py::overload_cast<>(&T::d2modBdzeta2), "")
     .def("d2modBds2", py::overload_cast<>(&T::d2modBds2), "")
     .def("d2modBdthetadzeta", py::overload_cast<>(&T::d2modBdthetadzeta), "")
     .def("d2modBdsdzeta", py::overload_cast<>(&T::d2modBdsdzeta), "")
     .def("d2modBdsdtheta", py::overload_cast<>(&T::d2modBdsdtheta), "")
     .def("G", py::overload_cast<>(&T::G), "")
     .def("psip", py::overload_cast<>(&T::psip), "")
     .def("iota", py::overload_cast<>(&T::iota), "")
     .def("dGds", py::overload_cast<>(&T::dGds), "")
     .def("diotads", py::overload_cast<>(&T::diotads), "")

     .def("modB_ref", py::overload_cast<>(&T::modB_ref), "")
     .def("dmodBdtheta_ref", py::overload_cast<>(&T::dmodBdtheta_ref), "")
     .def("dmodBdzeta_ref", py::overload_cast<>(&T::dmodBdzeta_ref), "")
     .def("dmodBds_ref", py::overload_cast<>(&T::dmodBds_ref), "")
     .def("d2modBdtheta2_ref", py::overload_cast<>(&T::d2modBdtheta2_ref), "")
     .def("d2modBdzeta2_ref", py::overload_cast<>(&T::d2modBdzeta2_ref), "")
     .def("d2modBds2_ref", py::overload_cast<>(&T::d2modBds2_ref), "")
     .def("d2modBdthetadzeta_ref", py::overload_cast<>(&T::d2modBdthetadzeta_ref), "")
     .def("d2modBdsdzeta_ref", py::overload_cast<>(&T::d2modBdsdzeta_ref), "")
     .def("d2modBdsdtheta_ref", py::overload_cast<>(&T::d2modBdsdtheta_ref), "")
     .def("G_ref", py::overload_cast<>(&T::G_ref), "")
     .def("psip_ref", py::overload_cast<>(&T::psip_ref), "")
     .def("iota_ref", py::overload_cast<>(&T::iota_ref), "")
     .def("dGds_ref", py::overload_cast<>(&T::dGds_ref), "")
     .def("diotads_ref", py::overload_cast<>(&T::diotads_ref), "")

     .def("invalidate_cache", &T::invalidate_cache, "Clear the cache. Called automatically after each call to `set_points[...]`.")
     .def("get_points", &T::get_points, "Get the point where the field should be evaluated in Boozer coordinates.")
     .def("get_points_ref", &T::get_points_ref, "As `get_points`, but returns a reference to the array (this array should be read only).")
     .def("set_points", &T::set_points, "");
}

void init_boozermagneticfields(py::module_ &m){
  auto mf = py::class_<PyBoozerMagneticField, PyBoozerMagneticFieldTrampoline<PyBoozerMagneticField>, py_shared_ptr<PyBoozerMagneticField>>(m, "BoozerMagneticField", "")
      .def(py::init<double>());
  register_common_field_methods<PyBoozerMagneticField>(mf);

  auto ifield = py::class_<PyInterpolatedBoozerField, py_shared_ptr<PyInterpolatedBoozerField>, PyBoozerMagneticField>(m, "InterpolatedBoozerField")
      .def(py::init<shared_ptr<PyBoozerMagneticField>, InterpolationRule, RangeTriplet, RangeTriplet, RangeTriplet, bool, int, bool>())
      .def(py::init<shared_ptr<PyBoozerMagneticField>, int, RangeTriplet, RangeTriplet, RangeTriplet, bool, int, bool>())
      .def("estimate_error_modB", &PyInterpolatedBoozerField::estimate_error_modB)
      .def("estimate_error_G", &PyInterpolatedBoozerField::estimate_error_G)
      .def("estimate_error_iota", &PyInterpolatedBoozerField::estimate_error_iota)
      .def_readonly("s_range", &PyInterpolatedBoozerField::s_range)
      .def_readonly("theta_range", &PyInterpolatedBoozerField::theta_range)
      .def_readonly("zeta_range", &PyInterpolatedBoozerField::zeta_range)
      .def_readonly("rule", &PyInterpolatedBoozerField::rule);

}
