#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
#include "xtensor-python/pytensor.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;
typedef xt::pytensor<double, 2, xt::layout_type::row_major> PyTensor;
using std::shared_ptr;
using std::vector;

namespace py = pybind11;
#include "boozermagneticfield.h"
#include "boozermagneticfield_interpolated.h"
#include "pyboozermagneticfield.h"
#include "regular_grid_interpolant_3d.h"
typedef InterpolatedBoozerField<xt::pytensor> PyInterpolatedBoozerField;
typedef BoozerMagneticField<xt::pytensor> PyBoozerMagneticField;

template <typename T, typename S> void register_common_field_methods(S &c) {
    c
     .def("nu", py::overload_cast<>(&T::nu), "Returns a `(npoints, 1)` array containing the difference between the Boozer and cylindrical toroidal angles, e.g. zeta_b = phi + nu.")
     .def("R", py::overload_cast<>(&T::R), "Returns a `(npoints, 1)` array containing the major radius as a function of Boozer coordinates.")
     .def("Z", py::overload_cast<>(&T::Z), "Returns a `(npoints, 1)` array containing the height as a function of Boozer coordinates.")
     .def("modB", py::overload_cast<>(&T::modB), "Returns a `(npoints, 1)` array containing the magnetic field strength in Boozer coordinates.")
     .def("dmodBdtheta", py::overload_cast<>(&T::dmodBdtheta), "Returns a `(npoints, 1)` array containing the derivative of the magnetic field strength wrt theta in Boozer coordinates.")
     .def("dmodBdzeta", py::overload_cast<>(&T::dmodBdzeta), "Returns a `(npoints, 1)` array containing the derivative of the magnetic field strength wrt zeta in Boozer coordinates.")
     .def("dmodBds", py::overload_cast<>(&T::dmodBds), "Returns a `(npoints, 1)` array containing the derivative of the magnetic field strength wrt s in Boozer coordinates.")
     .def("G", py::overload_cast<>(&T::G), "Returns a `(npoints, 1)` array containing the magnetic field toroidal covariant component in Boozer coordinates.")
     .def("I", py::overload_cast<>(&T::I), "Returns a `(npoints, 1)` array containing the magnetic field poloidal covariant component in Boozer coordinates.")
     .def("psip", py::overload_cast<>(&T::psip), "Returns a `(npoints, 1)` array containing the (poloidal flux)/(2*pi) in Boozer coordinates.")
     .def("iota", py::overload_cast<>(&T::iota), "Returns a `(npoints, 1)` array containing the rotational transform in Boozer coordinates.")
     .def("dGds", py::overload_cast<>(&T::dGds), "Returns a `(npoints, 1)` array containing the derivative of the magnetic field toroidal covariant component wrt s in Boozer coordinates.")
     .def("dIds", py::overload_cast<>(&T::dIds), "Returns a `(npoints, 1)` array containing the derivative of the magnetic field poloidal covariant component wrt s in Boozer coordinates.")
     .def("diotads", py::overload_cast<>(&T::diotads), "Returns a `(npoints, 1)` array containing the derivative of the rotational transform wrt s in Boozer coordinates.")

     .def("nu_ref", py::overload_cast<>(&T::nu_ref), "Same as `nu`, but returns a reference to the array (this array should be read only).")
     .def("R_ref", py::overload_cast<>(&T::R_ref), "Same as `R`, but returns a reference to the array (this array should be read only).")
     .def("Z_ref", py::overload_cast<>(&T::Z_ref), "Same as `Z`, but returns a reference to the array (this array should be read only).")
     .def("modB_ref", py::overload_cast<>(&T::modB_ref), "Same as `modB`, but returns a reference to the array (this array should be read only).")
     .def("dmodBdtheta_ref", py::overload_cast<>(&T::dmodBdtheta_ref), "Same as `dmodBdtheta`, but returns a reference to the array (this array should be read only).")
     .def("dmodBdzeta_ref", py::overload_cast<>(&T::dmodBdzeta_ref), "Same as `dmodBdzeta`, but returns a reference to the array (this array should be read only).")
     .def("dmodBds_ref", py::overload_cast<>(&T::dmodBds_ref), "Same as `dmodBds`, but returns a reference to the array (this array should be read only).")
     .def("G_ref", py::overload_cast<>(&T::G_ref), "Same as `G`, but returns a reference to the array (this array should be read only).")
     .def("I_ref", py::overload_cast<>(&T::I_ref), "Same as `I`, but returns a reference to the array (this array should be read only).")
     .def("psip_ref", py::overload_cast<>(&T::psip_ref), "Same as `psip`, but returns a reference to the array (this array should be read only).")
     .def("iota_ref", py::overload_cast<>(&T::iota_ref), "Same as `iota`, but returns a reference to the array (this array should be read only).")
     .def("dGds_ref", py::overload_cast<>(&T::dGds_ref), "Same as `dGds`, but returns a reference to the array (this array should be read only).")
     .def("dIds_ref", py::overload_cast<>(&T::dIds_ref), "Same as `dIds`, but returns a reference to the array (this array should be read only).")
     .def("diotads_ref", py::overload_cast<>(&T::diotads_ref), "Same as `diotads`, but returns a reference to the array (this array should be read only).")

     .def("invalidate_cache", &T::invalidate_cache, "Clear the cache. Called automatically after each call to `set_points[...]`.")
     .def("get_points", &T::get_points, "Get the point where the field should be evaluated in Boozer coordinates.")
     .def("get_points_ref", &T::get_points_ref, "As `get_points`, but returns a reference to the array (this array should be read only).")
     .def("set_points", &T::set_points, "");
}

void init_boozermagneticfields(py::module_ &m){
  auto mf = py::class_<PyBoozerMagneticField, PyBoozerMagneticFieldTrampoline<PyBoozerMagneticField>, shared_ptr<PyBoozerMagneticField>>(m, "BoozerMagneticField", "")
      .def(py::init<double>());
  register_common_field_methods<PyBoozerMagneticField>(mf);

  auto ifield = py::class_<PyInterpolatedBoozerField, shared_ptr<PyInterpolatedBoozerField>, PyBoozerMagneticField>(m, "InterpolatedBoozerField")
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
