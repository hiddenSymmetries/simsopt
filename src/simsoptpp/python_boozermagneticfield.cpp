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
#include "python_deprecation.h"
typedef InterpolatedBoozerField<xt::pytensor> PyInterpolatedBoozerField;
typedef BoozerMagneticField<xt::pytensor> PyBoozerMagneticField;

template <typename T, typename S> void register_common_field_methods(S &c) {
    c
     .def("dKdtheta", py::overload_cast<>(&T::dKdtheta), "Returns a `(npoints, 1)` array containing the theta derivative of the Boozer radial covariant component, K, where B = G nabla zeta + I nabla theta + K nabla psi.")
     .def("dKdzeta", py::overload_cast<>(&T::dKdzeta), "Returns a `(npoints, 1)` array containing the zeta derivative of the Boozer radial covariant component, K, where B = G nabla zeta + I nabla theta + K nabla psi.")
     .def("K_derivs", py::overload_cast<>(&T::K_derivs), "Returns a `(npoints, 2)` array containing the dKdtheta and dKdzeta, where K is the Boozer radial covariant component, and B = G nabla zeta + I nabla theta + K nabla psi.")
     .def("K", py::overload_cast<>(&T::K), "Returns a `(npoints, 1)` array containing the Boozer radial covariant component, K, where B = G nabla zeta + I nabla theta + K nabla psi.")
     .def("nu", py::overload_cast<>(&T::nu), "Returns a `(npoints, 1)` array containing the difference between the Boozer and cylindrical toroidal angles, e.g. zeta_b = phi + nu.")
     .def("dnudtheta", py::overload_cast<>(&T::dnudtheta), "Returns a `(npoints, 1)` array containing the derivative of nu wrt theta as a function of Boozer coordinates.")
     .def("dnudzeta", py::overload_cast<>(&T::dnudzeta), "Returns a `(npoints, 1)` array containing the derivative of nu wrt zeta as a function of Boozer coordinates.")
     .def("dnuds", py::overload_cast<>(&T::dnuds), "Returns a `(npoints, 1)` array containing the derivative of nu wrt s as a function of Boozer coordinates.")
     .def("nu_derivs", py::overload_cast<>(&T::nu_derivs), "Returns a `(npoints, 3)` array containing (dnuds,dnudtheta,dnudzeta).")
     .def("R", py::overload_cast<>(&T::R), "Returns a `(npoints, 1)` array containing the major radius as a function of Boozer coordinates.")
     .def("dRdtheta", py::overload_cast<>(&T::dRdtheta), "Returns a `(npoints, 1)` array containing the derivative of the major radius wrt theta as a function of Boozer coordinates.")
     .def("dRdzeta", py::overload_cast<>(&T::dRdzeta), "Returns a `(npoints, 1)` array containing the derivative of the major radius wrt zeta as a function of Boozer coordinates.")
     .def("dRds", py::overload_cast<>(&T::dRds), "Returns a `(npoints, 1)` array containing the derivative of the major radius wrt s as a function of Boozer coordinates.")
     .def("R_derivs", py::overload_cast<>(&T::R_derivs), "Returns a `(npoints, 3)` array containing (dRds,dRdtheta,dRdzeta).")
     .def("Z", py::overload_cast<>(&T::Z), "Returns a `(npoints, 1)` array containing the height as a function of Boozer coordinates.")
     .def("dZdtheta", py::overload_cast<>(&T::dZdtheta), "Returns a `(npoints, 1)` array containing the derivative of the height wrt theta as a function of Boozer coordinates.")
     .def("dZdzeta", py::overload_cast<>(&T::dZdzeta), "Returns a `(npoints, 1)` array containing the derivative of the height wrt zeta as a function of Boozer coordinates.")
     .def("dZds", py::overload_cast<>(&T::dZds), "Returns a `(npoints, 1)` array containing the derivative of the height wrt s as a function of Boozer coordinates.")     .def("dZds", py::overload_cast<>(&T::dZds), "Returns a `(npoints, 1)` array containing the derivative of the height wrt s as a function of Boozer coordinates.")
     .def("Z_derivs", py::overload_cast<>(&T::Z_derivs), "Returns a `(npoints, 3)` array containing (dZds,dZdtheta,dZdzeta).")
     .def("modB", py::overload_cast<>(&T::modB), "Returns a `(npoints, 1)` array containing the magnetic field strength in Boozer coordinates.")
     .def("dmodBdtheta", py::overload_cast<>(&T::dmodBdtheta), "Returns a `(npoints, 1)` array containing the derivative of the magnetic field strength wrt theta in Boozer coordinates.")
     .def("dmodBdzeta", py::overload_cast<>(&T::dmodBdzeta), "Returns a `(npoints, 1)` array containing the derivative of the magnetic field strength wrt zeta in Boozer coordinates.")
     .def("dmodBds", py::overload_cast<>(&T::dmodBds), "Returns a `(npoints, 1)` array containing the derivative of the magnetic field strength wrt s in Boozer coordinates.")
     .def("modB_derivs", py::overload_cast<>(&T::modB_derivs), "Returns a `(npoints, 3)` array containing (dmodBds,dmodBdtheta,dmodBdzeta).")
     .def("d2modBdtheta2", py::overload_cast<>(&T::d2modBdtheta2), "Returns a `(npoints, 1)` array containing the second derivative of the magnetic field strength wrt theta in Boozer coordinates.")
     .def("d2modBdzeta2", py::overload_cast<>(&T::d2modBdzeta2), "Returns a `(npoints, 1)` array containing the second derivative of the magnetic field strength wrt zeta in Boozer coordinates.")
     .def("d2modBdthetadzeta", py::overload_cast<>(&T::d2modBdthetadzeta), "Returns a `(npoints, 1)` array containing the second derivative of the magnetic field strength wrt theta and zeta in Boozer coordinates.")
     .def("G", py::overload_cast<>(&T::G), "Returns a `(npoints, 1)` array containing the magnetic field toroidal covariant component in Boozer coordinates.")
     .def("I", py::overload_cast<>(&T::I), "Returns a `(npoints, 1)` array containing the magnetic field poloidal covariant component in Boozer coordinates.")
     .def("psip", py::overload_cast<>(&T::psip), "Returns a `(npoints, 1)` array containing the (poloidal flux)/(2*pi) in Boozer coordinates.")
     .def("iota", py::overload_cast<>(&T::iota), "Returns a `(npoints, 1)` array containing the rotational transform in Boozer coordinates.")
     .def("dGds", py::overload_cast<>(&T::dGds), "Returns a `(npoints, 1)` array containing the derivative of the magnetic field toroidal covariant component wrt s in Boozer coordinates.")
     .def("dIds", py::overload_cast<>(&T::dIds), "Returns a `(npoints, 1)` array containing the derivative of the magnetic field poloidal covariant component wrt s in Boozer coordinates.")
     .def("diotads", py::overload_cast<>(&T::diotads), "Returns a `(npoints, 1)` array containing the derivative of the rotational transform wrt s in Boozer coordinates.")

     .def("invalidate_cache", &T::invalidate_cache, "Clear the cache. Called automatically after each call to `set_points[...]`.")
     .def("get_points", &T::get_points, "Get the point where the field should be evaluated in Boozer coordinates.")
     .def("get_points_ref", &T::get_points_ref, "As `get_points`, but returns a reference to the array (this array should be read only). The array is overwritten in place by later calls to `set_points` with the same number of points; copy it before holding on to it.")
     .def("set_points", &T::set_points, "");

    // Private cache accessors, each with a deprecated public alias kept for backward compatibility;
    // in-tree code must use the private names.
    def_cache_ref(c, "dKdtheta", py::overload_cast<>(&T::dKdtheta_ref));
    def_cache_ref(c, "dKdzeta", py::overload_cast<>(&T::dKdzeta_ref));
    def_cache_ref(c, "K_derivs", py::overload_cast<>(&T::K_derivs_ref));
    def_cache_ref(c, "K", py::overload_cast<>(&T::K_ref));
    def_cache_ref(c, "nu", py::overload_cast<>(&T::nu_ref));
    def_cache_ref(c, "dnudtheta", py::overload_cast<>(&T::dnudtheta_ref));
    def_cache_ref(c, "dnudzeta", py::overload_cast<>(&T::dnudzeta_ref));
    def_cache_ref(c, "dnuds", py::overload_cast<>(&T::dnuds_ref));
    def_cache_ref(c, "nu_derivs", py::overload_cast<>(&T::nu_derivs_ref));
    def_cache_ref(c, "R", py::overload_cast<>(&T::R_ref));
    def_cache_ref(c, "Z", py::overload_cast<>(&T::Z_ref));
    def_cache_ref(c, "dRdtheta", py::overload_cast<>(&T::dRdtheta_ref));
    def_cache_ref(c, "dRdzeta", py::overload_cast<>(&T::dRdzeta_ref));
    def_cache_ref(c, "dRds", py::overload_cast<>(&T::dRds_ref));
    def_cache_ref(c, "R_derivs", py::overload_cast<>(&T::R_derivs_ref));
    def_cache_ref(c, "dZdtheta", py::overload_cast<>(&T::dZdtheta_ref));
    def_cache_ref(c, "dZdzeta", py::overload_cast<>(&T::dZdzeta_ref));
    def_cache_ref(c, "dZds", py::overload_cast<>(&T::dZds_ref));
    def_cache_ref(c, "Z_derivs", py::overload_cast<>(&T::Z_derivs_ref));
    def_cache_ref(c, "modB", py::overload_cast<>(&T::modB_ref));
    def_cache_ref(c, "dmodBdtheta", py::overload_cast<>(&T::dmodBdtheta_ref));
    def_cache_ref(c, "dmodBdzeta", py::overload_cast<>(&T::dmodBdzeta_ref));
    def_cache_ref(c, "dmodBds", py::overload_cast<>(&T::dmodBds_ref));
    def_cache_ref(c, "modB_derivs", py::overload_cast<>(&T::modB_derivs_ref));
    def_cache_ref(c, "d2modBdtheta2", py::overload_cast<>(&T::d2modBdtheta2_ref));
    def_cache_ref(c, "d2modBdzeta2", py::overload_cast<>(&T::d2modBdzeta2_ref));
    def_cache_ref(c, "d2modBdthetadzeta", py::overload_cast<>(&T::d2modBdthetadzeta_ref));
    def_cache_ref(c, "G", py::overload_cast<>(&T::G_ref));
    def_cache_ref(c, "I", py::overload_cast<>(&T::I_ref));
    def_cache_ref(c, "psip", py::overload_cast<>(&T::psip_ref));
    def_cache_ref(c, "iota", py::overload_cast<>(&T::iota_ref));
    def_cache_ref(c, "dGds", py::overload_cast<>(&T::dGds_ref));
    def_cache_ref(c, "dIds", py::overload_cast<>(&T::dIds_ref));
    def_cache_ref(c, "diotads", py::overload_cast<>(&T::diotads_ref));
}

void init_boozermagneticfields(py::module_ &m){
  auto mf = py::class_<PyBoozerMagneticField, PyBoozerMagneticFieldTrampoline<PyBoozerMagneticField>, shared_ptr<PyBoozerMagneticField>>(m, "BoozerMagneticField", "")
      .def(py::init<double>());
  register_common_field_methods<PyBoozerMagneticField>(mf);

  auto ifield = py::class_<PyInterpolatedBoozerField, shared_ptr<PyInterpolatedBoozerField>, PyBoozerMagneticField>(m, "InterpolatedBoozerField")
      .def(py::init<shared_ptr<PyBoozerMagneticField>, InterpolationRule, RangeTriplet, RangeTriplet, RangeTriplet, bool, int, bool>())
      .def(py::init<shared_ptr<PyBoozerMagneticField>, int, RangeTriplet, RangeTriplet, RangeTriplet, bool, int, bool>())
      .def("estimate_error_K", &PyInterpolatedBoozerField::estimate_error_K)
      .def("estimate_error_modB", &PyInterpolatedBoozerField::estimate_error_modB)
      .def("estimate_error_R", &PyInterpolatedBoozerField::estimate_error_R)
      .def("estimate_error_Z", &PyInterpolatedBoozerField::estimate_error_Z)
      .def("estimate_error_nu", &PyInterpolatedBoozerField::estimate_error_nu)
      .def("estimate_error_G", &PyInterpolatedBoozerField::estimate_error_G)
      .def("estimate_error_I", &PyInterpolatedBoozerField::estimate_error_I)
      .def("estimate_error_iota", &PyInterpolatedBoozerField::estimate_error_iota)
      .def_readonly("s_range", &PyInterpolatedBoozerField::s_range)
      .def_readonly("theta_range", &PyInterpolatedBoozerField::theta_range)
      .def_readonly("zeta_range", &PyInterpolatedBoozerField::zeta_range)
      .def_readonly("rule", &PyInterpolatedBoozerField::rule);

}
