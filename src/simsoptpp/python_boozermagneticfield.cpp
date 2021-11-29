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
     .def("G", py::overload_cast<>(&T::G), "Returns a `(npoints, 1)` array containing the magnetic field toroidal covariant component in Boozer coordinates.")
     .def("I", py::overload_cast<>(&T::I), "Returns a `(npoints, 1)` array containing the magnetic field poloidal covariant component in Boozer coordinates.")
     .def("psip", py::overload_cast<>(&T::psip), "Returns a `(npoints, 1)` array containing the (poloidal flux)/(2*pi) in Boozer coordinates.")
     .def("iota", py::overload_cast<>(&T::iota), "Returns a `(npoints, 1)` array containing the rotational transform in Boozer coordinates.")
     .def("dGds", py::overload_cast<>(&T::dGds), "Returns a `(npoints, 1)` array containing the derivative of the magnetic field toroidal covariant component wrt s in Boozer coordinates.")
     .def("dIds", py::overload_cast<>(&T::dIds), "Returns a `(npoints, 1)` array containing the derivative of the magnetic field poloidal covariant component wrt s in Boozer coordinates.")
     .def("diotads", py::overload_cast<>(&T::diotads), "Returns a `(npoints, 1)` array containing the derivative of the rotational transform wrt s in Boozer coordinates.")

     .def("dKdtheta_ref", py::overload_cast<>(&T::dKdtheta_ref), "Same as `dKdtheta`, but returns a reference to the array (this array should be read only).")
     .def("dKdzeta_ref", py::overload_cast<>(&T::dKdzeta_ref), "Same as `dKdzeta`, but returns a reference to the array (this array should be read only).")
     .def("K_derivs_ref", py::overload_cast<>(&T::K_derivs_ref), "Same as `K_derivs`, but returns a reference to the array (this array should be read only).")
     .def("K_ref", py::overload_cast<>(&T::K_ref), "Same as `K`, but returns a reference to the array (this array should be read only).")
     .def("nu_ref", py::overload_cast<>(&T::nu_ref), "Same as `nu`, but returns a reference to the array (this array should be read only).")
     .def("dnudtheta_ref", py::overload_cast<>(&T::dnudtheta_ref), "Same as `dnudtheta`, but returns a reference to the array (this array should be read only).")
     .def("dnudzeta_ref", py::overload_cast<>(&T::dnudzeta_ref), "Same as `dnudzeta`, but returns a reference to the array (this array should be read only).")
     .def("dnuds_ref", py::overload_cast<>(&T::dnuds_ref), "Same as `dnuds`, but returns a reference to the array (this array should be read only).")
     .def("nu_derivs_ref", py::overload_cast<>(&T::nu_derivs_ref), "Same as `nu_derivs`, but returns a reference to the array (this array should be read only).")
     .def("R_ref", py::overload_cast<>(&T::R_ref), "Same as `R`, but returns a reference to the array (this array should be read only).")
     .def("Z_ref", py::overload_cast<>(&T::Z_ref), "Same as `Z`, but returns a reference to the array (this array should be read only).")
     .def("dRdtheta_ref", py::overload_cast<>(&T::dRdtheta_ref), "Same as `dRdtheta`, but returns a reference to the array (this array should be read only).")
     .def("dRdzeta_ref", py::overload_cast<>(&T::dRdzeta_ref), "Same as `dRdzeta`, but returns a reference to the array (this array should be read only).")
     .def("dRds_ref", py::overload_cast<>(&T::dRds_ref), "Same as `dRds`, but returns a reference to the array (this array should be read only).")
     .def("R_derivs_ref", py::overload_cast<>(&T::R_derivs_ref), "Same as `R_derivs`, but returns a reference to the array (this array should be read only).")
     .def("dZdtheta_ref", py::overload_cast<>(&T::dZdtheta_ref), "Same as `dZdtheta`, but returns a reference to the array (this array should be read only).")
     .def("dZdzeta_ref", py::overload_cast<>(&T::dZdzeta_ref), "Same as `dZdzeta`, but returns a reference to the array (this array should be read only).")
     .def("dZds_ref", py::overload_cast<>(&T::dZds_ref), "Same as `dZds`, but returns a reference to the array (this array should be read only).")
     .def("Z_derivs_ref", py::overload_cast<>(&T::Z_derivs_ref), "Same as `Z_derivs`, but returns a reference to the array (this array should be read only).")
     .def("modB_ref", py::overload_cast<>(&T::modB_ref), "Same as `modB`, but returns a reference to the array (this array should be read only).")
     .def("dmodBdtheta_ref", py::overload_cast<>(&T::dmodBdtheta_ref), "Same as `dmodBdtheta`, but returns a reference to the array (this array should be read only).")
     .def("dmodBdzeta_ref", py::overload_cast<>(&T::dmodBdzeta_ref), "Same as `dmodBdzeta`, but returns a reference to the array (this array should be read only).")
     .def("dmodBds_ref", py::overload_cast<>(&T::dmodBds_ref), "Same as `dmodBds`, but returns a reference to the array (this array should be read only).")
     .def("modB_derivs_ref", py::overload_cast<>(&T::modB_derivs_ref), "Same as `modB_derivs`, but returns a reference to the array (this array should be read only).")
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
