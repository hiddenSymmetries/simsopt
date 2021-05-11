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
#include "magneticfield.h"
#include "dommaschk.h"
#include "regular_grid_interpolant_3d.h"

namespace py = pybind11;

using std::vector;
using std::shared_ptr;

void init_surfaces(py::module_ &);
void init_curves(py::module_ &);


PYBIND11_MODULE(simsgeopp, m) {
    xt::import_numpy();

    init_curves(m);
    init_surfaces(m);

    m.def("biot_savart", &biot_savart);
    m.def("biot_savart_B", &biot_savart_B);
    m.def("biot_savart_vjp", &biot_savart_vjp);

    m.def("DommaschkB" , &DommaschkB);
    m.def("DommaschkdB", &DommaschkdB);


    py::class_<RegularGridInterpolant3D<PyArray, 1>>(m, "RegularGridInterpolant3D1")
        .def(py::init<int, int, int, int>())
        .def(py::init<RangeTriplet, RangeTriplet, RangeTriplet, int>())
        .def("interpolate", &RegularGridInterpolant3D<PyArray, 1>::interpolate)
        .def("interpolate_batch", &RegularGridInterpolant3D<PyArray, 1>::interpolate_batch)
        .def("evaluate_batch_with_transform", &RegularGridInterpolant3D<PyArray, 1>::evaluate_batch_with_transform)
        .def("evaluate_batch", &RegularGridInterpolant3D<PyArray, 1>::evaluate_batch)
        .def("evaluate", &RegularGridInterpolant3D<PyArray, 1>::evaluate)
        .def("estimate_error", &RegularGridInterpolant3D<PyArray, 1>::estimate_error);
    py::class_<RegularGridInterpolant3D<PyArray, 4>>(m, "RegularGridInterpolant3D4")
        .def(py::init<int, int, int, int>())
        .def(py::init<RangeTriplet, RangeTriplet, RangeTriplet, int>())
        .def("interpolate", &RegularGridInterpolant3D<PyArray, 4>::interpolate)
        .def("interpolate_batch", &RegularGridInterpolant3D<PyArray, 4>::interpolate_batch)
        .def("evaluate_batch_with_transform", &RegularGridInterpolant3D<PyArray, 4>::evaluate_batch_with_transform)
        .def("evaluate_batch", &RegularGridInterpolant3D<PyArray, 4>::evaluate_batch)
        .def("evaluate", &RegularGridInterpolant3D<PyArray, 4>::evaluate)
        .def("estimate_error", &RegularGridInterpolant3D<PyArray, 4>::estimate_error);

    py::class_<Current<PyArray>, shared_ptr<Current<PyArray>>>(m, "Current")
        .def(py::init<double>())
        .def("set_dofs", &Current<PyArray>::set_dofs)
        .def("get_dofs", &Current<PyArray>::get_dofs)
        .def("set_value", &Current<PyArray>::set_value)
        .def("get_value", &Current<PyArray>::get_value);
        

    py::class_<Coil<PyArray>, shared_ptr<Coil<PyArray>>>(m, "Coil");

    py::class_<MagneticField<PyArray>>(m, "MagneticField");
        //.def("B", py::overload_cast<>(&MagneticField<PyArray>::B));

    py::class_<BiotSavart<PyArray>, MagneticField<PyArray>>(m, "BiotSavart")
        .def(py::init<vector<shared_ptr<Coil<PyArray>>>>())
        //.def("B_impl", &BiotSavart<PyArray>::B_impl)
        .def("B", py::overload_cast<>(&BiotSavart<PyArray>::B))
        .def("invalidate_cache", &BiotSavart<PyArray>::invalidate_cache);
        //.def("B", py::overload_cast<const Array&>(&BiotSavart<PyArray>::B));

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
