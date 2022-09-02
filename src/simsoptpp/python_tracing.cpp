#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"
namespace py = pybind11;
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;
#include "xtensor-python/pytensor.hpp"     // Numpy bindings
typedef xt::pytensor<double, 2, xt::layout_type::row_major> PyTensor;
using std::shared_ptr;
using std::vector;
#include "tracing.h"

void init_tracing(py::module_ &m){


    py::class_<StoppingCriterion<PyArray>, shared_ptr<StoppingCriterion<PyArray>>>(m, "StoppingCriterion");
    py::class_<IterationStoppingCriterion<PyArray>, shared_ptr<IterationStoppingCriterion<PyArray>>, StoppingCriterion<PyArray>>(m, "IterationStoppingCriterion")
        .def(py::init<int>());
    py::class_<MaxToroidalFluxStoppingCriterion<PyArray>, shared_ptr<MaxToroidalFluxStoppingCriterion<PyArray>>, StoppingCriterion<PyArray>>(m, "MaxToroidalFluxStoppingCriterion")
        .def(py::init<double>());
    py::class_<MinToroidalFluxStoppingCriterion<PyArray>, shared_ptr<MinToroidalFluxStoppingCriterion<PyArray>>, StoppingCriterion<PyArray>>(m, "MinToroidalFluxStoppingCriterion")
        .def(py::init<double>());
    py::class_<ToroidalTransitStoppingCriterion<PyArray>, shared_ptr<ToroidalTransitStoppingCriterion<PyArray>>, StoppingCriterion<PyArray>>(m, "ToroidalTransitStoppingCriterion")
        .def(py::init<int,bool>());
    py::class_<LevelsetStoppingCriterion<PyTensor,PyArray>, shared_ptr<LevelsetStoppingCriterion<PyTensor,PyArray>>, StoppingCriterion<PyArray>>(m, "LevelsetStoppingCriterion")
        .def(py::init<shared_ptr<RegularGridInterpolant3D<PyTensor>>>());

    m.def("particle_guiding_center_boozer_tracing", &particle_guiding_center_boozer_tracing<xt::pytensor,PyArray>,
        py::arg("field"),
        py::arg("stz_init"),
        py::arg("m"),
        py::arg("q"),
        py::arg("vtotal"),
        py::arg("vtang"),
        py::arg("tmax"),
        py::arg("tol"),
        py::arg("vacuum"),
        py::arg("noK"),
        py::arg("zetas")=vector<double>{},
        py::arg("stopping_criteria")=vector<shared_ptr<StoppingCriterion<PyArray>>>{}
        );

    m.def("particle_guiding_center_tracing", &particle_guiding_center_tracing<xt::pytensor,PyArray>,
        py::arg("field"),
        py::arg("xyz_init"),
        py::arg("m"),
        py::arg("q"),
        py::arg("vtotal"),
        py::arg("vtang"),
        py::arg("tmax"),
        py::arg("tol"),
        py::arg("vacuum"),
        py::arg("phis")=vector<double>{},
        py::arg("stopping_criteria")=vector<shared_ptr<StoppingCriterion<PyArray>>>{}
        );

    m.def("particle_fullorbit_tracing", &particle_fullorbit_tracing<xt::pytensor,PyArray>,
        py::arg("field"),
        py::arg("xyz_init"),
        py::arg("v_init"),
        py::arg("m"),
        py::arg("q"),
        py::arg("tmax"),
        py::arg("tol"),
        py::arg("phis")=vector<double>{},
        py::arg("stopping_criteria")=vector<shared_ptr<StoppingCriterion<PyArray>>>{}
        );

    m.def("fieldline_tracing", &fieldline_tracing<xt::pytensor,PyArray>,
            py::arg("field"),
            py::arg("xyz_init"),
            py::arg("tmax"),
            py::arg("tol"),
            py::arg("phis")=vector<double>{},
            py::arg("stopping_criteria")=vector<shared_ptr<StoppingCriterion<PyArray>>>{});

    m.def("get_phi", &get_phi);
}
