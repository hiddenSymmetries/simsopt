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
    py::class_<StoppingCriterion, shared_ptr<StoppingCriterion>>(m, "StoppingCriterion");
    py::class_<IterationStoppingCriterion, shared_ptr<IterationStoppingCriterion>, StoppingCriterion>(m, "IterationStoppingCriterion")
        .def(py::init<int>());
    py::class_<MaxToroidalFluxStoppingCriterion, shared_ptr<MaxToroidalFluxStoppingCriterion>, StoppingCriterion>(m, "MaxToroidalFluxStoppingCriterion")
        .def(py::init<double>());
    py::class_<MinToroidalFluxStoppingCriterion, shared_ptr<MinToroidalFluxStoppingCriterion>, StoppingCriterion>(m, "MinToroidalFluxStoppingCriterion")
        .def(py::init<double>());
    py::class_<ToroidalTransitStoppingCriterion, shared_ptr<ToroidalTransitStoppingCriterion>, StoppingCriterion>(m, "ToroidalTransitStoppingCriterion")
        .def(py::init<int>());
    py::class_<VparStoppingCriterion, shared_ptr<VparStoppingCriterion>, StoppingCriterion>(m, "VparStoppingCriterion")
        .def(py::init<double>());
    py::class_<ZetaStoppingCriterion, shared_ptr<ZetaStoppingCriterion>, StoppingCriterion>(m, "ZetaStoppingCriterion")
        .def(py::init<int>());
    py::class_<StepSizeStoppingCriterion, shared_ptr<StepSizeStoppingCriterion>, StoppingCriterion>(m, "StepSizeStoppingCriterion")
        .def(py::init<double>());

    m.def("particle_guiding_center_boozer_tracing", &particle_guiding_center_boozer_tracing<xt::pytensor>,
        py::arg("field"),
        py::arg("stz_init"),
        py::arg("m"),
        py::arg("q"),
        py::arg("vtotal"),
        py::arg("vtang"),
        py::arg("tmax"),
        py::arg("dt"),
        py::arg("abstol"),
        py::arg("reltol"),
        py::arg("roottol"),
        py::arg("vacuum"),
        py::arg("noK"),
        py::arg("solveSympl")=false,
        py::arg("zetas")=vector<double>{},
        py::arg("omegas")=vector<double>{},
        py::arg("stopping_criteria")=vector<shared_ptr<StoppingCriterion>>{},
        py::arg("dt_save")=1e-6,
        py::arg("vpars")=vector<double>{},
        py::arg("zetas_stop")=false,
        py::arg("vpars_stop")=false,
        py::arg("forget_exact_path")=false,
        py::arg("axis")=0,
        py::arg("predictor_step")=true
        );

    m.def("particle_guiding_center_boozer_perturbed_tracing", &particle_guiding_center_boozer_perturbed_tracing<xt::pytensor>,
        py::arg("perturbed_field"),
        py::arg("stz_init"),
        py::arg("m"),
        py::arg("q"),
        py::arg("vtotal"),
        py::arg("vtang"),
        py::arg("mu"),
        py::arg("tmax"),
        py::arg("abstol"),
        py::arg("reltol"),
        py::arg("zetas")=vector<double>{},
        py::arg("omegas")=vector<double>{},
        py::arg("stopping_criteria")=vector<shared_ptr<StoppingCriterion>>{},
        py::arg("dt_save")=1e-6,
        py::arg("zetas_stop")=false,
        py::arg("vpars_stop")=false,
        py::arg("forget_exact_path")=false,
        py::arg("axis")=0,
        py::arg("vpars")=vector<double>{}
    );
}
