#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"
namespace py = pybind11;
using std::shared_ptr;
using std::vector;
#include "tracing.h"
#include "tracing_helpers.h"
#ifdef USE_GSL
    #include "symplectic.h"
#endif

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

    m.def("particle_guiding_center_boozer_tracing", &particle_guiding_center_boozer_tracing,
        py::arg("field"),
        py::arg("stz_init"),
        py::arg("m"),
        py::arg("q"),
        py::arg("vtotal"),
        py::arg("vtang"),
        py::arg("tmax"),
        py::arg("vacuum"),
        py::arg("noK"),   
        py::arg("zetas")=vector<double>{},
        py::arg("omegas")=vector<double>{},     
        py::arg("vpars")=vector<double>{},
        py::arg("stopping_criteria")=vector<shared_ptr<StoppingCriterion>>{},
        py::arg("dt_save")=1e-6,
        py::arg("forget_exact_path")=false,
        py::arg("zetas_stop")=false,
        py::arg("vpars_stop")=false,
        py::arg("axis")=0,
        py::arg("abstol")=1e-9,
        py::arg("reltol")=1e-9,
        py::arg("solveSympl")=false,
        py::arg("predictor_step")=true,
        py::arg("roottol")=1e-9,
        py::arg("dt")=1e-7
        );

    m.def("particle_guiding_center_boozer_perturbed_tracing", &particle_guiding_center_boozer_perturbed_tracing,
        py::arg("pertrurbed_field"),
        py::arg("stz_init"),
        py::arg("m"),
        py::arg("q"),
        py::arg("vtotal"),
        py::arg("vtang"),
        py::arg("mu"),
        py::arg("tmax"),
        py::arg("abstol"),
        py::arg("reltol"),
        py::arg("vacuum"),
        py::arg("noK"),
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
