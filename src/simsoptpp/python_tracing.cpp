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

extern "C" vector<double> gpu_tracing_saw(py::array_t<double> quad_pts, py::array_t<double> srange,
        py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tmax, double tol, double psi0, int nparticles, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, double saw_omega, int saw_nharmonics);

extern "C" py::array_t<double> test_interpolation(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> loc, int n);
extern "C" py::array_t<double> test_gpu_interpolation(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> loc, int n, int n_points);

extern "C" py::array_t<double> test_derivatives(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> loc, py::array_t<double> vpar, double v_total, double m, double q, double psi0, int n_points);
extern "C" vector<double> test_timestep(py::array_t<double> quad_pts, py::array_t<double> srange,
        py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tol, double psi0, int nparticles);

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

        m.def("gpu_tracing_saw", &gpu_tracing_saw,
        py::arg("quad_pts"),
        py::arg("srange"),
        py::arg("trange"),
        py::arg("zrange"),
        py::arg("stz_init"),
        py::arg("m"),
        py::arg("q"),
        py::arg("vtotal"),
        py::arg("vtang"),
        py::arg("tmax"),
        py::arg("tol"),
        py::arg("psi0"),
        py::arg("nparticles"),
        py::arg("saw_srange"),
        py::arg("saw_m"),
        py::arg("saw_n"),
        py::arg("saw_phihats"),
        py::arg("saw_omega"),
        py::arg("saw_nharmonics")
        );

    m.def("test_interpolation", &test_interpolation,
        py::arg("quad_pts"),
        py::arg("srange"),
        py::arg("trange"),
        py::arg("zrange"),
        py::arg("loc"),
        py::arg("n")
        );

    m.def("test_gpu_interpolation", &test_gpu_interpolation,
        py::arg("quad_pts"),
        py::arg("srange"),
        py::arg("trange"),
        py::arg("zrange"),
        py::arg("loc"),
        py::arg("n"),
        py::arg("n_points")
        );


    m.def("test_derivatives", &test_derivatives,
        py::arg("quad_pts"),
        py::arg("srange"),
        py::arg("trange"),
        py::arg("zrange"),
        py::arg("loc"),
        py::arg("vpar"),
        py::arg("v_total"),
        py::arg("m"),
        py::arg("q"),
        py::arg("psi0"),
        py::arg("n_points")
        );



    m.def("simsopt_derivs", &simsopt_derivs,
        py::arg("field"),
        py::arg("loc"),
        py::arg("m"),
        py::arg("q"),
        py::arg("vtotal"),
        py::arg("vtang")
        );

    m.def("test_timestep", &test_timestep,
        py::arg("quad_pts"),
        py::arg("srange"),
        py::arg("trange"),
        py::arg("zrange"),
        py::arg("stz_init"),
        py::arg("m"),
        py::arg("q"),
        py::arg("vtotal"),
        py::arg("vtang"),
        py::arg("tol"),
        py::arg("psi0"),
        py::arg("nparticles")
        );
}
