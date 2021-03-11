#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;

#include "surface.cpp"
#include "pysurface.cpp"
#include "surfacerzfourier.cpp"
typedef SurfaceRZFourier<PyArray> PySurfaceRZFourier;
#include "surfacexyzfourier.cpp"
typedef SurfaceXYZFourier<PyArray> PySurfaceXYZFourier;


#include "curve.cpp"
#include "pycurve.cpp"

#include "curvexyzfourier.cpp"
typedef CurveXYZFourier<PyArray> PyCurveXYZFourier;
#include "curverzfourier.cpp"
typedef CurveRZFourier<PyArray> PyCurveRZFourier;

#include "biot_savart.h"

namespace py = pybind11;

template <class PyCurveXYZFourierBase = PyCurveXYZFourier> class PyCurveXYZFourierTrampoline : public PyCurveTrampoline<PyCurveXYZFourierBase> {
    public:
        using PyCurveTrampoline<PyCurveXYZFourierBase>::PyCurveTrampoline; // Inherit constructors

        int num_dofs() override {
            return PyCurveXYZFourierBase::num_dofs();
        }

        void set_dofs_impl(const vector<double>& _dofs) override {
            PyCurveXYZFourierBase::set_dofs_impl(_dofs);
        }

        vector<double> get_dofs() override {
            return PyCurveXYZFourierBase::get_dofs();
        }

        void gamma_impl(PyArray& data, PyArray& quadpoints) override {
            PyCurveXYZFourierBase::gamma_impl(data, quadpoints);
        }
};

template <class PyCurveRZFourierBase = PyCurveRZFourier> class PyCurveRZFourierTrampoline : public PyCurveTrampoline<PyCurveRZFourierBase> {
    public:
        using PyCurveTrampoline<PyCurveRZFourierBase>::PyCurveTrampoline; // Inherit constructors

        int num_dofs() override {
            return PyCurveRZFourierBase::num_dofs();
        }

        void set_dofs_impl(const vector<double>& _dofs) override {
            PyCurveRZFourierBase::set_dofs_impl(_dofs);
        }

        vector<double> get_dofs() override {
            return PyCurveRZFourierBase::get_dofs();
        }

        void gamma_impl(PyArray& data, PyArray& quadpoints) override {
            PyCurveRZFourierBase::gamma_impl(data, quadpoints);
        }
};

template <class PySurfaceRZFourierBase = PySurfaceRZFourier> class PySurfaceRZFourierTrampoline : public PySurfaceRZFourierBase {
    public:
        using PySurfaceRZFourierBase::PySurfaceRZFourierBase;

        int num_dofs() override {
            return PySurfaceRZFourierBase::num_dofs();
        }

        void set_dofs_impl(const vector<double>& _dofs) override {
            PySurfaceRZFourierBase::set_dofs_impl(_dofs);
        }

        vector<double> get_dofs() override {
            return PySurfaceRZFourierBase::get_dofs();
        }

        void gamma_impl(PyArray& data) override {
            PySurfaceRZFourierBase::gamma_impl(data);
        }

        void fit_to_curve(PyCurve& curve, double radius) {
            PySurfaceRZFourierBase::fit_to_curve(curve, radius);
        }
};

template <class PySurfaceXYZFourierBase = PySurfaceXYZFourier> class PySurfaceXYZFourierTrampoline : public PySurfaceXYZFourierBase {
    public:
        using PySurfaceXYZFourierBase::PySurfaceXYZFourierBase;

        int num_dofs() override {
            return PySurfaceXYZFourierBase::num_dofs();
        }

        void set_dofs_impl(const vector<double>& _dofs) override {
            PySurfaceXYZFourierBase::set_dofs_impl(_dofs);
        }

        vector<double> get_dofs() override {
            return PySurfaceXYZFourierBase::get_dofs();
        }

        void gamma_impl(PyArray& data) override {
            PySurfaceXYZFourierBase::gamma_impl(data);
        }

        void fit_to_curve(PyCurve& curve, double radius) {
            PySurfaceXYZFourierBase::fit_to_curve(curve, radius);
        }
};


PYBIND11_MODULE(simsgeopp, m) {
    xt::import_numpy();
    py::class_<PySurface, std::shared_ptr<PySurface>, PySurfaceTrampoline<PySurface>>(m, "Surface")
        .def(py::init<vector<double>,vector<double>>())
        .def("gamma", &PySurface::gamma)
        .def("gammadash1", &PySurface::gammadash1)
        .def("gammadash2", &PySurface::gammadash2)
        .def("normal", &PySurface::normal)
        .def("dnormal_by_dcoeff", &PySurface::dnormal_by_dcoeff)
        .def("surface_area", &PySurface::surface_area)
        .def("dsurface_area_by_dcoeff", &PySurface::dsurface_area_by_dcoeff)
        .def("invalidate_cache", &PySurface::invalidate_cache)
        .def("set_dofs", &PySurface::set_dofs)
        .def("fit_to_curve", &PySurface::fit_to_curve, py::arg("curve"), py::arg("radius"), py::arg("flip_theta") = false)
        .def("scale", &PySurface::scale)
        .def("extend_via_normal", &PySurface::extend_via_normal)
        .def_readonly("quadpoints_phi", &PySurface::quadpoints_phi)
        .def_readonly("quadpoints_theta", &PySurface::quadpoints_theta);

    py::class_<PySurfaceRZFourier, std::shared_ptr<PySurfaceRZFourier>, PySurfaceRZFourierTrampoline<PySurfaceRZFourier>>(m, "SurfaceRZFourier")
        .def(py::init<int, int, int, bool, vector<double>,vector<double>>())
        .def_readwrite("rc", &PySurfaceRZFourier::rc)
        .def_readwrite("rs", &PySurfaceRZFourier::rs)
        .def_readwrite("zc", &PySurfaceRZFourier::zc)
        .def_readwrite("zs", &PySurfaceRZFourier::zs)
        .def("invalidate_cache", &PySurfaceRZFourier::invalidate_cache)
        .def("get_dofs", &PySurfaceRZFourier::get_dofs)
        .def("set_dofs", &PySurfaceRZFourier::set_dofs)
        .def("gamma", &PySurfaceRZFourier::gamma)
        .def("gammadash1", &PySurfaceRZFourier::gammadash1)
        .def("gammadash2", &PySurfaceRZFourier::gammadash2)
        .def("dgamma_by_dcoeff", &PySurfaceRZFourier::dgamma_by_dcoeff)
        .def("dgammadash1_by_dcoeff", &PySurfaceRZFourier::dgammadash1_by_dcoeff)
        .def("dgammadash2_by_dcoeff", &PySurfaceRZFourier::dgammadash2_by_dcoeff)
        .def("normal", &PySurfaceRZFourier::normal)
        .def("dnormal_by_dcoeff", &PySurfaceRZFourier::dnormal_by_dcoeff)
        .def("surface_area", &PySurfaceRZFourier::surface_area)
        .def("dsurface_area_by_dcoeff", &PySurfaceRZFourier::dsurface_area_by_dcoeff)
        .def("fit_to_curve", &PySurfaceRZFourier::fit_to_curve, py::arg("curve"), py::arg("radius"), py::arg("flip_theta") = false)
        .def("scale", &PySurfaceRZFourier::scale)
        .def("extend_via_normal", &PySurfaceRZFourier::extend_via_normal);


    py::class_<PySurfaceXYZFourier, std::shared_ptr<PySurfaceXYZFourier>, PySurfaceXYZFourierTrampoline<PySurfaceXYZFourier>>(m, "SurfaceXYZFourier")
        .def(py::init<int, int, int, bool, vector<double>,vector<double>>())
        .def_readwrite("xc", &PySurfaceXYZFourier::xc)
        .def_readwrite("xs", &PySurfaceXYZFourier::xs)
        .def_readwrite("yc", &PySurfaceXYZFourier::yc)
        .def_readwrite("ys", &PySurfaceXYZFourier::ys)
        .def_readwrite("zc", &PySurfaceXYZFourier::zc)
        .def_readwrite("zs", &PySurfaceXYZFourier::zs)
        .def("invalidate_cache", &PySurfaceXYZFourier::invalidate_cache)
        .def("get_dofs", &PySurfaceXYZFourier::get_dofs)
        .def("set_dofs", &PySurfaceXYZFourier::set_dofs)
        .def("gamma", &PySurfaceXYZFourier::gamma)
        .def("gammadash1", &PySurfaceXYZFourier::gammadash1)
        .def("gammadash2", &PySurfaceXYZFourier::gammadash2)
        .def("dgamma_by_dcoeff", &PySurfaceXYZFourier::dgamma_by_dcoeff)
        .def("dgammadash1_by_dcoeff", &PySurfaceXYZFourier::dgammadash1_by_dcoeff)
        .def("dgammadash2_by_dcoeff", &PySurfaceXYZFourier::dgammadash2_by_dcoeff)
        .def("normal", &PySurfaceXYZFourier::normal)
        .def("dnormal_by_dcoeff", &PySurfaceXYZFourier::dnormal_by_dcoeff)
        .def("surface_area", &PySurfaceXYZFourier::surface_area)
        .def("dsurface_area_by_dcoeff", &PySurfaceXYZFourier::dsurface_area_by_dcoeff)
        .def("fit_to_curve", &PySurfaceXYZFourier::fit_to_curve, py::arg("curve"), py::arg("radius"), py::arg("flip_theta") = false)
        .def("scale", &PySurfaceXYZFourier::scale)
        .def("extend_via_normal", &PySurfaceXYZFourier::extend_via_normal);


    py::class_<PyCurve, std::shared_ptr<PyCurve>, PyCurveTrampoline<PyCurve>>(m, "Curve")
        .def(py::init<vector<double>>())
        .def("gamma", &PyCurve::gamma)
        .def("gamma_impl", &PyCurve::gamma_impl)
        .def("gammadash", &PyCurve::gammadash)
        .def("gammadashdash", &PyCurve::gammadashdash)
        .def("gammadashdashdash", &PyCurve::gammadashdashdash)
        .def("dgamma_by_dcoeff", &PyCurve::dgamma_by_dcoeff)
        .def("dgammadash_by_dcoeff", &PyCurve::dgammadash_by_dcoeff)
        .def("dgammadashdash_by_dcoeff", &PyCurve::dgammadashdash_by_dcoeff)
        .def("dgammadashdashdash_by_dcoeff", &PyCurve::dgammadashdashdash_by_dcoeff)
        .def("incremental_arclength", &PyCurve::incremental_arclength)
        .def("dincremental_arclength_by_dcoeff", &PyCurve::dincremental_arclength_by_dcoeff)
        .def("kappa", &PyCurve::kappa)
        .def("dkappa_by_dcoeff", &PyCurve::dkappa_by_dcoeff)
        .def("torsion", &PyCurve::torsion)
        .def("dtorsion_by_dcoeff", &PyCurve::dtorsion_by_dcoeff)
        .def("invalidate_cache", &PyCurve::invalidate_cache)
        .def("least_squares_fit", &PyCurve::least_squares_fit)
        .def("set_dofs", &PyCurve::set_dofs)
        .def_readonly("quadpoints", &PyCurve::quadpoints);


    py::class_<PyCurveXYZFourier, std::shared_ptr<PyCurveXYZFourier>, PyCurveXYZFourierTrampoline<PyCurveXYZFourier>, PyCurve>(m, "CurveXYZFourier")
        //.def(py::init<int, int>())
        .def(py::init<vector<double>, int>())
        .def("gamma", &PyCurveXYZFourier::gamma)
        .def("gamma_impl", &PyCurveXYZFourier::gamma_impl)
        .def("dgamma_by_dcoeff", &PyCurveXYZFourier::dgamma_by_dcoeff)
        .def("dgamma_by_dcoeff_vjp", &PyCurveXYZFourier::dgamma_by_dcoeff_vjp)

        .def("gammadash", &PyCurveXYZFourier::gammadash)
        .def("dgammadash_by_dcoeff", &PyCurveXYZFourier::dgammadash_by_dcoeff)
        .def("dgammadash_by_dcoeff_vjp", &PyCurveXYZFourier::dgammadash_by_dcoeff_vjp)

        .def("gammadashdash", &PyCurveXYZFourier::gammadashdash)
        .def("dgammadashdash_by_dcoeff", &PyCurveXYZFourier::dgammadashdash_by_dcoeff)
        .def("dgammadashdash_by_dcoeff_vjp", &PyCurveXYZFourier::dgammadashdash_by_dcoeff_vjp)

        .def("gammadashdashdash", &PyCurveXYZFourier::gammadashdashdash)
        .def("dgammadashdashdash_by_dcoeff", &PyCurveXYZFourier::dgammadashdashdash_by_dcoeff)
        .def("dgammadashdashdash_by_dcoeff_vjp", &PyCurveXYZFourier::dgammadashdashdash_by_dcoeff_vjp)

        .def("incremental_arclength", &PyCurveXYZFourier::incremental_arclength)
        .def("dincremental_arclength_by_dcoeff", &PyCurveXYZFourier::dincremental_arclength_by_dcoeff)
        .def("kappa", &PyCurveXYZFourier::kappa)
        .def("dkappa_by_dcoeff", &PyCurveXYZFourier::dkappa_by_dcoeff)
        .def("torsion", &PyCurveXYZFourier::torsion)
        .def("dtorsion_by_dcoeff", &PyCurveXYZFourier::dtorsion_by_dcoeff)

        .def("least_squares_fit", &PyCurveXYZFourier::least_squares_fit)

        .def("get_dofs", &PyCurveXYZFourier::get_dofs)
        .def("set_dofs", &PyCurveXYZFourier::set_dofs)
        .def("num_dofs", &PyCurveXYZFourier::num_dofs)
        .def("invalidate_cache", &PyCurveXYZFourier::invalidate_cache)
        .def_readonly("dofs", &PyCurveXYZFourier::dofs)
        .def_readonly("quadpoints", &PyCurveXYZFourier::quadpoints);

    py::class_<PyCurveRZFourier, std::shared_ptr<PyCurveRZFourier>, PyCurveRZFourierTrampoline<PyCurveRZFourier>, PyCurve>(m, "CurveRZFourier")
        //.def(py::init<int, int>())
        .def(py::init<vector<double>, int, int, bool>())
        .def_readwrite("rc", &PyCurveRZFourier::rc)
        .def_readwrite("rs", &PyCurveRZFourier::rs)
        .def_readwrite("zc", &PyCurveRZFourier::zc)
        .def_readwrite("zs", &PyCurveRZFourier::zs)
        .def("gamma", &PyCurveRZFourier::gamma)
        .def("gamma_impl", &PyCurveRZFourier::gamma_impl)
        .def("dgamma_by_dcoeff", &PyCurveRZFourier::dgamma_by_dcoeff)
        .def("dgamma_by_dcoeff_vjp", &PyCurveRZFourier::dgamma_by_dcoeff_vjp)

        .def("gammadash", &PyCurveRZFourier::gammadash)
        .def("dgammadash_by_dcoeff", &PyCurveRZFourier::dgammadash_by_dcoeff)
        .def("dgammadash_by_dcoeff_vjp", &PyCurveRZFourier::dgammadash_by_dcoeff_vjp)

        .def("gammadashdash", &PyCurveRZFourier::gammadashdash)
        .def("dgammadashdash_by_dcoeff", &PyCurveRZFourier::dgammadashdash_by_dcoeff)
        .def("dgammadashdash_by_dcoeff_vjp", &PyCurveRZFourier::dgammadashdash_by_dcoeff_vjp)

        .def("gammadashdashdash", &PyCurveRZFourier::gammadashdashdash)
        .def("dgammadashdashdash_by_dcoeff", &PyCurveRZFourier::dgammadashdashdash_by_dcoeff)
        .def("dgammadashdashdash_by_dcoeff_vjp", &PyCurveRZFourier::dgammadashdashdash_by_dcoeff_vjp)

        .def("incremental_arclength", &PyCurveRZFourier::incremental_arclength)
        .def("dincremental_arclength_by_dcoeff", &PyCurveRZFourier::dincremental_arclength_by_dcoeff)
        .def("kappa", &PyCurveRZFourier::kappa)
        .def("dkappa_by_dcoeff", &PyCurveRZFourier::dkappa_by_dcoeff)
        .def("torsion", &PyCurveRZFourier::torsion)
        .def("dtorsion_by_dcoeff", &PyCurveRZFourier::dtorsion_by_dcoeff)

        .def("least_squares_fit", &PyCurveRZFourier::least_squares_fit)

        .def("get_dofs", &PyCurveRZFourier::get_dofs)
        .def("set_dofs", &PyCurveRZFourier::set_dofs)
        .def("num_dofs", &PyCurveRZFourier::num_dofs)
        .def("invalidate_cache", &PyCurveRZFourier::invalidate_cache)
        .def_readonly("quadpoints", &PyCurveRZFourier::quadpoints)
        .def_property_readonly("nfp", &PyCurveRZFourier::get_nfp);

    m.def("biot_savart", &biot_savart);
    m.def("biot_savart_B", &biot_savart_B);
    m.def("biot_savart_vjp", &biot_savart_vjp);


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
