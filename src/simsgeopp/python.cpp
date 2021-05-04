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
#include "surfacexyztensorfourier.cpp"
typedef SurfaceXYZTensorFourier<PyArray> PySurfaceXYZTensorFourier;


#include "curve.cpp"
#include "pycurve.cpp"

#include "curvexyzfourier.cpp"
typedef CurveXYZFourier<PyArray> PyCurveXYZFourier;
#include "curverzfourier.cpp"
typedef CurveRZFourier<PyArray> PyCurveRZFourier; 
#include "biot_savart_py.h"
#include "biot_savart_vjp_py.h"

#include "dommaschk.cpp"
#include "reiman.cpp"

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
        using PySurfaceRZFourierBase::mpol;
        using PySurfaceRZFourierBase::ntor;
        using PySurfaceRZFourierBase::nfp;
        using PySurfaceRZFourierBase::stellsym;

        int num_dofs() override {
            return PySurfaceRZFourierBase::num_dofs();
        }

        void set_dofs_impl(const vector<double>& _dofs) override {
            PySurfaceRZFourierBase::set_dofs_impl(_dofs);
        }

        vector<double> get_dofs() override {
            return PySurfaceRZFourierBase::get_dofs();
        }

        void gamma_impl(PyArray& data, PyArray& quadpoints_phi, PyArray& quadpoints_theta) override {
            PySurfaceRZFourierBase::gamma_impl(data, quadpoints_phi, quadpoints_theta);
        }

        void gamma_lin(PyArray& data, PyArray& quadpoints_phi, PyArray& quadpoints_theta) override {
            PySurfaceRZFourierBase::gamma_lin(data, quadpoints_phi, quadpoints_theta);
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

        void gamma_impl(PyArray& data, PyArray& quadpoints_phi, PyArray& quadpoints_theta) override {
            PySurfaceXYZFourierBase::gamma_impl(data, quadpoints_phi, quadpoints_theta);
        }

        void gamma_lin(PyArray& data, PyArray& quadpoints_phi, PyArray& quadpoints_theta) override {
            PySurfaceXYZFourierBase::gamma_lin(data, quadpoints_phi, quadpoints_theta);
        }

        void fit_to_curve(PyCurve& curve, double radius) {
            PySurfaceXYZFourierBase::fit_to_curve(curve, radius);
        }
};

template <class PySurfaceXYZTensorFourierBase = PySurfaceXYZTensorFourier> class PySurfaceXYZTensorFourierTrampoline : public PySurfaceXYZTensorFourierBase {
    public:
        using PySurfaceXYZTensorFourierBase::PySurfaceXYZTensorFourierBase;

        int num_dofs() override {
            return PySurfaceXYZTensorFourierBase::num_dofs();
        }

        void set_dofs_impl(const vector<double>& _dofs) override {
            PySurfaceXYZTensorFourierBase::set_dofs_impl(_dofs);
        }

        vector<double> get_dofs() override {
            return PySurfaceXYZTensorFourierBase::get_dofs();
        }

        void gamma_impl(PyArray& data, PyArray& quadpoints_phi, PyArray& quadpoints_theta) override {
            PySurfaceXYZTensorFourierBase::gamma_impl(data, quadpoints_phi, quadpoints_theta);
        }

        void gamma_lin(PyArray& data, PyArray& quadpoints_phi, PyArray& quadpoints_theta) override {
            PySurfaceXYZTensorFourierBase::gamma_lin(data, quadpoints_phi, quadpoints_theta);
        }


        void fit_to_curve(PyCurve& curve, double radius) {
            PySurfaceXYZTensorFourierBase::fit_to_curve(curve, radius);
        }
};

template <typename T, typename S> void register_common_surface_methods(S &s) {
    s.def("gamma", &T::gamma)
     .def("gamma_impl", &T::gamma_impl)
     .def("gamma_lin", &T::gamma_lin)
     .def("dgamma_by_dcoeff", &T::dgamma_by_dcoeff)
     .def("gammadash1", &T::gammadash1)
     .def("dgammadash1_by_dcoeff", &T::dgammadash1_by_dcoeff)
     .def("gammadash2", &T::gammadash2)
     .def("dgammadash2_by_dcoeff", &T::dgammadash2_by_dcoeff)
     .def("normal", &T::normal)
     .def("dnormal_by_dcoeff", &T::dnormal_by_dcoeff)
     .def("d2normal_by_dcoeffdcoeff", &T::d2normal_by_dcoeffdcoeff)
     .def("area", &T::area)
     .def("darea_by_dcoeff", &T::darea_by_dcoeff)
     .def("d2area_by_dcoeffdcoeff", &T::d2area_by_dcoeffdcoeff)
     .def("volume", &T::volume)
     .def("dvolume_by_dcoeff", &T::dvolume_by_dcoeff)
     .def("d2volume_by_dcoeffdcoeff", &T::d2volume_by_dcoeffdcoeff)
     .def("fit_to_curve", &T::fit_to_curve, py::arg("curve"), py::arg("radius"), py::arg("flip_theta") = false)
     .def("scale", &T::scale)
     .def("extend_via_normal", &T::extend_via_normal)
     .def("least_squares_fit", &T::least_squares_fit)
     .def("invalidate_cache", &T::invalidate_cache)
     .def("set_dofs", &T::set_dofs)
     .def("get_dofs", &T::get_dofs)
     .def_readonly("quadpoints_phi", &T::quadpoints_phi)
     .def_readonly("quadpoints_theta", &T::quadpoints_theta);
}
template <typename T, typename S> void register_common_curve_methods(S &c) {
    c.def("gamma", &T::gamma)
     .def("gamma_impl", &T::gamma_impl)
     .def("gammadash", &T::gammadash)
     .def("gammadashdash", &T::gammadashdash)
     .def("gammadashdashdash", &T::gammadashdashdash)

     .def("dgamma_by_dcoeff", &T::dgamma_by_dcoeff)
     .def("dgammadash_by_dcoeff", &T::dgammadash_by_dcoeff)
     .def("dgammadashdash_by_dcoeff", &T::dgammadashdash_by_dcoeff)
     .def("dgammadashdashdash_by_dcoeff", &T::dgammadashdashdash_by_dcoeff)

     .def("dgamma_by_dcoeff_vjp", &T::dgamma_by_dcoeff_vjp)
     .def("dgammadash_by_dcoeff_vjp", &T::dgammadash_by_dcoeff_vjp)
     .def("dgammadashdash_by_dcoeff_vjp", &T::dgammadashdash_by_dcoeff_vjp)
     .def("dgammadashdashdash_by_dcoeff_vjp", &T::dgammadashdashdash_by_dcoeff_vjp)

     .def("incremental_arclength", &T::incremental_arclength)
     .def("dincremental_arclength_by_dcoeff", &T::dincremental_arclength_by_dcoeff)
     .def("kappa", &T::kappa)
     .def("dkappa_by_dcoeff", &T::dkappa_by_dcoeff)
     .def("torsion", &T::torsion)
     .def("dtorsion_by_dcoeff", &T::dtorsion_by_dcoeff)
     .def("invalidate_cache", &T::invalidate_cache)
     .def("least_squares_fit", &T::least_squares_fit)

     .def("set_dofs", &T::set_dofs)
     .def("get_dofs", &T::get_dofs)
     .def("num_dofs", &T::num_dofs)
     .def_readonly("quadpoints", &T::quadpoints);
}

PYBIND11_MODULE(simsgeopp, m) {
    xt::import_numpy();
    auto pysurface = py::class_<PySurface, std::shared_ptr<PySurface>, PySurfaceTrampoline<PySurface>>(m, "Surface")
        .def(py::init<vector<double>,vector<double>>());
    register_common_surface_methods<PySurface>(pysurface);

    auto pysurfacerzfourier = py::class_<PySurfaceRZFourier, std::shared_ptr<PySurfaceRZFourier>, PySurfaceRZFourierTrampoline<PySurfaceRZFourier>>(m, "SurfaceRZFourier")
        .def(py::init<int, int, int, bool, vector<double>, vector<double>>())
        .def(py::init<int, int, int, bool, int, int>())
        .def_readwrite("rc", &PySurfaceRZFourier::rc)
        .def_readwrite("rs", &PySurfaceRZFourier::rs)
        .def_readwrite("zc", &PySurfaceRZFourier::zc)
        .def_readwrite("zs", &PySurfaceRZFourier::zs)
        .def_readwrite("mpol", &PySurfaceRZFourier::mpol)
        .def_readwrite("ntor", &PySurfaceRZFourier::ntor)
        .def_readwrite("nfp", &PySurfaceRZFourier::nfp)
        .def_readwrite("stellsym", &PySurfaceRZFourier::stellsym)
        .def("allocate", &PySurfaceRZFourier::allocate);
    register_common_surface_methods<PySurfaceRZFourier>(pysurfacerzfourier);

    auto pysurfacexyzfourier = py::class_<PySurfaceXYZFourier, std::shared_ptr<PySurfaceXYZFourier>, PySurfaceXYZFourierTrampoline<PySurfaceXYZFourier>>(m, "SurfaceXYZFourier")
        .def(py::init<int, int, int, bool, vector<double>, vector<double>>())
        .def(py::init<int, int, int, bool, int, int>())
        .def_readwrite("xc", &PySurfaceXYZFourier::xc)
        .def_readwrite("xs", &PySurfaceXYZFourier::xs)
        .def_readwrite("yc", &PySurfaceXYZFourier::yc)
        .def_readwrite("ys", &PySurfaceXYZFourier::ys)
        .def_readwrite("zc", &PySurfaceXYZFourier::zc)
        .def_readwrite("zs", &PySurfaceXYZFourier::zs)
        .def_readwrite("mpol",&PySurfaceXYZFourier::mpol)
        .def_readwrite("ntor",&PySurfaceXYZFourier::ntor)
        .def_readwrite("nfp", &PySurfaceXYZFourier::nfp)
        .def_readwrite("stellsym", &PySurfaceXYZFourier::stellsym);
    register_common_surface_methods<PySurfaceXYZFourier>(pysurfacexyzfourier);

    auto pysurfacexyztensorfourier = py::class_<PySurfaceXYZTensorFourier, std::shared_ptr<PySurfaceXYZTensorFourier>, PySurfaceXYZTensorFourierTrampoline<PySurfaceXYZTensorFourier>>(m, "SurfaceXYZTensorFourier")
        .def(py::init<int, int, int, bool, vector<bool>, vector<double>, vector<double>>())
        .def(py::init<int, int, int, bool, vector<bool>, int, int>())
        .def_readwrite("x", &PySurfaceXYZTensorFourier::x)
        .def_readwrite("y", &PySurfaceXYZTensorFourier::y)
        .def_readwrite("z", &PySurfaceXYZTensorFourier::z)
        .def_readwrite("ntor", &PySurfaceXYZTensorFourier::ntor)
        .def_readwrite("mpol", &PySurfaceXYZTensorFourier::mpol)
        .def_readwrite("nfp", &PySurfaceXYZTensorFourier::nfp)
        .def_readwrite("stellsym", &PySurfaceXYZTensorFourier::stellsym);
    register_common_surface_methods<PySurfaceXYZTensorFourier>(pysurfacexyztensorfourier);


    auto pycurve = py::class_<PyCurve, std::shared_ptr<PyCurve>, PyCurveTrampoline<PyCurve>>(m, "Curve")
        .def(py::init<vector<double>>());
    register_common_curve_methods<PyCurve>(pycurve);

    auto pycurvexyzfourier = py::class_<PyCurveXYZFourier, std::shared_ptr<PyCurveXYZFourier>, PyCurveXYZFourierTrampoline<PyCurveXYZFourier>, PyCurve>(m, "CurveXYZFourier")
        .def(py::init<vector<double>, int>())
        .def_readonly("dofs", &PyCurveXYZFourier::dofs);
    register_common_curve_methods<PyCurveXYZFourier>(pycurvexyzfourier);

    auto pycurverzfourier = py::class_<PyCurveRZFourier, std::shared_ptr<PyCurveRZFourier>, PyCurveRZFourierTrampoline<PyCurveRZFourier>, PyCurve>(m, "CurveRZFourier")
        //.def(py::init<int, int>())
        .def(py::init<vector<double>, int, int, bool>())
        .def_readwrite("rc", &PyCurveRZFourier::rc)
        .def_readwrite("rs", &PyCurveRZFourier::rs)
        .def_readwrite("zc", &PyCurveRZFourier::zc)
        .def_readwrite("zs", &PyCurveRZFourier::zs)
        .def_property_readonly("nfp", &PyCurveRZFourier::get_nfp);
    register_common_curve_methods<PyCurveRZFourier>(pycurverzfourier);

    m.def("biot_savart", &biot_savart);
    m.def("biot_savart_B", &biot_savart_B);
    m.def("biot_savart_vjp", &biot_savart_vjp);

    m.def("DommaschkB" , &DommaschkB);
    m.def("DommaschkdB", &DommaschkdB);

    m.def("ReimanB" , &ReimanB);
    m.def("ReimandB", &ReimandB);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
