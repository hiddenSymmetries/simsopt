#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;
using std::shared_ptr;

namespace py = pybind11;

#include "curve.h"
#include "pycurve.h"

#include "curvexyzfourier.h"
typedef CurveXYZFourier<PyArray> PyCurveXYZFourier;
#include "curverzfourier.h"
typedef CurveRZFourier<PyArray> PyCurveRZFourier; 

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

     .def("dgamma_by_dcoeff_vjp_impl", &T::dgamma_by_dcoeff_vjp_impl)
     .def("dgammadash_by_dcoeff_vjp_impl", &T::dgammadash_by_dcoeff_vjp_impl)
     .def("dgammadashdash_by_dcoeff_vjp_impl", &T::dgammadashdash_by_dcoeff_vjp_impl)
     .def("dgammadashdashdash_by_dcoeff_vjp_impl", &T::dgammadashdashdash_by_dcoeff_vjp_impl)

     .def("incremental_arclength", &T::incremental_arclength)
     .def("dincremental_arclength_by_dcoeff", &T::dincremental_arclength_by_dcoeff)
     .def("kappa", &T::kappa)
     .def("dkappa_by_dcoeff", &T::dkappa_by_dcoeff)
     .def("torsion", &T::torsion)
     .def("dtorsion_by_dcoeff", &T::dtorsion_by_dcoeff)
     .def("invalidate_cache", &T::invalidate_cache)
     .def("least_squares_fit", &T::least_squares_fit)

     .def("set_dofs", &T::set_dofs)
     .def("set_dofs_impl", &T::set_dofs_impl)
     .def("get_dofs", &T::get_dofs)
     .def("num_dofs", &T::num_dofs)
     .def_readonly("quadpoints", &T::quadpoints);
}

void init_curves(py::module_ &m) {
    auto pycurve = py::class_<PyCurve, shared_ptr<PyCurve>, PyCurveTrampoline<PyCurve>>(m, "Curve")
        .def(py::init<vector<double>>());
    register_common_curve_methods<PyCurve>(pycurve);

    auto pycurvexyzfourier = py::class_<PyCurveXYZFourier, shared_ptr<PyCurveXYZFourier>, PyCurveXYZFourierTrampoline<PyCurveXYZFourier>, PyCurve>(m, "CurveXYZFourier")
        .def(py::init<vector<double>, int>())
        .def_readonly("dofs", &PyCurveXYZFourier::dofs)
        .def_readonly("order", &PyCurveXYZFourier::order);
    register_common_curve_methods<PyCurveXYZFourier>(pycurvexyzfourier);

    auto pycurverzfourier = py::class_<PyCurveRZFourier, shared_ptr<PyCurveRZFourier>, PyCurveRZFourierTrampoline<PyCurveRZFourier>, PyCurve>(m, "CurveRZFourier")
        //.def(py::init<int, int>())
        .def(py::init<vector<double>, int, int, bool>())
        .def_readwrite("rc", &PyCurveRZFourier::rc)
        .def_readwrite("rs", &PyCurveRZFourier::rs)
        .def_readwrite("zc", &PyCurveRZFourier::zc)
        .def_readwrite("zs", &PyCurveRZFourier::zs)
        .def_readonly("order", &PyCurveRZFourier::order)
        .def_readonly("stellsym", &PyCurveRZFourier::stellsym)
        .def_readonly("nfp", &PyCurveRZFourier::nfp);
    register_common_curve_methods<PyCurveRZFourier>(pycurverzfourier);
}
