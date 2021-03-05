#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;


#include "curve.cpp"
#include "pycurve.cpp"

#include "fouriercurve.cpp"
typedef FourierCurve<PyArray> PyFourierCurve;
#include "magneticaxis.cpp"
typedef StellaratorSymmetricCylindricalFourierCurve<PyArray> PyStellaratorSymmetricCylindricalFourierCurve;

#include "biot_savart.h"

namespace py = pybind11;

template <class PyFourierCurveBase = PyFourierCurve> class PyFourierCurveTrampoline : public PyCurveTrampoline<PyFourierCurveBase> {
    public:
        using PyCurveTrampoline<PyFourierCurveBase>::PyCurveTrampoline; // Inherit constructors

        int num_dofs() override {
            return PyFourierCurveBase::num_dofs();
        }

        void set_dofs_impl(const vector<double>& _dofs) override {
            PyFourierCurveBase::set_dofs_impl(_dofs);
        }

        vector<double> get_dofs() override {
            return PyFourierCurveBase::get_dofs();
        }

        void gamma_impl(PyArray& data) override {
            PyFourierCurveBase::gamma_impl(data);
        }
};

template <class PyStellaratorSymmetricCylindricalFourierCurveBase = PyStellaratorSymmetricCylindricalFourierCurve> class PyStellaratorSymmetricCylindricalFourierCurveTrampoline : public PyCurveTrampoline<PyStellaratorSymmetricCylindricalFourierCurveBase> {
    public:
        using PyCurveTrampoline<PyStellaratorSymmetricCylindricalFourierCurveBase>::PyCurveTrampoline; // Inherit constructors

        int num_dofs() override {
            return PyStellaratorSymmetricCylindricalFourierCurveBase::num_dofs();
        }

        void set_dofs_impl(const vector<double>& _dofs) override {
            PyStellaratorSymmetricCylindricalFourierCurveBase::set_dofs_impl(_dofs);
        }

        vector<double> get_dofs() override {
            return PyStellaratorSymmetricCylindricalFourierCurveBase::get_dofs();
        }

        void gamma_impl(PyArray& data) override {
            PyStellaratorSymmetricCylindricalFourierCurveBase::gamma_impl(data);
        }
};

PYBIND11_MODULE(simsgeopp, m) {
    xt::import_numpy();

    py::class_<PyCurve, std::shared_ptr<PyCurve>, PyCurveTrampoline<PyCurve>>(m, "Curve")
        .def(py::init<vector<double>>())
        .def("gamma", &PyCurve::gamma)
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
        .def("invalidate_cache", &PyFourierCurve::invalidate_cache)
        .def("set_dofs", &PyFourierCurve::set_dofs)
        .def_readonly("quadpoints", &PyCurve::quadpoints);


    py::class_<PyFourierCurve, std::shared_ptr<PyFourierCurve>, PyFourierCurveTrampoline<PyFourierCurve>>(m, "FourierCurve")
        //.def(py::init<int, int>())
        .def(py::init<vector<double>, int>())
        .def("gamma", &PyFourierCurve::gamma)
        .def("dgamma_by_dcoeff", &PyFourierCurve::dgamma_by_dcoeff)
        .def("dgamma_by_dcoeff_vjp", &PyFourierCurve::dgamma_by_dcoeff_vjp)

        .def("gammadash", &PyFourierCurve::gammadash)
        .def("dgammadash_by_dcoeff", &PyFourierCurve::dgammadash_by_dcoeff)
        .def("dgammadash_by_dcoeff_vjp", &PyFourierCurve::dgammadash_by_dcoeff_vjp)

        .def("gammadashdash", &PyFourierCurve::gammadashdash)
        .def("dgammadashdash_by_dcoeff", &PyFourierCurve::dgammadashdash_by_dcoeff)
        .def("dgammadashdash_by_dcoeff_vjp", &PyFourierCurve::dgammadashdash_by_dcoeff_vjp)

        .def("gammadashdashdash", &PyFourierCurve::gammadashdashdash)
        .def("dgammadashdashdash_by_dcoeff", &PyFourierCurve::dgammadashdashdash_by_dcoeff)
        .def("dgammadashdashdash_by_dcoeff_vjp", &PyFourierCurve::dgammadashdashdash_by_dcoeff_vjp)

        .def("incremental_arclength", &PyFourierCurve::incremental_arclength)
        .def("dincremental_arclength_by_dcoeff", &PyFourierCurve::dincremental_arclength_by_dcoeff)
        .def("kappa", &PyFourierCurve::kappa)
        .def("dkappa_by_dcoeff", &PyFourierCurve::dkappa_by_dcoeff)
        .def("torsion", &PyFourierCurve::torsion)
        .def("dtorsion_by_dcoeff", &PyFourierCurve::dtorsion_by_dcoeff)

        .def("get_dofs", &PyFourierCurve::get_dofs)
        .def("set_dofs", &PyFourierCurve::set_dofs)
        .def("num_dofs", &PyFourierCurve::num_dofs)
        .def("invalidate_cache", &PyFourierCurve::invalidate_cache)
        .def_readonly("dofs", &PyFourierCurve::dofs)
        .def_readonly("quadpoints", &PyFourierCurve::quadpoints);

    py::class_<PyStellaratorSymmetricCylindricalFourierCurve, std::shared_ptr<PyStellaratorSymmetricCylindricalFourierCurve>, PyStellaratorSymmetricCylindricalFourierCurveTrampoline<PyStellaratorSymmetricCylindricalFourierCurve>>(m, "StellaratorSymmetricCylindricalFourierCurve")
        //.def(py::init<int, int>())
        .def(py::init<vector<double>, int, int>())
        .def("gamma", &PyStellaratorSymmetricCylindricalFourierCurve::gamma)
        .def("dgamma_by_dcoeff", &PyStellaratorSymmetricCylindricalFourierCurve::dgamma_by_dcoeff)
        .def("dgamma_by_dcoeff_vjp", &PyStellaratorSymmetricCylindricalFourierCurve::dgamma_by_dcoeff_vjp)

        .def("gammadash", &PyStellaratorSymmetricCylindricalFourierCurve::gammadash)
        .def("dgammadash_by_dcoeff", &PyStellaratorSymmetricCylindricalFourierCurve::dgammadash_by_dcoeff)
        .def("dgammadash_by_dcoeff_vjp", &PyStellaratorSymmetricCylindricalFourierCurve::dgammadash_by_dcoeff_vjp)

        .def("gammadashdash", &PyStellaratorSymmetricCylindricalFourierCurve::gammadashdash)
        .def("dgammadashdash_by_dcoeff", &PyStellaratorSymmetricCylindricalFourierCurve::dgammadashdash_by_dcoeff)
        .def("dgammadashdash_by_dcoeff_vjp", &PyStellaratorSymmetricCylindricalFourierCurve::dgammadashdash_by_dcoeff_vjp)

        .def("gammadashdashdash", &PyStellaratorSymmetricCylindricalFourierCurve::gammadashdashdash)
        .def("dgammadashdashdash_by_dcoeff", &PyStellaratorSymmetricCylindricalFourierCurve::dgammadashdashdash_by_dcoeff)
        .def("dgammadashdashdash_by_dcoeff_vjp", &PyStellaratorSymmetricCylindricalFourierCurve::dgammadashdashdash_by_dcoeff_vjp)

        .def("incremental_arclength", &PyStellaratorSymmetricCylindricalFourierCurve::incremental_arclength)
        .def("dincremental_arclength_by_dcoeff", &PyStellaratorSymmetricCylindricalFourierCurve::dincremental_arclength_by_dcoeff)
        .def("kappa", &PyStellaratorSymmetricCylindricalFourierCurve::kappa)
        .def("dkappa_by_dcoeff", &PyStellaratorSymmetricCylindricalFourierCurve::dkappa_by_dcoeff)
        .def("torsion", &PyStellaratorSymmetricCylindricalFourierCurve::torsion)
        .def("dtorsion_by_dcoeff", &PyStellaratorSymmetricCylindricalFourierCurve::dtorsion_by_dcoeff)

        .def("get_dofs", &PyStellaratorSymmetricCylindricalFourierCurve::get_dofs)
        .def("set_dofs", &PyStellaratorSymmetricCylindricalFourierCurve::set_dofs)
        .def("num_dofs", &PyStellaratorSymmetricCylindricalFourierCurve::num_dofs)
        .def("invalidate_cache", &PyStellaratorSymmetricCylindricalFourierCurve::invalidate_cache)
        .def_readonly("dofs", &PyStellaratorSymmetricCylindricalFourierCurve::dofs)
        .def_readonly("quadpoints", &PyStellaratorSymmetricCylindricalFourierCurve::quadpoints)
        .def_property_readonly("nfp", &PyStellaratorSymmetricCylindricalFourierCurve::get_nfp);

    m.def("biot_savart", &biot_savart);
    m.def("biot_savart_by_dcoilcoeff_all_vjp_full", &biot_savart_by_dcoilcoeff_all_vjp_full);


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
