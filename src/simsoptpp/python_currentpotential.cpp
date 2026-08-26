#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"
#include "xtensor-python/pyarray.hpp"
typedef xt::pyarray<double> PyArray;
using std::shared_ptr;
using std::vector;

#include "currentpotential.h"
#include "pycurrentpotential.h"
#include "pysurface.h"
#include "surface.h"
#include "currentpotentialfourier.h"
typedef CurrentPotentialFourier<PyArray> PyCurrentPotentialFourier;

template <class PyCurrentPotentialFourierBase = PyCurrentPotentialFourier> class PyCurrentPotentialFourierTrampoline : public PyCurrentPotentialTrampoline<PyCurrentPotentialFourierBase> {
    public:
        using PyCurrentPotentialTrampoline<PyCurrentPotentialFourierBase>::PyCurrentPotentialTrampoline;
        using PyCurrentPotentialFourierBase::mpol;
        using PyCurrentPotentialFourierBase::ntor;
        using PyCurrentPotentialFourierBase::nfp;
        using PyCurrentPotentialFourierBase::stellsym;

        int num_dofs() override {
            return PyCurrentPotentialFourierBase::num_dofs();
        }

        void set_dofs_impl(const vector<double>& _dofs) override {
            PyCurrentPotentialFourierBase::set_dofs_impl(_dofs);
        }

        vector<double> get_dofs() override {
            return PyCurrentPotentialFourierBase::get_dofs();
        }

        void Phi_impl(PyArray& data, PyArray& quadpoints_phi, PyArray& quadpoints_theta) override {
            PyCurrentPotentialFourierBase::Phi_impl(data, quadpoints_phi, quadpoints_theta);
        }
};

template <typename T, typename S> void register_common_currentpotential_methods(S &s) {
    s.def("Phi", pybind11::overload_cast<>(&T::Phi))
     .def("K_impl_helper", &T::K_impl_helper)
     .def("K_matrix_impl_helper", &T::K_matrix_impl_helper)
     .def("K_rhs_impl_helper", &T::K_rhs_impl_helper)
     .def("set_dofs_impl", &T::set_dofs_impl)
     .def("Phidash1", pybind11::overload_cast<>(&T::Phidash1))
     .def("Phidash2", pybind11::overload_cast<>(&T::Phidash2))
     .def("Phidash1_impl", &T::Phidash1_impl)
     .def("Phidash2_impl", &T::Phidash2_impl)
     .def("invalidate_cache", &T::invalidate_cache)
     .def("set_dofs", &T::set_dofs)
     .def("get_dofs", &T::get_dofs)
     .def_readonly("quadpoints_phi", &T::quadpoints_phi)
     .def_readonly("quadpoints_theta", &T::quadpoints_theta);
}

void init_currentpotential(pybind11::module_ &m){
    auto pycurrentpotential = pybind11::class_<PyCurrentPotential, shared_ptr<PyCurrentPotential>, PyCurrentPotentialTrampoline<PyCurrentPotential>>(m, "CurrentPotential")
        .def(pybind11::init<vector<double>,vector<double>, double, double>());
    register_common_currentpotential_methods<PyCurrentPotential>(pycurrentpotential);

    auto pycurrentpotentialfourier = pybind11::class_<PyCurrentPotentialFourier, shared_ptr<PyCurrentPotentialFourier>, PyCurrentPotentialFourierTrampoline<PyCurrentPotentialFourier>>(m, "CurrentPotentialFourier")
        .def(pybind11::init<int, int, int, bool, vector<double>, vector<double>, double, double>())
        .def_readwrite("phic", &PyCurrentPotentialFourier::phic)
        .def_readwrite("phis", &PyCurrentPotentialFourier::phis)
        .def_readwrite("mpol", &PyCurrentPotentialFourier::mpol)
        .def_readwrite("ntor", &PyCurrentPotentialFourier::ntor)
        .def_readwrite("nfp", &PyCurrentPotentialFourier::nfp)
        .def_readwrite("stellsym", &PyCurrentPotentialFourier::stellsym)
        .def_readwrite("net_poloidal_current_amperes", &PyCurrentPotentialFourier::net_poloidal_current_amperes)
        .def_readwrite("net_toroidal_current_amperes", &PyCurrentPotentialFourier::net_toroidal_current_amperes)
        .def("allocate", &PyCurrentPotentialFourier::allocate);
    register_common_currentpotential_methods<PyCurrentPotentialFourier>(pycurrentpotentialfourier);
}
