#pragma once

#include "currentpotential.h"
#include "xtensor-python/pyarray.hpp"
typedef xt::pyarray<double> PyArray;

typedef CurrentPotential<PyArray> PyCurrentPotential;

template <class CurrentPotentialBase = PyCurrentPotential> class PyCurrentPotentialTrampoline : public CurrentPotentialBase {
    public:
        using CurrentPotentialBase::CurrentPotentialBase;

        virtual int num_dofs() override {
            PYBIND11_OVERLOAD(int, CurrentPotentialBase, num_dofs);
        }
        virtual void set_dofs_impl(const vector<double>& _dofs) override {
            PYBIND11_OVERLOAD(void, CurrentPotentialBase, set_dofs_impl, _dofs);
        }
        virtual void set_dofs(const vector<double>& _dofs) override {
            PYBIND11_OVERLOAD(void, CurrentPotentialBase, set_dofs, _dofs);
        }
        virtual vector<double> get_dofs() override {
            PYBIND11_OVERLOAD(vector<double>, CurrentPotentialBase, get_dofs);
        }
        virtual void Phi_impl(PyArray& data, PyArray& quadpoints_phi, PyArray& quadpoints_theta) override {
            PYBIND11_OVERLOAD(void, CurrentPotentialBase, Phi_impl, data, quadpoints_phi, quadpoints_theta);
        }
        virtual void Phidash1_impl(PyArray& data) override {
            PYBIND11_OVERLOAD(void, CurrentPotentialBase, Phidash1_impl, data);
        }
        virtual void Phidash2_impl(PyArray& data) override {
            PYBIND11_OVERLOAD(void, CurrentPotentialBase, Phidash2_impl, data);
        }
};
