#pragma once

#include "surface.cpp"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;

typedef Surface<PyArray> PySurface;

// this allows the Python code to define children of surfaces if desired

template <class SurfaceBase = PySurface> class PySurfaceTrampoline : public SurfaceBase {
    public:
        using SurfaceBase::SurfaceBase;

        virtual int num_dofs() override {
            PYBIND11_OVERLOAD_PURE(int, SurfaceBase, num_dofs);
        }
        virtual void set_dofs_impl(const vector<double>& _dofs) override {
            PYBIND11_OVERLOAD_PURE(void, SurfaceBase, set_dofs_impl, _dofs);
        }
        virtual vector<double> get_dofs() override {
            PYBIND11_OVERLOAD_PURE(vector<double>, SurfaceBase, get_dofs);
        }
        virtual void gamma_impl(PyArray& data, PyArray& quadpoints_phi, PyArray& quadpoints_theta) override {
            PYBIND11_OVERLOAD_PURE(void, SurfaceBase, gamma_impl, data);
        }
        virtual void gamma_lin(PyArray& data, PyArray& quadpoints_phi, PyArray& quadpoints_theta) override {
            PYBIND11_OVERLOAD_PURE(void, SurfaceBase, gamma_lin, data);
        }
         virtual void gammadash1_impl(PyArray& data) override {
            PYBIND11_OVERLOAD_PURE(void, SurfaceBase, gammadash1_impl, data);
        }
        virtual void gammadash2_impl(PyArray& data) override {
            PYBIND11_OVERLOAD_PURE(void, SurfaceBase, gammadash2_impl, data);
        }
};
