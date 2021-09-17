#pragma once

#include "curve.h"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;

typedef Curve<PyArray> PyCurve;

template <class CurveBase = PyCurve> class PyCurveTrampoline : public CurveBase {
    public:
        using CurveBase::CurveBase;

        virtual int num_dofs() override {
            PYBIND11_OVERLOAD_PURE(int, CurveBase, num_dofs);
        }

        virtual void set_dofs_impl(const vector<double>& _dofs) override {
            PYBIND11_OVERLOAD_PURE(void, CurveBase, set_dofs_impl, _dofs);
        }

        virtual void set_dofs(const vector<double>& _dofs) override {
            PYBIND11_OVERLOAD(void, CurveBase, set_dofs, _dofs);
        }

        virtual vector<double> get_dofs() override {
            PYBIND11_OVERLOAD_PURE(vector<double>, CurveBase, get_dofs);
        }

        virtual void gamma_impl(PyArray& data, PyArray& quadpoints) override {
            PYBIND11_OVERLOAD_PURE(void, CurveBase, gamma_impl, data, quadpoints);
        }

        virtual void gammadash_impl(PyArray& data) override {
            PYBIND11_OVERLOAD(void, CurveBase, gammadash_impl, data);
        }

        virtual void gammadashdash_impl(PyArray& data) override {
            PYBIND11_OVERLOAD(void, CurveBase, gammadashdash_impl, data);
        }

        virtual void gammadashdashdash_impl(PyArray& data) override {
            PYBIND11_OVERLOAD(void, CurveBase, gammadashdashdash_impl, data);
        }

        virtual void dgamma_by_dcoeff_impl(PyArray& data) override {
            PYBIND11_OVERLOAD(void, CurveBase, dgamma_by_dcoeff_impl, data);
        }

        virtual void dgammadash_by_dcoeff_impl(PyArray& data) override {
            PYBIND11_OVERLOAD(void, CurveBase, dgammadash_by_dcoeff_impl, data);
        }

        virtual void dgammadashdash_by_dcoeff_impl(PyArray& data) override {
            PYBIND11_OVERLOAD(void, CurveBase, dgammadashdash_by_dcoeff_impl, data);
        }

        virtual void dgammadashdashdash_by_dcoeff_impl(PyArray& data) override {
            PYBIND11_OVERLOAD(void, CurveBase, dgammadashdashdash_by_dcoeff_impl, data);
        }

        virtual PyArray dgamma_by_dcoeff_vjp_impl(PyArray& v) override {
            PYBIND11_OVERLOAD(PyArray, CurveBase, dgamma_by_dcoeff_vjp_impl, v);
        }

        virtual PyArray dgammadash_by_dcoeff_vjp_impl(PyArray& v) override {
            PYBIND11_OVERLOAD(PyArray, CurveBase, dgammadash_by_dcoeff_vjp_impl, v);
        }

        virtual PyArray dgammadashdash_by_dcoeff_vjp_impl(PyArray& v) override {
            PYBIND11_OVERLOAD(PyArray, CurveBase, dgammadashdash_by_dcoeff_vjp_impl, v);
        }

        virtual PyArray dgammadashdashdash_by_dcoeff_vjp_impl(PyArray& v) override {
            PYBIND11_OVERLOAD(PyArray, CurveBase, dgammadashdashdash_by_dcoeff_vjp_impl, v);
        }

        virtual void kappa_impl(PyArray& data) override {
            PYBIND11_OVERLOAD(void, CurveBase, kappa_impl, data);
        }

        virtual void dkappa_by_dcoeff_impl(PyArray& data) override {
            PYBIND11_OVERLOAD(void, CurveBase, dkappa_by_dcoeff_impl, data);
        }

        virtual void torsion_impl(PyArray& data) override {
            PYBIND11_OVERLOAD(void, CurveBase, torsion_impl, data);
        }

        virtual void dtorsion_by_dcoeff_impl(PyArray& data) override {
            PYBIND11_OVERLOAD(void, CurveBase, dtorsion_by_dcoeff_impl, data);
        }
};
