#pragma once

#include "boozermagneticfield.h"
#include "xtensor-python/pytensor.hpp"     // Numpy bindings

typedef BoozerMagneticField<xt::pytensor> PyBoozerMagneticField;

// this allows the Python code to define children of BoozerMagneticFields

template <class BoozerMagneticFieldBase = PyBoozerMagneticField> class PyBoozerMagneticFieldTrampoline : public BoozerMagneticFieldBase {
    public:
        using BoozerMagneticFieldBase::BoozerMagneticFieldBase;

        virtual void _set_points() override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _set_points);
        }

        virtual void _nu_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _nu_impl, data);
        }

        virtual void _R_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _R_impl, data);
        }

        virtual void _Z_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _Z_impl, data);
        }

        virtual void _modB_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _modB_impl, data);
        }

        virtual void _dmodBdtheta_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _dmodBdtheta_impl, data);
        }

        virtual void _dmodBdzeta_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _dmodBdzeta_impl, data);
        }

        virtual void _dmodBds_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _dmodBds_impl, data);
        }

        virtual void _G_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _G_impl, data);
        }

        virtual void _I_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _I_impl, data);
        }

        virtual void _iota_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _iota_impl, data);
        }

        virtual void _dGds_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _dGds_impl, data);
        }

        virtual void _dIds_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _dIds_impl, data);
        }

        virtual void _diotads_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _diotads_impl, data);
        }

        virtual void _psip_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _psip_impl, data);
        }
};
