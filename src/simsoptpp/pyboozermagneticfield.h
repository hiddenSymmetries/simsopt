#pragma once

#include "boozermagneticfield.h"
#include "xtensor-python/pytensor.hpp"     // Numpy bindings

typedef BoozerMagneticField<xt::pytensor> PyBoozerMagneticField;

// this allows the Python code to define children of Magnetic Fields

template <class BoozerMagneticFieldBase = PyBoozerMagneticField> class PyBoozerMagneticFieldTrampoline : public BoozerMagneticFieldBase {
    public:
        using BoozerMagneticFieldBase::BoozerMagneticFieldBase;

        virtual void _set_points() override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _set_points);
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

        virtual void _d2modBdtheta2_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _d2modBdtheta2_impl, data);
        }

        virtual void _d2modBdzeta2_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _d2modBdzeta2_impl, data);
        }

        virtual void _d2modBds2_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _d2modBds2_impl, data);
        }

        virtual void _d2modBdthetadzeta_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _d2modBdthetadzeta_impl, data);
        }

        virtual void _d2modBdsdzeta_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _d2modBdsdzeta_impl, data);
        }

        virtual void _d2modBdsdtheta_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _d2modBdsdtheta_impl, data);
        }

        virtual void _G_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _G_impl, data);
        }

        virtual void _iota_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _iota_impl, data);
        }

        virtual void _dGds_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _dGds_impl, data);
        }

        virtual void _diotads_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _diotads_impl, data);
        }

        virtual void _psip_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _psip_impl, data);
        }
};
