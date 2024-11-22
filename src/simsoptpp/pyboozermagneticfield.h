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

        virtual void _K_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _K_impl, data);
        }

        virtual void _dKdtheta_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _dKdtheta_impl, data);
        }

        virtual void _dKdzeta_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _dKdzeta_impl, data);
        }

        virtual void _K_derivs_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _K_derivs_impl, data);
        }

        virtual void _nu_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _nu_impl, data);
        }

        virtual void _dnudtheta_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _dnudtheta_impl, data);
        }

        virtual void _dnudzeta_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _dnudzeta_impl, data);
        }

        virtual void _dnuds_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _dnuds_impl, data);
        }

        virtual void _nu_derivs_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _nu_derivs_impl, data);
        }

        virtual void _R_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _R_impl, data);
        }

        virtual void _dRdtheta_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _dRdtheta_impl, data);
        }

        virtual void _dZdtheta_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _dZdtheta_impl, data);
        }

        virtual void _dRdzeta_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _dRdzeta_impl, data);
        }

        virtual void _dZdzeta_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _dZdzeta_impl, data);
        }

        virtual void _dRds_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _dRds_impl, data);
        }

        virtual void _dZds_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _dZds_impl, data);
        }

        virtual void _R_derivs_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _R_derivs_impl, data);
        }

        virtual void _Z_derivs_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _Z_derivs_impl, data);
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

        virtual void _modB_derivs_impl(typename BoozerMagneticFieldBase::Tensor2& data) override {
            PYBIND11_OVERLOAD(void, BoozerMagneticFieldBase, _modB_derivs_impl, data);
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
