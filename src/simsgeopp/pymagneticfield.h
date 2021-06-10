#pragma once

#include "magneticfield.h"
#include "xtensor-python/pytensor.hpp"     // Numpy bindings

typedef MagneticField<xt::pytensor> PyMagneticField;

// this allows the Python code to define children of Magnetic Fields

template <class MagneticFieldBase = PyMagneticField> class PyMagneticFieldTrampoline : public MagneticFieldBase {
    public:
        using MagneticFieldBase::MagneticFieldBase;

        virtual void _set_points_cb() override {
            PYBIND11_OVERLOAD(void, MagneticFieldBase, _set_points_cb);
        }

        virtual void _B_impl(typename MagneticFieldBase::Tensor2& data) override { 
            PYBIND11_OVERLOAD(void, MagneticFieldBase, _B_impl, data);
        }
        virtual void _dB_by_dX_impl(typename MagneticFieldBase::Tensor3& data) override { 
            PYBIND11_OVERLOAD(void, MagneticFieldBase, _dB_by_dX_impl, data);
        }
        virtual void _d2B_by_dXdX_impl(typename MagneticFieldBase::Tensor4& data) override { 
            PYBIND11_OVERLOAD(void, MagneticFieldBase, _d2B_by_dXdX_impl, data);
        }
        virtual void _A_impl(typename MagneticFieldBase::Tensor2& data) override { 
            PYBIND11_OVERLOAD(void, MagneticFieldBase, _A_impl, data);
        }
        virtual void _dA_by_dX_impl(typename MagneticFieldBase::Tensor3& data) override { 
            PYBIND11_OVERLOAD(void, MagneticFieldBase, _dA_by_dX_impl, data);
        }
        virtual void _d2A_by_dXdX_impl(typename MagneticFieldBase::Tensor4& data) override { 
            PYBIND11_OVERLOAD(void, MagneticFieldBase, _d2A_by_dXdX_impl, data);
        }
};
