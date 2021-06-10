#pragma once

#include "magneticfield.h"
#include "xtensor-python/pytensor.hpp"     // Numpy bindings

typedef MagneticField<xt::pytensor> PyMagneticField;

// this allows the Python code to define children of Magnetic Fields

template <class MagneticFieldBase = PyMagneticField> class PyMagneticFieldTrampoline : public MagneticFieldBase {
    public:
        using MagneticFieldBase::MagneticFieldBase;

        virtual void __set_points_cb() override {
            PYBIND11_OVERLOAD(void, MagneticFieldBase, __set_points_cb);
        }

        virtual void __B_impl(typename MagneticFieldBase::Tensor2& data) override { 
            PYBIND11_OVERLOAD(void, MagneticFieldBase, __B_impl, data);
        }
        virtual void __dB_by_dX_impl(typename MagneticFieldBase::Tensor3& data) override { 
            PYBIND11_OVERLOAD(void, MagneticFieldBase, __dB_by_dX_impl, data);
        }
        virtual void __d2B_by_dXdX_impl(typename MagneticFieldBase::Tensor4& data) override { 
            PYBIND11_OVERLOAD(void, MagneticFieldBase, __d2B_by_dXdX_impl, data);
        }
        virtual void __A_impl(typename MagneticFieldBase::Tensor2& data) override { 
            PYBIND11_OVERLOAD(void, MagneticFieldBase, __A_impl, data);
        }
        virtual void __dA_by_dX_impl(typename MagneticFieldBase::Tensor3& data) override { 
            PYBIND11_OVERLOAD(void, MagneticFieldBase, __dA_by_dX_impl, data);
        }
        virtual void __d2A_by_dXdX_impl(typename MagneticFieldBase::Tensor4& data) override { 
            PYBIND11_OVERLOAD(void, MagneticFieldBase, __d2A_by_dXdX_impl, data);
        }
};
