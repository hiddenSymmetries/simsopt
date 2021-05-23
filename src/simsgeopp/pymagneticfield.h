#pragma once

#include "magneticfield.h"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;

typedef MagneticField<PyArray> PyMagneticField;

// this allows the Python code to define children of Magnetic Fields

template <class MagneticFieldBase = PyMagneticField> class PyMagneticFieldTrampoline : public MagneticFieldBase {
    public:
        using MagneticFieldBase::MagneticFieldBase;

        virtual void set_points_cb() override {
            PYBIND11_OVERLOAD(void, MagneticFieldBase, set_points_cb);
        }

        virtual void B_impl(PyArray& data) override { 
            PYBIND11_OVERLOAD(void, MagneticFieldBase, B_impl, data);
        }
        virtual void dB_by_dX_impl(PyArray& data) override { 
            PYBIND11_OVERLOAD(void, MagneticFieldBase, dB_by_dX_impl, data);
        }
        virtual void d2B_by_dXdX_impl(PyArray& data) override { 
            PYBIND11_OVERLOAD(void, MagneticFieldBase, d2B_by_dXdX_impl, data);
        }
        virtual void A_impl(PyArray& data) override { 
            PYBIND11_OVERLOAD(void, MagneticFieldBase, A_impl, data);
        }
        virtual void dA_by_dX_impl(PyArray& data) override { 
            PYBIND11_OVERLOAD(void, MagneticFieldBase, dA_by_dX_impl, data);
        }
        virtual void d2A_by_dXdX_impl(PyArray& data) override { 
            PYBIND11_OVERLOAD(void, MagneticFieldBase, d2A_by_dXdX_impl, data);
        }
};
