#pragma once

#include "current.h"
typedef xt::pyarray<double> PyArray;
typedef CurrentBase<PyArray> PyCurrentBase;
#include "xtensor-python/pyarray.hpp"     // Numpy bindings

class PyCurrentBaseTrampoline : public PyCurrentBase {
    public:
        using PyCurrentBase::PyCurrentBase;

        double get_value() override {
            PYBIND11_OVERLOAD_PURE(double, PyCurrentBase, get_value);
        }
};
