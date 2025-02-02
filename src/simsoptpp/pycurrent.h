#pragma once

#include "current.h"
typedef xt::pyarray<std::complex<double>> PyArray;
typedef CurrentBase<PyArray> PyCurrentBase;
#include "xtensor-python/pyarray.hpp"     // Numpy bindings

class PyCurrentBaseTrampoline : public PyCurrentBase {
    public:
        using PyCurrentBase::PyCurrentBase;

        std::complex<double> get_value() override {
            PYBIND11_OVERLOAD_PURE(std::complex<double>, PyCurrentBase, get_value);
        }
};
