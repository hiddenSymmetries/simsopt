#pragma once
#include "shearalfvenwave.h"
#include "pybind11/pybind11.h"

template <class ShearAlfvenWaveBase = ShearAlfvenWave> 
class ShearAlfvenWaveTrampoline : public ShearAlfvenWaveBase {
public:
    using ShearAlfvenWaveBase::ShearAlfvenWaveBase; // Inherit constructors

    void _Phi_impl(typename ShearAlfvenWaveBase::Array2& data) override {
        PYBIND11_OVERLOAD(void, ShearAlfvenWaveBase, _Phi_impl, data);
    }

    void _dPhidpsi_impl(typename ShearAlfvenWaveBase::Array2& data) override {
        PYBIND11_OVERLOAD(void, ShearAlfvenWaveBase, _dPhidpsi_impl, data);
    }

    void _dPhidtheta_impl(typename ShearAlfvenWaveBase::Array2& data) override {
        PYBIND11_OVERLOAD(void, ShearAlfvenWaveBase, _dPhidtheta_impl, data);
    }

    void _dPhidzeta_impl(typename ShearAlfvenWaveBase::Array2& data) override {
        PYBIND11_OVERLOAD(void, ShearAlfvenWaveBase, _dPhidzeta_impl, data);
    }

    void _Phidot_impl(typename ShearAlfvenWaveBase::Array2& data) override {
        PYBIND11_OVERLOAD(void, ShearAlfvenWaveBase, _Phidot_impl, data);
    }

    void _alpha_impl(typename ShearAlfvenWaveBase::Array2& data) override {
        PYBIND11_OVERLOAD(void, ShearAlfvenWaveBase, _alpha_impl, data);
    }

    void _alphadot_impl(typename ShearAlfvenWaveBase::Array2& data) override {
        PYBIND11_OVERLOAD(void, ShearAlfvenWaveBase, _alphadot_impl, data);
    }

    void _dalphadtheta_impl(typename ShearAlfvenWaveBase::Array2& data) override {
        PYBIND11_OVERLOAD(void, ShearAlfvenWaveBase, _dalphadtheta_impl, data);
    }

    void _dalphadpsi_impl(typename ShearAlfvenWaveBase::Array2& data) override {
        PYBIND11_OVERLOAD(void, ShearAlfvenWaveBase, _dalphadpsi_impl, data);
    }

    void _dalphadzeta_impl(typename ShearAlfvenWaveBase::Array2& data) override {
        PYBIND11_OVERLOAD(void, ShearAlfvenWaveBase, _dalphadzeta_impl, data);
    }
};
