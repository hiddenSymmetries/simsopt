#include "magneticfield_biotsavart.h"
#include "biot_savart_impl.h"

template<class Array>
void set_array_to_zero(Array& data){
    std::fill(data.begin(), data.end(), 0.);
}


template<template<class, std::size_t, xt::layout_type> class T, class Array>
void BiotSavart<T, Array>::compute(int derivatives) {
    //fmt::print("Calling compute({})\n", derivatives);
    auto points = this->get_points_cart_ref();
    this->fill_points(points);
    Array dummyjac = xt::zeros<double>({1, 1, 1});
    Array dummyhess = xt::zeros<double>({1, 1, 1, 1});
    Tensor3 _dummyjac = xt::zeros<double>({1, 1, 1});
    Tensor4 _dummyhess = xt::zeros<double>({1, 1, 1, 1});
    int ncoils = this->coils.size();
    Tensor2& B = data_B.get_or_create({npoints, 3});
    Tensor3& dB = derivatives >= 1 ? data_dB.get_or_create({npoints, 3, 3}) : _dummyjac;
    Tensor4& ddB = derivatives >= 2 ? data_ddB.get_or_create({npoints, 3, 3, 3}) : _dummyhess;

    set_array_to_zero(B);
    set_array_to_zero(dB);
    set_array_to_zero(ddB);

    // Creating new xtensor arrays from an openmp thread doesn't appear
    // to be safe. so we do that here in serial.
    for (int i = 0; i < ncoils; ++i) {
        this->coils[i]->curve->gamma();
        this->coils[i]->curve->gammadash();
        field_cache.get_or_create(fmt::format("B_{}", i), {npoints, 3});
        if(derivatives > 0)
            field_cache.get_or_create(fmt::format("dB_{}", i), {npoints, 3, 3});
        if(derivatives > 1)
            field_cache.get_or_create(fmt::format("ddB_{}", i), {npoints, 3, 3, 3});
    }

#pragma omp parallel for
    for (int i = 0; i < ncoils; ++i) {
        Array& Bi = field_cache.get_or_create(fmt::format("B_{}", i), {npoints, 3});
        set_array_to_zero(Bi);
        Array& gamma = this->coils[i]->curve->gamma();
        Array& gammadash = this->coils[i]->curve->gammadash();
        double current = this->coils[i]->current->get_value();
        if(derivatives == 0){
            biot_savart_kernel<Array, 0>(pointsx, pointsy, pointsz, gamma, gammadash, Bi, dummyjac, dummyhess);
        } else {
            Array& dBi = field_cache.get_or_create(fmt::format("dB_{}", i), {npoints, 3, 3});
            set_array_to_zero(dBi);
            if(derivatives == 1) {
                biot_savart_kernel<Array, 1>(pointsx, pointsy, pointsz, gamma, gammadash, Bi, dBi, dummyhess);
            } else {
                Array& ddBi = field_cache.get_or_create(fmt::format("ddB_{}", i), {npoints, 3, 3, 3});
                set_array_to_zero(ddBi);
                if (derivatives == 2) {
                    biot_savart_kernel<Array, 2>(pointsx, pointsy, pointsz, gamma, gammadash, Bi, dBi, ddBi);
                } else {
                    throw logic_error("Only two derivatives of Biot Savart implemented");
                }
            }
        }
    }
    for (int i = 0; i < ncoils; ++i) {
        Array& Bi = field_cache.get_or_create(fmt::format("B_{}", i), {npoints, 3});
        double current = this->coils[i]->current->get_value();
        xt::noalias(B) = B + current * Bi;
    }
    if(derivatives>=1) {
        for (int i = 0; i < ncoils; ++i) {
            Array& dBi = field_cache.get_or_create(fmt::format("dB_{}", i), {npoints, 3, 3});
            double current = this->coils[i]->current->get_value();
            xt::noalias(dB) = dB + current * dBi;
        }
    }
    if(derivatives>=2) {
        for (int i = 0; i < ncoils; ++i) {
            Array& ddBi = field_cache.get_or_create(fmt::format("ddB_{}", i), {npoints, 3, 3, 3});
            double current = this->coils[i]->current->get_value();
            xt::noalias(ddB) = ddB + current * ddBi;
        }
    }
}


#include "xtensor-python/pyarray.hpp"     // Numpy bindings
#include "xtensor-python/pytensor.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;
template class BiotSavart<xt::pytensor, PyArray>;
