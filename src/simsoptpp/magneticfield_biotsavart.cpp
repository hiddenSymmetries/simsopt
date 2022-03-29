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

    std::vector<double> currents(ncoils, 0.);
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
        currents[i] = this->coils[i]->current->get_value();
    }

#pragma omp parallel for
    for (int i = 0; i < ncoils; ++i) {
        Array& Bi = field_cache.get_or_create(fmt::format("B_{}", i), {npoints, 3});
        set_array_to_zero(Bi);
        Array& gamma = this->coils[i]->curve->gamma();
        Array& gammadash = this->coils[i]->curve->gammadash();
        double current = currents[i];
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


template<template<class, std::size_t, xt::layout_type> class T, class Array>
void BiotSavart<T, Array>::compute_A(int derivatives) {
    //fmt::print("Calling compute({})\n", derivatives);
    auto points = this->get_points_cart_ref();
    this->fill_points(points);
    Array dummyjac = xt::zeros<double>({1, 1, 1});
    Array dummyhess = xt::zeros<double>({1, 1, 1, 1});
    Tensor3 _dummyjac = xt::zeros<double>({1, 1, 1});
    Tensor4 _dummyhess = xt::zeros<double>({1, 1, 1, 1});
    int ncoils = this->coils.size();
    Tensor2& A = data_A.get_or_create({npoints, 3});
    Tensor3& dA = derivatives >= 1 ? data_dA.get_or_create({npoints, 3, 3}) : _dummyjac;
    Tensor4& ddA = derivatives >= 2 ? data_ddA.get_or_create({npoints, 3, 3, 3}) : _dummyhess;

    set_array_to_zero(A);
    set_array_to_zero(dA);
    set_array_to_zero(ddA);

    // Creating new xtensor arrays from an openmp thread doesn't appear
    // to be safe. so we do that here in serial.
    // We also acquire all currents here. The reason for that is that some
    // coils point at the same current in the background, and if the
    // `get_value` function for that is implemented in python, then this will
    // freeze in parallel.
    std::vector<double> currents(ncoils, 0.);
    for (int i = 0; i < ncoils; ++i) {
        this->coils[i]->curve->gamma();
        this->coils[i]->curve->gammadash();
        field_cache.get_or_create(fmt::format("A_{}", i), {npoints, 3});
        if(derivatives > 0)
            field_cache.get_or_create(fmt::format("dA_{}", i), {npoints, 3, 3});
        if(derivatives > 1)
            field_cache.get_or_create(fmt::format("ddA_{}", i), {npoints, 3, 3, 3});
        currents[i] = this->coils[i]->current->get_value();
    }

#pragma omp parallel for
    for (int i = 0; i < ncoils; ++i) {
        Array& Ai = field_cache.get_or_create(fmt::format("A_{}", i), {npoints, 3});
        set_array_to_zero(Ai);
        Array& gamma = this->coils[i]->curve->gamma();
        Array& gammadash = this->coils[i]->curve->gammadash();
        double current = currents[i];
        if(derivatives == 0){
            biot_savart_kernel_A<Array, 0>(pointsx, pointsy, pointsz, gamma, gammadash, Ai, dummyjac, dummyhess);
        } else {
            Array& dAi = field_cache.get_or_create(fmt::format("dA_{}", i), {npoints, 3, 3});
            set_array_to_zero(dAi);
            if(derivatives == 1) {
                biot_savart_kernel_A<Array, 1>(pointsx, pointsy, pointsz, gamma, gammadash, Ai, dAi, dummyhess);
            } else {
                Array& ddAi = field_cache.get_or_create(fmt::format("ddA_{}", i), {npoints, 3, 3, 3});
                set_array_to_zero(ddAi);
                if (derivatives == 2) {
                    biot_savart_kernel_A<Array, 2>(pointsx, pointsy, pointsz, gamma, gammadash, Ai, dAi, ddAi);
                } else {
                    throw logic_error("Only two derivatives of Biot Savart vector potential implemented");
                }
            }
        }
    }
    for (int i = 0; i < ncoils; ++i) {
        Array& Ai = field_cache.get_or_create(fmt::format("A_{}", i), {npoints, 3});
        double current = this->coils[i]->current->get_value();
        xt::noalias(A) = A + current * Ai;
    }
    if(derivatives>=1) {
        for (int i = 0; i < ncoils; ++i) {
            Array& dAi = field_cache.get_or_create(fmt::format("dA_{}", i), {npoints, 3, 3});
            double current = this->coils[i]->current->get_value();
            xt::noalias(dA) = dA + current * dAi;
        }
    }
    if(derivatives>=2) {
        for (int i = 0; i < ncoils; ++i) {
            Array& ddAi = field_cache.get_or_create(fmt::format("ddA_{}", i), {npoints, 3, 3, 3});
            double current = this->coils[i]->current->get_value();
            xt::noalias(ddA) = ddA + current * ddAi;
        }
    }
}


#include "xtensor-python/pyarray.hpp"     // Numpy bindings
#include "xtensor-python/pytensor.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;
template class BiotSavart<xt::pytensor, PyArray>;
