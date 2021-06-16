#include "magneticfield_biotsavart.h"
#include "biot_savart_impl.h"

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
    //fmt::print("B at {}, dB at {}, ddB at {}\n", fmt::ptr(B.data()), fmt::ptr(dB.data()), fmt::ptr(ddB.data()));

    B *= 0; // TODO Actually set to zero, multiplying with zero doesn't get rid of NANs
    dB *= 0;
    ddB *= 0;

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

    //fmt::print("Start B(0, :) = ({}, {}, {}) at {}\n", B(0, 0), B(0, 1), B(0, 2), fmt::ptr(B.data()));
#pragma omp parallel for
    for (int i = 0; i < ncoils; ++i) {
        Array& Bi = field_cache.get_or_create(fmt::format("B_{}", i), {npoints, 3});
        Bi *= 0;
        Array& gamma = this->coils[i]->curve->gamma();
        Array& gammadash = this->coils[i]->curve->gammadash();
        double current = this->coils[i]->current->get_value();
        if(derivatives == 0){
            biot_savart_kernel<Array, 0>(pointsx, pointsy, pointsz, gamma, gammadash, Bi, dummyjac, dummyhess);
        } else {
            Array& dBi = field_cache.get_or_create(fmt::format("dB_{}", i), {npoints, 3, 3});
            dBi *= 0;
            if(derivatives == 1) {
                biot_savart_kernel<Array, 1>(pointsx, pointsy, pointsz, gamma, gammadash, Bi, dBi, dummyhess);
            } else {
                Array& ddBi = field_cache.get_or_create(fmt::format("ddB_{}", i), {npoints, 3, 3, 3});
                ddBi *= 0;
                if (derivatives == 2) {
                    biot_savart_kernel<Array, 2>(pointsx, pointsy, pointsz, gamma, gammadash, Bi, dBi, ddBi);
                } else {
                    throw logic_error("Only two derivatives of Biot Savart implemented");
                }
                //fmt::print("ddBi(0, 0, 0, :) = ({}, {}, {})\n", ddBi(0, 0, 0, 0), ddBi(0, 0, 0, 1), ddBi(0, 0, 0, 2));
#pragma omp critical
                {
                    xt::noalias(ddB) = ddB + current * ddBi;
                }
            }
#pragma omp critical
            {
                xt::noalias(dB) = dB + current * dBi;
            }
        }
        //fmt::print("i={}, Bi(0, :) = ({}, {}, {}) at {}\n", i, Bi(0, 0), Bi(0, 1), Bi(0, 2), fmt::ptr(Bi.data()));
        //fmt::print("i={},  B(0, :) = ({}, {}, {}) at {}\n", i, B(0, 0), B(0, 1), B(0, 2), fmt::ptr(B.data()));
#pragma omp critical
        {
            xt::noalias(B) = B + current * Bi;
        }
        //fmt::print("i={},  B(0, :) = ({}, {}, {}) at {}\n", i, B(0, 0), B(0, 1), B(0, 2), fmt::ptr(B.data()));
    }
    //fmt::print("Finish B(0, :) = ({}, {}, {}) at {}\n", B(0, 0), B(0, 1), B(0, 2), fmt::ptr(B.data()));
}


#include "xtensor-python/pyarray.hpp"     // Numpy bindings
#include "xtensor-python/pytensor.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;
template class BiotSavart<xt::pytensor, PyArray>;
