#pragma once

#include <vector>
#include "xtensor/xarray.hpp"
#include "xtensor/xlayout.hpp"

using std::vector;
using std::array;

template<template<class, std::size_t, xt::layout_type> class Tensor, std::size_t rank>
class CachedTensor {
    private:
        using T = Tensor<double, rank, xt::layout_type::row_major>;
        T data = {};
        bool status;
        array<int, rank> dims;
    public:
        using Shape = std::array<int, rank>;

        CachedTensor(const Shape& dims) : status(true), dims(dims) {
            data = xt::zeros<double>(dims);
        }

        CachedTensor() : status(false) {
            dims.fill(1);
            data = xt::zeros<double>(dims);
        }

        inline T& get_or_create(const Shape& new_dims){
            if(dims != new_dims){
                data = xt::zeros<double>(new_dims);
                //fmt::print("Dims ({} != {}) don't match, create a new Tensor.\n", dims, new_dims);
                dims = new_dims;
            }
            status = true;
            return data;
        }

        inline bool get_status() const {
            return status;
        }

        inline T& get_or_create_and_fill(const Shape& new_dims, const std::function<void(T&)>& impl){
            if(status)
                return data;
            if(dims != new_dims){
                data = xt::zeros<double>(new_dims);
                //fmt::print("Dims ({} != {}) don't match, create a new Tensor.\n", dims, new_dims);
                dims = new_dims;
            }
            impl(data);
            status = true;
            return data;
        }

        inline void invalidate_cache() {
            status = false;
        }

};
