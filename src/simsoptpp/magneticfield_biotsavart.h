#pragma once 

#include <vector>
#include "xtensor/xarray.hpp"
#include "xtensor/xlayout.hpp"
#include "simdhelpers.h"
#include "magneticfield.h"
#include "coil.h"

typedef vector_type AlignedVector;

template<template<class, std::size_t, xt::layout_type> class T, class Array>
class BiotSavart : public MagneticField<T> {
     //This class describes a Magnetic field induced by a list of coils. It
     //computes the Biot Savart law to evaluate the field.
    public:
        using typename MagneticField<T>::Tensor2;
        using typename MagneticField<T>::Tensor3;
        using typename MagneticField<T>::Tensor4;
        const vector<shared_ptr<Coil<Array>>> coils;

    private:
        Cache<Array> field_cache;

        // this vectors are aligned in memory for fast simd usage.
        AlignedVector pointsx = AlignedVector(xsimd::simd_type<double>::size, 0.);
        AlignedVector pointsy = AlignedVector(xsimd::simd_type<double>::size, 0.);
        AlignedVector pointsz = AlignedVector(xsimd::simd_type<double>::size, 0.);

        inline void fill_points(const Tensor2& points) {
            // allocating these aligned vectors is not super cheap, so reuse
            // whenever possible.
            if(pointsx.size() != npoints)
                pointsx = AlignedVector(npoints, 0.);
            if(pointsy.size() != npoints)
                pointsy = AlignedVector(npoints, 0.);
            if(pointsz.size() != npoints)
                pointsz = AlignedVector(npoints, 0.);
            for (int i = 0; i < npoints; ++i) {
                pointsx[i] = points(i, 0);
                pointsy[i] = points(i, 1);
                pointsz[i] = points(i, 2);
            }
        }

    protected:

        void _B_impl(Tensor2& B) override {
            this->compute(0);
        }
        
        void _dB_by_dX_impl(Tensor3& dB_by_dX) override {
            this->compute(1);
        }

        void _d2B_by_dXdX_impl(Tensor4& d2B_by_dXdX) override {
            this->compute(2);
        }


    public:
        using MagneticField<T>::npoints;
        using MagneticField<T>::data_B;
        using MagneticField<T>::data_dB;
        using MagneticField<T>::data_ddB;

        BiotSavart(vector<shared_ptr<Coil<Array>>> coils) : MagneticField<T>(), coils(coils) {

        }

        void compute(int derivatives);       virtual void invalidate_cache() override {
            MagneticField<T>::invalidate_cache();
            this->field_cache.invalidate_cache();
        }

        Array& fieldcache_get_or_create(string key, vector<int> dims){
            return this->field_cache.get_or_create(key, dims);
        }

        bool fieldcache_get_status(string key){
            return this->field_cache.get_status(key);
        }

};

