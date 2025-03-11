#pragma once 

#include <vector>
#include "xtensor/xarray.hpp"
#include "xtensor/xlayout.hpp"
#include "simdhelpers.h"
#include "magneticfield.h"

template<template<class, std::size_t, xt::layout_type> class T, class Array, class IntArray>
class WireframeField : public MagneticField<T> {
     /* 
      * This class describes a magnetic field induced by a wireframe. It
      * computes the Biot-Savart law to evaluate the field. It is closely
      * modeled on the BiotSavart class that describes a magnetic field from
      * a list of Coil class instances.
      *
      */
    public:
        using typename MagneticField<T>::Tensor2;
        using typename MagneticField<T>::Tensor3;
        using typename MagneticField<T>::Tensor4;
        //const vector<shared_ptr<Coil<Array>>> coils;
        vector<Array> nodes;
        IntArray segments;
        vector<double> seg_signs;
        Array currents;

    private:
        Cache<Array> field_cache;

        #if defined(USE_XSIMD)
        // this vectors are aligned in memory for fast simd usage.
        AlignedPaddedVec pointsx = AlignedPaddedVec(xsimd::simd_type<double>::size, 0.);
        AlignedPaddedVec pointsy = AlignedPaddedVec(xsimd::simd_type<double>::size, 0.);
        AlignedPaddedVec pointsz = AlignedPaddedVec(xsimd::simd_type<double>::size, 0.);

        inline void fill_points(const Tensor2& points) {
            // allocating these aligned vectors is not super cheap, so reuse
            // whenever possible.
            if(pointsx.size() != npoints)
                pointsx = AlignedPaddedVec(npoints, 0.);
            if(pointsy.size() != npoints)
                pointsy = AlignedPaddedVec(npoints, 0.);
            if(pointsz.size() != npoints)
                pointsz = AlignedPaddedVec(npoints, 0.);
            for (int i = 0; i < npoints; ++i) {
                pointsx[i] = points(i, 0);
                pointsy[i] = points(i, 1);
                pointsz[i] = points(i, 2);
            }
        }
        #else
        AlignedPaddedVec pointsx;
        AlignedPaddedVec pointsy;
        AlignedPaddedVec pointsz;

        inline void fill_points(const Tensor2& points) {
            // allocating these aligned vectors is not super cheap, so reuse
            // whenever possible.
            if(pointsx.size() != npoints){
                pointsx.clear();
                pointsx.resize(npoints, 0.0);
            }
            if(pointsy.size() != npoints){
                pointsy.clear();
                pointsy.resize(npoints, 0.0);
            }
            if(pointsz.size() != npoints){
                pointsz.clear();
                pointsz.resize(npoints, 0.0);
            }
            for (int i = 0; i < npoints; ++i) {
                pointsx[i] = points(i, 0);
                pointsy[i] = points(i, 1);
                pointsz[i] = points(i, 2);
            }
        }
        #endif

    protected:

        void _B_impl(Tensor2& B) override {
            this->compute(0);
        }
        
        void _dB_by_dX_impl(Tensor3& dB_by_dX) override {
            this->compute(1);
        }

    public:
        using MagneticField<T>::npoints;
        using MagneticField<T>::data_B;
        using MagneticField<T>::data_dB;
        using MagneticField<T>::data_ddB;
        using MagneticField<T>::data_A;
        using MagneticField<T>::data_dA;
        using MagneticField<T>::data_ddA;


        // Constructor
        WireframeField(vector<Array> _nodes, IntArray _segments, vector<double> _seg_signs, Array _currents) : MagneticField<T>(), nodes(_nodes), segments(_segments), seg_signs(_seg_signs), currents(_currents) {

        }

        void compute(int derivatives);

        virtual void invalidate_cache() override {
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

