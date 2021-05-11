#pragma once
#include "xtensor/xarray.hpp"
#include "cachedarray.h"
#include <fmt/core.h>
#include "biot_savart_impl.h"

using std::vector;
using std::shared_ptr;

template<class Array>
class MagneticField {
    private:
        map<string, CachedArray<Array>> cache;

    public:
        Array points;

        MagneticField() {
            points = xt::zeros<double>({1, 3});
        }

        Array& check_the_cache(string key, vector<int> dims){
            auto loc = cache.find(key);
            if(loc == cache.end()){ // Key not found --> allocate array
                loc = cache.insert(std::make_pair(key, CachedArray<Array>(xt::zeros<double>(dims)))).first; 
            } else if((loc->second).data.shape(0) != dims[0]) { // key found but not the right number of points
                loc->second = CachedArray<Array>(xt::zeros<double>(dims));
            }
            (loc->second).status = true;
            return (loc->second).data;
        }

        Array& check_the_cache_and_fill(string key, vector<int> dims, std::function<void(Array&)> impl){
            auto loc = cache.find(key);
            if(loc == cache.end()){ // Key not found --> allocate array
                loc = cache.insert(std::make_pair(key, CachedArray<Array>(xt::zeros<double>(dims)))).first; 
            } else if((loc->second).data.shape(0) != dims[0]) { // key found but not the right number of points
                loc->second = CachedArray<Array>(xt::zeros<double>(dims));
            }

            if(!((loc->second).status)){ // needs recomputing
                impl((loc->second).data);
                (loc->second).status = true;
            }
            return (loc->second).data;
        }

        void invalidate_cache() {
            for (auto it = cache.begin(); it != cache.end(); ++it) {
                (it->second).status = false;
            }
        }

        MagneticField& set_points(Array& p) {
            points = p;
            return *this;
        }

        virtual void B_impl(Array& B) { throw logic_error("B_impl was not implemented"); }
        virtual void dB_by_dX_impl(Array& dB_by_dX) { throw logic_error("dB_by_dX_impl was not implemented"); }
        virtual void d2B_by_dXdX_impl(Array& d2B_by_dXdX) { throw logic_error("d2B_by_dXdX_impl was not implemented"); }

        Array& B() {
            return check_the_cache_and_fill("B", {static_cast<int>(points.shape(0)), 3}, [this](Array& B) { return B_impl(B);});
        }

        Array& dB_by_dX() {
            return check_the_cache_and_fill("dB_by_dX", {static_cast<int>(points.shape(0)), 3, 3}, [this](Array& dB_by_dX) { return dB_by_dX_impl(dB_by_dX);});
        }

        Array& d2B_by_dXdX() {
            return check_the_cache_and_fill("d2B_by_dXdX", {static_cast<int>(points.shape(0)), 3, 3, 3}, [this](Array& d2B_by_dXdX) { return d2B_by_dXdX_impl(d2B_by_dXdX);});
        }

};

template<class Array>
class Current {
    private:
        double value;
    public:
        Current(double value) : value(value) {}
        inline void set_dofs(Array& dofs) { value=dofs.data()[0]; };
        inline Array get_dofs() { return Array({value}); };
        inline double get_value() { return value; }
        inline void set_value(double val) { value = val; }
};


template<class Array>
class Coil {
    public:
        const shared_ptr<Curve<Array>> curve;
        const shared_ptr<Current<Array>> current;
        Coil(shared_ptr<Curve<Array>> curve, shared_ptr<Current<Array>> current) :
            curve(curve), current(current) { }
};


typedef vector_type AlignedVector;

template<class Array>
class BiotSavart : public MagneticField<Array> {
    private:

        vector<shared_ptr<Coil<Array>>> coils;

        using MagneticField<Array>::check_the_cache;
        using MagneticField<Array>::points;
        AlignedVector pointsx = AlignedVector(xsimd::simd_type<double>::size, 0.);
        AlignedVector pointsy = AlignedVector(xsimd::simd_type<double>::size, 0.);
        AlignedVector pointsz = AlignedVector(xsimd::simd_type<double>::size, 0.);

        void fill_points(const Array& points) {
            int npoints = points.shape(0);
            if(pointsx.size() < npoints)
                pointsx = AlignedVector(npoints, 0.);
            if(pointsy.size() < npoints)
                pointsy = AlignedVector(npoints, 0.);
            if(pointsz.size() < npoints)
                pointsz = AlignedVector(npoints, 0.);
            for (int i = 0; i < npoints; ++i) {
                pointsx[i] = points(i, 0);
                pointsy[i] = points(i, 1);
                pointsz[i] = points(i, 2);
            }
        }

    public:
        BiotSavart(vector<shared_ptr<Coil<Array>>> coils) : coils(coils) {

        }

        void compute(int derivatives) {
            this->fill_points(points);
            Array dummyjac = xt::zeros<double>({1, 1, 1});
            Array dummyhess = xt::zeros<double>({1, 1, 1, 1});
            Array& B = check_the_cache("B", {static_cast<int>(points.shape(0)), 3});
            Array& dB = dummyjac;
            Array& ddB = dummyhess;
            if(derivatives > 0)
                 dB = check_the_cache("dB", {static_cast<int>(points.shape(0)), 3, 3});
            if(derivatives > 1)
                 ddB = check_the_cache("ddB", {static_cast<int>(points.shape(0)), 3, 3, 3});

            for (int i = 0; i < this->coils.size(); ++i) {
                Array& Bi = check_the_cache(fmt::format("B_{}", i), {static_cast<int>(points.shape(0)), 3});
                Array& gamma = this->coils[i]->curve->gamma();
                Array& gammadash = this->coils[i]->curve->gammadash();
                double current = this->coils[i]->current->get_value();
                if(derivatives == 0){
                    biot_savart_kernel<Array, 0>(pointsx, pointsy, pointsz, gamma, gammadash, Bi, dummyjac, dummyhess);
                } else {
                    Array& dBi = check_the_cache(fmt::format("dB_{}", i), {static_cast<int>(points.shape(0)), 3, 3});
                    if(derivatives == 1) {
                        biot_savart_kernel<Array, 1>(pointsx, pointsy, pointsz, gamma, gammadash, Bi, dBi, dummyhess);
                    } else {
                        Array& ddBi = check_the_cache(fmt::format("ddB_{}", i), {static_cast<int>(points.shape(0)), 3, 3, 3});
                        if (derivatives == 2) {
                            biot_savart_kernel<Array, 2>(pointsx, pointsy, pointsz, gamma, gammadash, Bi, dBi, ddBi);
                        } else {
                            throw logic_error("Only two derivatives of Biot Savart implemented");
                        }
                        ddB += current * ddBi;
                    }
                    dB += current * dBi;
                }
                B += current * Bi;
            }
        }


        void B_impl(Array& B) override {
            this->compute(0);
        }
        
        void dB_by_dX_impl(Array& dB_by_dX) override {
            this->compute(1);
        }

        void d2B_by_dXdX_impl(Array& d2B_by_dXdX) override {
            this->compute(2);
        }

        //Array dB_by_dcoeff_vjp(Array& vec) {
        //    int num_coils = this->coilcollection.curves.size();
        //    Array dummy = Array();
        //    auto res_gamma = std::vector<Array>(num_coils, Array());
        //    auto res_gammadash = std::vector<Array>(num_coils, Array());
        //    auto res_current = std::vector<Array>(num_coils, Array());
        //    for(int i=0; i<num_coils; i++) {
        //        int num_points = this->coils[i].curve.gamma().shape(0);
        //        res_gamma[i] = xt::zeros<double>({num_points, 3});
        //        res_gammadash[i] = xt::zeros<double>({num_points, 3});
        //        res_current[i] = xt::zeros<double>({1});
        //    }
        //    this->fill_points(points);
        //    for(int i=0; i<num_coils; i++) {
        //            biot_savart_vjp_kernel<Array, 0>(pointsx, pointsy, pointsz, this->coilcollection.curves[i].gamma(), this->coilcollection.curves[i].gammadash(),
        //                    vec, res_gamma[i], res_gammadash[i], dummy, dummy, dummy);
        //    }

        //    int npoints = points.shape(0);
        //    for(int i=0; i<num_coils; i++) {
        //        Array& Bi = check_the_cache(fmt::format("B_{}", i), {static_cast<int>(points.shape(0)), 3});
        //        for (int j = 0; j < npoints; ++j) {
        //            res_current[i] += Bi(j, 0)*vec(j, 0) + Bi(j, 1)*vec(j, 1) + Bi(j, 2)*vec(j, 2);
        //        }
        //    }

        //    // TODO: figure out how to add these in the right way, when some correspond to coil dofs, others correspond to coil currents etc
        //    return this->coilcollection.dgamma_by_dcoeff_vjp(res_gamma)
        //        + this->coilcollection.dgammadash_by_dcoeff_vjp(res_gammadash)
        //        + this->coilcollection.dcurrent_by_dcoeff_vjp(res_current);

        //}

};



