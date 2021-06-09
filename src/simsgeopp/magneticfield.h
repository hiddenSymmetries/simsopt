#pragma once
#include <xtensor/xarray.hpp>
#include <xtensor/xnoalias.hpp>
#include <stdexcept>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>


#include "cachedarray.h"
#include "biot_savart_impl.h"
#include "curve.h"
#include "current.h"
#include "coil.h"

using std::logic_error;
using std::vector;
using std::array;
using std::shared_ptr;
using std::make_shared;


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

template<class Array>
class Cache {
    private:
        std::map<string, CachedArray<Array>> cache;
    public:
        bool get_status(string key) const {
            auto loc = cache.find(key);
            if(loc == cache.end()){ // Key not found
                return false;
            }
            if(!(loc->second.status)){ // needs recomputing
                return false;
            }
            return true;
        }
        Array& get_or_create(string key, vector<int> dims){
            auto loc = cache.find(key);
            if(loc == cache.end()){ // Key not found --> allocate array
                loc = cache.insert(std::make_pair(key, CachedArray<Array>(xt::zeros<double>(dims)))).first; 
                //fmt::print("Create a new array for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
            } else if(loc->second.data.shape(0) != dims[0]) { // key found but not the right number of points
                loc->second = CachedArray<Array>(xt::zeros<double>(dims));
                //fmt::print("Create a new array for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
            } else {
                //fmt::print("Existing array found for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
            }
            loc->second.status = true;
            return loc->second.data;
        }

        Array& get_or_create_and_fill(string key, vector<int> dims, std::function<void(Array&)> impl) {
            auto loc = cache.find(key);
            if(loc == cache.end()){ // Key not found --> allocate array
                loc = cache.insert(std::make_pair(key, CachedArray<Array>(xt::zeros<double>(dims)))).first; 
                //fmt::print("Create a new array for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
            } else if(loc->second.data.shape(0) != dims[0]) { // key found but not the right number of points
                loc->second = CachedArray<Array>(xt::zeros<double>(dims));
                //fmt::print("Create a new array for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
            }
            if(!(loc->second.status)){ // needs recomputing
                //fmt::print("Fill array for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
                impl(loc->second.data);
                loc->second.status = true;
            }
            return loc->second.data;
        }

        void invalidate_cache(){
            for (auto it = cache.begin(); it != cache.end(); ++it) {
                it->second.status = false;
            }
        }
};



template<template<class, std::size_t, xt::layout_type> class T>
class MagneticField {
    /*
     * This is the abstract base class for a magnetic field B and it's potential A.
     * The usage is as follows:
     * Bfield = InstanceOfMagneticField(...)
     * points = some array of shape (n, 3) where to evaluate the B field
     * Bfield.set_points(points)
     * B = Bfield.B() // to get the magnetic field at `points`, a (n, 3) array
     * A = Bfield.A() // to get the potential field at `points`, a (n, 3) array
     * gradB = Bfield.dB_by_dX() // to get the gradient of the magnetic field at `points`, a (n, 3, 3) array
     * Some performance notes:
     *  - this class has an internal cache that is cleared everytime set_points() is called
     *  - all functions have a `_ref` version, e.g. `Bfield.B_ref()` which
     *    returns a reference to the array in the cache. this should be used when
     *    performance is key and when the user guarantees that the array is only
     *    read and not modified.
     */
    protected:
        CachedTensor<T, 2> points_cart;
        CachedTensor<T, 2> points_cyl;
        CachedTensor<T, 2> data_B, data_A, data_GradAbsB, data_AbsB, data_Bcyl;
        CachedTensor<T, 3> data_dB, data_dA;
        CachedTensor<T, 4> data_ddB, data_ddA;
        int npoints;

    public:
        using Tensor1 = T<double, 1, xt::layout_type::row_major>;
        using Tensor2 = T<double, 2, xt::layout_type::row_major>;
        using Tensor3 = T<double, 3, xt::layout_type::row_major>;
        using Tensor4 = T<double, 4, xt::layout_type::row_major>;

        MagneticField() {
            Tensor2 vals({{0., 0., 0.}});
            this->set_points_cart(vals);
        }

        virtual void invalidate_cache() {
            data_B.invalidate_cache();
            data_dB.invalidate_cache();
            data_ddB.invalidate_cache();
            data_A.invalidate_cache();
            data_dA.invalidate_cache();
            data_ddA.invalidate_cache();
            data_AbsB.invalidate_cache();
            data_GradAbsB.invalidate_cache();
            data_Bcyl.invalidate_cache();
        }

        virtual void set_points_cb() {

        }

        virtual MagneticField& set_points_cyl(Tensor2& p) {
            this->invalidate_cache();
            this->points_cart.invalidate_cache();
            this->points_cyl.invalidate_cache();
            npoints = p.shape(0);
            Tensor2& points = points_cyl.get_or_create({npoints, 3});
            memcpy(points.data(), p.data(), 3*npoints*sizeof(double));
            for (int i = 0; i < npoints; ++i) {
                points(i, 1) = std::fmod(points(i, 1), 2*M_PI);
            }
            this->set_points_cb();
            return *this;
        }

        virtual MagneticField& set_points_cart(Tensor2& p) {
            this->invalidate_cache();
            this->points_cart.invalidate_cache();
            this->points_cyl.invalidate_cache();
            npoints = p.shape(0);
            Tensor2& points = points_cart.get_or_create({npoints, 3});
            memcpy(points.data(), p.data(), 3*npoints*sizeof(double));
            this->set_points_cb();
            return *this;
        }


        virtual MagneticField& set_points(Tensor2& p) {
            return set_points_cart(p);
        }

        Tensor2 get_points_cyl() {
            return get_points_cyl_ref();
        }

        Tensor2& get_points_cyl_ref() {
            return points_cyl.get_or_create_and_fill({npoints, 3}, [this](Tensor2& B) { return get_points_cyl_impl(B);});
        }

        virtual void get_points_cyl_impl(Tensor2& points_cyl) {
            if(!points_cart.get_status())
                throw logic_error("To compute points_cyl, points_cart needs to exist in the cache.");
            Tensor2& points_cart = get_points_cart_ref();
            for (int i = 0; i < npoints; ++i) {
                double x = points_cart(i, 0);
                double y = points_cart(i, 1);
                double z = points_cart(i, 2);
                points_cyl(i, 0) = std::sqrt(x*x + y*y);
                double phi = std::atan2(y, x);
                if(phi < 0)
                    phi += 2*M_PI;
                points_cyl(i, 1) = phi;
                points_cyl(i, 2) = z;
            }
        }

        Tensor2 get_points_cart() {
            return get_points_cart_ref();
        }

        Tensor2& get_points_cart_ref() {
            return points_cart.get_or_create_and_fill({npoints, 3}, [this](Tensor2& B) { return get_points_cart_impl(B);});
        }

        virtual void get_points_cart_impl(Tensor2& points_cart) {
            if(!points_cyl.get_status())
                throw logic_error("To compute points_cart, points_cyl needs to exist in the cache.");
            Tensor2& points_cyl = get_points_cyl_ref();
            for (int i = 0; i < npoints; ++i) {
                double r = points_cyl(i, 0);
                double phi = points_cyl(i, 1);
                double z = points_cyl(i, 2);
                points_cart(i, 0) = r * std::cos(phi);
                points_cart(i, 1) = r * std::sin(phi);
                points_cart(i, 2) = z;
            }
        }

        virtual void B_impl(Tensor2& B) { throw logic_error("B_impl was not implemented"); }
        virtual void dB_by_dX_impl(Tensor3& dB_by_dX) { throw logic_error("dB_by_dX_impl was not implemented"); }
        virtual void d2B_by_dXdX_impl(Tensor4& d2B_by_dXdX) { throw logic_error("d2B_by_dXdX_impl was not implemented"); }
        virtual void A_impl(Tensor2& A) { throw logic_error("A_impl was not implemented"); }
        virtual void dA_by_dX_impl(Tensor3& dA_by_dX) { throw logic_error("dA_by_dX_impl was not implemented"); }
        virtual void d2A_by_dXdX_impl(Tensor4& d2A_by_dXdX) { throw logic_error("d2A_by_dXdX_impl was not implemented"); }

        Tensor2& B_ref() {
            return data_B.get_or_create_and_fill({npoints, 3}, [this](Tensor2& B) { return B_impl(B);});
        }
        Tensor3& dB_by_dX_ref() {
            return data_dB.get_or_create_and_fill({npoints, 3, 3}, [this](Tensor3& dB_by_dX) { return dB_by_dX_impl(dB_by_dX);});
        }
        Tensor4& d2B_by_dXdX_ref() {
            return data_ddB.get_or_create_and_fill({npoints, 3, 3, 3}, [this](Tensor4& d2B_by_dXdX) { return d2B_by_dXdX_impl(d2B_by_dXdX);});
        }
        Tensor2 B() { return B_ref(); }
        Tensor3 dB_by_dX() { return dB_by_dX_ref(); }
        Tensor4 d2B_by_dXdX() { return d2B_by_dXdX_ref(); }

        Tensor2& A_ref() {
            return data_A.get_or_create_and_fill({npoints, 3}, [this](Tensor2& A) { return A_impl(A);});
        }
        Tensor3& dA_by_dX_ref() {
            return data_dA.get_or_create_and_fill({npoints, 3, 3}, [this](Tensor3& dA_by_dX) { return dA_by_dX_impl(dA_by_dX);});
        }
        Tensor4& d2A_by_dXdX_ref() {
            return data_ddA.get_or_create_and_fill({npoints, 3, 3, 3}, [this](Tensor4& d2A_by_dXdX) { return d2A_by_dXdX_impl(d2A_by_dXdX);});
        }
        Tensor2 A() { return A_ref(); }
        Tensor3 dA_by_dX() { return dA_by_dX_ref(); }
        Tensor4 d2A_by_dXdX() { return d2A_by_dXdX_ref(); }

        void B_cyl_impl(Tensor2& B_cyl) {
            Tensor2& B = this->B_ref();
            Tensor2& rphiz = this->get_points_cyl_ref();
            int npoints = B.shape(0);
            for (int i = 0; i < npoints; ++i) {
                double phi = rphiz(i, 1);
                B_cyl(i, 0) = std::cos(phi)*B(i, 0) + std::sin(phi)*B(i, 1);
                B_cyl(i, 1) = std::cos(phi)*B(i, 1) - std::sin(phi)*B(i, 0);
                B_cyl(i, 2) = B(i, 2);
            }
        }

        Tensor2 B_cyl() {
            return B_cyl_ref();
        }

        Tensor2& B_cyl_ref() {
            return data_Bcyl.get_or_create_and_fill({npoints, 3}, [this](Tensor2& B) { return B_cyl_impl(B);});
        }

        void AbsB_impl(Tensor2& AbsB) {
            Tensor2& B = this->B_ref();
            int npoints = B.shape(0);
            for (int i = 0; i < npoints; ++i) {
                AbsB(i) = std::sqrt(B(i, 0)*B(i, 0) + B(i, 1)*B(i, 1) + B(i, 2)*B(i, 2));
            }
        }

        Tensor2 AbsB() {
            return AbsB_ref();
        }

        Tensor2& AbsB_ref() {
            return data_AbsB.get_or_create_and_fill({npoints}, [this](Tensor2& AbsB) { return AbsB_impl(AbsB);});
        }

        virtual void GradAbsB_impl(Tensor2& GradAbsB) {
            Tensor2& B = this->B_ref();
            Tensor3& GradB = this->dB_by_dX_ref();
            int npoints = B.shape(0);
            for (int i = 0; i < npoints; ++i) {
                double AbsB = std::sqrt(B(i, 0)*B(i, 0) + B(i, 1)*B(i, 1) + B(i, 2)*B(i, 2));
                GradAbsB(i, 0) = (B(i, 0) * GradB(i, 0, 0) + B(i, 1) * GradB(i, 0, 1) + B(i, 2) * GradB(i, 0, 2))/AbsB;
                GradAbsB(i, 1) = (B(i, 0) * GradB(i, 1, 0) + B(i, 1) * GradB(i, 1, 1) + B(i, 2) * GradB(i, 1, 2))/AbsB;
                GradAbsB(i, 2) = (B(i, 0) * GradB(i, 2, 0) + B(i, 1) * GradB(i, 2, 1) + B(i, 2) * GradB(i, 2, 2))/AbsB;
            }
        }

        Tensor2 GradAbsB() {
            return GradAbsB_ref();
        }

        Tensor2& GradAbsB_ref() {
            return data_GradAbsB.get_or_create_and_fill({npoints, 3}, [this](Tensor2& B) { return GradAbsB_impl(B);});
        }

};

typedef vector_type AlignedVector;

template<template<class, std::size_t, xt::layout_type> class T, class Array>
class BiotSavart : public MagneticField<T> {
     //This class describes a Magnetic field induced by a list of coils. It
     //computes the Biot Savart law to evaluate the field.
    public:
        using typename MagneticField<T>::Tensor2;
        using typename MagneticField<T>::Tensor3;
        using typename MagneticField<T>::Tensor4;

    private:
        Cache<Array> field_cache;


        vector<shared_ptr<Coil<Array>>> coils;
        // this vectors are aligned in memory for fast simd usage.
        AlignedVector pointsx = AlignedVector(xsimd::simd_type<double>::size, 0.);
        AlignedVector pointsy = AlignedVector(xsimd::simd_type<double>::size, 0.);
        AlignedVector pointsz = AlignedVector(xsimd::simd_type<double>::size, 0.);

        void fill_points(const Tensor2& points) {
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

    public:

        using MagneticField<T>::npoints;
        using MagneticField<T>::data_B;
        using MagneticField<T>::data_dB;
        using MagneticField<T>::data_ddB;
        BiotSavart(vector<shared_ptr<Coil<Array>>> coils) : MagneticField<T>(), coils(coils) {

        }

        void compute(int derivatives) {
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


        void B_impl(Tensor2& B) override {
            this->compute(0);
        }
        
        void dB_by_dX_impl(Tensor3& dB_by_dX) override {
            this->compute(1);
        }

        void d2B_by_dXdX_impl(Tensor4& d2B_by_dXdX) override {
            this->compute(2);
        }

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


#include "regular_grid_interpolant_3d.h"

template<template<class, std::size_t, xt::layout_type> class T>
class InterpolatedField : public MagneticField<T> {
    public:
        using typename MagneticField<T>::Tensor2;
    private:

        std::function<Vec(double, double, double)> f_B;
        std::function<Vec(Vec, Vec, Vec)> fbatch_B;
        std::function<Vec(Vec, Vec, Vec)> fbatch_GradAbsB;
        shared_ptr<RegularGridInterpolant3D<Tensor2>> interp_B, interp_GradAbsB;
        bool status_B = false;
        bool status_GradAbsB = false;
        const bool extrapolate;


    public:
        const shared_ptr<MagneticField<T>> field;
        const RangeTriplet r_range, phi_range, z_range;
        using MagneticField<T>::npoints;
        const InterpolationRule rule;

        InterpolatedField(shared_ptr<MagneticField<T>> field, InterpolationRule rule, RangeTriplet r_range, RangeTriplet phi_range, RangeTriplet z_range, bool extrapolate) :
            field(field), rule(rule), r_range(r_range), phi_range(phi_range), z_range(z_range), extrapolate(extrapolate)
             
        {
            fbatch_B = [this](Vec r, Vec phi, Vec z) {
                int npoints = r.size();
                Tensor2 points = xt::zeros<double>({npoints, 3});
                for(int i=0; i<npoints; i++) {
                    points(i, 0) = r[i];
                    points(i, 1) = phi[i];
                    points(i, 2) = z[i];
                }
                this->field->set_points_cyl(points);
                auto B = this->field->B();
                fmt::print("B: Actual size: ({}, {}), 3*npoints={}\n", B.shape(0), B.shape(1), 3*npoints);
                auto res = Vec(B.data(), B.data()+3*npoints);
                return res;
            };

            fbatch_GradAbsB = [this](Vec r, Vec phi, Vec z) {
                int npoints = r.size();
                Tensor2 points = xt::zeros<double>({npoints, 3});
                for(int i=0; i<npoints; i++) {
                    points(i, 0) = r[i];
                    points(i, 1) = phi[i];
                    points(i, 2) = z[i];
                }
                this->field->set_points_cyl(points);
                auto GradAbsB = this->field->GradAbsB();
                fmt::print("GradAbsB: Actual size: ({}, {}), 3*npoints={}\n", GradAbsB.shape(0), GradAbsB.shape(1), 3*npoints);
                auto res = Vec(GradAbsB.data(), GradAbsB.data() + 3*npoints);
                return res;
            };
        }

        InterpolatedField(shared_ptr<MagneticField<T>> field, int degree, RangeTriplet r_range, RangeTriplet phi_range, RangeTriplet z_range, bool extrapolate) : InterpolatedField(field, UniformInterpolationRule(degree), r_range, phi_range, z_range, extrapolate) {}

        void B_impl(Tensor2& B) override {
            if(!interp_B)
                interp_B = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, r_range, phi_range, z_range, 3, extrapolate);
            if(!status_B) {
                Tensor2 old_points = this->field->get_points_cart();
                interp_B->interpolate_batch(fbatch_B);
                this->field->set_points_cart(old_points);
                status_B = true;
            }
            interp_B->evaluate_batch(this->get_points_cyl_ref(), B);
        }

        void GradAbsB_impl(Tensor2& GradAbsB) override {
            if(!interp_GradAbsB)
                interp_GradAbsB = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, r_range, phi_range, z_range, 3, extrapolate);
            if(!status_GradAbsB) {
                Tensor2 old_points = this->field->get_points_cart();
                interp_GradAbsB->interpolate_batch(fbatch_GradAbsB);
                this->field->set_points_cart(old_points);
                status_GradAbsB = true;
            }
            interp_GradAbsB->evaluate_batch(this->get_points_cyl_ref(), GradAbsB);
        }

        std::pair<double, double> estimate_error_B(int samples) {
            if(!interp_B)
                interp_B = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, r_range, phi_range, z_range, 3, extrapolate);
            if(!status_B) {
                Tensor2 old_points = this->field->get_points_cart();
                interp_B->interpolate_batch(fbatch_B);
                this->field->set_points_cart(old_points);
                status_B = true;
            }
            return interp_B->estimate_error(this->fbatch_B, samples);
        }
        std::pair<double, double> estimate_error_GradAbsB(int samples) {
            if(!interp_GradAbsB)
                interp_GradAbsB = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, r_range, phi_range, z_range, 3, extrapolate);
            if(!status_GradAbsB) {
                Tensor2 old_points = this->field->get_points_cart();
                interp_GradAbsB->interpolate_batch(fbatch_GradAbsB);
                this->field->set_points_cart(old_points);
                status_GradAbsB = true;
            }
            return interp_GradAbsB->estimate_error(this->fbatch_GradAbsB, samples);
        }
};
