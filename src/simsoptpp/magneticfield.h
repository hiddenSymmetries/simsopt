#pragma once
#include <xtensor/xarray.hpp>
#include <xtensor/xnoalias.hpp>
#include <stdexcept>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>


#include "cachedarray.h"
#include "cache.h"
#include "cachedtensor.h"

using std::logic_error;
using std::vector;
using std::shared_ptr;
using std::make_shared;




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
    public:
        using Tensor1 = T<double, 1, xt::layout_type::row_major>;
        using Tensor2 = T<double, 2, xt::layout_type::row_major>;
        using Tensor3 = T<double, 3, xt::layout_type::row_major>;
        using Tensor4 = T<double, 4, xt::layout_type::row_major>;

    protected:
        void get_points_cyl_impl(Tensor2& points_cyl) {
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

        void get_points_cart_impl(Tensor2& points_cart) {
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

        virtual void _set_points_cb() { }

        virtual void _AbsB_impl(Tensor2& AbsB) {
            Tensor2& B = this->B_ref();
            int npoints = B.shape(0);
            for (int i = 0; i < npoints; ++i) {
                AbsB(i, 0) = std::sqrt(B(i, 0)*B(i, 0) + B(i, 1)*B(i, 1) + B(i, 2)*B(i, 2));
            }
        }

        virtual void _GradAbsB_impl(Tensor2& GradAbsB) {
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

        virtual void _B_cyl_impl(Tensor2& B_cyl) {
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

        virtual void _GradAbsB_cyl_impl(Tensor2& GradAbsB_cyl) {
            Tensor2& GradAbsB = this->GradAbsB_ref();
            Tensor2& rphiz = this->get_points_cyl_ref();
            int npoints = GradAbsB.shape(0);
            for (int i = 0; i < npoints; ++i) {
                double phi = rphiz(i, 1);
                GradAbsB_cyl(i, 0) = std::cos(phi)*GradAbsB(i, 0) + std::sin(phi)*GradAbsB(i, 1);
                GradAbsB_cyl(i, 1) = std::cos(phi)*GradAbsB(i, 1) - std::sin(phi)*GradAbsB(i, 0);
                GradAbsB_cyl(i, 2) = GradAbsB(i, 2);
            }
        }

        virtual void _B_impl(Tensor2& B) { throw logic_error("_B_impl was not implemented"); }
        virtual void _dB_by_dX_impl(Tensor3& dB_by_dX) { throw logic_error("_dB_by_dX_impl was not implemented"); }
        virtual void _d2B_by_dXdX_impl(Tensor4& d2B_by_dXdX) { throw logic_error("_d2B_by_dXdX_impl was not implemented"); }
        virtual void _A_impl(Tensor2& A) { throw logic_error("_A_impl was not implemented"); }
        virtual void _dA_by_dX_impl(Tensor3& dA_by_dX) { throw logic_error("_dA_by_dX_impl was not implemented"); }
        virtual void _d2A_by_dXdX_impl(Tensor4& d2A_by_dXdX) { throw logic_error("_d2A_by_dXdX_impl was not implemented"); }

        CachedTensor<T, 2> points_cart;
        CachedTensor<T, 2> points_cyl;
        CachedTensor<T, 2> data_B, data_A, data_GradAbsB, data_AbsB, data_Bcyl, data_GradAbsBcyl;
        CachedTensor<T, 3> data_dB, data_dA;
        CachedTensor<T, 4> data_ddB, data_ddA;
        int npoints;

    public:
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
            data_GradAbsBcyl.invalidate_cache();
        }

        MagneticField& set_points_cyl(Tensor2& p) {
            this->invalidate_cache();
            this->points_cart.invalidate_cache();
            this->points_cyl.invalidate_cache();
            npoints = p.shape(0);
            Tensor2& points = points_cyl.get_or_create({npoints, 3});
            memcpy(points.data(), p.data(), 3*npoints*sizeof(double));
            for (int i = 0; i < npoints; ++i) {
                points(i, 1) = std::fmod(points(i, 1), 2*M_PI);
            }
            this->_set_points_cb();
            return *this;
        }

        MagneticField& set_points_cart(Tensor2& p) {
            this->invalidate_cache();
            this->points_cart.invalidate_cache();
            this->points_cyl.invalidate_cache();
            npoints = p.shape(0);
            Tensor2& points = points_cart.get_or_create({npoints, 3});
            memcpy(points.data(), p.data(), 3*npoints*sizeof(double));
            this->_set_points_cb();
            return *this;
        }

        MagneticField& set_points(Tensor2& p) {
            return set_points_cart(p);
        }

        Tensor2 get_points_cyl() {
            return get_points_cyl_ref();
        }

        Tensor2& get_points_cyl_ref() {
            return points_cyl.get_or_create_and_fill({npoints, 3}, [this](Tensor2& B) { return get_points_cyl_impl(B);});
        }

        Tensor2 get_points_cart() {
            return get_points_cart_ref();
        }

        Tensor2& get_points_cart_ref() {
            return points_cart.get_or_create_and_fill({npoints, 3}, [this](Tensor2& B) { return get_points_cart_impl(B);});
        }

        Tensor2& B_ref() {
            return data_B.get_or_create_and_fill({npoints, 3}, [this](Tensor2& B) { return _B_impl(B);});
        }
        Tensor3& dB_by_dX_ref() {
            return data_dB.get_or_create_and_fill({npoints, 3, 3}, [this](Tensor3& dB_by_dX) { return _dB_by_dX_impl(dB_by_dX);});
        }
        Tensor4& d2B_by_dXdX_ref() {
            return data_ddB.get_or_create_and_fill({npoints, 3, 3, 3}, [this](Tensor4& d2B_by_dXdX) { return _d2B_by_dXdX_impl(d2B_by_dXdX);});
        }
        Tensor2 B() { return B_ref(); }
        Tensor3 dB_by_dX() { return dB_by_dX_ref(); }
        Tensor4 d2B_by_dXdX() { return d2B_by_dXdX_ref(); }

        Tensor2& A_ref() {
            return data_A.get_or_create_and_fill({npoints, 3}, [this](Tensor2& A) { return _A_impl(A);});
        }
        Tensor3& dA_by_dX_ref() {
            return data_dA.get_or_create_and_fill({npoints, 3, 3}, [this](Tensor3& dA_by_dX) { return _dA_by_dX_impl(dA_by_dX);});
        }
        Tensor4& d2A_by_dXdX_ref() {
            return data_ddA.get_or_create_and_fill({npoints, 3, 3, 3}, [this](Tensor4& d2A_by_dXdX) { return _d2A_by_dXdX_impl(d2A_by_dXdX);});
        }
        Tensor2 A() { return A_ref(); }
        Tensor3 dA_by_dX() { return dA_by_dX_ref(); }
        Tensor4 d2A_by_dXdX() { return d2A_by_dXdX_ref(); }


        Tensor2 B_cyl() {
            return B_cyl_ref();
        }

        Tensor2& B_cyl_ref() {
            return data_Bcyl.get_or_create_and_fill({npoints, 3}, [this](Tensor2& B) { return _B_cyl_impl(B);});
        }


        Tensor2 AbsB() {
            return AbsB_ref();
        }

        Tensor2& AbsB_ref() {
            return data_AbsB.get_or_create_and_fill({npoints, 1}, [this](Tensor2& AbsB) { return _AbsB_impl(AbsB);});
        }


        Tensor2 GradAbsB() {
            return GradAbsB_ref();
        }

        Tensor2& GradAbsB_ref() {
            return data_GradAbsB.get_or_create_and_fill({npoints, 3}, [this](Tensor2& GradAbsB) { return _GradAbsB_impl(GradAbsB);});
        }

        Tensor2 GradAbsB_cyl() {
            return GradAbsB_cyl_ref();
        }

        Tensor2& GradAbsB_cyl_ref() {
            return data_GradAbsBcyl.get_or_create_and_fill({npoints, 3}, [this](Tensor2& GradAbsB_cyl) { return _GradAbsB_cyl_impl(GradAbsB_cyl);});
        }

};


