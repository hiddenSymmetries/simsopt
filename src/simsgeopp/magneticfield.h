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
using std::shared_ptr;
using std::make_shared;

template<class Array>
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
    private:
        std::map<string, CachedArray<Array>> cache;

    public:
        int npoints;

        MagneticField() {
        }

        bool cache_get_status(string key){
            auto loc = cache.find(key);
            if(loc == cache.end()){ // Key not found
                return false;
            }
            if(!(loc->second.status)){ // needs recomputing
                return false;
            }
            return true;
        }

        Array& cache_get_or_create(string key, vector<int> dims){
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

        Array& cache_get_or_create_and_fill(string key, vector<int> dims, std::function<void(Array&)> impl){
            auto loc = cache.find(key);
            if(loc == cache.end()){ // Key not found --> allocate array
                loc = cache.insert(std::make_pair(key, CachedArray<Array>(xt::zeros<double>(dims)))).first; 
                //fmt::print("Create a new array for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
            } else if(loc->second.data.shape(0) != dims[0]) { // key found but not the right number of points
                loc->second = CachedArray<Array>(xt::zeros<double>(dims));
                //fmt::print("Create a new array for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
            }
            if(!(loc->second.status)){ // needs recomputing
                impl(loc->second.data);
                loc->second.status = true;
            }
            return loc->second.data;
        }

        void invalidate_cache() {
            for (auto it = cache.begin(); it != cache.end(); ++it) {
                it->second.status = false;
            }
        }

        virtual void set_points_cb() {

        }

        virtual MagneticField& set_points_cyl(Array& p) {
            this->invalidate_cache();
            npoints = p.shape(0);
            Array& points = cache_get_or_create("points_cyl", {npoints, 3});
            points = p;
            for (int i = 0; i < npoints; ++i) {
                points(i, 1) = std::fmod(points(i, 1), 2*M_PI);
            }
            this->set_points_cb();
            return *this;
        }

        virtual MagneticField& set_points_cart(Array& p) {
            this->invalidate_cache();
            npoints = p.shape(0);
            Array& points = cache_get_or_create("points_cart", {npoints, 3});
            points = p;
            this->set_points_cb();
            return *this;
        }


        virtual MagneticField& set_points(Array& p) {
            return set_points_cart(p);
        }

        Array get_points_cyl() {
            return get_points_cyl_ref();
        }

        Array& get_points_cyl_ref() {
            return cache_get_or_create_and_fill("points_cyl", {npoints, 3}, [this](Array& B) { return get_points_cyl_impl(B);});
        }

        virtual void get_points_cyl_impl(Array& points_cyl) {
            if(!cache_get_status("points_cart"))
                throw logic_error("To compute points_cyl, points_cart needs to exist in the cache.");
            Array& points_cart = get_points_cart_ref();
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

        Array get_points_cart() {
            return get_points_cart_ref();
        }

        Array& get_points_cart_ref() {
            return cache_get_or_create_and_fill("points_cart", {npoints, 3}, [this](Array& B) { return get_points_cart_impl(B);});
        }

        virtual void get_points_cart_impl(Array& points_cart) {
            if(!cache_get_status("points_cyl"))
                throw logic_error("To compute points_cart, points_cyl needs to exist in the cache.");
            Array& points_cyl = get_points_cyl_ref();
            for (int i = 0; i < npoints; ++i) {
                double r = points_cyl(i, 0);
                double phi = points_cyl(i, 1);
                double z = points_cyl(i, 2);
                points_cart(i, 0) = r * std::cos(phi);
                points_cart(i, 1) = r * std::sin(phi);
                points_cart(i, 2) = z;
            }
        }

        virtual void B_impl(Array& B) { throw logic_error("B_impl was not implemented"); }
        virtual void dB_by_dX_impl(Array& dB_by_dX) { throw logic_error("dB_by_dX_impl was not implemented"); }
        virtual void d2B_by_dXdX_impl(Array& d2B_by_dXdX) { throw logic_error("d2B_by_dXdX_impl was not implemented"); }
        virtual void A_impl(Array& A) { throw logic_error("A_impl was not implemented"); }
        virtual void dA_by_dX_impl(Array& dA_by_dX) { throw logic_error("dA_by_dX_impl was not implemented"); }
        virtual void d2A_by_dXdX_impl(Array& d2A_by_dXdX) { throw logic_error("d2A_by_dXdX_impl was not implemented"); }

        Array& B_ref() {
            return cache_get_or_create_and_fill("B", {npoints, 3}, [this](Array& B) { return B_impl(B);});
        }
        Array& dB_by_dX_ref() {
            return cache_get_or_create_and_fill("dB_by_dX", {npoints, 3, 3}, [this](Array& dB_by_dX) { return dB_by_dX_impl(dB_by_dX);});
        }
        Array& d2B_by_dXdX_ref() {
            return cache_get_or_create_and_fill("d2B_by_dXdX", {npoints, 3, 3, 3}, [this](Array& d2B_by_dXdX) { return d2B_by_dXdX_impl(d2B_by_dXdX);});
        }
        Array B() { return B_ref(); }
        Array dB_by_dX() { return dB_by_dX_ref(); }
        Array d2B_by_dXdX() { return d2B_by_dXdX_ref(); }

        Array& A_ref() {
            return cache_get_or_create_and_fill("A", {npoints, 3}, [this](Array& A) { return A_impl(A);});
        }
        Array& dA_by_dX_ref() {
            return cache_get_or_create_and_fill("dA_by_dX", {npoints, 3, 3}, [this](Array& dA_by_dX) { return dA_by_dX_impl(dA_by_dX);});
        }
        Array& d2A_by_dXdX_ref() {
            return cache_get_or_create_and_fill("d2A_by_dXdX", {npoints, 3, 3, 3}, [this](Array& d2A_by_dXdX) { return d2A_by_dXdX_impl(d2A_by_dXdX);});
        }
        Array A() { return A_ref(); }
        Array dA_by_dX() { return dA_by_dX_ref(); }
        Array d2A_by_dXdX() { return d2A_by_dXdX_ref(); }

        void B_cyl_impl(Array& B_cyl) {
            Array& B = this->B_ref();
            Array& rphiz = this->get_points_cyl_ref();
            int npoints = B.shape(0);
            for (int i = 0; i < npoints; ++i) {
                double phi = rphiz(i, 1);
                B_cyl(i, 0) = std::cos(phi)*B(i, 0) + std::sin(phi)*B(i, 1);
                B_cyl(i, 1) = std::cos(phi)*B(i, 1) - std::sin(phi)*B(i, 0);
                B_cyl(i, 2) = B(i, 2);
            }
        }

        Array B_cyl() {
            return B_cyl_ref();
        }

        Array& B_cyl_ref() {
            return cache_get_or_create_and_fill("B_cyl", {npoints, 3}, [this](Array& B) { return B_cyl_impl(B);});
        }


    //Bs = biotsavart.B(compute_derivatives=1)
    //GradBs = biotsavart.dB_by_dX(compute_derivatives=1)
    //AbsBs = np.linalg.norm(Bs, axis=1)
    //GradAbsBs = (Bs[:, None, 0]*GradBs[:, :, 0] + Bs[:, None, 1]*GradBs[:, :, 1] + Bs[:, None, 2]*GradBs[:, :, 2])/AbsBs[:, None]
    //
        void GradAbsB_impl(Array& GradAbsB) {
            Array& B = this->B_ref();
            Array& GradB = this->dB_by_dX_ref();
            int npoints = B.shape(0);
            for (int i = 0; i < npoints; ++i) {
                double AbsB = std::sqrt(B(i, 0)*B(i, 0) + B(i, 1)*B(i, 1) + B(i, 2)*B(i, 2));
                GradAbsB(i, 0) = (B(i, 0) * GradB(i, 0, 0) + B(i, 1) * GradB(i, 0, 1) + B(i, 2) * GradB(i, 0, 2))/AbsB;
                GradAbsB(i, 1) = (B(i, 0) * GradB(i, 1, 0) + B(i, 1) * GradB(i, 1, 1) + B(i, 2) * GradB(i, 1, 2))/AbsB;
                GradAbsB(i, 2) = (B(i, 0) * GradB(i, 2, 0) + B(i, 1) * GradB(i, 2, 1) + B(i, 2) * GradB(i, 2, 2))/AbsB;
            }
        }

        Array GradAbsB() {
            return GradAbsB_ref();
        }

        Array& GradAbsB_ref() {
            return cache_get_or_create_and_fill("GradAbsB", {npoints, 3}, [this](Array& B) { return GradAbsB_impl(B);});
        }

};

typedef vector_type AlignedVector;

template<class Array>
class BiotSavart : public MagneticField<Array> {
    /*
     * This class describes a Magnetic field induced by a list of coils. It
     * computes the Biot Savart law to evaluate the field.
     */
    private:

        vector<shared_ptr<Coil<Array>>> coils;
        // this vectors are aligned in memory for fast simd usage.
        AlignedVector pointsx = AlignedVector(xsimd::simd_type<double>::size, 0.);
        AlignedVector pointsy = AlignedVector(xsimd::simd_type<double>::size, 0.);
        AlignedVector pointsz = AlignedVector(xsimd::simd_type<double>::size, 0.);

        void fill_points(const Array& points) {
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
        using MagneticField<Array>::cache_get_or_create;
        using MagneticField<Array>::cache_get_status;
        using MagneticField<Array>::npoints;
        BiotSavart(vector<shared_ptr<Coil<Array>>> coils) : coils(coils) {

        }

        void compute(int derivatives) {
            //fmt::print("Calling compute({})\n", derivatives);
            auto points = this->get_points_cart_ref();
            this->fill_points(points);
            Array dummyjac = xt::zeros<double>({1, 1, 1});
            Array dummyhess = xt::zeros<double>({1, 1, 1, 1});
            int ncoils = this->coils.size();
            Array& B = cache_get_or_create("B", {npoints, 3});
            Array& dB = derivatives > 0 ? cache_get_or_create("dB_by_dX", {npoints, 3, 3}) : dummyjac;
            Array& ddB = derivatives > 1 ? cache_get_or_create("d2B_by_dXdX", {npoints, 3, 3, 3}) : dummyhess;

            B *= 0; // TODO Actually set to zero, multiplying with zero doesn't get rid of NANs
            dB *= 0;
            ddB *= 0;

            // Creating new xtensor arrays from an openmp thread doesn't appear
            // to be safe. so we do that here in serial.
            for (int i = 0; i < ncoils; ++i) {
                this->coils[i]->curve->gamma();
                this->coils[i]->curve->gammadash();
                cache_get_or_create(fmt::format("B_{}", i), {npoints, 3});
                if(derivatives > 0)
                    cache_get_or_create(fmt::format("dB_{}", i), {npoints, 3, 3});
                if(derivatives > 1)
                    cache_get_or_create(fmt::format("ddB_{}", i), {npoints, 3, 3, 3});
            }

            //fmt::print("B(0, :) = ({}, {}, {}) at {}\n", B(0, 0), B(0, 1), B(0, 2), fmt::ptr(B.data()));
#pragma omp parallel for
            for (int i = 0; i < ncoils; ++i) {
                Array& Bi = cache_get_or_create(fmt::format("B_{}", i), {npoints, 3});
                Bi *= 0;
                Array& gamma = this->coils[i]->curve->gamma();
                Array& gammadash = this->coils[i]->curve->gammadash();
                double current = this->coils[i]->current->get_value();
                if(derivatives == 0){
                    biot_savart_kernel<Array, 0>(pointsx, pointsy, pointsz, gamma, gammadash, Bi, dummyjac, dummyhess);
                } else {
                    Array& dBi = cache_get_or_create(fmt::format("dB_{}", i), {npoints, 3, 3});
                    dBi *= 0;
                    if(derivatives == 1) {
                        biot_savart_kernel<Array, 1>(pointsx, pointsy, pointsz, gamma, gammadash, Bi, dBi, dummyhess);
                    } else {
                        Array& ddBi = cache_get_or_create(fmt::format("ddB_{}", i), {npoints, 3, 3, 3});
                        ddBi *= 0;
                        if (derivatives == 2) {
                            biot_savart_kernel<Array, 2>(pointsx, pointsy, pointsz, gamma, gammadash, Bi, dBi, ddBi);
                        } else {
                            throw logic_error("Only two derivatives of Biot Savart implemented");
                        }
                        ////fmt::print("ddBi(0, 0, 0, :) = ({}, {}, {})\n", ddBi(0, 0, 0, 0), ddBi(0, 0, 0, 1), ddBi(0, 0, 0, 2));
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
                //fmt::print("i={}, Bi(0, :) = ({}, {}, {}) at {}\n", i, Bi(0, 0), Bi(0, 1), Bi(0, 2), fmt::ptr(B.data()));
                //fmt::print("i={},  B(0, :) = ({}, {}, {}) at {}\n", i, B(0, 0), B(0, 1), B(0, 2), fmt::ptr(B.data()));
#pragma omp critical
                {
                    xt::noalias(B) = B + current * Bi;
                }
                //fmt::print("i={},  B(0, :) = ({}, {}, {}) at {}\n", i, B(0, 0), B(0, 1), B(0, 2), fmt::ptr(B.data()));
            }
            //fmt::print("B(0, :) = ({}, {}, {}) at {}\n", B(0, 0), B(0, 1), B(0, 2), fmt::ptr(B.data()));
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
        //        Array& Bi = cache_get_or_create(fmt::format("B_{}", i), {static_cast<int>(points.shape(0)), 3});
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


#include "regular_grid_interpolant_3d.h"

//template<class Array>
//struct CachedInterpolant {
//    RegularGridInterpolant3D<Array> interp;
//    bool status;
//    CachedInterpolant(RegularGridInterpolant3D<Array> interp) : interp(interp), status(false) {}
//};


template<class Array>
class InterpolatedField : public MagneticField<Array> {
    private:

        InterpolationRule rule;
        std::function<Vec(double, double, double)> f_B;
        std::function<Vec(Vec, Vec, Vec)> fbatch_B;
        std::function<Vec(Vec, Vec, Vec)> fbatch_GradAbsB;
        shared_ptr<RegularGridInterpolant3D<Array>> interp_B, interp_GradAbsB;
        bool status_B = false;
        bool status_GradAbsB = false;


    public:
        const shared_ptr<MagneticField<Array>> field;
        const RangeTriplet r_range, phi_range, z_range;
        using MagneticField<Array>::npoints;

        InterpolatedField(shared_ptr<MagneticField<Array>> field, InterpolationRule rule, RangeTriplet r_range, RangeTriplet phi_range, RangeTriplet z_range) :
            field(field), rule(rule), r_range(r_range), phi_range(phi_range), z_range(z_range)
             
        {
            //f_B = [this](double r, double phi, double z) {
            //    Array old_points = this->field->get_points_cart();
            //    Array points = xt::zeros<double>({1, 3});
            //    points(0, 0) = r * std::cos(phi);
            //    points(0, 1) = r * std::sin(phi);
            //    points(0, 2) = z;
            //    this->field->set_points(points);
            //    auto B = this->field->B();
            //    this->field->set_points_cart(old_points);
            //    return Vec{ B(0, 0), B(0, 1), B(0, 2) };
            //};

            fbatch_B = [this](Vec r, Vec phi, Vec z) {
                int npoints = r.size();
                Array points = xt::zeros<double>({npoints, 3});
                for(int i=0; i<npoints; i++) {
                    points(i, 0) = r[i];
                    points(i, 1) = phi[i];
                    points(i, 2) = z[i];
                }
                this->field->set_points_cyl(points);
                auto B = this->field->B();
                auto res = Vec(B.data(), B.data()+3*npoints);
                return res;
            };

            fbatch_GradAbsB = [this](Vec r, Vec phi, Vec z) {
                int npoints = r.size();
                Array points = xt::zeros<double>({npoints, 3});
                for(int i=0; i<npoints; i++) {
                    points(i, 0) = r[i];
                    points(i, 1) = phi[i];
                    points(i, 2) = z[i];
                }
                this->field->set_points_cyl(points);
                auto GradAbsB = this->field->GradAbsB();
                auto res = Vec(GradAbsB.data(), GradAbsB.data() + 3*npoints);
                return res;
            };
        }

        InterpolatedField(shared_ptr<MagneticField<Array>> field, int degree, RangeTriplet r_range, RangeTriplet phi_range, RangeTriplet z_range) : InterpolatedField(field, ChebyshevInterpolationRule(degree), r_range, phi_range, z_range) {}

        void B_impl(Array& B) override {
            if(!interp_B)
                interp_B = std::make_shared<RegularGridInterpolant3D<Array>>(rule, r_range, phi_range, z_range, 3);
            if(!status_B) {
                Array old_points = this->field->get_points_cart();
                interp_B->interpolate_batch(fbatch_B);
                this->field->set_points_cart(old_points);
                status_B = true;
            }
            interp_B->evaluate_batch(this->get_points_cyl_ref(), B);
        }

        void GradAbsB_impl(Array& GradAbsB) override {
            if(!interp_GradAbsB)
                interp_GradAbsB = std::make_shared<RegularGridInterpolant3D<Array>>(rule, r_range, phi_range, z_range, 3);
            if(!status_GradAbsB) {
                Array old_points = this->field->get_points_cart();
                interp_GradAbsB->interpolate_batch(fbatch_GradAbsB);
                this->field->set_points_cart(old_points);
                status_GradAbsB = true;
            }
            interp_GradAbsB->evaluate_batch(this->get_points_cyl_ref(), GradAbsB);
        }

        std::pair<double, double> estimate_error_B(int samples) {
            if(!interp_B)
                interp_B = std::make_shared<RegularGridInterpolant3D<Array>>(rule, r_range, phi_range, z_range, 3);
            if(!status_B) {
                Array old_points = this->field->get_points_cart();
                interp_B->interpolate_batch(fbatch_B);
                this->field->set_points_cart(old_points);
                status_B = true;
            }
            return interp_B->estimate_error(this->fbatch_B, samples);
        }
        std::pair<double, double> estimate_error_GradAbsB(int samples) {
            if(!interp_GradAbsB)
                interp_GradAbsB = std::make_shared<RegularGridInterpolant3D<Array>>(rule, r_range, phi_range, z_range, 3);
            if(!status_GradAbsB) {
                Array old_points = this->field->get_points_cart();
                interp_GradAbsB->interpolate_batch(fbatch_GradAbsB);
                this->field->set_points_cart(old_points);
                status_GradAbsB = true;
            }
            return interp_GradAbsB->estimate_error(this->fbatch_GradAbsB, samples);
        }

};
