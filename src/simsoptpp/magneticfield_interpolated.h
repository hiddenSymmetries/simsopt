#pragma once 

#include "magneticfield.h"
#include "xtensor/xlayout.hpp"
#include "regular_grid_interpolant_3d.h"

template<template<class, std::size_t, xt::layout_type> class T>
class InterpolatedField : public MagneticField<T> {
    public:
        using typename MagneticField<T>::Tensor2;
    private:

        CachedTensor<T, 2> points_cyl_sym;
        std::function<Vec(Vec, Vec, Vec)> fbatch_B;
        std::function<Vec(Vec, Vec, Vec)> fbatch_GradAbsB;
        std::function<std::vector<bool>(Vec, Vec, Vec)> skip;
        shared_ptr<RegularGridInterpolant3D<Tensor2>> interp_B, interp_GradAbsB;
        bool status_B = false;
        bool status_GradAbsB = false;
        const bool extrapolate;
        const bool stellsym = false;
        const int nfp = 1;
        vector<bool> symmetries = vector<bool>(1, false);

    protected:
        void _B_cyl_impl(Tensor2& B_cyl) override {
            if(!interp_B)
                interp_B = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, r_range, phi_range, z_range, 3, extrapolate, skip);
            if(!status_B) {
                Tensor2 old_points = this->field->get_points_cart();
                interp_B->interpolate_batch(fbatch_B);
                this->field->set_points_cart(old_points);
                status_B = true;
            }
            if(nfp > 1 || stellsym){
                Tensor2& rphiz = this->get_points_cyl_ref();
                Tensor2& rphiz_sym = points_cyl_sym.get_or_create({npoints, 3});
                exploit_symmetries_points(rphiz, rphiz_sym);
                interp_B->evaluate_batch(rphiz_sym, B_cyl);
                apply_symmetries_to_B_cyl(B_cyl);
            } else {
                interp_B->evaluate_batch(this->get_points_cyl_ref(), B_cyl);
            }
        }

        void _GradAbsB_cyl_impl(Tensor2& GradAbsB_cyl) override {
            if(!interp_GradAbsB)
                interp_GradAbsB = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, r_range, phi_range, z_range, 3, extrapolate, skip);
            if(!status_GradAbsB) {
                Tensor2 old_points = this->field->get_points_cart();
                interp_GradAbsB->interpolate_batch(fbatch_GradAbsB);
                this->field->set_points_cart(old_points);
                status_GradAbsB = true;
            }
            if(nfp > 1 || stellsym){
                Tensor2& rphiz = this->get_points_cyl_ref();
                Tensor2& rphiz_sym = points_cyl_sym.get_or_create({npoints, 3});
                exploit_symmetries_points(rphiz, rphiz_sym);
                interp_GradAbsB->evaluate_batch(rphiz_sym, GradAbsB_cyl);
                apply_symmetries_to_GradAbsB_cyl(GradAbsB_cyl);
            } else {
                interp_GradAbsB->evaluate_batch(this->get_points_cyl_ref(), GradAbsB_cyl);
            }
        }

        void _B_impl(Tensor2& B) override {
            Tensor2& B_cyl = this->B_cyl_ref();
            Tensor2& rphiz = this->get_points_cyl_ref();
            int npoints = B.shape(0);
            for (int i = 0; i < npoints; ++i) {
                double phi = rphiz(i, 1);
                B(i, 0) = std::cos(phi)*B_cyl(i, 0) - std::sin(phi)*B_cyl(i, 1);
                B(i, 1) = std::sin(phi)*B_cyl(i, 0) + std::cos(phi)*B_cyl(i, 1);
                B(i, 2) = B_cyl(i, 2);
            }
        }

        void _GradAbsB_impl(Tensor2& GradAbsB) override {
            Tensor2& GradAbsB_cyl = this->GradAbsB_cyl_ref();
            Tensor2& rphiz = this->get_points_cyl_ref();
            int npoints = GradAbsB.shape(0);
            for (int i = 0; i < npoints; ++i) {
                double phi = rphiz(i, 1);
                GradAbsB(i, 0) = std::cos(phi)*GradAbsB_cyl(i, 0) - std::sin(phi)*GradAbsB_cyl(i, 1);
                GradAbsB(i, 1) = std::sin(phi)*GradAbsB_cyl(i, 0) + std::cos(phi)*GradAbsB_cyl(i, 1);
                GradAbsB(i, 2) = GradAbsB_cyl(i, 2);
            }
        }

        void exploit_symmetries_points(Tensor2& rphiz, Tensor2& rphiz_sym){
            int npoints = rphiz.shape(0);
            if(symmetries.size() != npoints)
                symmetries = vector<bool>(npoints, false);
            double period = (2*M_PI)/nfp;
            double* dataptr = &(rphiz(0, 0));
            double* datasymptr = &(rphiz_sym(0, 0));
            for (int i = 0; i < npoints; ++i) {
                double r = dataptr[3*i+0];
                double phi = dataptr[3*i+1];
                double z = dataptr[3*i+2];
                //fmt::print("(r, phi, z) = ({}, {}, {})", r, phi, z);
                if(z < 0 && stellsym) {
                    z = -z;
                    phi = 2*M_PI-phi;
                    int phi_mult = int(phi/period);
                    phi = phi - phi_mult * period;
                    symmetries[i] = true;
                }else{
                    int phi_mult = int(phi/period);
                    phi = phi - phi_mult * period;
                    symmetries[i] = false;
                }
                //fmt::print(" -> (r, phi, z) = ({}, {}, {})\n", r, phi, z);
                datasymptr[3*i+0] = r;
                datasymptr[3*i+1] = phi;
                datasymptr[3*i+2] = z;
            }
        }

        void apply_symmetries_to_B_cyl(Tensor2& field){
            int npoints = field.shape(0);
            for (int i = 0; i < npoints; ++i) {
                if(symmetries[i])
                    field(i, 0) = -field(i, 0);
            }
        }

        void apply_symmetries_to_GradAbsB_cyl(Tensor2& field){
            int npoints = field.shape(0);
            for (int i = 0; i < npoints; ++i) {
                if(symmetries[i]){
                    field(i, 1) = -field(i, 1);
                    field(i, 2) = -field(i, 2);
                }
            }
        }


    public:
        const shared_ptr<MagneticField<T>> field;
        const RangeTriplet r_range, phi_range, z_range;
        using MagneticField<T>::npoints;
        const InterpolationRule rule;

        InterpolatedField(
                shared_ptr<MagneticField<T>> field, InterpolationRule rule,
                RangeTriplet r_range, RangeTriplet phi_range, RangeTriplet z_range,
                bool extrapolate, int nfp, bool stellsym, std::function<std::vector<bool>(Vec, Vec, Vec)> skip) :
            field(field), rule(rule), r_range(r_range), phi_range(phi_range), z_range(z_range), extrapolate(extrapolate), nfp(nfp), stellsym(stellsym),
            skip(skip)
             
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
                auto B_cyl = this->field->B_cyl();
                //fmt::print("B: Actual size: ({}, {}), 3*npoints={}\n", B.shape(0), B.shape(1), 3*npoints);
                auto res = Vec(B_cyl.data(), B_cyl.data()+3*npoints);
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
                auto GradAbsB_cyl = this->field->GradAbsB_cyl();
                //fmt::print("GradAbsB: Actual size: ({}, {}), 3*npoints={}\n", GradAbsB.shape(0), GradAbsB.shape(1), 3*npoints);
                auto res = Vec(GradAbsB_cyl.data(), GradAbsB_cyl.data() + 3*npoints);
                return res;
            };
        }

        InterpolatedField(
                shared_ptr<MagneticField<T>> field, int degree,
                RangeTriplet r_range, RangeTriplet phi_range, RangeTriplet z_range,
                bool extrapolate, int nfp, bool stellsym, std::function<std::vector<bool>(Vec, Vec, Vec)> skip) : InterpolatedField(field, UniformInterpolationRule(degree), r_range, phi_range, z_range, extrapolate, nfp, stellsym, skip) {}

        std::pair<double, double> estimate_error_B(int samples) {
            if(!interp_B)
                interp_B = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, r_range, phi_range, z_range, 3, extrapolate, skip);
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
                interp_GradAbsB = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, r_range, phi_range, z_range, 3, extrapolate, skip);
            if(!status_GradAbsB) {
                Tensor2 old_points = this->field->get_points_cart();
                interp_GradAbsB->interpolate_batch(fbatch_GradAbsB);
                this->field->set_points_cart(old_points);
                status_GradAbsB = true;
            }
            return interp_GradAbsB->estimate_error(this->fbatch_GradAbsB, samples);
        }
};
