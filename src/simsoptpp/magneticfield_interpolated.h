#pragma once 

#include "magneticfield.h"
#include "xtensor/xlayout.hpp"
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

    protected:
        void _B_impl(Tensor2& B) override {
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

        void _GradAbsB_impl(Tensor2& GradAbsB) override {
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
                //fmt::print("B: Actual size: ({}, {}), 3*npoints={}\n", B.shape(0), B.shape(1), 3*npoints);
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
                //fmt::print("GradAbsB: Actual size: ({}, {}), 3*npoints={}\n", GradAbsB.shape(0), GradAbsB.shape(1), 3*npoints);
                auto res = Vec(GradAbsB.data(), GradAbsB.data() + 3*npoints);
                return res;
            };
        }

        InterpolatedField(shared_ptr<MagneticField<T>> field, int degree, RangeTriplet r_range, RangeTriplet phi_range, RangeTriplet z_range, bool extrapolate) : InterpolatedField(field, UniformInterpolationRule(degree), r_range, phi_range, z_range, extrapolate) {}

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
