#pragma once
#include <memory>
#include <vector>
#include "magneticfield.h"
#include "regular_grid_interpolant_3d.h"

using std::shared_ptr;
using std::vector;
using std::tuple;

double get_phi(double x, double y, double phi_near);


class StoppingCriterion {
    public:
        // Should return true if the Criterion is satisfied.
        virtual bool operator()(int iter, double t, double x, double y, double z) = 0;
        virtual ~StoppingCriterion() {}
};

class ToroidalTransitStoppingCriterion : public StoppingCriterion {
    private:
        int max_transits;
        double phi_last;
        double phi_init;
    public:
        ToroidalTransitStoppingCriterion(int max_transits) : max_transits(max_transits) {
        };
        bool operator()(int iter, double t, double x, double y, double z) override {
            if (iter == 1) {
              phi_last = M_PI;
            }
            // if (iter > 0) {
            double phi = get_phi(x, y, phi_last);
            if (iter == 1) {
              phi_init = phi;
            }
            phi_last = phi;
              // if (phi_init == -2*M_PI) {
              //   phi_init = phi;
              // }
              // bool phi_hit = std::floor((phi-phi_init)/(2*M_PI)) != std::floor((phi_last-phi_init)/(2*M_PI));
              // phi_last = phi;
              // if (phi_hit) {
              //   ntransits += 1;
              // }
            // }
            int ntransits = std::abs(std::floor((phi-phi_init)/(2*M_PI)));
            return ntransits>=max_transits;
        };
};

class IterationStoppingCriterion : public StoppingCriterion{
    private:
        int max_iter;
    public:
        IterationStoppingCriterion(int max_iter) : max_iter(max_iter) {};
        bool operator()(int iter, double t, double x, double y, double z) override {
            return iter>max_iter;
        };
};

template<class Array>
class LevelsetStoppingCriterion : public StoppingCriterion{
    private:
        shared_ptr<RegularGridInterpolant3D<Array>> levelset;
    public:
        LevelsetStoppingCriterion(shared_ptr<RegularGridInterpolant3D<Array>> levelset) : levelset(levelset) { };
        bool operator()(int iter, double t, double x, double y, double z) override {
            double r = std::sqrt(x*x + y*y);
            double phi = std::atan2(y, x);
            if(phi < 0)
                phi += 2*M_PI;
            double f = levelset->evaluate(r, phi, z)[0];
            //fmt::print("Levelset at xyz=({}, {}, {}), rphiz=({}, {}, {}), f={}\n", x, y, z, r, phi, z, f);
            return f<0;
        };
};


template<template<class, std::size_t, xt::layout_type> class T>
tuple<vector<array<double, 5>>, vector<array<double, 6>>>
particle_guiding_center_tracing(
        shared_ptr<MagneticField<T>> field, array<double, 3> xyz_init,
        double m, double q, double vtotal, double vtang, double tmax, double tol, bool vacuum,
        vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria);

template<template<class, std::size_t, xt::layout_type> class T>
tuple<vector<array<double, 7>>, vector<array<double, 8>>>
particle_fullorbit_tracing(
        shared_ptr<MagneticField<T>> field, array<double, 3> xyz_init, array<double, 3> v_init,
        double m, double q, double tmax, double tol, vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria);

template<template<class, std::size_t, xt::layout_type> class T>
tuple<vector<array<double, 4>>, vector<array<double, 5>>>
fieldline_tracing(
        shared_ptr<MagneticField<T>> field, array<double, 3> xyz_init,
        double tmax, double tol, vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria);
