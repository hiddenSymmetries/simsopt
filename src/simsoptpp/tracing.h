#pragma once
#include <memory>
#include <vector>
#include "boozermagneticfield.h"
#include "regular_grid_interpolant_3d.h"

using std::shared_ptr;
using std::vector;
using std::tuple;

class StoppingCriterion {
    public:
        // Should return true if the Criterion is satisfied.
        virtual bool operator()(int iter, double dt, double t, double x, double y, double z, double vpar=0) = 0;
        virtual ~StoppingCriterion() {}
};

class ZetaStoppingCriterion : public StoppingCriterion {
    private:
        int nfp;
    public:
        ZetaStoppingCriterion(int nfp) : nfp(nfp) {
        };
        bool operator()(int iter, double dt, double t, double s, double theta, double zeta, double vpar=0) override {
            return std::abs(zeta)>=2*M_PI/nfp;
        };
};

// template<class Array>
class VparStoppingCriterion : public StoppingCriterion {
    private:
        double vpar_crit;
    public:
        VparStoppingCriterion(double vpar_crit) : vpar_crit(vpar_crit) {
        };
        bool operator()(int iter, double dt, double t, double x, double y, double z, double vpar) override {
            return std::abs(vpar)<=vpar_crit;
        };
};

// template<class Array>
class ToroidalTransitStoppingCriterion : public StoppingCriterion {
    private:
        int max_transits;
        double zeta_last;
        double zeta_init;
    public:
        ToroidalTransitStoppingCriterion(int max_transits) : max_transits(max_transits) {
        };
        bool operator()(int iter, double dt, double t, double s, double theta, double zeta, double vpar=0) override {
            if (iter == 1) {
              zeta_last = M_PI;
            }
            double this_zeta = zeta;
            if (iter == 1) {
              zeta_init = this_zeta;
            }
            zeta_last = this_zeta;
            int ntransits = std::abs(std::floor((this_zeta-zeta_init)/(2*M_PI)));
            return ntransits>=max_transits;
        };
};

// template<class Array>
class MaxToroidalFluxStoppingCriterion : public StoppingCriterion {
    private:
        double max_s;
    public:
        MaxToroidalFluxStoppingCriterion(double max_s) : max_s(max_s) {};
        bool operator()(int iter, double dt, double t, double s, double theta, double zeta, double vpar=0) override {
            return s>=max_s;
        };
};

// template<class Array>
class MinToroidalFluxStoppingCriterion : public StoppingCriterion {
    private:
        double min_s;
    public:
        MinToroidalFluxStoppingCriterion(double min_s) : min_s(min_s) {};
        bool operator()(int iter, double dt, double t, double s, double theta, double zeta, double vpar=0) override {
            return s<=min_s;
        };
};

class IterationStoppingCriterion : public StoppingCriterion {
    private:
        int max_iter;
    public:
        IterationStoppingCriterion(int max_iter) : max_iter(max_iter) {};
        bool operator()(int iter, double dt, double t, double s, double theta, double zeta, double vpar=0) override {
            return iter>max_iter;
        };
};

class StepSizeStoppingCriterion : public StoppingCriterion {
    private:
        int min_dt;
    public:
        StepSizeStoppingCriterion(int min_dt) : min_dt(min_dt) {};
        bool operator()(int iter, double dt, double t, double s, double theta, double zeta, double vpar=0) override {
            return dt<min_dt;
        };
};

template<template<class, std::size_t, xt::layout_type> class T>
tuple<vector<array<double, 6>>, vector<array<double, 7>>>
particle_guiding_center_boozer_perturbed_tracing(
        shared_ptr<BoozerMagneticField<T>> field, array<double, 3> stz_init,
        double m, double q, double vtotal, double vtang, double mu, double tmax, double abstol, double reltol,
        bool vacuum, bool noK, vector<double> zetas, vector<double> omegas,
        vector<shared_ptr<StoppingCriterion>> stopping_criteria, double dt_save=1e-6,
        bool zetas_stop=false, bool vpars_stop=false,
        double alphahat=0, double omega=0, int alpham=0, int alphan=0, double phase=0,
        bool forget_exact_path=false, int axis=0, vector<double> vpars={});

	template<template<class, std::size_t, xt::layout_type> class T>
	tuple<vector<array<double, 5>>, vector<array<double, 6>>>
	particle_guiding_center_boozer_tracing(
			  shared_ptr<BoozerMagneticField<T>> field, array<double, 3> stz_init,
        double m, double q, double vtotal, double vtang, double tmax, double dt, double abstol, double reltol, double roottol,
        bool vacuum, bool noK, bool solveSympl, vector<double> zetas, vector<double> omegas, vector<shared_ptr<StoppingCriterion>> stopping_criteria, double dt_save = 1e-6,
        bool forget_exact_path=false, int axis=0, bool predictor_step=true,
        bool zetas_stop=false, bool vpars_stop=false, vector<double> vpars={});

