#pragma once
#include <memory>
#include <vector>
#include "boozermagneticfield.h"
#include "regular_grid_interpolant_3d.h"

using std::shared_ptr;
using std::vector;
using std::tuple;

double get_phi(double x, double y, double phi_near);


// template<class Array>
class StoppingCriterion {
    public:
        // Should return true if the Criterion is satisfied.
        virtual bool operator()(int iter, double dt, double t, double x, double y, double z, double vpar=0) = 0;
        // virtual bool operator()(int iter, double t, double x, double y, double z) = 0;
        virtual ~StoppingCriterion() {}
};

class ZetaStoppingCriterion : public StoppingCriterion {
    private:
        int nfp;
    public:
        ZetaStoppingCriterion(int nfp) : nfp(nfp) {
        };
        bool operator()(int iter, double dt, double t, double s, double theta, double zeta, double vpar=0) override {
            // return std::abs(std::fmod(zeta,2*M_PI/nfp))<=zeta_crit;
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
        // bool operator()(int iter, double t, Array& ) override {
        bool operator()(int iter, double dt, double t, double x, double y, double z, double vpar) override {
            return std::abs(vpar)<=vpar_crit;
        };
};

// template<class Array>
class ToroidalTransitStoppingCriterion : public StoppingCriterion {
    private:
        int max_transits;
        double phi_last;
        double phi_init;
        bool flux;
    public:
        ToroidalTransitStoppingCriterion(int max_transits, bool flux) : max_transits(max_transits), flux(flux) {
        };
        // bool operator()(int iter, double t, Array& y) override {
        bool operator()(int iter, double dt, double t, double x, double y, double z, double vpar=0) override {
            if (iter == 1) {
              phi_last = M_PI;
            }
            double phi = z;
            // double phi = z;
            if (!flux) {
                // phi = get_phi(y[0], y[1], phi_last);
              phi = get_phi(x, y, phi_last);
            }
            if (iter == 1) {
              phi_init = phi;
            }
            phi_last = phi;
            int ntransits = std::abs(std::floor((phi-phi_init)/(2*M_PI)));
            return ntransits>=max_transits;
        };
};

// template<class Array>
class MaxToroidalFluxStoppingCriterion : public StoppingCriterion {
    private:
        double max_s;
    public:
        MaxToroidalFluxStoppingCriterion(double max_s) : max_s(max_s) {};
        // bool operator()(int iter, double t, Array& y) override {
        bool operator()(int iter, double dt, double t, double s, double theta, double zeta, double vpar=0) override {
            return s>=max_s;
            // return y[0]>=max_s;
        };
};

// template<class Array>
class MinToroidalFluxStoppingCriterion : public StoppingCriterion {
    private:
        double min_s;
    public:
        MinToroidalFluxStoppingCriterion(double min_s) : min_s(min_s) {};
        // bool operator()(int iter, double t, Array& y) override {
        bool operator()(int iter, double dt, double t, double s, double theta, double zeta, double vpar=0) override {
            return s<=min_s;
            // return y[0]<=min_s;
        };
};

// template<class Array>
class IterationStoppingCriterion : public StoppingCriterion {
    private:
        int max_iter;
    public:
        IterationStoppingCriterion(int max_iter) : max_iter(max_iter) {};
        bool operator()(int iter, double dt, double t, double x, double y, double z, double vpar=0) override {
        // bool operator()(int iter, double t, Array& y) override {
            return iter>max_iter;
        };
};

class StepSizeStoppingCriterion : public StoppingCriterion {
    private:
        int min_dt;
    public:
			  StepSizeStoppingCriterion(int min_dt) : min_dt(min_dt) {};
			  bool operator()(int iter, double dt, double t, double x, double y, double z, double vpar=0) override {
			  // bool operator()(int iter, double t, Array& y) override {
					return dt<min_dt;
			  };
	};

	template<class Array>
	class LevelsetStoppingCriterion : public StoppingCriterion {
		 private:
			  shared_ptr<RegularGridInterpolant3D<Array>> levelset;
		 public:
			  LevelsetStoppingCriterion(shared_ptr<RegularGridInterpolant3D<Array>> levelset) : levelset(levelset) { };
			  // bool operator()(int iter, double t, Array2& y) override {
			  bool operator()(int iter, double dt, double t, double x, double y, double z, double vpar=0) override {
					double r = std::sqrt(x*x + y*y);
					// double r = std::sqrt(y[0]*y[0] + y[1]*y[1]);
					double phi = std::atan2(y, x);
					// double phi = std::atan2(y[1],y[0]);
					if(phi < 0)
						 phi += 2*M_PI;
					double f = levelset->evaluate(r, phi, z)[0];
					// double f = levelset->evaluate(r, phi, y[2])[0];
					//fmt::print("Levelset at xyz=({}, {}, {}), rphiz=({}, {}, {}), f={}\n", x, y, z, r, phi, z, f);
					return f<0;
			  };
	};

	template<template<class, std::size_t, xt::layout_type> class T>
	tuple<vector<array<double, 6>>, vector<array<double, 7>>>
	particle_guiding_center_boozer_perturbed_tracing(
			  shared_ptr<BoozerMagneticField<T>> field, array<double, 3> stz_init,
			  double m, double q, double vtotal, double vtang, double mu, double tmax, double abstol, double reltol,
			  bool vacuum, bool noK, vector<double> zetas, vector<double> omegas,
			  vector<shared_ptr<StoppingCriterion>> stopping_criteria,
			  bool zetas_stop=false, bool vpars_stop=false,
			  double alphahat=0, double omega=0, int alpham=0, int alphan=0, double phase=0,
			  bool forget_exact_path=false, int axis=0,  vector<double> vpars={});

	template<template<class, std::size_t, xt::layout_type> class T>
	tuple<vector<array<double, 5>>, vector<array<double, 6>>>
	particle_guiding_center_boozer_tracing(
			  shared_ptr<BoozerMagneticField<T>> field, array<double, 3> stz_init,
        double m, double q, double vtotal, double vtang, double tmax, double dt, double abstol, double reltol, double roottol,
        bool vacuum, bool noK, bool GPU, bool solveSympl, vector<double> zetas, vector<double> omegas,
        vector<shared_ptr<StoppingCriterion>> stopping_criteria,
        bool forget_exact_path=false, int axis=0, bool predictor_step=true,
        bool zetas_stop=false, bool vpars_stop=false, vector<double> vpars={});

