#pragma once

#include <boost/math/tools/roots.hpp>
#include "tracing_helpers.h"
#include <array>
#include <vector>
#include <tuple>
#include <memory>
#include <functional>
#include <iostream>

using std::array;
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

class MaxToroidalFluxStoppingCriterion : public StoppingCriterion {
    private:
        double max_s;
    public:
        MaxToroidalFluxStoppingCriterion(double max_s) : max_s(max_s) {};
        bool operator()(int iter, double dt, double t, double s, double theta, double zeta, double vpar=0) override {
            return s>=max_s;
        };
};

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

template<std::size_t m, std::size_t n>
array<double, m+n> join(const array<double, m>& a, const array<double, n>& b)
{
     array<double, m+n> res;
     for (int i = 0; i < m; ++i) {
         res[i] = a[i];
     }
     for (int i = 0; i < n; ++i) {
         res[i+m] = b[i];
     }
     return res;
}

template<class RHS, class DENSE>
bool check_stopping_criteria(RHS rhs, typename RHS::State y, int iter, vector<array<double, RHS::Size+1>> &res, 
    vector<array<double, RHS::Size+2>> &res_hits, DENSE dense, double t_last, double t_current, double dt, 
    double zeta_last, double zeta_current, double vpar_last, double vpar_current, double abstol, vector<double> zetas, 
    vector<double> omegas, vector<shared_ptr<StoppingCriterion>> stopping_criteria, vector<double> vpars, 
    bool zetas_stop, bool vpars_stop)
{
    typedef typename RHS::State State;
    // abstol?
    boost::math::tools::eps_tolerance<double> roottol(-int(std::log2(abstol)));
    uintmax_t rootmaxit = 200;
    State temp;

    bool stop = false;
    array<double, RHS::Size> ykeep = {};

    // Now check whether we have hit any of the vpar planes
    for (int i = 0; i < vpars.size(); ++i) {
        double vpar = vpars[i];
        if((vpar_last-vpar != 0) && (vpar_current-vpar != 0) && (((vpar_last-vpar > 0) ? 1 : ((vpar_last-vpar < 0) ? -1 : 0)) != ((vpar_current-vpar > 0) ? 1 : ((vpar_current-vpar < 0) ? -1 : 0)))){ // check whether vpar = vpars[i] was crossed
            std::function<double(double)> rootfun = [&dense, &temp, &vpar_last, &vpar](double t){
                dense.calc_state(t, temp);
                if (vpar == 0) {
                    return (temp[3]-vpar);
                } else {
                    // Normalize by vpar since it can be large in magnitude
                    return (temp[3]-vpar)/vpar; 
                }
            };
            auto root = toms748_solve(rootfun, t_last, t_current, vpar_last-vpar, vpar_current-vpar, roottol, rootmaxit);
            double f0 = rootfun(root.first);
            double f1 = rootfun(root.second);
            double troot = std::abs(f0) < std::abs(f1) ? root.first : root.second;
            dense.calc_state(troot, ykeep);
            if (rhs.axis==1) {
                ykeep[0] = pow(temp[0],2) + pow(temp[1],2);
                ykeep[1] = atan2(temp[1],temp[0]);
            } else if (rhs.axis==2) {
                ykeep[0] = sqrt(pow(temp[0],2) + pow(temp[1],2));
                ykeep[1] = atan2(temp[1],temp[0]);
            }
            res_hits.push_back(join<2, RHS::Size>({troot, double(i) + zetas.size()}, ykeep));
            if (vpars_stop) {
                stop = true;
                break;
            }
        }
    }
    // Now check whether we have hit any of the (zeta - omega t) planes
    for (int i = 0; i < zetas.size(); ++i) {
        double zeta = zetas[i];
        double omega = omegas[i];
        double phase_last = zeta_last - omega*t_last;
        double phase_current = zeta_current - omega*t_current;
        if((std::floor((phase_last - zeta)/(2*M_PI)) != std::floor((phase_current-zeta)/(2*M_PI)))) { // check whether zeta+k*2pi for some k was crossed
            int fak = std::round(((phase_last+phase_current)/2-zeta)/(2*M_PI));
            double phase_shift = fak*2*M_PI + zeta;
            assert((phase_last <= phase_shift && phase_shift <= phase_current) || (phase_current <= phase_shift && phase_shift <= phase_last));

            std::function<double(double)> rootfun = [&phase_shift, &zeta_last, &omega, &dense, &temp](double t){
                dense.calc_state(t, temp);
                return temp[2] - omega*t - phase_shift;
            };
            auto root = toms748_solve(rootfun, t_last, t_current, phase_last - phase_shift, phase_current - phase_shift, roottol, rootmaxit);
            double f0 = rootfun(root.first);
            double f1 = rootfun(root.second);
            double troot = std::abs(f0) < std::abs(f1) ? root.first : root.second;
            dense.calc_state(troot, temp);
            ykeep = temp;
            if (rhs.axis==1) {
                ykeep[0] = pow(temp[0],2) + pow(temp[1],2);
                ykeep[1] = atan2(temp[1],temp[0]);
            } else if (rhs.axis==2) {
                ykeep[0] = sqrt(pow(temp[0],2) + pow(temp[1],2));
                ykeep[1] = atan2(temp[1],temp[0]);
            }
            res_hits.push_back(join<2, RHS::Size>({troot, double(i)}, ykeep));
            if (zetas_stop && !stop) {
                stop = true;
                break;
            }
        }
    }
    // check whether we have satisfied any of the extra stopping criteria (e.g. left a surface)
    for (int i = 0; i < stopping_criteria.size(); ++i) {
        ykeep = y;
        if (rhs.axis==1) {
            ykeep[0] = pow(y[0],2) + pow(y[1],2);
            ykeep[1] = atan2(y[1],y[0]);
        } else if (rhs.axis==2) {
            ykeep[0] = sqrt(pow(y[0],2) + pow(y[1],2));
            ykeep[1] = atan2(y[1],y[0]);
        }
        if(stopping_criteria[i] && (*stopping_criteria[i])(iter, dt, t_current, ykeep[0], ykeep[1], ykeep[2], ykeep[3])){
            stop = true;
            res_hits.push_back(join<2, RHS::Size>({t_current, -1-double(i)}, ykeep));
            res.push_back(join<1, RHS::Size>({t_current}, {ykeep})); 
            break;
        }
    }

    return stop;
}