#include <memory>
#include <vector>
#include <functional>
#include "magneticfield.h"
#include <cassert>
#include <stdexcept>
#include "tracing.h"
using std::shared_ptr;
using std::vector;
using std::tuple;
using std::pair;
using std::function;

#include "xtensor-python/pyarray.hpp"     // Numpy bindings
#include "xtensor-python/pytensor.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;




#if WITH_BOOST
#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint.hpp>
//#include <boost/numeric/odeint/stepper/bulirsch_stoer_dense_out.hpp>
using boost::math::tools::toms748_solve;
using namespace boost::numeric::odeint;
#else
#endif


double get_phi(double x, double y, double phi_near){
    double phi = std::atan2(y, x);
    if(phi < 0)
        phi += 2*M_PI;
    double phi_near_mod = std::fmod(phi_near, 2*M_PI);
    double nearest_multiple = std::round(phi_near/(2*M_PI))*2*M_PI;
    double opt1 = nearest_multiple - 2*M_PI + phi;
    double opt2 = nearest_multiple + phi;
    double opt3 = nearest_multiple + 2*M_PI + phi;
    double dist1 = std::abs(opt1-phi_near);
    double dist2 = std::abs(opt2-phi_near);
    double dist3 = std::abs(opt3-phi_near);
    if(dist1 <= std::min(dist2, dist3))
        return opt1;
    else if(dist2 <= std::min(dist1, dist3))
        return opt2;
    else
        return opt3;
}


template<class RHS>
tuple<vector<array<double, RHS::Size+1>>, vector<array<double, RHS::Size+2>>>
solve(RHS rhs, typename RHS::State y, double tmax, double dt, double dtmax, double tol, vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria)
{
#if WITH_BOOST
    vector<array<double, RHS::Size+1>> res = {};
    vector<array<double, RHS::Size+2>> res_phi_hits = {};
    typedef typename RHS::State State;
    typedef typename boost::numeric::odeint::result_of::make_dense_output<runge_kutta_dopri5<State>>::type dense_stepper_type;
    dense_stepper_type dense = make_dense_output(tol, tol, dtmax, runge_kutta_dopri5<State>());
    double t = 0;
    dense.initialize(y, t, dt);
    int iter = 0;
    bool stop = false;
    double phi_last = get_phi(y[0], y[1], M_PI);
    double phi_current;
    boost::math::tools::eps_tolerance<double> roottol(-int(std::log2(tol)));
    uintmax_t rootmaxit = 200;
    State temp;
    do {
        res.push_back(join<1, RHS::Size>({t}, y));
        tuple<double, double> step = dense.do_step(rhs);
        iter++;
        t = dense.current_time();
        y = dense.current_state();
        phi_current = get_phi(y[0], y[1], phi_last);
        double tlast = std::get<0>(step);
        double tcurrent = std::get<1>(step);
        // Now check whether we have hit any of the phi planes
        for (int i = 0; i < phis.size(); ++i) {
            double phi = phis[i];
            if(std::floor((phi_last-phi)/(2*M_PI)) != std::floor((phi_current-phi)/(2*M_PI))){ // check whether phi+k*2pi for some k was crossed
                int fak = std::round(((phi_last+phi_current)/2-phi)/(2*M_PI));
                double phi_shift = fak*2*M_PI + phi;
                assert((phi_last <= phi_shift && phi_shift <= phi_current) || (phi_current <= phi_shift && phi_shift <= phi_last));

                std::function<double(double)> rootfun = [&dense, &phi_shift, &temp, &phi_last](double t){
                    dense.calc_state(t, temp);
                    double diff = get_phi(temp[0], temp[1], phi_last)-phi_shift;
                    return diff;
                };

                auto root = toms748_solve(rootfun, tlast, tcurrent, phi_last - phi_shift, phi_current-phi_shift, roottol, rootmaxit);
                double f0 = rootfun(root.first);
                double f1 = rootfun(root.second);
                double troot = std::abs(f0) < std::abs(f1) ? root.first : root.second;
                dense.calc_state(troot, temp);
                double rroot = std::sqrt(temp[0]*temp[0] + temp[1]*temp[1]);
                double phiroot = std::atan2(temp[1], temp[0]);
                if(phiroot<0)
                    phiroot += 2*M_PI;
                //fmt::print("root=({:.5f}, {:.5f}), tlast={:.5f}, phi_last={:.5f}, tcurrent={:.5f}, phi_current={:.5f}, phi_shift={:.5f}, phi_root={:.5f}\n", std::get<0>(root), std::get<1>(root), tlast, phi_last, tcurrent, phi_current, phi_shift, get_phi(temp[0], temp[1], phi_last));
                //fmt::print("t={:.5f}, xyz=({:.5f}, {:.5f}, {:.5f}), rphiz=({}, {}, {})\n", troot, temp[0], temp[1], temp[2], rroot, phiroot, temp[2]);
                //fmt::print("x={}, y={}, phi={}\n", temp[0], temp[1], std::atan2(temp[1], temp[0]));
                res_phi_hits.push_back(join<2, RHS::Size>({troot, double(i)}, temp));
            }
        }
        // check whether we have satisfied any of the extra stopping criteria (e.g. left a surface)
        for (int i = 0; i < stopping_criteria.size(); ++i) {
            if(stopping_criteria[i] && (*stopping_criteria[i])(iter, t, y[0], y[1], y[2])){
                stop = true;
                res_phi_hits.push_back(join<2, RHS::Size>({t, -1-double(i)}, y));
                break;
            }
        }
        phi_last = phi_current;
    } while(t < tmax && !stop);
    if(!stop){
        dense.calc_state(tmax, y);
        res.push_back(join<1, RHS::Size>({tmax}, y));
    }
    return std::make_tuple(res, res_phi_hits);
#else
    throw std::runtime_error("Guiding center computation not available without boost.");
#endif
}



template<template<class, std::size_t, xt::layout_type> class T>
tuple<vector<array<double, 5>>, vector<array<double, 6>>>
particle_guiding_center_tracing(
        shared_ptr<MagneticField<T>> field, double xinit, double yinit, double zinit,
        double m, double q, double vtotal, double vtang, double tmax, double tol, vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria)
{
    typename MagneticField<T>::Tensor2 xyz({{xinit, yinit, zinit}});
    field->set_points(xyz);
    double AbsB = field->AbsB_ref()(0);
    double vperp2 = vtotal*vtotal - vtang*vtang;
    double mu = vperp2/(2*AbsB);

    auto rhs_class = GuidingCenterRHS<T>(field, m, q, mu);
    array<double, 4> y = {xinit, yinit, zinit, vtang};

    double dtmax = 0.01/vtotal; // can at most move 1cm per step
    double dt = 0.001 * dtmax; // initial guess for first timestep, will be adjusted by adaptive timestepper

    return solve(rhs_class, y, tmax, dt, dtmax, tol, phis, stopping_criteria);
}

template
tuple<vector<array<double, 5>>, vector<array<double, 6>>> particle_guiding_center_tracing<xt::pytensor>(
        shared_ptr<MagneticField<xt::pytensor>> field, double xinit, double yinit, double zinit,
        double m, double q, double vtotal, double vtang, double tmax, double tol, vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria);


template<template<class, std::size_t, xt::layout_type> class T>
tuple<vector<array<double, 4>>, vector<array<double, 5>>>
fieldline_tracing(
    shared_ptr<MagneticField<T>> field, double xinit, double yinit, double zinit,
    double tmax, double tol, vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria)
{
    auto rhs_class = FieldlineRHS<T>(field);
    array<double, 3> y = {xinit, yinit, zinit};
    double dtmax = 0.1; // todo: better guess for dtmax (maybe bound so that one can't do more than half a rotation per step or so)
    double dt = 0.001 * dtmax; // initial guess for first timestep, will be adjusted by adaptive timestepper
    return solve(rhs_class, y, tmax, dt, dtmax, tol, phis, stopping_criteria);
}

template
tuple<vector<array<double, 4>>, vector<array<double, 5>>>
fieldline_tracing(
    shared_ptr<MagneticField<xt::pytensor>> field, double xinit, double yinit, double zinit,
    double tmax, double tol, vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria);
