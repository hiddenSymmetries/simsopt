#include <memory>
#include <vector>
#include <functional>
#include "magneticfield.h"
#include <cassert>
#include "tracing.h"
using std::shared_ptr;
using std::vector;
using std::tuple;
using std::pair;
using std::function;

#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;




#if WITH_BOOST
#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint.hpp>
using boost::math::tools::toms748_solve;
using namespace boost::numeric::odeint;
#else
#endif

template<class Array>
tuple<vector<double>, vector<vector<double>>> particle_guiding_center_tracing(
        shared_ptr<MagneticField<Array>> field, double xinit, double yinit, double zinit,
        double m, double q, double vtotal, double vtang, double tmax, double tol, vector<shared_ptr<StoppingCriterion>> stopping_criteria)
{
#if WITH_BOOST

    Array xyz({{xinit, yinit, zinit}});
    field->set_points(xyz);
    double AbsB = field->AbsB_ref()(0);
    double vperp2 = vtotal*vtotal - vtang*vtang;
    double mu = vperp2/(2*AbsB);


    runge_kutta_dopri5<vector<double>> stepper;
    //runge_kutta_fehlberg78<vector<double>> stepper;
    auto rhs_class = GuidingCenterRHS<Array>(field, m, q, mu);
    GuidingCenterRHS<Array>& rhs_ref = rhs_class; 
    vector<double> y = {xinit, yinit, zinit, vtang};

    double dtmax = 0.1/vtotal; // can at most move 1cm per step
    double dt = 0.001 * dtmax; // initial guess for first timestep, will be adjusted by adaptive timestepper
    auto res_y = vector<vector<double>>();
    res_y.reserve(2*int(tmax/dtmax));
    auto res_t = vector<double>();
    res_t.reserve(2*int(tmax/dtmax));
    function<void(const vector<double> &, double)> observer =
        [&res_y, &res_t](const vector<double> &y, double t) {
            res_y.push_back(y);
            res_t.push_back(t);
        };
    
    typedef boost::numeric::odeint::result_of::make_dense_output<runge_kutta_dopri5<vector<double>>>::type dense_stepper_type;
    dense_stepper_type dense = make_dense_output(tol, tol, dtmax, runge_kutta_dopri5<vector<double>>());
    double t = 0;
    dense.initialize(y, t, dt);
    int iter = 0;
    bool stop = false;
    do {
        res_t.push_back(t);
        res_y.push_back(y);
        dense.do_step(rhs_ref);
        iter++;
        t = dense.current_time();
        y = dense.current_state();
        for (int i = 0; i < stopping_criteria.size(); ++i) {
            if(stopping_criteria[i] && (*stopping_criteria[i])(iter, t, y)){
                stop = true;
                break;
            }
        }
    } while(t < tmax && !stop);
    if(!stop){
        res_t.push_back(tmax);
        vector<double> yfinal(4, 0.);
        dense.calc_state(tmax, yfinal);
        res_y.push_back(yfinal);
    }

    return make_tuple(res_t, res_y);

#else
    throw runtime_error("Guiding center computation not available without boost.")
#endif
}

template
tuple<vector<double>, vector<vector<double>>> particle_guiding_center_tracing<Array>(
        shared_ptr<MagneticField<Array>> field, double xinit, double yinit, double zinit,
        double m, double q, double vtotal, double vtang, double tmax, double tol, vector<shared_ptr<StoppingCriterion>> stopping_criteria);

double get_phi(const vector<double>& pt, double phi_near){
    double phi = std::atan2(pt[1], pt[0]);
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



template<class Array>
tuple<vector<double>, vector<vector<double>>, vector<vector<vector<double>>>> fieldline_tracing(
        shared_ptr<MagneticField<Array>> field, double xinit, double yinit, double zinit,
        double tmax, double tol, vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria)
{
#if WITH_BOOST
    runge_kutta_dopri5<vector<double>> stepper;
    //runge_kutta_fehlberg78<vector<double>> stepper;
    auto rhs_class = FieldlineRHS<Array>(field);
    vector<double> y = {xinit, yinit, zinit};

    double dtmax = 0.1; // can at most move 1cm per step
    double dt = 0.001 * dtmax; // initial guess for first timestep, will be adjusted by adaptive timestepper

    auto res_y = vector<vector<double>>();
    res_y.reserve(2*int(tmax/dtmax));
    auto res_t = vector<double>();
    res_t.reserve(2*int(tmax/dtmax));
    vector<vector<vector<double>>> res_phi_hits(phis.size(), vector<vector<double>>());
    function<void(const vector<double> &, double)> observer =
        [&res_y, &res_t](const vector<double> &y, double t) {
            res_y.push_back(y);
            res_t.push_back(t);
        };
    
    typedef boost::numeric::odeint::result_of::make_dense_output<runge_kutta_dopri5<vector<double>>>::type dense_stepper_type;
    dense_stepper_type dense = make_dense_output(tol, tol, dtmax, runge_kutta_dopri5<vector<double>>());
    double t = 0;
    dense.initialize(y, t, dt);
    int iter = 0;
    bool stop = false;
    double phi_last = get_phi(y, M_PI);
    double phi_current;
    boost::math::tools::eps_tolerance<double> roottol(-int(std::log2(tol)));
    vector<double> temp = {0., 0., 0.};
    do {
        res_t.push_back(t);
        res_y.push_back(y);
        tuple<double, double> step = dense.do_step(rhs_class);
        iter++;
        t = dense.current_time();
        y = dense.current_state();
        phi_current = get_phi(y, phi_last);
        double tlast = std::get<0>(step);
        double tcurrent = std::get<1>(step);
        //fmt::print("tlast={:5f}, tcurrent={:5f}, phi_last={:5f}, phi_current={:5f}\n", tlast, tcurrent, phi_last, phi_current);
        for (int i = 0; i < phis.size(); ++i) {
            double phi = phis[i];
            if(std::floor((phi_last-phi)/(2*M_PI)) != std::floor((phi_current-phi)/(2*M_PI))){ // check whether phi+k*2pi for some k was crossed
                int fak = std::round(((phi_last+phi_current)/2-phi)/(2*M_PI));
                double phi_shift = fak*2*M_PI + phi;
                assert((phi_last <= phi_shift && phi_shift <= phi_current) || (phi_current <= phi_shift && phi_shift <= phi_last));

                std::function<double(double)> rootfun = [&dense, &phi_shift, &temp, &phi_last](double t){
                    dense.calc_state(t, temp);
                    return get_phi(temp, phi_last)-phi_shift;
                };

                uintmax_t maxit = 100;
                auto root = toms748_solve(rootfun, tlast, tcurrent, phi_last - phi_shift, phi_current-phi_shift, roottol, maxit);
                auto troot = (std::get<0>(root)+std::get<1>(root))/2;
                dense.calc_state(troot, temp);
                double rroot = std::sqrt(temp[0]*temp[0] + temp[1]*temp[1]);
                double phiroot = std::atan2(temp[1], temp[0]);
                if(phiroot<0)
                    phiroot += 2*M_PI;
                //fmt::print("t={:.5f}, xyz=({:.5f}, {:.5f}, {:.5f}), rphiz=({}, {}, {})\n", troot, temp[0], temp[1], temp[2], rroot, phiroot, temp[2]);
                res_phi_hits[i].push_back({troot, rroot, phiroot, temp[2]});
            }
        }
        phi_last = phi_current;
        for (int i = 0; i < stopping_criteria.size(); ++i) {
            if(stopping_criteria[i] && (*stopping_criteria[i])(iter, t, y)){
                stop = true;
                break;
            }
        }
    } while(t < tmax && !stop);
    if(!stop){
        res_t.push_back(tmax);
        vector<double> yfinal(3, 0.);
        dense.calc_state(tmax, yfinal);
        res_y.push_back(yfinal);
    }

    return make_tuple(res_t, res_y, res_phi_hits);
#else
    throw runtime_error("Fieldline computation not available without boost.")
#endif
}

template
tuple<vector<double>, vector<vector<double>>, vector<vector<vector<double>>>> fieldline_tracing(
        shared_ptr<MagneticField<Array>> field, double xinit, double yinit, double zinit,
        double tmax, double tol, vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria);
