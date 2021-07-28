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



#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint.hpp>
//#include <boost/numeric/odeint/stepper/bulirsch_stoer_dense_out.hpp>
using boost::math::tools::toms748_solve;
using namespace boost::numeric::odeint;

template<template<class, std::size_t, xt::layout_type> class T>
class GuidingCenterVacuumRHS {
    /*
     * The state consists of :math:`[x, y, z, v_par]` with
     *
     *   [\dot x, \dot y, \dot z] &= v_{||}\frac{B}{|B|} + \frac{m}{q|B|^3}  (0.5v_\perp^2 + v_{||}^2)  B\times \nabla(|B|)
     *   \dot v_{||}              &= -\mu  (B \cdot \nabla(|B|))
     *
     * where v_perp = 2*mu*|B|
     */
    private:
        std::array<double, 3> BcrossGradAbsB = {0., 0., 0.};
        typename MagneticField<T>::Tensor2 rphiz = xt::zeros<double>({1, 3});
        shared_ptr<MagneticField<T>> field;
        double m, q, mu;
    public:
        static constexpr int Size = 4;
        using State = std::array<double, Size>;


        GuidingCenterVacuumRHS(shared_ptr<MagneticField<T>> field, double m, double q, double mu)
            : field(field), m(m), q(q), mu(mu) {

            }

        void operator()(const State &ys, array<double, 4> &dydt,
                const double t) {
            double x = ys[0];
            double y = ys[1];
            double z = ys[2];
            double v_par = ys[3];

            rphiz(0, 0) = std::sqrt(x*x+y*y);
            rphiz(0, 1) = std::atan2(y, x);
            if(rphiz(0, 1) < 0)
                rphiz(0, 1) += 2*M_PI;
            rphiz(0, 2) = z;

            field->set_points_cyl(rphiz);
            auto& GradAbsB = field->GradAbsB_ref();
            auto& B = field->B_ref();
            double AbsB = field->AbsB_ref()(0);
            BcrossGradAbsB[0] = (B(0, 1) * GradAbsB(0, 2)) - (B(0, 2) * GradAbsB(0, 1));
            BcrossGradAbsB[1] = (B(0, 2) * GradAbsB(0, 0)) - (B(0, 0) * GradAbsB(0, 2));
            BcrossGradAbsB[2] = (B(0, 0) * GradAbsB(0, 1)) - (B(0, 1) * GradAbsB(0, 0));
            double v_perp2 = 2*mu*AbsB;
            double fak1 = (v_par/AbsB);
            double fak2 = (m/(q*pow(AbsB, 3)))*(0.5*v_perp2 + v_par*v_par);
            dydt[0] = fak1*B(0, 0) + fak2*BcrossGradAbsB[0];
            dydt[1] = fak1*B(0, 1) + fak2*BcrossGradAbsB[1];
            dydt[2] = fak1*B(0, 2) + fak2*BcrossGradAbsB[2];
            dydt[3] = -mu*(B(0, 0)*GradAbsB(0, 0) + B(0, 1)*GradAbsB(0, 1) + B(0, 2)*GradAbsB(0, 2))/AbsB;
        }
};

template<template<class, std::size_t, xt::layout_type> class T>
class FullorbitRHS {
    // Right hand side for full orbit tracing of particles, the state is
    // (x, y, z, \dot x, \dot y, \dot z) and the rhs is
    // (\dot x, \dot y, \dot z, \dot\dot x, \dot\dot y, \dot\dot z).
    // Using F=m*a and F = q * v \cross B, we get a = (q/m) * v\cross B
    // and hence \dot\dot (x, y, z) = (q/m)* \dot(x,y,z) \cross B
    // where we used v = \dot (x,y,z)
    private:
        typename MagneticField<T>::Tensor2 rphiz = xt::zeros<double>({1, 6});
        shared_ptr<MagneticField<T>> field;
        const double qoverm;
    public:
        static constexpr int Size = 6;
        using State = std::array<double, Size>;

        FullorbitRHS(shared_ptr<MagneticField<T>> field, double m, double q)
            : field(field), qoverm(q/m) {

            }
        void operator()(const array<double, 6> &ys, array<double, 6> &dydt,
                const double t) {
            double x = ys[0];
            double y = ys[1];
            double z = ys[2];
            double vx = ys[3];
            double vy = ys[4];
            double vz = ys[5];
            rphiz(0, 0) = std::sqrt(x*x+y*y);
            rphiz(0, 1) = std::atan2(y, x);
            if(rphiz(0, 1) < 0)
                rphiz(0, 1) += 2*M_PI;
            rphiz(0, 2) = z;
            field->set_points_cyl(rphiz);
            auto& B = field->B_ref();
            double Bx = B(0, 0);
            double By = B(0, 1);
            double Bz = B(0, 2);
            dydt[0] = vx;
            dydt[1] = vy;
            dydt[2] = vz;
            dydt[3] = qoverm * (vy*Bz-vz*By);
            dydt[4] = qoverm * (vz*Bx-vx*Bz);
            dydt[5] = qoverm * (vx*By-vy*Bx);
        }
};
template<template<class, std::size_t, xt::layout_type> class T>
class FieldlineRHS {
    private:
        typename MagneticField<T>::Tensor2 rphiz = xt::zeros<double>({1, 3});
        shared_ptr<MagneticField<T>> field;
    public:
        static constexpr int Size = 3;
        using State = std::array<double, Size>;

        FieldlineRHS(shared_ptr<MagneticField<T>> field)
            : field(field) {

            }
        void operator()(const array<double, 3> &ys, array<double, 3> &dydt,
                const double t) {
            double x = ys[0];
            double y = ys[1];
            double z = ys[2];
            rphiz(0, 0) = std::sqrt(x*x+y*y);
            rphiz(0, 1) = std::atan2(y, x);
            if(rphiz(0, 1) < 0)
                rphiz(0, 1) += 2*M_PI;
            rphiz(0, 2) = z;
            field->set_points_cyl(rphiz);
            auto& B = field->B_ref();
            dydt[0] = B(0, 0);
            dydt[1] = B(0, 1);
            dydt[2] = B(0, 2);
        }
};

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

template<std::size_t m, std::size_t n>
std::array<double, m+n> join(const std::array<double, m>& a, const std::array<double, n>& b){
     std::array<double, m+n> res;
     for (int i = 0; i < m; ++i) {
         res[i] = a[i];
     }
     for (int i = 0; i < n; ++i) {
         res[i+m] = b[i];
     }
     return res;
}



template<class RHS>
tuple<vector<array<double, RHS::Size+1>>, vector<array<double, RHS::Size+2>>>
solve(RHS rhs, typename RHS::State y, double tmax, double dt, double dtmax, double tol, vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria)
{
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
}



template<template<class, std::size_t, xt::layout_type> class T>
tuple<vector<array<double, 5>>, vector<array<double, 6>>>
particle_guiding_center_tracing(
        shared_ptr<MagneticField<T>> field, array<double, 3> xyz_init,
        double m, double q, double vtotal, double vtang, double tmax, double tol, bool vacuum, vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria)
{
    typename MagneticField<T>::Tensor2 xyz({{xyz_init[0], xyz_init[1], xyz_init[2]}});
    field->set_points(xyz);
    double AbsB = field->AbsB_ref()(0);
    double vperp2 = vtotal*vtotal - vtang*vtang;
    double mu = vperp2/(2*AbsB);

    array<double, 4> y = {xyz_init[0], xyz_init[1], xyz_init[2], vtang};
    double r0 = std::sqrt(xyz_init[0]*xyz_init[0] + xyz_init[1]*xyz_init[1]);
    double dtmax = r0*0.5*M_PI/vtotal; // can at most do quarter of a revolution per step
    double dt = 1e-3 * dtmax; // initial guess for first timestep, will be adjusted by adaptive timestepper

    if(vacuum){
        auto rhs_class = GuidingCenterVacuumRHS<T>(field, m, q, mu);
        return solve(rhs_class, y, tmax, dt, dtmax, tol, phis, stopping_criteria);
    }
    else
        throw std::logic_error("Guiding center right hand side currently only implemented for vacuum fields.");
}

template
tuple<vector<array<double, 5>>, vector<array<double, 6>>> particle_guiding_center_tracing<xt::pytensor>(
        shared_ptr<MagneticField<xt::pytensor>> field, array<double, 3> xyz_init,
        double m, double q, double vtotal, double vtang, double tmax, double tol, bool vacuum,
        vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria);


template<template<class, std::size_t, xt::layout_type> class T>
tuple<vector<array<double, 7>>, vector<array<double, 8>>>
particle_fullorbit_tracing(
        shared_ptr<MagneticField<T>> field, array<double, 3> xyz_init, array<double, 3> v_init,
        double m, double q, double tmax, double tol, vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria)
{

    auto rhs_class = FullorbitRHS<T>(field, m, q);
    array<double, 6> y = {xyz_init[0], xyz_init[1], xyz_init[2], v_init[0], v_init[1], v_init[2]};

    double vtotal = std::sqrt(std::pow(v_init[0], 2) + std::pow(v_init[1], 2) + std::pow(v_init[2], 2));
    double r0 = std::sqrt(xyz_init[0]*xyz_init[0] + xyz_init[1]*xyz_init[1]);
    double dtmax = r0*0.5*M_PI/vtotal; // can at most do quarter of a revolution per step
    double dt = 1e-3 * dtmax; // initial guess for first timestep, will be adjusted by adaptive timestepper

    return solve(rhs_class, y, tmax, dt, dtmax, tol, phis, stopping_criteria);
}

template
tuple<vector<array<double, 7>>, vector<array<double, 8>>> particle_fullorbit_tracing<xt::pytensor>(
        shared_ptr<MagneticField<xt::pytensor>> field, array<double, 3> xyz_init, array<double, 3> v_init,
        double m, double q, double tmax, double tol, vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria);

template<template<class, std::size_t, xt::layout_type> class T>
tuple<vector<array<double, 4>>, vector<array<double, 5>>>
fieldline_tracing(
    shared_ptr<MagneticField<T>> field, array<double, 3> xyz_init,
    double tmax, double tol, vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria)
{
    auto rhs_class = FieldlineRHS<T>(field);
    double r0 = std::sqrt(xyz_init[0]*xyz_init[0] + xyz_init[1]*xyz_init[1]);
    typename MagneticField<T>::Tensor2 xyz({{xyz_init[0], xyz_init[1], xyz_init[2]}});
    field->set_points(xyz);
    double AbsB = field->AbsB_ref()(0);
    double dtmax = r0*0.5*M_PI/AbsB; // can at most do quarter of a revolution per step
    double dt = 1e-5 * dtmax; // initial guess for first timestep, will be adjusted by adaptive timestepper
    return solve(rhs_class, xyz_init, tmax, dt, dtmax, tol, phis, stopping_criteria);
}

template
tuple<vector<array<double, 4>>, vector<array<double, 5>>>
fieldline_tracing(
    shared_ptr<MagneticField<xt::pytensor>> field, array<double, 3> xyz_init,
    double tmax, double tol, vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria);
