#include <memory>
#include <vector>
#include <functional>
#include "boozermagneticfield.h"
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
using boost::math::tools::toms748_solve;
using namespace boost::numeric::odeint;

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
#include <iomanip>

template<template<class, std::size_t, xt::layout_type> class T>
class GuidingCenterVacuumBoozerRHS {
    /*
     * The state consists of :math:`[s, t, z, v_par]` with
     *
     *    \dot s = -|B|_{,\theta} m(v_{||}^2/|B| + \mu)/(q \psi_0)
     *    \dot \theta = |B|_{,s} m(v_{||}^2/|B| + \mu)/(q \psi_0) + \iota v_{||} |B|/G
     *    \dot \zeta = v_{||}|B|/G
     *    \dot v_{||} = -(\iota |B|_{,\theta} + |B|_{,\zeta})\mu |B|/G,
     *
     *  where :math:`q` is the charge, :math:`m` is the mass, and :math:`v_\perp = 2\mu|B|`.
     *
     */
    private:
        typename BoozerMagneticField<T>::Tensor2 stz = xt::zeros<double>({1, 3});
        shared_ptr<BoozerMagneticField<T>> field;
        double m, q, mu;
    public:
        int axis;
        static constexpr int Size = 4;
        using State = std::array<double, Size>;

        GuidingCenterVacuumBoozerRHS(shared_ptr<BoozerMagneticField<T>> field, double m, double q, double mu, int axis)
            : field(field), m(m), q(q), mu(mu), axis(axis) {
            }

        void operator()(const State &ys, array<double, 4> &dydt,
                const double t) {
            double v_par = ys[3];
            double s, theta;
            if (axis==1) {
                s = pow(ys[0],2)+pow(ys[1],2);
                theta = atan2(ys[1],ys[0]);          
            } else if (axis==2) {
                s = sqrt(pow(ys[0],2)+pow(ys[1],2));
                theta = atan2(ys[1],ys[0]); 
            } else {
                s = ys[0];
                theta = ys[1];
            }  

            stz(0, 0) = s;
            stz(0, 1) = theta;
            stz(0, 2) = ys[2];

            field->set_points(stz);
            auto psi0 = field->psi0;
            double modB = field->modB_ref()(0);
            double G = field->G_ref()(0);
            double iota = field->iota_ref()(0);
            double dmodBds = field->modB_derivs_ref()(0);
            double dmodBdtheta = field->modB_derivs_ref()(1);
            double dmodBdzeta = field->modB_derivs_ref()(2);
            double v_perp2 = 2*mu*modB;
            double fak1 = m*v_par*v_par/modB + m*mu;

            double sdot = -dmodBdtheta*fak1/(q*psi0);
            double tdot = dmodBds*fak1/(q*psi0) + iota*v_par*modB/G; 

            if (axis==1) {
                dydt[0] = sdot*cos(theta)/(2*sqrt(s)) - sqrt(s) * sin(theta) * tdot;
                dydt[1] = sdot*sin(theta)/(2*sqrt(s)) + sqrt(s) * cos(theta) * tdot;
            } else if (axis==2) {
                dydt[0] = sdot*cos(theta) - s * sin(theta) * tdot;
                dydt[1] = sdot*sin(theta) + s * cos(theta) * tdot; 
            } else {
                dydt[0] = sdot;
                dydt[1] = tdot;
            }
            dydt[2] = v_par*modB/G;
            dydt[3] = -(iota*dmodBdtheta + dmodBdzeta)*mu*modB/G;
        }
};

template<template<class, std::size_t, xt::layout_type> class T>
class GuidingCenterVacuumBoozerPerturbedRHS {
    /*
     * The state consists of :math:`[s, t, z, v_par]` with
     *
     *    \dot s = -|B|_{,\theta} m(v_{||}^2/|B| + \mu)/(q \psi_0)
     *    \dot \theta = |B|_{,s} m(v_{||}^2/|B| + \mu)/(q \psi_0) + \iota v_{||} |B|/G
     *    \dot \zeta = v_{||}|B|/G
     *    \dot v_{||} = -(\iota |B|_{,\theta} + |B|_{,\zeta})\mu |B|/G,
     *
     *  where :math:`q` is the charge, :math:`m` is the mass, and :math:`v_\perp = 2\mu|B|`.
     *
     */
    private:
        typename BoozerMagneticField<T>::Tensor2 stz = xt::zeros<double>({1, 3});
        shared_ptr<BoozerMagneticField<T>> field;
        double m, q, mu, Phihat, omega, phase;
        int Phim, Phin;
    public:
        int axis;
        static constexpr int Size = 5;
        using State = std::array<double, Size>;

        GuidingCenterVacuumBoozerPerturbedRHS(shared_ptr<BoozerMagneticField<T>> field,
            double m, double q, double mu, double Phihat, double omega, int Phim,
            int Phin, double phase, int axis)
            : field(field), m(m), q(q), mu(mu), Phihat(Phihat), omega(omega),
              Phim(Phim), Phin(Phin), phase(phase), axis(axis) {
            }

        void operator()(const State &ys, array<double, 5> &dydt,
                const double t) {
            double v_par = ys[3];
            double time = ys[4];
            double s, theta;
            if (axis==1) {
                s = pow(ys[0],2)+pow(ys[1],2);
                theta = atan2(ys[1],ys[0]);          
            } else if (axis==2) {
                s = sqrt(pow(ys[0],2)+pow(ys[1],2));
                theta = atan2(ys[1],ys[0]); 
            } else {
                s = ys[0];
                theta = ys[1];
            }  

            stz(0, 0) = s;
            stz(0, 1) = theta;
            stz(0, 2) = ys[2];

            field->set_points(stz);
            auto psi0 = field->psi0;
            double modB = field->modB_ref()(0);
            double G = field->G_ref()(0);
            double iota = field->iota_ref()(0);
            double diotadpsi = field->diotads_ref()(0)/psi0;
            double dmodBdpsi = field->modB_derivs_ref()(0)/psi0;
            double dmodBdtheta = field->modB_derivs_ref()(1);
            double dmodBdzeta = field->modB_derivs_ref()(2);
            double v_perp2 = 2*mu*modB;
            double fak1 = m*v_par*v_par/modB + m*mu;
            double Phi = Phihat * sin(Phim * theta - Phin * ys[2] + omega * time + phase);
            double dPhidpsi = 0;
            double Phidot = Phihat * omega * cos(Phim * theta - Phin * ys[2] + omega * time + phase);
            double dPhidtheta = Phidot * Phim / omega;
            double dPhidzeta = - Phidot * Phin / omega;
            double alpha = - Phi * (iota*Phim - Phin)/(omega*G);
            double alphadot = - Phidot * (iota*Phim - Phin)/(omega*G);
            double dalphadtheta = - dPhidtheta * (iota*Phim - Phin)/(omega*G);
            double dalphadzeta = - dPhidzeta * (iota*Phim - Phin)/(omega*G);
            double dalphadpsi = - dPhidpsi * (iota*Phim - Phin)/(omega*G) \
                - Phi * (diotadpsi*Phim)/(omega*G);

            double sdot = (-dmodBdtheta*fak1/q + dalphadtheta*modB*v_par - dPhidtheta)/psi0;
            double tdot = dmodBdpsi*fak1/q + (iota - dalphadpsi*G)*v_par*modB/G + dPhidpsi;
            if (axis==1) {
                dydt[0] = sdot*cos(theta)/(2*sqrt(s)) - sqrt(s) * sin(theta) * tdot;
                dydt[1] = sdot*sin(theta)/(2*sqrt(s)) + sqrt(s) * cos(theta) * tdot;
            } else if (axis==2) {
                dydt[0] = sdot*cos(theta) - s * sin(theta) * tdot;
                dydt[1] = sdot*sin(theta) + s * cos(theta) * tdot; 
            } else {
                dydt[0] = sdot;
                dydt[1] = tdot;
            }
            dydt[2] = v_par*modB/G;
            dydt[3] = -modB/(G*m) * (m*mu*(dmodBdzeta + dalphadtheta*dmodBdpsi*G \
                    + dmodBdtheta*(iota - dalphadpsi*G)) + q*(alphadot*G \
                    + dalphadtheta*G*dPhidpsi + (iota - dalphadpsi*G)*dPhidtheta + dPhidzeta)) \
                    + v_par/modB * (dmodBdtheta*dPhidpsi - dmodBdpsi*dPhidtheta);
            dydt[4] = 1;
        }
};


template<template<class, std::size_t, xt::layout_type> class T>
class GuidingCenterNoKBoozerPerturbedRHS {
    /*
     * The state consists of :math:`[s, t, z, v_par]` with
     *
     *    \dot s = -|B|_{,\theta} m(v_{||}^2/|B| + \mu)/(q \psi_0)
     *    \dot \theta = |B|_{,s} m(v_{||}^2/|B| + \mu)/(q \psi_0) + \iota v_{||} |B|/G
     *    \dot \zeta = v_{||}|B|/G
     *    \dot v_{||} = -(\iota |B|_{,\theta} + |B|_{,\zeta})\mu |B|/G,
     *
     *  where :math:`q` is the charge, :math:`m` is the mass, and :math:`v_\perp = 2\mu|B|`.
     *
     */
    private:
        typename BoozerMagneticField<T>::Tensor2 stz = xt::zeros<double>({1, 3});
        shared_ptr<BoozerMagneticField<T>> field;
        double m, q, mu, Phihat, omega, phase;
        int Phim, Phin;
    public:
        int axis;
        static constexpr int Size = 5;
        using State = std::array<double, Size>;

        GuidingCenterNoKBoozerPerturbedRHS(shared_ptr<BoozerMagneticField<T>> field,
            double m, double q, double mu, double Phihat, double omega, int Phim,
            int Phin, double phase, int axis)
            : field(field), m(m), q(q), mu(mu), Phihat(Phihat), omega(omega),
              Phim(Phim), Phin(Phin), phase(phase), axis(axis) {
            }

        void operator()(const State &ys, array<double, 5> &dydt,
                const double t) {
            double v_par = ys[3];
            double time = ys[4];
            double s, theta;
            if (axis==1) {
                s = pow(ys[0],2)+pow(ys[1],2);
                theta = atan2(ys[1],ys[0]);          
            } else if (axis==2) {
                s = sqrt(pow(ys[0],2)+pow(ys[1],2));
                theta = atan2(ys[1],ys[0]); 
            } else {
                s = ys[0];
                theta = ys[1];
            }  

            stz(0, 0) = s;
            stz(0, 1) = theta;
            stz(0, 2) = ys[2];

            field->set_points(stz);
            auto psi0 = field->psi0;
            double modB = field->modB_ref()(0);
            double G = field->G_ref()(0);
            double I = field->I_ref()(0);
            double dGdpsi = field->dGds_ref()(0)/psi0;
            double dIdpsi = field->dIds_ref()(0)/psi0;
            double iota = field->iota_ref()(0);
            double diotadpsi = field->diotads_ref()(0)/psi0;
            double dmodBdpsi = field->modB_derivs_ref()(0)/psi0;
            double dmodBdtheta = field->modB_derivs_ref()(1);
            double dmodBdzeta = field->modB_derivs_ref()(2);
            double v_perp2 = 2*mu*modB;
            double fak1 = m*v_par*v_par/modB + m*mu;
            double Phi = Phihat * sin(Phim * theta - Phin * ys[2] + omega * time + phase);
            double dPhidpsi = 0;
            double Phidot = Phihat * omega * cos(Phim * theta - Phin * ys[2] + omega * time + phase);
            double dPhidtheta = Phidot * Phim / omega;
            double dPhidzeta = - Phidot * Phin / omega;
            double alpha = - Phi * (iota*Phim - Phin)/(omega*(G+iota*I));
            double alphadot = - Phidot * (iota*Phim - Phin)/(omega*(G+iota*I));
            double dalphadtheta = - dPhidtheta * (iota*Phim - Phin)/(omega*(G+iota*I));
            double dalphadzeta = -dPhidzeta * (iota*Phim - Phin)/(omega*(G+iota*I));
            double dalphadpsi = - dPhidpsi * (iota*Phim - Phin)/(omega*(G+iota*I)) \
                - (Phi/omega) * (diotadpsi*Phim/(G+iota*I) \
                - (iota*Phim - Phin)/((G+iota*I)*(G+iota*I)) * (dGdpsi + diotadpsi*I + iota*dIdpsi));
            double denom = q*(G + I*(-alpha*dGdpsi + iota) + alpha*G*dIdpsi) + m*v_par/modB * (-dGdpsi*I + G*dIdpsi); // q G in vacuum

            double sdot = (-G*dPhidtheta*q + I*dPhidzeta*q + modB*q*v_par*(dalphadtheta*G-dalphadzeta*I) + (-dmodBdtheta*G + dmodBdzeta*I)*fak1)/(denom*psi0);
            double tdot = (G*q*dPhidpsi + modB*q*v_par*(-dalphadpsi*G - alpha*dGdpsi + iota) - dGdpsi*m*v_par*v_par \
                      + dmodBdpsi*G*fak1)/denom;
            if (axis==1) {
                dydt[0] = sdot*cos(theta)/(2*sqrt(s)) - sqrt(s) * sin(theta) * tdot;
                dydt[1] = sdot*sin(theta)/(2*sqrt(s)) + sqrt(s) * cos(theta) * tdot;
            } else if (axis==2) {
                dydt[0] = sdot*cos(theta) - s * sin(theta) * tdot;
                dydt[1] = sdot*sin(theta) + s * cos(theta) * tdot; 
            } else {
                dydt[0] = sdot;
                dydt[1] = tdot;
            }
            dydt[2] = (-I*(dmodBdpsi*m*mu + dPhidpsi*q) + modB*q*v_par*(1 + dalphadpsi*I + alpha*dIdpsi) \
                      + m*v_par*v_par/modB * (modB*dIdpsi - dmodBdpsi*I))/denom;
            dydt[3] = (modB*q/m * ( -m*mu * (dmodBdzeta*(1 + dalphadpsi*I + alpha*dIdpsi) \
                      + dmodBdpsi*(dalphadtheta*G - dalphadzeta*I) + dmodBdtheta*(iota - alpha*dGdpsi - dalphadpsi*G)) \
                      - q*(alphadot*(G + I*(iota - alpha*dGdpsi) + alpha*G*dIdpsi) \
                      + (dalphadtheta*G - dalphadzeta*I)*dPhidpsi \
                      + (iota - alpha*dGdpsi - dalphadpsi*G)*dPhidtheta \
                      + (1 + alpha*dIdpsi + dalphadpsi*I)*dPhidzeta)) \
                      + q*v_par/modB * ((dmodBdtheta*G - dmodBdzeta*I)*dPhidpsi \
                      + dmodBdpsi*(I*dPhidzeta - G*dPhidtheta)) \
                      + v_par*(m*mu*(dmodBdtheta*dGdpsi - dmodBdzeta*dIdpsi) \
                      + q*(alphadot*(dGdpsi*I-G*dIdpsi) + dGdpsi*dPhidtheta - dIdpsi*dPhidzeta)))/denom;
            dydt[4] = 1;
        }
};

template<template<class, std::size_t, xt::layout_type> class T>
class GuidingCenterNoKBoozerRHS {
    /*
     * The state consists of :math:`[s, t, z, v_par]` with
     *
     *  \dot s = (I |B|_{,\zeta} - G |B|_{,\theta})m(v_{||}^2/|B| + \mu)/(\iota D \psi_0)
     *  \dot \theta = (G |B|_{,\psi} m(v_{||}^2/|B| + \mu) - (-q \iota + m v_{||} G' / |B|) v_{||} |B|)/(\iota D)
     *  \dot \zeta = \left((q + m v_{||} I'/|B|) v_{||} |B| - |B|_{,\psi} m(\rho_{||}^2 |B| + \mu) I\right)/(\iota D)
     *  \dot v_{||} = ((-q\iota + m v_{||} G'/|B|)|B|_{,\theta} - (q + m v_{||}I'/|B|)|B|_{,\zeta})\mu |B|/(\iota D)
     *  D = ((q + m v_{||} I'/|B|)*G - (-q \iota + m v_{||} G'/|B|) I)/\iota
     *
     *  where primes indicate differentiation wrt :math:`\psi`, :math:`q` is the charge,
     *  :math:`m` is the mass, and :math:`v_\perp = 2\mu|B|`. This corresponds
     *  with the limit K = 0.
     */
    private:
        typename BoozerMagneticField<T>::Tensor2 stz = xt::zeros<double>({1, 3});
        shared_ptr<BoozerMagneticField<T>> field;
        double m, q, mu;
    public:
        int axis;
        static constexpr int Size = 4;
        using State = std::array<double, Size>;


        GuidingCenterNoKBoozerRHS(shared_ptr<BoozerMagneticField<T>> field, double m, double q, double mu, int axis)
            : field(field), m(m), q(q), mu(mu), axis(axis) {
            }

        void operator()(const State &ys, array<double, 4> &dydt,
                const double t) {
            double v_par = ys[3];
            double s, theta;
            if (axis==1) {
                s = pow(ys[0],2)+pow(ys[1],2);
                theta = atan2(ys[1],ys[0]);          
            } else if (axis==2) {
                s = sqrt(pow(ys[0],2)+pow(ys[1],2));
                theta = atan2(ys[1],ys[0]); 
            } else {
                s = ys[0];
                theta = ys[1];
            }  
            stz(0, 0) = s;
            stz(0, 1) = theta;
            stz(0, 2) = ys[2];
            field->set_points(stz);
            auto psi0 = field->psi0;
            double modB = field->modB_ref()(0);
            double G = field->G_ref()(0);
            double I = field->I_ref()(0);
            double dGdpsi = field->dGds_ref()(0)/psi0;
            double dIdpsi = field->dIds_ref()(0)/psi0;
            double iota = field->iota_ref()(0);
            double dmodBdpsi = field->modB_derivs_ref()(0)/psi0;
            double dmodBdtheta = field->modB_derivs_ref()(1);
            double dmodBdzeta = field->modB_derivs_ref()(2);
            double v_perp2 = 2*mu*modB;
            double fak1 = m*v_par*v_par/modB + m*mu;
            double D = ((q + m*v_par*dIdpsi/modB)*G - (-q*iota + m*v_par*dGdpsi/modB)*I)/iota;
            double F = (q + m*v_par*dIdpsi/modB);
            double C = (-q*iota + m*v_par*dGdpsi/modB);

            double sdot = (I*dmodBdzeta - G*dmodBdtheta)*fak1/(D*iota*psi0);
            double tdot = (G*dmodBdpsi*fak1 - (-q*iota + m*v_par*dGdpsi/modB)*v_par*modB)/(D*iota);
            if (axis==1) {
                dydt[0] = sdot*cos(theta)/(2*sqrt(s)) - sqrt(s) * sin(theta) * tdot;
                dydt[1] = sdot*sin(theta)/(2*sqrt(s)) + sqrt(s) * cos(theta) * tdot;
            } else if (axis==2) {
                dydt[0] = sdot*cos(theta) - s * sin(theta) * tdot;
                dydt[1] = sdot*sin(theta) + s * cos(theta) * tdot; 
            } else {
                dydt[0] = sdot;
                dydt[1] = tdot;
            }
            dydt[2] = ((q + m*v_par*dIdpsi/modB)*v_par*modB - dmodBdpsi*fak1*I)/(D*iota);
            dydt[3] = modB*mu*(dmodBdtheta*C - dmodBdzeta*F)/(F*G-C*I);
            // dydt[3] = - (mu / v_par) * (dmodBdpsi * sdot * psi0 + dmodBdtheta * tdot + dmodBdzeta * dydt[2]);
        }
};

template<template<class, std::size_t, xt::layout_type> class T>
class GuidingCenterBoozerRHS {
    /*
     * The state consists of :math:`[s, t, z, v_par]` with
     *
     *  \dot s = (I |B|_{,\zeta} - G |B|_{,\theta})m(v_{||}^2/|B| + \mu)/(\iota D \psi_0)
     *  \dot \theta = ((G |B|_{,\psi} - K |B|_{,\zeta}) m(v_{||}^2/|B| + \mu) - C v_{||} |B|)/(\iota D)
     *  \dot \zeta = (F v_{||} |B| - (|B|_{,\psi} I - |B|_{,\theta} K) m(\rho_{||}^2 |B| + \mu) )/(\iota D)
     *  \dot v_{||} = (C|B|_{,\theta} - F|B|_{,\zeta})\mu |B|/(\iota D)
     *  C = - m v_{||} K_{,\zeta}/|B|  - q \iota + m v_{||}G'/|B|
     *  F = - m v_{||} K_{,\theta}/|B| + q + m v_{||}I'/|B|
     *  D = (F G - C I))/\iota
     *
     *  where primes indicate differentiation wrt :math:`\psi`, :math:`q` is the charge,
     *  :math:`m` is the mass, and :math:`v_\perp = 2\mu|B|`.
     */
    private:
        typename BoozerMagneticField<T>::Tensor2 stz = xt::zeros<double>({1, 3});
        shared_ptr<BoozerMagneticField<T>> field;
        double m, q, mu;
    public:
        static constexpr int Size = 4;
        using State = std::array<double, Size>;
        int axis;

        GuidingCenterBoozerRHS(shared_ptr<BoozerMagneticField<T>> field, double m, double q, double mu, int axis)
            : field(field), m(m), q(q), mu(mu), axis(axis) {
            }

        void operator()(const State &ys, array<double, 4> &dydt,
                const double t) {
            double v_par = ys[3];
            double s, theta;
            if (axis==1) {
                s = pow(ys[0],2)+pow(ys[1],2);
                theta = atan2(ys[1],ys[0]);          
            } else if (axis==2) {
                s = sqrt(pow(ys[0],2)+pow(ys[1],2));
                theta = atan2(ys[1],ys[0]); 
            } else {
                s = ys[0];
                theta = ys[1];
            }  
            stz(0, 0) = s;
            stz(0, 1) = theta;
            stz(0, 2) = ys[2];

            field->set_points(stz);
            auto psi0 = field->psi0;
            double modB = field->modB_ref()(0);
            double K = field->K_ref()(0);
            double dKdtheta = field->K_derivs_ref()(0);
            double dKdzeta = field->K_derivs_ref()(1);

            double G = field->G_ref()(0);
            double I = field->I_ref()(0);
            double dGdpsi = field->dGds_ref()(0)/psi0;
            double dIdpsi = field->dIds_ref()(0)/psi0;
            double iota = field->iota_ref()(0);
            double dmodBdpsi = field->modB_derivs_ref()(0)/psi0;
            double dmodBdtheta = field->modB_derivs_ref()(1);
            double dmodBdzeta = field->modB_derivs_ref()(2);
            double v_perp2 = 2*mu*modB;
            double fak1 = m*v_par*v_par/modB + m*mu; // dHdB
            double C = -m*v_par*(dKdzeta-dGdpsi)/modB - q*iota;
            double F = -m*v_par*(dKdtheta-dIdpsi)/modB + q;
            double D = (F*G-C*I)/iota;

            double sdot = (I*dmodBdzeta - G*dmodBdtheta)*fak1/(D*iota*psi0);
            double tdot = (G*dmodBdpsi*fak1 - C*v_par*modB - K*fak1*dmodBdzeta)/(D*iota);
            if (axis==1) {
                dydt[0] = sdot*cos(theta)/(2*sqrt(s)) - sqrt(s) * sin(theta) * tdot;
                dydt[1] = sdot*sin(theta)/(2*sqrt(s)) + sqrt(s) * cos(theta) * tdot;
            } else if (axis==2) {
                dydt[0] = sdot*cos(theta) - s * sin(theta) * tdot;
                dydt[1] = sdot*sin(theta) + s * cos(theta) * tdot; 
            } else {
                dydt[0] = sdot;
                dydt[1] = tdot;
            }
            dydt[2] = (F*v_par*modB - dmodBdpsi*fak1*I + K*fak1*dmodBdtheta)/(D*iota);
            // dydt[3] = - (mu / v_par) * (dmodBdpsi * sdot * psi0 + dmodBdtheta * tdot + dmodBdzeta * dydt[2]);
            dydt[3] = modB*mu*(dmodBdtheta*C - dmodBdzeta*F)/(F*G-C*I);
        }
};

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
solve(RHS rhs, typename RHS::State y, double tmax, double dt, double dtmax, double abstol, double reltol, vector<double> zetas, vector<double> omegas, vector<shared_ptr<StoppingCriterion>> stopping_criteria, vector<double> vpars, bool zetas_stop=false, bool vpars_stop=false, bool forget_exact_path=false) {
    
    if (zetas.size() > 0 && omegas.size() == 0) {
        omegas.insert(omegas.end(), zetas.size(), 0.);
    } else if (zetas.size() !=  omegas.size()) {
        throw std::invalid_argument("zetas and omegas need to have matching length.");
    }

    vector<array<double, RHS::Size+1>> res = {};
    vector<array<double, RHS::Size+2>> res_hits = {};
    array<double, RHS::Size> ykeep = {};
    typedef typename RHS::State State;
    typedef typename boost::numeric::odeint::result_of::make_dense_output<runge_kutta_dopri5<State>>::type dense_stepper_type;
    dense_stepper_type dense = make_dense_output(abstol, reltol, dtmax, runge_kutta_dopri5<State>());
    double t = 0;
    dense.initialize(y, t, dt);
    int iter = 0;
    bool stop = false;
    double zeta_last;
    double vpar_last = 0;
    double t_last = 0;
    t_last = t;
    zeta_last = y[2];
    vpar_last = y[3];
    double zeta_current, vpar_current, t_current;
    State temp;
    do {
        if (!forget_exact_path || t==0) {
            ykeep = y;
            if (rhs.axis==1) {
                ykeep[0] = pow(y[0],2) + pow(y[1],2);
                ykeep[1] = atan2(y[1],y[0]);
            } else if (rhs.axis==2) {
                ykeep[0] = sqrt(pow(y[0],2) + pow(y[1],2));
                ykeep[1] = atan2(y[1],y[0]);
            }
            res.push_back(join<1, RHS::Size>({t}, ykeep));
        }
        tuple<double, double> step = dense.do_step(rhs);
        iter++;
        t = dense.current_time();
        y = dense.current_state();
        zeta_current = y[2];
        vpar_current = y[3];
        double t_last = std::get<0>(step);
        double t_current = std::get<1>(step);
        dt = t_current - t_last;
        stop = check_stopping_criteria(rhs, y, iter, res, res_hits, dense, t_last, t_current, zeta_last, zeta_current, vpar_last, 
                                vpar_current, abstol, zetas, omegas, stopping_criteria, vpars, zetas_stop, vpars_stop, forget_exact_path);
        zeta_last = zeta_current;
        vpar_last = vpar_current;
    } while(t < tmax && !stop);
    if(!stop){
        dense.calc_state(tmax, y);
        ykeep = y;
        if (rhs.axis==1) {
            ykeep[0] = pow(y[0],2) + pow(y[1],2);
            ykeep[1] = atan2(y[1],y[0]);
        } else if (rhs.axis==2) {
            ykeep[0] = sqrt(pow(y[0],2) + pow(y[1],2));
            ykeep[1] = atan2(y[1],y[0]);
        }      
        res.push_back(join<1, RHS::Size>({tmax}, ykeep));
    }
    return std::make_tuple(res, res_hits);
}

template<class RHS, class DENSE>
bool check_stopping_criteria(RHS rhs, typename RHS::State y, int iter, vector<array<double, RHS::Size+1>> &res, vector<array<double, RHS::Size+2>> &res_hits, DENSE dense, double t_last, double t_current, double zeta_last, double zeta_current, double vpar_last, double vpar_current, double abstol, vector<double> zetas, vector<double> omegas, vector<shared_ptr<StoppingCriterion>> stopping_criteria, vector<double> vpars, bool zetas_stop, bool vpars_stop, bool forget_exact_path) {

    typedef typename RHS::State State;
    // abstol?
    boost::math::tools::eps_tolerance<double> roottol(-int(std::log2(abstol)));
    uintmax_t rootmaxit = 200;
    State temp;

    bool stop = false;
    array<double, RHS::Size> ykeep = {};
    double dt = t_current - t_last;

    // Now check whether we have hit any of the vpar planes
    for (int i = 0; i < vpars.size(); ++i) {
        double vpar = vpars[i];
        if(t_last!=0 && (vpar_last-vpar != 0) && (vpar_current-vpar != 0) && (((vpar_last-vpar > 0) ? 1 : ((vpar_last-vpar < 0) ? -1 : 0)) != ((vpar_current-vpar > 0) ? 1 : ((vpar_current-vpar < 0) ? -1 : 0)))){ // check whether vpar = vpars[i] was crossed
            std::function<double(double)> rootfun = [&dense, &temp, &vpar_last, &vpar](double t){
                dense.calc_state(t, temp);
                return temp[3]-vpar;
            };
            auto root = toms748_solve(rootfun, t_last, t_current, vpar_last-vpar, vpar_current-vpar, roottol, rootmaxit);
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
            res_hits.push_back(join<2, RHS::Size>({troot, double(i) + zetas.size()}, ykeep));
            if (vpars_stop) {
                res.push_back(join<1, RHS::Size>({troot}, ykeep));
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
        if(t_last!=0 && (std::floor((phase_last - zeta)/(2*M_PI)) != std::floor((phase_current-zeta)/(2*M_PI)))) { // check whether zeta+k*2pi for some k was crossed
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
                res.push_back(join<1, RHS::Size>({troot}, ykeep));
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
            break;
        }
    }
    return stop;
}

template<template<class, std::size_t, xt::layout_type> class T>
class SymplField
{
    public:
    // Covaraint components of vector potential
    double  Atheta, Azeta;
    // htheta = G/B, hzeta = I/B
    double  htheta, hzeta;
    double  modB;
    double ptheta;

    // Derivatives of above quantities wrt (s, theta, zeta)
    double dAtheta[3], dAzeta[3];
    double dhtheta[3], dhzeta[3];
    double dmodB[3];
 
    // H = vpar^2/2 + mu B
    // vpar = (pzeta - q Azeta)/(m hzeta)
    // pzeta = m vpar * hzeta + q Azeta
    double H, pth, vpar;
    double dvpar[4], dH[4], dptheta[4];

    // mu = vperp^2/(2 B)
    // q = charge, m = mass
    double mu, q, m;

    shared_ptr<BoozerMagneticField<T>> field;
    typename BoozerMagneticField<T>::Tensor2 stz = xt::zeros<double>({1, 3});

    static constexpr int Size = 4;
    using State = std::array<double, Size>;
    static constexpr bool axis = false;

    SymplField(shared_ptr<BoozerMagneticField<T>> field, double m, double q, double mu) :
        field(field), m(m), q(q), mu(mu) {
    }

    //
    // Evaluates magnetic field in Boozer canonical coordinates (r, theta, zeta)
    // and stores results in the SymplField object
    //
    void eval_field(double s, double theta, double zeta)
    {
        double Btheta, Bzeta, dBtheta, dBzeta, modB2;

        stz[0, 0] = s; stz[0, 1] = theta; stz[0, 2] = zeta;
        field->set_points(stz);

        // A = psi \nabla \theta - psip \nabla \zeta
        Atheta = s*field->psi0;
        Azeta =  -field->psip()(0);
        dAtheta[0] = field->psi0; // dAthetads
        dAzeta[0] = -field->iota()(0)*field->psi0; // dAzetads
        for (int i=1; i<3; i++)
        {
            dAtheta[i] = 0.0;
            dAzeta[i] = 0.0;
        }

        modB = field->modB()(0);
        dmodB[0] = field->modB_derivs_ref()(0);
        dmodB[1] = field->modB_derivs_ref()(1);
        dmodB[2] = field->modB_derivs_ref()(2);

        Btheta = field->I()(0);
        Bzeta = field->G()(0);
        dBtheta = field->dIds()(0);
        dBzeta = field->dGds()(0);

        modB2 = pow(modB, 2);

        htheta = Btheta/modB;
        hzeta = Bzeta/modB;
        dhtheta[0] = dBtheta/modB - Btheta*dmodB[0]/modB2;
        dhzeta[0] = dBzeta/modB - Bzeta*dmodB[0]/modB2;

        for (int i=1; i<3; i++)
        {
            dhtheta[i] = -Btheta*dmodB[i]/modB2;
            dhzeta[i] = -Bzeta*dmodB[i]/modB2;
        }

    }

    // compute pzeta for given vpar
    double get_pzeta(double vpar)
    {
        return vpar*hzeta*m + q*Azeta; // q*psi0
    }

    // computes values of H, ptheta and vpar at z=(s, theta, zeta, pzeta)
    void get_val(double pzeta)
    {
        vpar = (pzeta - q*Azeta)/(hzeta*m);
        H = m*pow(vpar,2)/2.0 + m*mu*modB;
        ptheta = m*htheta*vpar + q*Atheta;
    }

    // computes H, ptheta and vpar at z=(s, theta, zeta, pzeta) and their derivatives
    void get_derivatives(double pzeta)
    {
        get_val(pzeta);

        for (int i=0; i<3; i++)
            dvpar[i] = -q*dAzeta[i]/(hzeta*m) - (vpar/hzeta)*dhzeta[i];

        dvpar[3]   = 1.0/(hzeta*m); // dvpardpzeta

        for (int i=0; i<3; i++)
            dH[i] = m*vpar*dvpar[i] + m*mu*dmodB[i];
        dH[3]   = m*vpar*dvpar[3]; // dHdpzeta

        for (int i=0; i<3; i++)
            dptheta[i] = m*dvpar[i]*htheta + m*vpar*dhtheta[i] + q*dAtheta[i];

        dptheta[3] = m*htheta*dvpar[3]; // dpthetadpzeta
    }

    double get_dsdt()
    {
        return (-dH[1] + dptheta[3]*dH[2] - dptheta[2]*dH[3])/dptheta[0];
    }

    double get_dthdt()
    {
        return dH[0]/dptheta[0];
    }

    double get_dzedt()
    {
        return (vpar - dH[0]/dptheta[0]*htheta)/hzeta;
    }

    double get_dvpardt()
    {
        double dsdt = (-dH[1] + dptheta[3]*dH[2] - dptheta[2]*dH[3])/dptheta[0];
        double dthdt = dH[0]/dptheta[0];
        double dzdt = (vpar - dH[0]/dptheta[0]*htheta)/hzeta;
        double dpzdt = (-dH[2] + dH[0]*dptheta[2]/dptheta[0]);

        return dvpar[0] * dsdt + dvpar[1] * dthdt + dvpar[2] * dzdt + dvpar[3] * dpzdt;
    }
};

template<template<class, std::size_t, xt::layout_type> class T>
struct f_quasi_params{
    double ptheta_old;
    double dt;
    array<double, 4> z;
    SymplField<T> f;
};

template<template<class, std::size_t, xt::layout_type> class T>
int f_euler_quasi_func(const gsl_vector* x, void* p, gsl_vector* f)
{
    struct f_quasi_params<T> * params = (struct f_quasi_params<T> *)p;
    const double ptheta_old = (params->ptheta_old);
    const double dt = (params->dt);
    auto z = (params->z);
    SymplField<T> field = (params->f);
    
    const double x0 = gsl_vector_get(x,0);
    const double x1 = gsl_vector_get(x,1);

    field.eval_field(x0, z[1], z[2]);
    field.get_derivatives(x1);

    // const double f0 = (field.dptheta[0]*(field.ptheta - ptheta_old)
    //       + dt*(field.dH[1]*field.dptheta[0] - field.dH[0]*field.dptheta[1])); // corresponds with (2.6) in JPP 2020
    // const double f1  = (field.dptheta[0]*(x1 - z[3])
    //       + dt*(field.dH[2]*field.dptheta[0] - field.dH[0]*field.dptheta[2])); // corresponds with (2.7) in JPP 2020

    const double f0 = (field.dptheta[0]*(field.ptheta - ptheta_old)
          + dt*(field.dH[1]*field.dptheta[0] - field.dH[0]*field.dptheta[1]))/pow(field.field->psi0 * field.q, 2); // corresponds with (2.6) in JPP 2020
    const double f1  = (field.dptheta[0]*(x1 - z[3])
          + dt*(field.dH[2]*field.dptheta[0] - field.dH[0]*field.dptheta[2]))/pow(field.field->psi0 * field.q, 2); // corresponds with (2.7) in JPP 2020

    gsl_vector_set(f, 0, f0);
    gsl_vector_set(f, 1, f1);

    return GSL_SUCCESS;
}

template<template<class, std::size_t, xt::layout_type> class T>
int f_euler_quasi_exact(const gsl_vector* x, void* p, gsl_vector* f)
{
    struct f_quasi_params<T> * params = (struct f_quasi_params<T> *)p;
    const double ptheta_old = (params->ptheta_old);
    auto z = (params->z);
    SymplField<T> field = (params->f);
    
    const double x0 = gsl_vector_get(x,0);

    field.eval_field(x0, z[1], z[2]);
    field.get_derivatives(z[3]);

    const double f0 = field.ptheta - ptheta_old;

    gsl_vector_set(f, 0, f0);

    return GSL_SUCCESS;
}

template<template<class, std::size_t, xt::layout_type> class T>
int f_midpoint_quasi_func(const gsl_vector* x, void* p, gsl_vector* f)
{
    struct f_quasi_params<T> * params = (struct f_quasi_params<T> *)p;
    const double ptheta_old = (params->ptheta_old);
    const double dt = (params->dt);
    auto z = (params->z);
    SymplField<T> field = (params->f);
    
    // s, theta, zeta, pzeta
    const double x0 = gsl_vector_get(x,0);
    const double x1 = gsl_vector_get(x,1);
    const double x2 = gsl_vector_get(x,2);
    const double x3 = gsl_vector_get(x,3);
    // s_m
    const double x4 = gsl_vector_get(x,4);

    field.eval_field(x4, 0.5*(x1 + z[1]), 0.5*(x2 + z[2]));
    field.get_derivatives(0.5*(x3 + z[3]));

    const double f1 = field.dptheta[0]*(x1 - z[1]) - dt*field.H[0]; // theta
    const double f2 = field.dptheta[0]*(x2 - z[2])*field.hzeta 
            - dt*(field.dptheta[0]*field.vpar - field.dH[0]*field.htheta); // zeta
    const double f3 = field.dptheta[0]*(x3 - z[3]) 
            + dt*(field.dH[2]*field.dptheta[0] - field.dH[0]*field.dptheta[2]); // pzeta
    const double f4 = field.dptheta[0]*(field.ptheta - ptheta_old)
          + 0.5*dt*(field.dH[1]*field.dptheta[0] - field.dH[0]*field.dptheta[1]); // ptheta

    double dptheta_m = field.dptheta[0];
    double dpthdt = field.dH[1]*field.dptheta[0] - field.dH[0]*field.dptheta[1];

    field.eval_field(x0, x1, x2);
    field.get_derivatives(x3);
    const double f0 = dptheta_m*(field.ptheta - ptheta_old) + dt*dpthdt;

    gsl_vector_set(f, 0, f0);
    gsl_vector_set(f, 1, f1);
    gsl_vector_set(f, 2, f2);
    gsl_vector_set(f, 3, f3);
    gsl_vector_set(f, 4, f4);

    return GSL_SUCCESS;
}

double cubic_hermite_interp(double t_last, double t_current, double y_last, double y_current, double dy_last, double dy_current, double t)
{
    double dt = t_current - t_last;
    return (3*dt*pow(t-t_last,2) - 2*pow(t-t_last,3))/pow(dt,3) * y_current 
            + (pow(dt,3)-3*dt*pow(t-t_last,2)+2*pow(t-t_last,3))/pow(dt,3) * y_last
            + pow(t-t_last,2)*(t-t_current)/pow(dt,2) * dy_current
            + (t-t_last)*pow(t-t_current,2)/pow(dt,2) * dy_last;
}

template<template<class, std::size_t, xt::layout_type> class T>
struct sympl_dense {
    // for interpolation
    array<double, 2> bracket_s = {}; 
    array<double, 2> bracket_dsdt = {};
    array<double, 2> bracket_theta = {}; 
    array<double, 2> bracket_dthdt = {}; 
    array<double, 2> bracket_zeta = {}; 
    array<double, 2> bracket_dzedt = {};
    array<double, 2> bracket_vpar = {}; 
    array<double, 2> bracket_dvpardt = {}; 
    typedef typename SymplField<T>::State State;

    double tlast = 0.0;
    double tcurrent = 0.0;

    void update(double t, double dt, array<double, 4>  y, SymplField<T> f) {
        tlast = t;
        tcurrent = t+dt;

        bracket_s[0] = bracket_s[1];
        bracket_theta[0] = bracket_theta[1];
        bracket_zeta[0] = bracket_zeta[1];
        bracket_vpar[0] = bracket_vpar[1];

        bracket_dsdt[0] =  bracket_dsdt[1];
        bracket_dthdt[0] = bracket_dthdt[1];
        bracket_dzedt[0] = bracket_dzedt[1];
        bracket_dvpardt[0] = bracket_dvpardt[1];

        bracket_s[1] = y[0];
        bracket_theta[1] = y[1];
        bracket_zeta[1] = y[2];
        bracket_vpar[1] = y[3];

        bracket_dsdt[1] = f.get_dsdt();
        bracket_dthdt[1] = f.get_dthdt();
        bracket_dzedt[1] = f.get_dzedt();
        bracket_dvpardt[1] = f.get_dvpardt();
    }

    void calc_state(double eval_t, State &temp) {
        temp[0] = cubic_hermite_interp(tlast, tcurrent, bracket_s[0], bracket_s[1], bracket_dsdt[0], bracket_dsdt[1], eval_t);
        temp[1] = cubic_hermite_interp(tlast, tcurrent, bracket_theta[0], bracket_theta[1], bracket_dthdt[0], bracket_dthdt[1], eval_t);
        temp[2] = cubic_hermite_interp(tlast, tcurrent, bracket_zeta[0], bracket_zeta[1], bracket_dzedt[0], bracket_dzedt[1], eval_t);
        temp[3] = cubic_hermite_interp(tlast, tcurrent, bracket_vpar[0], bracket_vpar[1], bracket_dvpardt[0], bracket_dvpardt[1], eval_t);
    }
};

// see https://github.com/itpplasma/SIMPLE/blob/master/SRC/
//         orbit_symplectic_quasi.f90:timestep_euler1_quasi
template<template<class, std::size_t, xt::layout_type> class T>
tuple<vector<array<double, SymplField<T>::Size+1>>, vector<array<double, SymplField<T>::Size+2>>>
solve_sympl(SymplField<T> f, typename SymplField<T>::State y, double tmax, double dt, double roottol, vector<double> zetas, vector<double> omegas, vector<shared_ptr<StoppingCriterion>> stopping_criteria, vector<double> vpars, bool zetas_stop=false, bool vpars_stop=false, bool forget_exact_path = false, bool predictor_step = true)
{
    double abstol = 0;
    if (zetas.size() > 0 && omegas.size() == 0) {
        omegas.insert(omegas.end(), zetas.size(), 0.);
    } else if (zetas.size() !=  omegas.size()) {
        throw std::invalid_argument("zetas and omegas need to have matching length.");
    }

    typedef typename SymplField<T>::State State;
    vector<array<double, SymplField<T>::Size+1>> res = {};
    vector<array<double, SymplField<T>::Size+2>> res_hits = {};
    double t = 0.0;
    bool stop = false;

    State  z = {}; // s, theta, zeta, pzeta
    // y = [s, theta, zeta, vpar]
        
    // Translate y to z
    // y = [s, theta, zeta, vpar]
    // z = [s, theta, zeta, pzeta]
    // pzeta = m*vpar*hzeta + q*Azeta
    z[0] = y[0];
    z[1] = y[1];
    z[2] = y[2];
    f.eval_field(z[0], z[1], z[2]);
    z[3] = f.get_pzeta(y[3]);
    f.get_derivatives(z[3]);
    double ptheta_old = f.ptheta;

    double t_last = t;
    double zeta_last = y[2];
    double vpar_last = y[3];

    // for interpolation
    sympl_dense<T> dense;
    dense.update(t, dt, y, f);

    // set up root solvers
    const gsl_multiroot_fsolver_type * Newt = gsl_multiroot_fsolver_hybrids;
    gsl_multiroot_fsolver *s_euler;
    s_euler = gsl_multiroot_fsolver_alloc(Newt, 2);

    struct f_quasi_params<T> params = {ptheta_old, dt, z, f};
    gsl_multiroot_function F_euler_quasi = {&f_euler_quasi_func<T>, 2, &params};
    gsl_vector* xvec_quasi = gsl_vector_alloc(2);

    int status;
    int iter = 0;
    double s_guess = z[0];
    double pzeta_guess = z[3];

    // //testing
    // int total_root_iter = 0;
    // auto rhs_class = GuidingCenterNoKBoozerRHS<T>(f.field, f.m, f.q, f.mu, f.axis);
    // double backup_ptheta = ptheta_old;
    // printf("dt = % .3e\n", dt);
    // printf("roottol = % .3e\n", roottol);
    // printf("predictor_step %s\n", predictor_step ? "on" : "off");
    do {

        if (!forget_exact_path || t==0){
            res.push_back(join<1,SymplField<T>::Size>({t}, y));
        }

        params.ptheta_old = ptheta_old;
        params.z = z;
        params.dt = dt;
        gsl_vector_set(xvec_quasi, 0,s_guess);
        gsl_vector_set(xvec_quasi, 1, pzeta_guess);
        gsl_multiroot_fsolver_set(s_euler, &F_euler_quasi, xvec_quasi);

        int root_iter = 0;
        // Solve implicit part of time-step with some quasi-Newton
        // applied to f_euler1_quasi. This corresponds with (2.6)-(2.7) in JPP 2020,
        // which are solved for x = [s, pzeta].
        do
          {
            root_iter++;

            status = gsl_multiroot_fsolver_iterate(s_euler);
            //  printf("iter = %3u x = % .10e % .10e "
            //            "f(x) = % .10e % .10e\n",
            //            iter,
            //            gsl_vector_get (s_euler->x, 0),
            //            gsl_vector_get (s_euler->x, 1),
            //            gsl_vector_get (s_euler->f, 0),
            //            gsl_vector_get (s_euler->f, 1));

            if (status) {  /* check if solver is stuck */
                printf("iter = %3u x = % .10e % .10e "
                        "f(x) = % .10e % .10e\n",
                        root_iter,
                        gsl_vector_get (s_euler->x, 0),
                        gsl_vector_get (s_euler->x, 1),
                        gsl_vector_get (s_euler->f, 0),
                        gsl_vector_get (s_euler->f, 1));
              printf ("status = %s\n", gsl_strerror (status));
              break;
            }
            status = gsl_multiroot_test_residual(s_euler->f, roottol); //tolerance --> roottol ~ 1e-15
          }
        while (status == GSL_CONTINUE && root_iter < 20);
        iter++;

        z[0] = gsl_vector_get(s_euler->x, 0);  // s
        z[3] = gsl_vector_get(s_euler->x, 1);  // pzeta

        // We now evaluate the explicit part of the time-step at [s, pzeta]
        // given by the Euler step.
        f.eval_field(z[0], z[1], z[2]);
        f.get_derivatives(z[3]);

        // z[1] = theta
        // z[2] = zeta
        // dH[0] = dH/dr
        // dptheta[0] = dptheta/dr
        // htheta = G/B
        // hzeta = I/B
        z[1] = z[1] + dt*f.dH[0]/f.dptheta[0]; // (2.9) in JPP 2020
        z[2] = z[2] + dt*(f.vpar - f.dH[0]/f.dptheta[0]*f.htheta)/f.hzeta; // (2.10) in JPP 2020

        // Translate z back to y
        // y = [s, theta, zeta, vpar]
        // z = [s, theta, zeta, pzeta]
        // pzeta = m*vpar*hzeta + q*Azeta
        f.eval_field(z[0], z[1], z[2]);
        f.get_derivatives(z[3]);
        y[0] = z[0];
        y[1] = z[1];
        y[2] = z[2];
        y[3] = f.vpar;
        ptheta_old = f.ptheta;

        dense.update(t, dt, y, f); // tlast = t; tcurrent = t+dt;

        // // testing derivatives
        // array<double, 4> dydt;
        // rhs_class(y, dydt, t);
        // assert(f.get_dsdt()==dydt[0]);
        // assert(f.get_dthdt()==dydt[1]);
        // assert(f.get_dzedt()==dydt[2]);
        // assert(f.get_dvpardt()==dydt[3]);
                
        // // predictor step test
        // total_root_iter += root_iter;
        if (predictor_step) {
            s_guess = z[0] + dt*(-f.dH[1] + f.dptheta[3]*f.dH[2] - f.dptheta[2]*f.dH[3])/f.dptheta[0];
            pzeta_guess = z[3] + dt*(- f.dH[2] + f.dH[0]*f.dptheta[2]/f.dptheta[0]); // corresponds with (2.7s) in JPP 2020
        } else {
            s_guess = z[0];
            pzeta_guess = z[3];
        }

        t += dt;

        double t_current = t;
        double zeta_current = y[2];
        double vpar_current = y[3];

        stop = check_stopping_criteria(f, y, iter, res, res_hits, dense, t_last, t_current, zeta_last, zeta_current, vpar_last, 
                                vpar_current, abstol, zetas, omegas, stopping_criteria, vpars, zetas_stop, vpars_stop, forget_exact_path);

        t_last = t_current;
        zeta_last = zeta_current;
        vpar_last = vpar_current;
    } while(t < tmax && !stop);
    if(!stop){
        dense.calc_state(tmax, y);
        res.push_back(join<1,SymplField<T>::Size>({tmax}, y));
    }
    gsl_multiroot_fsolver_free (s_euler);
    gsl_vector_free(xvec_quasi);

    // // test
    // printf("total_root_iter = %7u\n",
    //                        total_root_iter);
    // std::cout << "=======" << std::flush;
    return std::make_tuple(res, res_hits);
}

template<template<class, std::size_t, xt::layout_type> class T>
tuple<vector<array<double, 6>>, vector<array<double, 7>>>
particle_guiding_center_boozer_perturbed_tracing(
        shared_ptr<BoozerMagneticField<T>> field, array<double, 3> stz_init,
        double m, double q, double vtotal, double vtang, double mu, double tmax, double abstol, double reltol,
        bool vacuum, bool noK, vector<double> zetas, vector<double> omegas,
        vector<shared_ptr<StoppingCriterion>> stopping_criteria,
        bool zetas_stop, bool vpars_stop, double Phihat, double omega, int Phim,
        int Phin, double phase, bool forget_exact_path, int axis, vector<double> vpars)
{
    typename BoozerMagneticField<T>::Tensor2 stz({{stz_init[0], stz_init[1], stz_init[2]}});
    field->set_points(stz);
    double modB = field->modB()(0);
    array<double, 5> y;
    double G0 = std::abs(field->G()(0));
    double r0 = G0/modB;
    double dtmax = r0*0.5*M_PI/vtotal; // can at most do quarter of a revolution per step
    double dt = 1e-3 * dtmax; // initial guess for first timestep, will be adjusted by adaptive timestepper
    
    if (axis==1) {
      y = {sqrt(stz_init[0]) * cos(stz_init[1]), sqrt(stz_init[0]) * sin(stz_init[1]), stz_init[2], vtang, 0};
    } else if (axis==2) {
      y = {stz_init[0] * cos(stz_init[1]), stz_init[0] * sin(stz_init[1]), stz_init[2], vtang, 0};
    } else {
      y = {stz_init[0], stz_init[1], stz_init[2], vtang, 0};
    }
    if (vacuum) {
      auto rhs_class = GuidingCenterVacuumBoozerPerturbedRHS<T>(field, m, q, mu, Phihat, omega,
        Phim, Phin, phase, axis);
      return solve(rhs_class, y, tmax, dt, dtmax, abstol, reltol, zetas, omegas, stopping_criteria, vpars, zetas_stop, vpars_stop, forget_exact_path);
  } else {
      auto rhs_class = GuidingCenterNoKBoozerPerturbedRHS<T>(field, m, q, mu, Phihat, omega,
        Phim, Phin, phase, axis);
      return solve(rhs_class, y, tmax, dt, dtmax, abstol, reltol, zetas, omegas, stopping_criteria, vpars, zetas_stop, vpars_stop, forget_exact_path);
  }
}

template<template<class, std::size_t, xt::layout_type> class T>
tuple<vector<array<double, 5>>, vector<array<double, 6>>>
particle_guiding_center_boozer_tracing(
        shared_ptr<BoozerMagneticField<T>> field, array<double, 3> stz_init,
        double m, double q, double vtotal, double vtang, double tmax, double dt, double abstol, double reltol, double roottol,
        bool vacuum, bool noK, bool solveSympl, vector<double> zetas, vector<double> omegas,
        vector<shared_ptr<StoppingCriterion>> stopping_criteria, bool forget_exact_path, int axis, bool predictor_step,
        bool zetas_stop, bool vpars_stop, vector<double> vpars)
{
    typename BoozerMagneticField<T>::Tensor2 stz({{stz_init[0], stz_init[1], stz_init[2]}});
    field->set_points(stz);
    double modB = field->modB()(0);
    double vperp2 = vtotal*vtotal - vtang*vtang;
    double mu = vperp2/(2*modB);

    array<double, 4> y;

    double G0 = std::abs(field->G()(0));
    double r0 = G0/modB;
    double dtmax = r0*0.5*M_PI/vtotal; // can at most do quarter of a revolution per step

    if (!solveSympl){
        dt = 1e-3 * dtmax; // initial guess for first timestep, will be adjusted by adaptive timestepper
    }
    if (dt<0) {
        throw std::invalid_argument("dt needs to be positive.");
    }
    
    if (axis==1) {
      y = {sqrt(stz_init[0]) * cos(stz_init[1]), sqrt(stz_init[0]) * sin(stz_init[1]), stz_init[2], vtang};
    } else if (axis==2) {
      y = {stz_init[0] * cos(stz_init[1]), stz_init[0] * sin(stz_init[1]), stz_init[2], vtang};
    } else {
      y = {stz_init[0], stz_init[1], stz_init[2], vtang};
    }

    if (solveSympl) {
        // if (vacuum) {
        //     auto f = SymplField<T>(field, m, q, mu);
        //     return solve_sympl(f, y, tmax, dt, roottol, zetas, omegas, stopping_criteria,
        //     vpars, phis_stop, vpars_stop, forget_exact_path, predictor_step);
        // } else if (noK) {
        //     auto f = SymplField<T>(field, m, q, mu);
        //     return solve_sympl(f, y, tmax, dt, roottol, zetas, omegas, stopping_criteria,
        //     vpars, phis_stop, vpars_stop, forget_exact_path, predictor_step);
        // } else {
        // auto f = SymplField<T>(field, m, q, mu);
        //     return solve_sympl(f, y, tmax, dt, roottol, zetas, omegas, stopping_criteria,
        //     vpars, phis_stop, vpars_stop, forget_exact_path, predictor_step);
        // }
        auto f = SymplField<T>(field, m, q, mu);
        return solve_sympl(f, y, tmax, dt, roottol, zetas, omegas, stopping_criteria,
        vpars, zetas_stop, vpars_stop, forget_exact_path, predictor_step);
    } else {
        if (vacuum) {
          auto rhs_class = GuidingCenterVacuumBoozerRHS<T>(field, m, q, mu, axis);
          return solve(rhs_class, y, tmax, dt, dtmax, abstol, reltol, zetas, omegas, stopping_criteria, vpars, zetas_stop, vpars_stop, forget_exact_path);
        } else if (noK) {
          auto rhs_class = GuidingCenterNoKBoozerRHS<T>(field, m, q, mu, axis);
          return solve(rhs_class, y, tmax, dt, dtmax, abstol, reltol, zetas, omegas, stopping_criteria, vpars, zetas_stop, vpars_stop, forget_exact_path);
        } else {
          auto rhs_class = GuidingCenterBoozerRHS<T>(field, m, q, mu, axis);
          return solve(rhs_class, y, tmax, dt, dtmax, abstol, reltol, zetas, omegas, stopping_criteria, vpars, zetas_stop, vpars_stop, forget_exact_path);
        } 
    }
}

template
tuple<vector<array<double, 6>>, vector<array<double, 7>>>
particle_guiding_center_boozer_perturbed_tracing<xt::pytensor>(
        shared_ptr<BoozerMagneticField<xt::pytensor>> field, array<double, 3> stz_init,
        double m, double q, double vtotal, double vtang, double mu, double tmax, double abstol, double reltol,
        bool vacuum, bool noK, vector<double> zetas, vector<double> omegas,
        vector<shared_ptr<StoppingCriterion>> stopping_criteria,
        bool zetas_stop, bool vpars_stop, double Phihat,
        double omega, int Phim, int Phin, double phase, bool forget_exact_path, int axis, vector<double> vpars);

template
tuple<vector<array<double, 5>>, vector<array<double, 6>>>
particle_guiding_center_boozer_tracing<xt::pytensor>(
        shared_ptr<BoozerMagneticField<xt::pytensor>> field, array<double, 3> stz_init, double m, double q, double vtotal, double vtang, double tmax, double dt, double abstol, double reltol, double roottol, bool vacuum, bool noK, bool solveSympl, vector<double> zetas, vector<double> omegas, vector<shared_ptr<StoppingCriterion>> stopping_criteria, bool forget_exact_path, int axis, bool predictor_step, bool zetas_stop, bool vpars_stop, vector<double> vpars);