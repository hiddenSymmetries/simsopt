#include "tracing_helpers.h"
#include "boozermagneticfield.h"
#include "tracing.h"
#ifdef USE_GSL
    #include "symplectic.h"
#endif

#include <memory>
#include <vector>
#include <functional>
#include <cassert>
#include <stdexcept>
#include <iomanip>
#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint.hpp>

using std::shared_ptr;
using std::tuple;
using std::function;
using std::array;

using boost::math::tools::toms748_solve;
using namespace boost::numeric::odeint;
using Array2 = BoozerMagneticField::Array2;

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
        Array2 stz = xt::zeros<double>({1, 3});
        shared_ptr<BoozerMagneticField> field;
        double m, q, mu;
    public:
        int axis;
        static constexpr int Size = 4;
        using State = array<double, Size>;

        GuidingCenterVacuumBoozerRHS(shared_ptr<BoozerMagneticField> field, double m, double q, double mu, int axis)
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
        Array2 stz = xt::zeros<double>({1, 3});
        shared_ptr<BoozerMagneticField> field;
        double m, q, mu, Phihat, omega, phase;
        int Phim, Phin;
    public:
        int axis;
        static constexpr int Size = 5;
        using State = array<double, Size>;

        GuidingCenterVacuumBoozerPerturbedRHS(shared_ptr<BoozerMagneticField> field,
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
        Array2 stz = xt::zeros<double>({1, 3});
        shared_ptr<BoozerMagneticField> field;
        double m, q, mu, Phihat, omega, phase;
        int Phim, Phin;
    public:
        int axis;
        static constexpr int Size = 5;
        using State = array<double, Size>;

        GuidingCenterNoKBoozerPerturbedRHS(shared_ptr<BoozerMagneticField> field,
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
        Array2 stz = xt::zeros<double>({1, 3});
        shared_ptr<BoozerMagneticField> field;
        double m, q, mu;
    public:
        int axis;
        static constexpr int Size = 4;
        using State = array<double, Size>;


        GuidingCenterNoKBoozerRHS(shared_ptr<BoozerMagneticField> field, double m, double q, double mu, int axis)
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
        Array2 stz = xt::zeros<double>({1, 3});
        shared_ptr<BoozerMagneticField> field;
        double m, q, mu;
    public:
        static constexpr int Size = 4;
        using State = array<double, Size>;
        int axis;

        GuidingCenterBoozerRHS(shared_ptr<BoozerMagneticField> field, double m, double q, double mu, int axis)
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

template<class RHS>
tuple<vector<array<double, RHS::Size+1>>, vector<array<double, RHS::Size+2>>>
solve(RHS rhs, typename RHS::State y, double tmax, double dt, double dtmax, double abstol, double reltol, vector<double> zetas, vector<double> omegas, vector<shared_ptr<StoppingCriterion>> stopping_criteria, double dt_save, vector<double> vpars, bool zetas_stop=false, bool vpars_stop=false, bool forget_exact_path=false) {
    
    if (zetas.size() > 0 && omegas.size() == 0) {
        omegas.insert(omegas.end(), zetas.size(), 0.);
    } else if (zetas.size() !=  omegas.size()) {
        throw std::invalid_argument("zetas and omegas need to have matching length.");
    }

    vector<array<double, RHS::Size+1>> res = {};
    vector<array<double, RHS::Size+2>> res_hits = {};
    typedef typename RHS::State State;
    State temp;
    State ykeep;
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

    // Save initial state
    ykeep = y;
    if (rhs.axis==1) {
        ykeep[0] = pow(y[0],2) + pow(y[1],2);
        ykeep[1] = atan2(y[1],y[0]);
    } else if (rhs.axis==2) {
        ykeep[0] = sqrt(pow(y[0],2) + pow(y[1],2));
        ykeep[1] = atan2(y[1],y[0]);
    }
    res.push_back(join<1, RHS::Size>({0}, ykeep));

    double zeta_current, vpar_current, t_current;
    do {
        tuple<double, double> step = dense.do_step(rhs);
        iter++;
        t = dense.current_time();
        y = dense.current_state();
        zeta_current = y[2];
        vpar_current = y[3];
        double t_last = std::get<0>(step);
        double t_current = std::get<1>(step);
        dt = t_current - t_last;
        stop = check_stopping_criteria(rhs, y, iter, res, res_hits, dense,      
                                t_last, t_current, zeta_last, zeta_current, vpar_last, vpar_current, abstol, zetas, omegas, stopping_criteria, vpars, zetas_stop, vpars_stop, forget_exact_path, dt_save);
        zeta_last = zeta_current;
        vpar_last = vpar_current;
    } while(t < tmax && !stop);
    // Now save time = t 
    ykeep = y;
    if (rhs.axis==1) {
        ykeep[0] = pow(y[0],2) + pow(y[1],2);
        ykeep[1] = atan2(y[1],y[0]);
    } else if (rhs.axis==2) {
        ykeep[0] = sqrt(pow(y[0],2) + pow(y[1],2));
        ykeep[1] = atan2(y[1],y[0]);        
    }
    res.push_back(join<1, RHS::Size>({t}, {ykeep}));

    return std::make_tuple(res, res_hits);
}


tuple<vector<array<double, 6>>, vector<array<double, 7>>>
particle_guiding_center_boozer_perturbed_tracing(
        shared_ptr<BoozerMagneticField> field, array<double, 3> stz_init,
        double m, double q, double vtotal, double vtang, double mu, double tmax, double abstol, double reltol,
        bool vacuum, bool noK, vector<double> zetas, vector<double> omegas,
        vector<shared_ptr<StoppingCriterion>> stopping_criteria, double dt_save,
        bool zetas_stop, bool vpars_stop, double Phihat, double omega, int Phim,
        int Phin, double phase, bool forget_exact_path, int axis, vector<double> vpars)
{
    Array2 stz({{stz_init[0], stz_init[1], stz_init[2]}});
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
      auto rhs_class = GuidingCenterVacuumBoozerPerturbedRHS(field, m, q, mu, Phihat, omega,
        Phim, Phin, phase, axis);
      return solve<GuidingCenterVacuumBoozerPerturbedRHS>(rhs_class, y, tmax, dt, dtmax, abstol, reltol, zetas, omegas, stopping_criteria, dt_save, vpars, zetas_stop, vpars_stop, forget_exact_path);
  } else {
      auto rhs_class = GuidingCenterNoKBoozerPerturbedRHS(field, m, q, mu, Phihat, omega,
        Phim, Phin, phase, axis);
      return solve<GuidingCenterNoKBoozerPerturbedRHS>(rhs_class, y, tmax, dt, dtmax, abstol, reltol, zetas, omegas, stopping_criteria, dt_save, vpars, zetas_stop, vpars_stop, forget_exact_path);
  }
}

/**
 * @brief Traces the guiding center of a particle in a Boozer magnetic field.
 *
 * @param field Shared pointer to the BoozerMagneticField object.
 * @param stz_init Initial position of the particle in Boozer coordinates (s, theta, zeta).
 * @param m Mass of the particle.
 * @param q Charge of the particle.
 * @param vtotal Total velocity of the particle.
 * @param vtang Tangential velocity of the particle.
 * @param tmax Maximum time for the simulation.
 * @param dt Initial time step for the simulation.
 * @param abstol Absolute tolerance for the adaptive time stepper.
 * @param reltol Relative tolerance for the adaptive time stepper.
 * @param roottol Tolerance for root finding.
 * @param vacuum Boolean flag indicating if the field is a vacuum field.
 * @param noK Boolean flag indicating if the K term should be ignored.
 * @param solveSympl Boolean flag indicating if the symplectic solver should be used.
 * @param zetas Vector of zeta values for stopping criteria.
 * @param omegas Vector of omega values for stopping criteria.
 * @param stopping_criteria Vector of shared pointers to stopping criteria objects.
 * @param dt_save Time step for saving the results.
 * @param vpars Vector of additional parameters for the velocity.
 * @param zetas_stop Boolean flag indicating if zeta stopping criteria should be used.
 * @param vpars_stop Boolean flag indicating if velocity parameter stopping criteria should be used.
 * @param forget_exact_path Boolean flag indicating if the exact path should be forgotten.
 * @param axis Axis of symmetry (1, 2, or other).
 * @param predictor_step Boolean flag indicating if predictor step should be used.
 * @return A tuple containing two vectors: the first vector contains arrays of size 5, and the second vector contains arrays of size 6.
 * 
 * @throws std::invalid_argument if dt is not positive.
 */
tuple<vector<array<double, 5>>, vector<array<double, 6>>>
particle_guiding_center_boozer_tracing(
        shared_ptr<BoozerMagneticField> field, array<double, 3> stz_init,
        double m, double q, double vtotal, double vtang, double tmax, double dt, double abstol, double reltol, double roottol,
        bool vacuum, bool noK, bool solveSympl, vector<double> zetas, vector<double> omegas,
        vector<shared_ptr<StoppingCriterion>> stopping_criteria,  double dt_save, vector<double> vpars,
        bool zetas_stop, bool vpars_stop, bool forget_exact_path, int axis, bool predictor_step)
{
    Array2 stz({{stz_init[0], stz_init[1], stz_init[2]}});
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
#ifdef USE_GSL
        auto f = SymplField(field, m, q, mu);
        return solve_sympl(f, y, tmax, dt, roottol, zetas, omegas, stopping_criteria, vpars, zetas_stop, vpars_stop, forget_exact_path, predictor_step, dt_save);
#else
        throw std::invalid_argument("Symplectic solver not available. Please recompile with GSL support.");
#endif
    } else {
        if (vacuum) {
          auto rhs_class = GuidingCenterVacuumBoozerRHS(field, m, q, mu, axis);
          return solve<GuidingCenterVacuumBoozerRHS>(rhs_class, y, tmax, dt, dtmax, abstol, reltol, zetas, omegas, stopping_criteria, dt_save, vpars, zetas_stop, vpars_stop, forget_exact_path);
        } else if (noK) {
          auto rhs_class = GuidingCenterNoKBoozerRHS(field, m, q, mu, axis);
          return solve<GuidingCenterNoKBoozerRHS>(rhs_class, y, tmax, dt, dtmax, abstol, reltol, zetas, omegas, stopping_criteria, dt_save, vpars, zetas_stop, vpars_stop, forget_exact_path);
        } else {
          auto rhs_class = GuidingCenterBoozerRHS(field, m, q, mu, axis);
          return solve<GuidingCenterBoozerRHS>(rhs_class, y, tmax, dt, dtmax, abstol, reltol, zetas, omegas, stopping_criteria, dt_save, vpars, zetas_stop, vpars_stop, forget_exact_path);
        } 
    }
}
