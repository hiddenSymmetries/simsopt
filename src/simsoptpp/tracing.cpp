#include "tracing_helpers.h"
#include "boozermagneticfield.h"
#include "shearalfvenwave.h"
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
     * The state consists of :math:`[s, theta, zeta, v_par]` with
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
            auto modB_derivs = field->modB_derivs_ref();
            double dmodBds = modB_derivs(0);
            double dmodBdtheta = modB_derivs(1);
            double dmodBdzeta = modB_derivs(2);
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
     * The state consists of :math:`[s, theta, zeta, v_par, t]` with
     *
     *    \dot s      = (-|B|_{,\theta} m(v_{||}^2/|B| + \mu)/q 
     *                  + \alpha_{,\theta}|B|v_{||} - \Phi_{\theta})/psi0;
     *    \dot \theta = |B|_{,\psi} m (v_{||}^2/|B| + \mu)/q 
     *                  + (\iota - \alpha_{,psi} G) v_{||}|B|/G + \Phi_{,\psi};
     *    \dot \zeta  = v_{||}|B|/G
     *    \dot v_{||} = -|B|/(Gm) (m\mu(|B|_{,\zeta} 
     *                          + \alpha_{,\theta}|B|_{,\psi}G 
     *                          + |B|_{,\theta}(\iota - \alpha_{,\psi}G))
     *                  + q(\dot\alpha G + \alpha_{,\theta}G\Phi_{,\psi} 
     *                  + (\iota - \alpha_{\psi}*G)*\Phi_{\theta}
     *                  + \Phi_{,\zeta})) 
     *                  + v_{||}/|B|(|B|_{,\theta}\Phi_{,\psi} 
     *                             - |B|_{,\psi} \Phi_{,\theta})
     *
     *  where :math:`q` is the charge, :math:`m` is the mass, and :math:`v_\perp = 2\mu|B|`.
     *
     */
    private:
        Array2 stzt = xt::zeros<double>({1, 4});
        shared_ptr<ShearAlfvenWave> perturbed_field;
        double m, q, mu;
    public:
        int axis;
        static constexpr int Size = 5;
        using State = array<double, Size>;

        GuidingCenterVacuumBoozerPerturbedRHS(
            shared_ptr<ShearAlfvenWave> perturbed_field,
            double m,
            double q,
            double mu,
            int axis
        ): 
            perturbed_field(perturbed_field),
            m(m),
            q(q),
            mu(mu),
            axis(axis) {}

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

            stzt(0, 0) = s;
            stzt(0, 1) = theta;
            stzt(0, 2) = ys[2];
            stzt(0, 3) = ys[4];

            perturbed_field->set_points(stzt);
            auto field = perturbed_field->get_B0();
            auto psi0 = field->psi0;
            double modB = field->modB_ref()(0);
            double G = field->G_ref()(0);
            double iota = field->iota_ref()(0);
            double diotadpsi = field->diotads_ref()(0)/psi0;
            auto modB_derivs = field->modB_derivs_ref();
            double dmodBdpsi = modB_derivs(0)/psi0;
            double dmodBdtheta = modB_derivs(1);
            double dmodBdzeta = modB_derivs(2);
            double v_perp2 = 2*mu*modB;
            double fak1 = m*v_par*v_par/modB + m*mu;
            double Phi = perturbed_field->Phi_ref()(0);
            double Phidot = perturbed_field->Phidot_ref()(0);
            double dPhidpsi = perturbed_field->dPhidpsi_ref()(0);
            double dPhidtheta = perturbed_field->dPhidtheta_ref()(0);
            double dPhidzeta = perturbed_field->dPhidzeta_ref()(0);
            double alpha = perturbed_field->alpha_ref()(0);
            double alphadot = perturbed_field->alphadot_ref()(0);
            double dalphadpsi = perturbed_field->dalphadpsi_ref()(0);
            double dalphadtheta = perturbed_field->dalphadtheta_ref()(0);
            double dalphadzeta = perturbed_field->dalphadzeta_ref()(0);

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
     * The state consists of :math:`[s, theta, zeta, v_par, t]` with
     *
     *    \dot s = (-G \Phi_{,\theta}q + I\Phi_{,\zeta}q
     *               + |B|qv_{||}(\alpha_{\theta}G-\alpha_{,\zeta}I)
     *               + (-|B|_{,\theta}G + |B|_{,\zeta}I)
     *               * (mv_{||}}^2/|B| + m\mu))/(D psi0)
     *    \dot theta = (G q \Phi_{,\psi}
     *               + |B| q v_{||} (-\alpha_{,\psi} G - \alpha G_{,\psi} + \iota)
     *               - G_{,\psi} m v_{||}^2 + |B|_{,\psi} G (mv_{||}}^2/|B| + m\mu))/D
     *    \dot \zeta = (-I (|B|_{,\psi} m \mu + \Phi_{,\psi} q) 
     *               + |B| q v_{||} (1 + \alpha_{,\psi}) I + \alpha I'(\psi))
     *               + m v_{||}^2/|B| (|B| I'(\psi) - |B|_{,\psi} I))/D
     *    \dot v_{||} = (|B|q/m ( -m mu (|B|_{,\zeta}(1 + \alpha_{,\psi} I + \alpha I'(\psi)) 
     *                + |B|_{,\psi} (\alpha_{,\theta} G - \alpha_{,\zeta} I) 
     *                + |B|_{,\theta} (\iota - \alpha G'(\psi) - \alpha_{,\psi} G)) 
     *                - q (\dot \alpha (G + I (\iota - \alpha G'(\psi)) + \alpha G I'(\psi))
     *                + (\alpha_{,\theta} G - \alpha_{,\zeta} I) \Phi_{,\psi} 
     *                + (\iota - \alpha G_{,\psi} - \alpha_{,\psi} G) \Phi_{,\theta} 
     *                + (1 + \alpha I'(\psi) + \alpha_{,\psi} I) Phi_{,\zeta})) 
     *                + q v_{||}/|B| ((|B|_{,\theta} G - |B|_{,\zeta} I) \Phi_{,\psi} 
     *                + |B|_{,\psi} (I \Phi_{,\zeta} - G \Phi_{,\theta})) 
     *                + v_{||} (m \mu (|B|_{,\theta} G'(\psi) - |B|_{,\zeta} I'(\psi)) 
     *                + q (\dot \alpha (G'(\psi) I - G I'(\psi))
     *                + G'(\psi) \Phi_{,\theta} - I'(\psi)\Phi_{,\zeta})))/D
     *    D = (q(G + I(-\alpha G_{,\psi} + \iota) + \alpha G I'(\psi) 
     *          + mv_{||}/|B| (-G'(\psi) I + G I'(\psi)))
     *  where :math:`q` is the charge, :math:`m` is the mass, and :math:`v_\perp = 2\mu|B|`.
     *
     */
    private:
        Array2 stzt = xt::zeros<double>({1, 4});
        shared_ptr<ShearAlfvenWave> perturbed_field;
        double m, q, mu;
    public:
        int axis;
        static constexpr int Size = 5;
        using State = array<double, Size>;

        GuidingCenterNoKBoozerPerturbedRHS(
            shared_ptr<ShearAlfvenWave> perturbed_field,
            double m,
            double q,
            double mu,
            int axis
        ): 
        perturbed_field(perturbed_field),
        m(m),
        q(q),
        mu(mu),
        axis(axis) {}

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

            stzt(0, 0) = s;
            stzt(0, 1) = theta;
            stzt(0, 2) = ys[2];
            stzt(0, 3) = ys[4];
            perturbed_field->set_points(stzt);
            auto field = perturbed_field->get_B0();

            auto psi0 = field->psi0;
            double modB = field->modB_ref()(0);
            double G = field->G_ref()(0);
            double I = field->I_ref()(0);
            double dGdpsi = field->dGds_ref()(0)/psi0;
            double dIdpsi = field->dIds_ref()(0)/psi0;
            double iota = field->iota_ref()(0);
            double diotadpsi = field->diotads_ref()(0)/psi0;
            auto modB_derivs = field->modB_derivs_ref();
            double dmodBdpsi = modB_derivs(0)/psi0;
            double dmodBdtheta = modB_derivs(1);
            double dmodBdzeta = modB_derivs(2);
            double v_perp2 = 2*mu*modB;
            double fak1 = m*v_par*v_par/modB + m*mu;
            double Phi = perturbed_field->Phi_ref()(0);
            double Phidot = perturbed_field->Phidot_ref()(0);
            double dPhidpsi = perturbed_field->dPhidpsi_ref()(0);
            double dPhidtheta = perturbed_field->dPhidtheta_ref()(0);
            double dPhidzeta = perturbed_field->dPhidzeta_ref()(0);
            double alpha = perturbed_field->alpha_ref()(0);
            double alphadot = perturbed_field->alphadot_ref()(0);
            double dalphadpsi = perturbed_field->dalphadpsi_ref()(0);
            double dalphadtheta = perturbed_field->dalphadtheta_ref()(0);
            double dalphadzeta = perturbed_field->dalphadzeta_ref()(0);
            double denom = (q*(G + I*(-alpha*dGdpsi + iota) + alpha*G*dIdpsi) 
                + m*v_par/modB * (-dGdpsi*I + G*dIdpsi)); // q*G in vacuum

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
            auto modB_derivs = field->modB_derivs_ref();
            double dmodBdpsi = modB_derivs(0)/psi0;
            double dmodBdtheta = modB_derivs(1);
            double dmodBdzeta = modB_derivs(2);
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
            auto K_derivs = field->K_derivs_ref();
            double dKdtheta = K_derivs(0);
            double dKdzeta = K_derivs(1);

            double G = field->G_ref()(0);
            double I = field->I_ref()(0);
            double dGdpsi = field->dGds_ref()(0)/psi0;
            double dIdpsi = field->dIds_ref()(0)/psi0;
            double iota = field->iota_ref()(0);
            auto modB_derivs = field->modB_derivs_ref();
            double dmodBdpsi = modB_derivs(0)/psi0;
            double dmodBdtheta = modB_derivs(1);
            double dmodBdzeta = modB_derivs(2);
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
        t_last = std::get<0>(step);
        t_current = std::get<1>(step);
        dt = t_current - t_last;

        // Save path if forget_exact_path = False
        if (forget_exact_path == 0) {
            // This will give the first save point after t_last
            double t_save_last = dt_save * std::ceil(t_last/dt_save);
            for (double t_save = t_save_last; t_save <= t_current; t_save += dt_save) {
                if (t_save != 0) {
                    vpar_last = res.back()[4];
                    zeta_last = res.back()[3];
                    dense.calc_state(t_save, temp);
                    vpar_current = temp[3];
                    zeta_current = temp[2];
                    // Only save if we have not hit any stopping criteria
                    stop = check_stopping_criteria<RHS,dense_stepper_type>(rhs, 
                        temp, iter, res, res_hits, dense,      
                        t_save - dt_save, t_save, dt, zeta_last, zeta_current, vpar_last, vpar_current, abstol, zetas, omegas, stopping_criteria, vpars, zetas_stop, vpars_stop);
                    if (stop) {
                        break;
                    } else {
                        ykeep = temp;
                        if (rhs.axis==1) {
                            ykeep[0] = pow(temp[0],2) + pow(temp[1],2);
                            ykeep[1] = atan2(temp[1],temp[0]);
                        } else if (rhs.axis==2) {
                            ykeep[0] = sqrt(pow(temp[0],2) + pow(temp[1],2));
                            ykeep[1] = atan2(temp[1],temp[0]);        
                        }
                        res.push_back(join<1, RHS::Size>({t_save}, ykeep));
                    }
                }
            }
        } else {
            stop = check_stopping_criteria<RHS,dense_stepper_type>(rhs, y, 
                iter, res, res_hits, dense,      
                t_last, t_current, dt, zeta_last, zeta_current, vpar_last, vpar_current, abstol, zetas, omegas, stopping_criteria, vpars, zetas_stop, vpars_stop);
        }
        
        zeta_last = zeta_current;
        vpar_last = vpar_current;
    } while(t < tmax && !stop);
    // Save t = tmax
    if(!stop){
        dense.calc_state(tmax, y);
        t = tmax; 
        ykeep = y;
        if (rhs.axis==1) {
            ykeep[0] = pow(y[0],2) + pow(y[1],2);
            ykeep[1] = atan2(y[1],y[0]);
        } else if (rhs.axis==2) {
            ykeep[0] = sqrt(pow(y[0],2) + pow(y[1],2));
            ykeep[1] = atan2(y[1],y[0]);        
        }
        res.push_back(join<1, RHS::Size>({t}, {ykeep})); 
    
    } 
   
    return std::make_tuple(res, res_hits);
}

tuple<vector<array<double, 6>>, vector<array<double, 7>>>
particle_guiding_center_boozer_perturbed_tracing(
        shared_ptr<ShearAlfvenWave> perturbed_field,
        array<double, 3> stz_init,
        double m,
        double q,
        double vtotal,
        double vtang,
        double mu,
        double tmax,
        double abstol,
        double reltol,
        bool vacuum,
        bool noK,
        vector<double> zetas,
        vector<double> omegas,
        vector<shared_ptr<StoppingCriterion>> stopping_criteria,
        double dt_save,
        bool zetas_stop,
        bool vpars_stop,
        bool forget_exact_path,
        int axis,
        vector<double> vpars)
{
    Array2 stzt({{stz_init[0], stz_init[1], stz_init[2], 0.0}});
    perturbed_field->set_points(stzt);
    auto field = perturbed_field->get_B0();
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
      auto rhs_class = GuidingCenterVacuumBoozerPerturbedRHS(
          perturbed_field, m, q, mu, axis
      );
      return solve<GuidingCenterVacuumBoozerPerturbedRHS>(rhs_class, y, tmax, dt, dtmax, abstol, reltol, zetas, omegas, stopping_criteria, dt_save, vpars, zetas_stop, vpars_stop, forget_exact_path);
  } else {
      auto rhs_class = GuidingCenterNoKBoozerPerturbedRHS(
          perturbed_field, m, q, mu, axis
      );
      return solve<GuidingCenterNoKBoozerPerturbedRHS>(rhs_class, y, tmax, dt, dtmax, abstol, reltol, zetas, omegas, stopping_criteria, dt_save, vpars, zetas_stop, vpars_stop, forget_exact_path);
  }
}

/**
See trace_particles_boozer() defined in tracing.py for details on the parameters.
**/
tuple<vector<array<double, 5>>, vector<array<double, 6>>>
particle_guiding_center_boozer_tracing(
        shared_ptr<BoozerMagneticField> field, 
        array<double, 3> stz_init,
        double m, 
        double q, 
        double vtotal, 
        double vtang, 
        double tmax, 
        bool vacuum, 
        bool noK, 
        vector<double> zetas, 
        vector<double> omegas,
        vector<double> vpars,
        vector<shared_ptr<StoppingCriterion>> stopping_criteria,  
        double dt_save, 
        bool forget_exact_path,
        bool zetas_stop, 
        bool vpars_stop, 
        int axis, 
        double abstol, 
        double reltol,
        bool solveSympl,
        bool predictor_step, 
        double roottol,
        double dt
        )
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
