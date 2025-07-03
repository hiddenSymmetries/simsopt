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
        double vnorm, tnorm;
        static constexpr int Size = 4;
        using State = array<double, Size>;
        State stzv, stzvdot; 

        GuidingCenterVacuumBoozerRHS(shared_ptr<BoozerMagneticField> field, double m, double q, double mu, int axis, double vnorm=1, double tnorm=1)
            : field(field), m(m), q(q), mu(mu), axis(axis), vnorm(vnorm), tnorm(tnorm) {
            }

        void operator()(const State &ys, array<double, 4> &dydt,
                const double t) {

            y_to_stzvt<GuidingCenterVacuumBoozerRHS>(ys, stzv, *this);

            stz(0, 0) = stzv[0];
            stz(0, 1) = stzv[1];
            stz(0, 2) = stzv[2];
            double v_par = stzv[3];

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

            stzvdot[0] = -dmodBdtheta*fak1/(q*psi0);
            stzvdot[1] =  dmodBds*fak1/(q*psi0) + iota*v_par*modB/G; 
            stzvdot[2] = v_par*modB/G;
            stzvdot[3] = -(iota*dmodBdtheta + dmodBdzeta)*mu*modB/G;

            stzvtdot_to_ydot<GuidingCenterVacuumBoozerRHS>(stzvdot, stzv, dydt, *this);
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
        double vnorm, tnorm;
        static constexpr int Size = 5;
        using State = array<double, Size>;
        State stzvt, stzvtdot;

        GuidingCenterVacuumBoozerPerturbedRHS(
            shared_ptr<ShearAlfvenWave> perturbed_field,
            double m,
            double q,
            double mu,
            int axis, 
            double vnorm=1,
            double tnorm=1
        ): 
            perturbed_field(perturbed_field),
            m(m),
            q(q),
            mu(mu),
            axis(axis), 
            vnorm(vnorm),
            tnorm(tnorm) {}

        void operator()(const State &ys, array<double, 5> &dydt,
                const double t) {

            y_to_stzvt<GuidingCenterVacuumBoozerPerturbedRHS>(ys, stzvt, *this);

            stzt(0, 0) = stzvt[0];
            stzt(0, 1) = stzvt[1];
            stzt(0, 2) = stzvt[2];
            stzt(0, 3) = stzvt[4];
            double v_par = stzvt[3];
            double time = stzvt[4];

            perturbed_field->set_points(stzt);
            auto field = perturbed_field->get_B0();
            auto psi0 = field->psi0;
            double modB = field->modB_ref()(0);
            double G = field->G_ref()(0);
            double iota = field->iota_ref()(0);
            auto modB_derivs = field->modB_derivs_ref();
            double dmodBdpsi = modB_derivs(0)/psi0;
            double dmodBdtheta = modB_derivs(1);
            double dmodBdzeta = modB_derivs(2);
            double v_perp2 = 2*mu*modB;
            double fak1 = m*v_par*v_par/modB + m*mu;
            double dPhidpsi = perturbed_field->dPhidpsi_ref()(0);
            double dPhidtheta = perturbed_field->dPhidtheta_ref()(0);
            double dPhidzeta = perturbed_field->dPhidzeta_ref()(0);
            double alphadot = perturbed_field->alphadot_ref()(0);
            double dalphadpsi = perturbed_field->dalphadpsi_ref()(0);
            double dalphadtheta = perturbed_field->dalphadtheta_ref()(0);

            stzvtdot[0] = (-dmodBdtheta*fak1/q + dalphadtheta*modB*v_par - dPhidtheta)/psi0;
            stzvtdot[1] = dmodBdpsi*fak1/q + (iota - dalphadpsi*G)*v_par*modB/G + dPhidpsi;
            stzvtdot[2] = v_par*modB/G;
            stzvtdot[3] = -modB/(G*m) * (m*mu*(dmodBdzeta + dalphadtheta*dmodBdpsi*G \
                    + dmodBdtheta*(iota - dalphadpsi*G)) + q*(alphadot*G \
                    + dalphadtheta*G*dPhidpsi + (iota - dalphadpsi*G)*dPhidtheta + dPhidzeta)) \
                    + v_par/modB * (dmodBdtheta*dPhidpsi - dmodBdpsi*dPhidtheta);
            stzvtdot[4] = 1;

            stzvtdot_to_ydot<GuidingCenterVacuumBoozerPerturbedRHS>(stzvtdot, stzvt, dydt, *this);
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
        double vnorm, tnorm;
        static constexpr int Size = 5;
        using State = array<double, Size>;
        State stzvt, stzvtdot;

        GuidingCenterNoKBoozerPerturbedRHS(
            shared_ptr<ShearAlfvenWave> perturbed_field,
            double m,
            double q,
            double mu,
            int axis, 
            double vnorm=1,
            double tnorm=1
        ): 
        perturbed_field(perturbed_field),
        m(m),
        q(q),
        mu(mu),
        axis(axis),
        vnorm(vnorm),
        tnorm(tnorm) {}

        void operator()(const State &ys, array<double, 5> &dydt,
                const double t) {

            y_to_stzvt<GuidingCenterNoKBoozerPerturbedRHS>(ys, stzvt, *this);

            stzt(0, 0) = stzvt[0];
            stzt(0, 1) = stzvt[1];
            stzt(0, 2) = stzvt[2];
            stzt(0, 3) = stzvt[4];
            double v_par = stzvt[3];

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

            stzvtdot[0] = (-G*dPhidtheta*q + I*dPhidzeta*q + modB*q*v_par*(dalphadtheta*G-dalphadzeta*I) + (-dmodBdtheta*G + dmodBdzeta*I)*fak1)/(denom*psi0);
            stzvtdot[1] = (G*q*dPhidpsi + modB*q*v_par*(-dalphadpsi*G - alpha*dGdpsi + iota) - dGdpsi*m*v_par*v_par \
                      + dmodBdpsi*G*fak1)/denom;
            stzvtdot[2] = (-I*(dmodBdpsi*m*mu + dPhidpsi*q) + modB*q*v_par*(1 + dalphadpsi*I + alpha*dIdpsi) \
                      + m*v_par*v_par/modB * (modB*dIdpsi - dmodBdpsi*I))/denom;
            stzvtdot[3] = (modB*q/m * ( -m*mu * (dmodBdzeta*(1 + dalphadpsi*I + alpha*dIdpsi) \
                      + dmodBdpsi*(dalphadtheta*G - dalphadzeta*I) + dmodBdtheta*(iota - alpha*dGdpsi - dalphadpsi*G)) \
                      - q*(alphadot*(G + I*(iota - alpha*dGdpsi) + alpha*G*dIdpsi) \
                      + (dalphadtheta*G - dalphadzeta*I)*dPhidpsi \
                      + (iota - alpha*dGdpsi - dalphadpsi*G)*dPhidtheta \
                      + (1 + alpha*dIdpsi + dalphadpsi*I)*dPhidzeta)) \
                      + q*v_par/modB * ((dmodBdtheta*G - dmodBdzeta*I)*dPhidpsi \
                      + dmodBdpsi*(I*dPhidzeta - G*dPhidtheta)) \
                      + v_par*(m*mu*(dmodBdtheta*dGdpsi - dmodBdzeta*dIdpsi) \
                      + q*(alphadot*(dGdpsi*I-G*dIdpsi) + dGdpsi*dPhidtheta - dIdpsi*dPhidzeta)))/denom;
            stzvtdot[4] = 1;

            stzvtdot_to_ydot<GuidingCenterNoKBoozerPerturbedRHS>(stzvtdot, stzvt, dydt, *this);
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
        double vnorm, tnorm;
        static constexpr int Size = 4;
        using State = array<double, Size>;
        State stzv, stzvdot; 

        GuidingCenterNoKBoozerRHS(shared_ptr<BoozerMagneticField> field, double m, double q, double mu, int axis, double vnorm=1, double tnorm=1)
            : field(field), m(m), q(q), mu(mu), axis(axis), vnorm(vnorm), tnorm(tnorm) {
            }

        void operator()(const State &ys, array<double, 4> &dydt,
                const double t) {
            y_to_stzvt<GuidingCenterNoKBoozerRHS>(ys, stzv, *this);

            stz(0, 0) = stzv[0];
            stz(0, 1) = stzv[1];
            stz(0, 2) = stzv[2];
            double v_par = stzv[3];
                    
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

            stzvdot[0] = (I*dmodBdzeta - G*dmodBdtheta)*fak1/(D*iota*psi0);
            stzvdot[1] = (G*dmodBdpsi*fak1 - (-q*iota + m*v_par*dGdpsi/modB)*v_par*modB)/(D*iota);
            stzvdot[2] = ((q + m*v_par*dIdpsi/modB)*v_par*modB - dmodBdpsi*fak1*I)/(D*iota);
            stzvdot[3] = modB*mu*(dmodBdtheta*C - dmodBdzeta*F)/(F*G-C*I);

            stzvtdot_to_ydot<GuidingCenterNoKBoozerRHS>(stzvdot, stzv, dydt, *this);

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
        double vnorm, tnorm; 
        State stzv, stzvdot; 

        GuidingCenterBoozerRHS(shared_ptr<BoozerMagneticField> field, double m, double q, double mu, int axis, double vnorm=1, double tnorm=1)
            : field(field), m(m), q(q), mu(mu), axis(axis), vnorm(vnorm), tnorm(tnorm) {
            }

        void operator()(const State &ys, array<double, 4> &dydt,
                const double t) {

            y_to_stzvt<GuidingCenterBoozerRHS>(ys, stzv, *this);

            stz(0, 0) = stzv[0];
            stz(0, 1) = stzv[1];
            stz(0, 2) = stzv[2];
            double v_par = stzv[3];

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

            stzvdot[0] = (I*dmodBdzeta - G*dmodBdtheta)*fak1/(D*iota*psi0);
            stzvdot[1] = (G*dmodBdpsi*fak1 - C*v_par*modB - K*fak1*dmodBdzeta)/(D*iota);
            stzvdot[2] = (F*v_par*modB - dmodBdpsi*fak1*I + K*fak1*dmodBdtheta)/(D*iota);
            // dydt[3] = - (mu / v_par) * (dmodBdpsi * sdot * psi0 + dmodBdtheta * tdot + dmodBdzeta * dydt[2]);
            stzvdot[3] = modB*mu*(dmodBdtheta*C - dmodBdzeta*F)/(F*G-C*I);

            stzvtdot_to_ydot<GuidingCenterBoozerRHS>(stzvdot, stzv, dydt, *this);
        }
};

template<class RHS>
tuple<vector<array<double, RHS::Size+1>>, vector<array<double, RHS::Size+2>>>
solve(RHS rhs, typename RHS::State stzvt, double tau_max, double dtau, double dtau_max, double abstol, double reltol, vector<double> thetas, vector<double> zetas, 
    vector<double> omega_thetas, vector<double> omega_zetas, vector<shared_ptr<StoppingCriterion>> stopping_criteria, double dtau_save, vector<double> vpars, 
    bool thetas_stop=false, bool zetas_stop=false, bool vpars_stop=false, bool forget_exact_path=false) {

    if (zetas.size() > 0 && omega_zetas.size() == 0) {
        omega_zetas.insert(omega_zetas.end(), zetas.size(), 0.);
    } else if (zetas.size() !=  omega_zetas.size()) {
        throw std::invalid_argument("zetas and omega_zetas need to have matching length.");
    }
    if (thetas.size() > 0 && omega_thetas.size() == 0) {
        omega_thetas.insert(omega_thetas.end(), thetas.size(), 0.);
    } else if (thetas.size() != omega_thetas.size() and thetas.size() != 0) {
        throw std::invalid_argument("thetas and omega_thetas need to have matching length.");
    }

    vector<array<double, RHS::Size+1>> res = {};
    vector<array<double, RHS::Size+2>> res_hits = {};
    typedef typename RHS::State State;
    State y, temp; 
    typedef typename boost::numeric::odeint::result_of::make_dense_output<runge_kutta_dopri5<State>>::type dense_stepper_type;
    dense_stepper_type dense = make_dense_output(abstol, reltol, dtau_max, runge_kutta_dopri5<State>());
    double tau = 0;
    int iter = 0;
    bool stop = false;
    double tau_last = 0;
    double tau_current;
    tau_last = tau;

    // Save initial state
    res.push_back(join<1, RHS::Size>({0}, stzvt));

    stzvt_to_y<RHS>(stzvt, y, rhs);
    dense.initialize(y, tau, dtau);

    do {
        tuple<double, double> step = dense.do_step(rhs);
        iter++;
        tau = dense.current_time();
        y = dense.current_state();
        tau_last = std::get<0>(step);
        tau_current = std::get<1>(step);
        dtau = tau_current - tau_last; // Timestep taken

        // Check if we have hit a stopping criterion between tau_last and tau_current
        stop = check_stopping_criteria<RHS,dense_stepper_type>(rhs, iter, res_hits, dense, tau_last, 
            tau_current, dtau, abstol, thetas, zetas, omega_thetas, omega_zetas, stopping_criteria, vpars, thetas_stop, zetas_stop, vpars_stop);

        // Save path if forget_exact_path = False
        if (forget_exact_path == 0) {
            // If we have hit a stopping criterion, we still want to save the trajectory up to that point
            if (stop) {
                tau_current = res_hits.back()[0] / rhs.tnorm;
            }
            // This will give the first save point after tau_last
            double tau_save_last = std::ceil(tau_last/dtau_save) * dtau_save;
            for (double tau_save = tau_save_last; tau_save <= tau_current; tau_save += dtau_save) {
                if (tau_save != 0) {  // tau = 0 is already saved. 
                    dense.calc_state(tau_save, temp);
                    double t_save = tau_save * rhs.tnorm;
                    y_to_stzvt<RHS>(temp, stzvt, rhs);
                    res.push_back(join<1, RHS::Size>({t_save}, stzvt));
                }
            }
        } 
    } while(tau < tau_max && !stop);
    // Save t = tmax
    if (stop) {
        tau_max = res_hits.back()[0] / rhs.tnorm;
    }
    double t_max = tau_max * rhs.tnorm;
    dense.calc_state(tau_max, y);
    y_to_stzvt<RHS>(y, stzvt, rhs);
    res.push_back(join<1, RHS::Size>({t_max}, stzvt));

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
        vector<double> thetas,
        vector<double> zetas,
        vector<double> omega_thetas,
        vector<double> omega_zetas,
        vector<shared_ptr<StoppingCriterion>> stopping_criteria,
        double dt_save,
        bool thetas_stop,
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
    array<double, 5> stzvt;
    double G0 = std::abs(field->G()(0));
    double r0 = G0/modB;
    double vnorm = vtotal; // Normalizing velocity = vtotal
    double tnorm = r0*2*M_PI/vtotal; // Normalizing time = time for one toroidal revolution
    double dtau_max = 0.25; // can at most do quarter of a revolution per step
    double dtau = 1e-3 * dtau_max; // initial guess for first timestep, will be adjusted by adaptive timestepper

    if (dtau<0) {
        throw std::invalid_argument("dtau needs to be positive.");
    }

    // Normalize tmax and dt_save
    double tau_max = tmax / tnorm;
    double dtau_save = dt_save / tnorm;

    // Initial conditions are passed as (s, theta, zeta, v_par, t)
    // While, tracing is done in mapped coordinates y 
    stzvt[0] = stz_init[0];
    stzvt[1] = stz_init[1];
    stzvt[2] = stz_init[2]; 
    stzvt[3] = vtang; 
    stzvt[4] = 0; 

    if (vacuum) {
      auto rhs_class = GuidingCenterVacuumBoozerPerturbedRHS(
          perturbed_field, m, q, mu, axis, vnorm, tnorm
      );
      return solve<GuidingCenterVacuumBoozerPerturbedRHS>(rhs_class, stzvt, tau_max, dtau, dtau_max, abstol, reltol, thetas, zetas, omega_thetas, omega_zetas, 
            stopping_criteria, dtau_save, vpars, thetas_stop, zetas_stop, vpars_stop, forget_exact_path);
  } else {
      auto rhs_class = GuidingCenterNoKBoozerPerturbedRHS(
          perturbed_field, m, q, mu, axis, vnorm, tnorm
      );
      return solve<GuidingCenterNoKBoozerPerturbedRHS>(rhs_class, stzvt, tau_max, dtau, dtau_max, abstol, reltol, thetas, zetas, omega_thetas, omega_zetas, 
            stopping_criteria, dtau_save, vpars, thetas_stop, zetas_stop, vpars_stop, forget_exact_path);
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
        vector<double> thetas,
        vector<double> zetas, 
        vector<double> omega_thetas,
        vector<double> omega_zetas,
        vector<double> vpars,
        vector<shared_ptr<StoppingCriterion>> stopping_criteria,  
        double dt_save, 
        bool forget_exact_path,
        bool thetas_stop,
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
    array<double, 4> stzv;
    double vnorm, tnorm, dtau_max, dtau; 

    if (!solveSympl){
        double G0 = std::abs(field->G()(0));
        double r0 = G0/modB;
        vnorm = vtotal; // Normalizing velocity = vtotal
        tnorm = r0*2*M_PI/vtotal; // Normalizing time = time for one toroidal revolution
        dtau_max = 0.25; // can at most do quarter of a revolution per step
        dtau = 1e-3 * dtau_max; // initial guess for first timestep, will be adjusted by adaptive timestepper
    } else {
        vnorm = 1;
        tnorm = 1; 
        dtau = dt / tnorm;
    }
    if (dtau<0) {
        throw std::invalid_argument("dtau needs to be positive.");
    }
    // Normalize tmax and dt_save
    double tau_max = tmax / tnorm;
    double dtau_save = dt_save / tnorm;

    stzv[0] = stz_init[0];
    stzv[1] = stz_init[1];
    stzv[2] = stz_init[2]; 
    stzv[3] = vtang; 

    if (solveSympl) {
#ifdef USE_GSL
        auto f = SymplField(field, m, q, mu, vnorm, tnorm);
        return solve_sympl(f, stzv, tau_max, dtau, roottol, thetas, zetas, omega_thetas, omega_zetas, stopping_criteria, vpars, thetas_stop, zetas_stop, vpars_stop, forget_exact_path, predictor_step, dtau_save);
#else
        throw std::invalid_argument("Symplectic solver not available. Please recompile with GSL support.");
#endif
    } else {
        if (vacuum) {
          auto rhs_class = GuidingCenterVacuumBoozerRHS(field, m, q, mu, axis, vnorm, tnorm);
          return solve<GuidingCenterVacuumBoozerRHS>(rhs_class, stzv, tau_max, dtau, dtau_max, abstol, reltol, thetas, zetas, omega_thetas, omega_zetas, stopping_criteria, dtau_save, vpars, thetas_stop, zetas_stop, vpars_stop, forget_exact_path);
        } else if (noK) {
          auto rhs_class = GuidingCenterNoKBoozerRHS(field, m, q, mu, axis, vnorm, tnorm);
          return solve<GuidingCenterNoKBoozerRHS>(rhs_class, stzv, tau_max, dtau, dtau_max, abstol, reltol, thetas, zetas, omega_thetas, omega_zetas, stopping_criteria, dt_save, vpars, thetas_stop, zetas_stop, vpars_stop, forget_exact_path);
        } else {
          auto rhs_class = GuidingCenterBoozerRHS(field, m, q, mu, axis, vnorm, tnorm);
          return solve<GuidingCenterBoozerRHS>(rhs_class, stzv, tau_max, dtau, dtau_max, abstol, reltol, thetas, zetas, omega_thetas, omega_zetas, stopping_criteria, dtau_save, vpars, thetas_stop, zetas_stop, vpars_stop, forget_exact_path);
        }
    }
}
