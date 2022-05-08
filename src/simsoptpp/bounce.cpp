#include "boozermagneticfield.h"
#include "bounce.h"
#include <vector>
using Vec = std::vector<double>;
#include "xtensor-python/pyarray.hpp"
typedef xt::pyarray<double> Array;
#include "xtensor-python/pytensor.hpp"
#include <xtensor/xio.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xsort.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint/integrate/integrate.hpp>
#include <boost/numeric/odeint.hpp>

#include <tuple>
using namespace boost::numeric::odeint;

template
std::vector<double> find_bounce_points<xt::pytensor>(
      shared_ptr<BoozerMagneticField<xt::pytensor>> field, double s,
      double theta0, double lam, int nfp, int option, int nmax, int nzeta,
      int digits, double derivative_tol, double argmin_tol, double root_tol);

template
Array bounce_integral<xt::pytensor>(std::vector<double> bouncel, std::vector<double> bouncer,
        shared_ptr<BoozerMagneticField<xt::pytensor>> field, double s,
            double theta0, double lam, int nfp, bool jpar,
            bool psidot, bool alphadot, bool ihat, bool khat, bool dkhatdalpha, bool tau,
            double step_size, double tol, bool adjust);

template<template<class, std::size_t, xt::layout_type> class T>
std::vector<double> find_bounce_points(shared_ptr<BoozerMagneticField<T>> field, double s,
    double theta0, double lam, int nfp, int option, int nmax, int nzeta, int digits, double derivative_tol, double argmin_tol, double root_tol) {
    ///
    // field: instance of BoozerMagneticField
    // s (double): normalized toroidal flux
    // theta0 (double): initial poloidal angle s.t. field line label is given
    //                  by alpha = theta0
    // lam (double): field line label = v_perp^2/(2*B*v^2)
    // nfp (int): number of field periods
    // option: 0 = find left bounce points, 1 = find right bounce points,
    //         2 = find maxima along field line
    // nzeta: number of gridpoints in toroidal angle grid along field line
    // nmax (int): number of periods for toroidal angle grid along field line
    // digits (int): number of digits for root solve
    // derivative_tol: tolerance for derivative to distinguish maxima
    // argmin_tol: tolerance for finding gridpoints in zeta close to bounce point
    // root_tol: tolerance for root solve to determine bounce point
    ///
    typename BoozerMagneticField<T>::Tensor2 point = xt::zeros<double>({1, 3});
    point(0,0) = s;
    field->set_points(point);
    double iota = field->iota()(0,0);

    // Set up grid in zeta - note that grid points at zeta = 0 and zeta = 2*pi*nmax/nfp
    // are included so that we can diagnose points that lie in the last interval
    Vec zeta(nzeta, 0.);
    for (int i = 0; i < nzeta; ++i) {
      zeta[i] = i*2*M_PI*nmax/(nfp*(nzeta-1));
    }

    typename BoozerMagneticField<T>::Tensor2 points = xt::zeros<double>({nzeta, 3});
    for (int i = 0; i < nzeta; ++i) {
      points(i, 0) = s;
      points(i, 1) = theta0 + iota * zeta[i];
      points(i, 2) = zeta[i];
    }

    // Compute modB on this grid
    field->set_points(points);
    auto modB = field->modB();

    // We want to perform a root solve provided with left and right bounds.
    // Using the grid along the field line, we identify potential left bounding
    // points of the bounce point, where the corresponding right bounding point
    // is the grid point to the immediate right.
    std::vector<double> bounce_try_l;
    std::vector<double> bounce_try_r;
    if (option != 2) {
      for (int i = 0; i < nzeta-1; ++i) {
        // Find points such that modB brackets 1/lam on either side and
        // left point has larger modB, right points has smaller modB
        // This yields the set of potential left bounce points.
        if (option == 0) {
          if (modB(i,0) > 1/lam && modB(i+1,0) < 1/lam) {
            bounce_try_l.push_back(zeta[i]);
            bounce_try_r.push_back(zeta[i+1]);
          }
        } else if (option == 1) {
          if (modB(i,0) < 1/lam && modB(i+1,0) > 1/lam) {
            bounce_try_l.push_back(zeta[i]);
            bounce_try_r.push_back(zeta[i+1]);
          }
        }
      }
    } else {
      // The above does not work well if looking for a bounce point near the maxima
      // Instead try performing the root solve near the locations where
      // |modB - 1/lam| is minimized.
      auto argmin = xt::argwhere(xt::abs(1 - lam*modB) < argmin_tol);
      int argmin_left, argmin_right, index_left, index_right;
      for (int i = 0; i < argmin.size(); ++i) {
        if (argmin[i][0] == nzeta - 1) {
            bounce_try_l.push_back(zeta[nzeta-2]);
            bounce_try_r.push_back(zeta[nzeta-2] + 2*2*M_PI*nmax/(nfp*(nzeta-1)));
        // For case of > , there is one potential left point to add but does
        // not need to be added since it is equivalent to adding i = 0
        } else if (argmin[i][0] == 0) {
            bounce_try_l.push_back(zeta[1] - 2*2*M_PI*nmax/(nfp*(nzeta-1)));
            bounce_try_r.push_back(zeta[1]);
        } else {
            bounce_try_l.push_back(zeta[argmin[i][0]-1]);
            bounce_try_r.push_back(zeta[argmin[i][0]+1]);
        }
      }
    }

    // Function handle for modB along field line
    std::function<double(double)> modBf = [iota,theta0,field,point](double zeta) mutable {
      point(0,1) = theta0 + iota * zeta;
      point(0,2) = zeta;
      field->set_points(point);
      auto modB = field->modB();
      return modB(0,0);
    };

    // Function handle for derivative of modB along field line
    std::function<double(double)> modBprimef = [iota,theta0,field,point](double zeta) mutable {
      point(0,1) = theta0 + iota * zeta;
      point(0,2) = zeta;
      field->set_points(point);
      auto dmodBdtheta = field->dmodBdtheta();
      auto dmodBdzeta = field->dmodBdzeta();
      return iota*dmodBdtheta(0,0) + dmodBdzeta(0,0);
    };

    // Function handle for root
    std::function<std::tuple<double,double>(double)> rootf = [modBf,modBprimef,lam](double zeta) mutable {
      return std::make_tuple(modBf(zeta) - 1/lam,modBprimef(zeta));
    };

    // Iterate over points bracketing potential left points. Peform root solve
    // to find left bounce points.
    std::vector<double> bounce;
    for (int ir = 0; ir < bounce_try_l.size(); ++ir)  {
      double zetal = bounce_try_l[ir];
      double zetar = bounce_try_r[ir];
      auto guess = 0.5*(zetal + zetar);
      auto root = boost::math::tools::newton_raphson_iterate(rootf, guess, zetal, zetar, digits);
      if (std::abs(1 - lam*modBf(root)) < root_tol) {
        // Check that field is decreasing along field line
        // if seeking left bounce points
        if (option == 0) {
          if (modBprimef(root) <= 0) {
            bounce.push_back(root);
          }
        } else if (option == 1) {
          if (modBprimef(root) >= 0) {
            bounce.push_back(root);
          }
        } else {
          if (std::abs(modBprimef(root)) < derivative_tol) {
            bounce.push_back(root);
          }
        }
      }
    }
    return bounce;
}

template<template<class, std::size_t, xt::layout_type> class T>
Array bounce_integral(std::vector<double> bouncel, std::vector<double> bouncer, shared_ptr<BoozerMagneticField<T>> field, double s,
    double theta0, double lam, int nfp, bool jpar, bool psidot, bool alphadot, bool ihat, bool khat, bool dkhatdalpha,
    bool tau, double step_size, double tol, bool adjust) {
    // bouncel = left bounce points in zeta
    // bouncer = guess for right bounce point in zeta
    // field: instance of BoozerMagneticField
    // s (double): normalized toroidal flux
    // theta0 (double): initial poloidal angle s.t. field line label is given
    //                  by theta0
    // lam (double): field line label = v_perp^2/(2*B*v^2)
    // nfp (int): number of field periods
    // step_size = Time step between observer calls in integration
    // tol = integration tolerance
    // adjust = if True, then the guess for the right bounce point, bouncer, is
    //          adjusted if another bounce point to the left is detected
    ///
    typedef std::vector< double > state_type;
    typedef boost::numeric::odeint::runge_kutta_dopri5< state_type > error_stepper_type;

    typename BoozerMagneticField<T>::Tensor2 point = xt::zeros<double>({1, 3});
    point(0,0) = s;

    field->set_points(point);
    double iota = field->iota_ref()(0);
    double G = field->G_ref()(0);
    double I = field->I_ref()(0);
    double jacfac = (G + iota*I);
    double dIdpsi = field->dIds_ref()(0)/field->psi0;
    double dGdpsi = field->dGds_ref()(0)/field->psi0;
    double diotadpsi = field->diotads_ref()(0)/field->psi0;

    std::function<double(double)> modBf = [theta0,iota,field,point](double zeta) mutable {
      point(0,1) = theta0 + iota * zeta;
      point(0,2) = zeta;
      field->set_points(point);
      auto modB = field->modB();
      return modB(0,0);
    };

    std::function<double(double)> dmodBdthetaf = [theta0,iota,field,point](double zeta) mutable {
      point(0,1) = theta0 + iota * zeta;
      point(0,2) = zeta;
      field->set_points(point);
      auto dmodBdtheta = field->dmodBdtheta();
      return dmodBdtheta(0,0);
    };

    std::function<double(double)> dmodBdzetaf = [theta0,iota,field,point](double zeta) mutable {
      point(0,1) = theta0 + iota * zeta;
      point(0,2) = zeta;
      field->set_points(point);
      auto dmodBdzeta = field->dmodBdzeta();
      return dmodBdzeta(0,0);
    };

    std::function<double(double)> dmodBdpsif = [theta0,iota,field,point](double zeta) mutable {
      point(0,1) = theta0 + iota * zeta;
      point(0,2) = zeta;
      field->set_points(point);
      auto dmodBdpsi = field->dmodBds()/field->psi0;
      return dmodBdpsi(0,0);
    };

    std::function<double(double)> Kf = [theta0,iota,field,point](double zeta) mutable {
      point(0,1) = theta0 + iota * zeta;
      point(0,2) = zeta;
      field->set_points(point);
      auto K = field->K();
      return K(0,0);
    };

    std::function<double(double)> dKdthetaf = [theta0,iota,field,point](double zeta) mutable {
      point(0,1) = theta0 + iota * zeta;
      point(0,2) = zeta;
      field->set_points(point);
      auto dKdtheta = field->dKdtheta();
      return dKdtheta(0,0);
    };

    std::function<double(double)> dKdzetaf = [theta0,iota,field,point](double zeta) mutable {
      point(0,1) = theta0 + iota * zeta;
      point(0,2) = zeta;
      field->set_points(point);
      auto dKdzeta = field->dKdzeta();
      return dKdzeta(0,0);
    };

    double zetamax, zetal;
    bool adjusted;

    std::function<void(const state_type&, const double)> factor_observer = [adjust,modBf,lam,&zetal,&zetamax,&adjusted](const state_type &x , const double t) mutable {
      if (adjust && (1 <= lam*modBf(t)) && (t <= zetamax) && (t > zetal)) {
          // Store smallest value of right bounce point
          zetamax = t;
          // Remember that we have reached a bounce point
          adjusted = true;
      };
    };

    std::function<void(const state_type&, state_type&, const double)> ihatf = [modBf,lam,jacfac,&zetamax](const state_type &x , state_type &dxdt , const double t) mutable {
        if ((1 >= lam*modBf(t)) && (t <= zetamax)) {
          dxdt[0] = std::sqrt(1 - lam*modBf(t))*jacfac/(modBf(t)*modBf(t));
        } else {
          dxdt[0] = 0;
        }
    };

    std::function<void(const state_type&, state_type&, const double)> jparf = [modBf,lam,jacfac,&zetamax](const state_type &x , state_type &dxdt , const double t) mutable {
      if ((1 >= lam*modBf(t)) && (t <= zetamax)) {
          dxdt[0] = std::sqrt(std::abs(1 - lam*modBf(t)))*jacfac/modBf(t);
        } else {
          dxdt[0] = 0;
        }
    };

    std::function<void(const state_type&, state_type&, const double)> dkhatdalphaf = [modBf,dmodBdthetaf,lam,jacfac,&zetamax](const state_type &x , state_type &dxdt , const double t) mutable {
      if ((1 >= lam*modBf(t)) && (t <= zetamax)) {
          dxdt[0] = std::sqrt(1 - lam*modBf(t))*dmodBdthetaf(t)*(-1.5*lam - 2*(1 - lam*modBf(t))/modBf(t))*jacfac/(modBf(t)*modBf(t));
        } else {
          dxdt[0] = 0;
        }
    };

    std::function<void(const state_type&, state_type&, const double)> alphadotf = [modBf,dmodBdthetaf,dmodBdzetaf,dmodBdpsif,Kf,dKdthetaf,dKdzetaf,G,I,dGdpsi,dIdpsi,diotadpsi,lam,jacfac,iota,&zetamax](const state_type &x , state_type &dxdt , const double t) mutable {
        auto modB = modBf(t);
        auto dmodBdtheta = dmodBdthetaf(t);
        auto dmodBdzeta = dmodBdzetaf(t);
        auto dmodBdpsi = dmodBdpsif(t);
        auto K = Kf(t);
        auto dKdtheta = dKdthetaf(t);
        auto dKdzeta = dKdzetaf(t);
        auto fac1 = K * (-iota*dmodBdtheta - dmodBdzeta) \
            + I * (-diotadpsi*t*dmodBdzeta + iota*dmodBdpsi) \
            + G * (dmodBdpsi + diotadpsi*t*dmodBdtheta);
        auto fac2 = -dIdpsi*iota - dGdpsi + dKdzeta + iota*dKdtheta;
        if ((1 > lam*modBf(t)) && (t < zetamax)) {
          dxdt[0] = (2 - lam*modB)*fac1/(std::sqrt(1 - lam*modB)*2*modB*modB) \
              + std::sqrt(1 - lam*modB)*fac2/modB;
        } else {
          dxdt[0] = 0;
        }
    };

    std::function<void(const state_type&, state_type&, const double)> psidotf = [modBf,dmodBdthetaf,dmodBdzetaf,G,I,lam,jacfac,&zetamax](const state_type &x , state_type &dxdt , const double t) mutable {
        auto modB = modBf(t);
        auto dmodBdtheta = dmodBdthetaf(t);
        auto dmodBdzeta = dmodBdzetaf(t);
        if ((1 > lam*modBf(t)) && (t < zetamax)) {
          dxdt[0] = (2 - lam*modB)*(I*dmodBdzeta-G*dmodBdtheta)/(std::sqrt(1 - lam*modB)*2*modB*modB);
        } else {
          dxdt[0] = 0;
        }
    };

    state_type x(1);
    // Now perform bounce integral for each pair of bounce points
    xt::xtensor<double, 2>::shape_type my_shape = {bouncel.size(), 9};
    xt::xarray<double> integrals = xt::zeros<double>(my_shape);
    double zetar;
    for (int i = 0; i < bouncel.size(); ++i) {
      zetal = bouncel[i];
      zetar = bouncer[i];
      zetamax = zetar;
      adjusted = false;
      if (jpar) {
        x[0] = 0;
        boost::numeric::odeint::integrate_adaptive(make_controlled< error_stepper_type >( tol , tol ), jparf, x, zetal, zetar, step_size, factor_observer);
        integrals(i,0) = x[0];
        zetar = zetamax;
      }
      if (psidot) {
        x[0] = 0;
        boost::numeric::odeint::integrate_adaptive(make_controlled< error_stepper_type >( tol , tol ), psidotf, x, zetal, zetar, step_size, factor_observer);
        integrals(i,1) = x[0];
        zetar = zetamax;
      }
      if (alphadot) {
        x[0] = 0;
        boost::numeric::odeint::integrate_adaptive(make_controlled< error_stepper_type >( tol , tol ), alphadotf, x, zetal, zetar, step_size, factor_observer);
        integrals(i,2) = x[0];
        zetar = zetamax;
      }
      if (ihat) {
        x[0] = 0;
        boost::numeric::odeint::integrate_adaptive(make_controlled< error_stepper_type >( tol , tol ), ihatf, x, zetal, zetar, step_size, factor_observer);
        integrals(i,3) = x[0];
        zetar = zetamax;
      }
      if (dkhatdalpha) {
        x[0] = 0;
        boost::numeric::odeint::integrate_adaptive(make_controlled< error_stepper_type >( tol , tol ), dkhatdalphaf, x, zetal, zetar, step_size, factor_observer);
        integrals(i,5) = x[0];
        zetar = zetamax;
      }
      integrals(i,7) = bouncel[i];
      integrals(i,8) = zetar;
    }
    return integrals;
}
