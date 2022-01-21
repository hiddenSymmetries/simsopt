// using namespace boost::numeric::odeint;
#include "boozermagneticfield.h"
#include "bounce.h"
#include <vector>
using Vec = std::vector<double>;
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
#include "xtensor-python/pytensor.hpp"     // Numpy bindings
#include <xtensor/xio.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xsort.hpp>
// #define BOOST_MATH_INSTRUMENT
#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint/integrate/integrate.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4_controller.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4_dense_output.hpp>
#include <boost/numeric/odeint.hpp>

#include <tuple>
// using boost::math::tools::toms748_solve;
using namespace boost::numeric::odeint;

template
std::vector<double> find_bounce_points<xt::pytensor>(
    shared_ptr<BoozerMagneticField<xt::pytensor>> field, double s,
    double theta0, double zeta0, int nzeta, double lam, int nfp, int nmax, int digits, int option,
    double derivative_tol, double argmin_tol, double root_tol);

template
Array bounce_integral<xt::pytensor>(std::vector<double> bouncel, std::vector<double> bouncer,
        shared_ptr<BoozerMagneticField<xt::pytensor>> field, double s,
            double theta0, double lam, int nfp, int ntransitmax, bool jpar,
            bool psidot, bool alphadot, bool ihat, bool khat, bool dkhatdalpha, bool tau,
            double step_size, double tol, double dt_max, bool adjust);

template
double vprime<xt::pytensor>(shared_ptr<BoozerMagneticField<xt::pytensor>> field, double s,
    double theta0, int nzeta, int nfp, int nmax, double step_size, int digits);

template<template<class, std::size_t, xt::layout_type> class T>
double vprime(shared_ptr<BoozerMagneticField<T>> field, double s,
    double theta0, int nzeta, int nfp, int nmax, double step_size, int digits) {

      typename BoozerMagneticField<T>::Tensor2 points0({{s, 0, 0}});
      field->set_points(points0);

      double iota = field->iota_ref()(0);
      double G = field->G_ref()(0);
      double I = field->I_ref()(0);
      double jacfac = (G + iota*I);

      typename BoozerMagneticField<T>::Tensor2 point = xt::zeros<double>({1, 3});
      point(0,0) = s;

      typedef std::vector< double > state_type;

      std::function<void(const state_type&, state_type&, const double)> vprimef = [jacfac,field,point,theta0,iota](const state_type &x , state_type &dxdt , const double t) mutable {
          point(0,1) = theta0 + iota * t;
          point(0,2) = t;
          field->set_points(point);
          auto modB = field->modB();
          dxdt[0] = jacfac/(modB(0,0)*modB(0,0));
      };

      state_type x(1);
      x[0] = 0;
      double zetaend = 2*M_PI*nmax/nfp;
      boost::numeric::odeint::integrate(vprimef, x, 0., zetaend, step_size);
      return x[0];
}


template<template<class, std::size_t, xt::layout_type> class T>
std::vector<double> find_bounce_points(shared_ptr<BoozerMagneticField<T>> field, double s,
    double theta0, double zeta0, int nzeta, double lam, int nfp, int nmax, int digits, int option,
    double derivative_tol, double argmin_tol, double root_tol) {
      ///
      // option: 0 = left, 1 = right, 2 = max
      // derivative_tol: tolerance for derivative
      ///

    typename BoozerMagneticField<T>::Tensor2 point = xt::zeros<double>({1, 3});
    point(0,0) = s;
    field->set_points(point);
    double iota = field->iota()(0,0);

    // Set up grid in zeta - note that grid points at zeta = 0 and zeta = 2*pi*nmax/nfp
    // are included so that we can diagnose points that lie in the last interval
    Vec zeta(nzeta, 0.);
    for (int i = 0; i < nzeta; ++i) {
      zeta[i] = i*2*M_PI*nmax/(nfp*(nzeta-1)) + zeta0;
    }

    typename BoozerMagneticField<T>::Tensor2 points = xt::zeros<double>({nzeta, 3});
    for (int i = 0; i < nzeta; ++i) {
      points(i, 0) = s;
      points(i, 1) = theta0 + iota * (zeta[i] - zeta0);
      points(i, 2) = zeta[i];
    }

    // Compute modB on this grid
    field->set_points(points);
    auto modB = field->modB();

    std::vector<int> bounce_try;
    if (option != 2) {
      for (int i = 0; i < nzeta-1; ++i) {
        // Find points such that modB brackets 1/lam on either side and
        // left point has larger modB, right points has smaller modB
        // This yields the set of potential left bounce points.
        if (option == 0) {
          if (modB(i,0) > 1/lam && modB(i+1,0) < 1/lam) {
            bounce_try.push_back(i);
          }
        } else if (option == 1) {
          if (modB(i,0) < 1/lam && modB(i+1,0) > 1/lam) {
            bounce_try.push_back(i);
          }
        }
      }
    }

    // The above does not work well if looking for a bounce point near the maxima
    // Instead try performing the root solve near the locations where
    // |modB - 1/lam| is minimized
    auto argmin = xt::argwhere(xt::abs(1 - lam*modB) < argmin_tol);
    // std::cout << "argmin.shape(): " << argmin.size() << std::endl;
    // std::cout << "argmin: " << argmin << std::endl;
    int argmin_left, argmin_right, index_left, index_right;
    for (int i = 0; i < argmin.size(); ++i) {
      if (argmin[i][0] == nzeta - 1) {
          argmin_left = argmin[i][0]-1;
          argmin_right = 1;
          // For the case of = , there are two potential left points to add
          // One does not need to be considered since it is equivalent to adding i = 0
          if (std::abs(1 - lam*modB(argmin_left,0)) <= std::abs(1 - lam*modB(argmin_right,0))) {
              if (std::count(bounce_try.begin(),bounce_try.end(),argmin_left)==0) {
                bounce_try.push_back(argmin_left);
              }
          }
          // For case of > , there is one potential left point to add but does
          // not need to be added since it is equivalent to adding i = 0
      } else if (argmin[i][0] == 0) {
          argmin_left = nzeta - 2;
          argmin_right = argmin[i][0]+1;
          if (std::abs(1 - lam*modB(argmin_left,0)) >= std::abs(1 - lam*modB(argmin_right,0))) {
            if (std::count(bounce_try.begin(),bounce_try.end(),argmin[i][0])==0) {
              bounce_try.push_back(argmin[i][0]);
            }
          }
      } else {
        argmin_left = argmin[i][0]-1;
        argmin_right = argmin[i][0]+1;
        if (std::abs(1 - lam*modB(argmin_left,0)) >= std::abs(1 - lam*modB(argmin_right,0))) {
          if (std::count(bounce_try.begin(),bounce_try.end(),argmin[i][0])==0) {
            bounce_try.push_back(argmin[i][0]);
          }
        }
        if (std::abs(1 - lam*modB(argmin_left,0)) <= std::abs(1 - lam*modB(argmin_right,0))) {
          if (std::count(bounce_try.begin(),bounce_try.end(),argmin_left)==0) {
            bounce_try.push_back(argmin_left);
          }
        }
      }

    }

    // Function handle for modB along field line
    std::function<double(double)> modBf = [iota,theta0,zeta0,field,point](double zeta) mutable {
      point(0,1) = theta0 + iota * (zeta - zeta0);
      point(0,2) = zeta;
      field->set_points(point);
      auto modB = field->modB();
      return modB(0,0);
    };

    // Function handle for derivative of modB along field line
    std::function<double(double)> modBprimef = [iota,theta0,zeta0,field,point](double zeta) mutable {
      point(0,1) = theta0 + iota * (zeta - zeta0);
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
    for (int ir = 0; ir < bounce_try.size(); ++ir)  {
      double zetal = zeta[bounce_try[ir]];
      double zetar = zeta[bounce_try[ir]+1];
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
    double theta0, double lam, int nfp, int ntransitmax,
    bool jpar, bool psidot, bool alphadot, bool ihat, bool khat, bool dkhatdalpha,
    bool tau, double step_size, double tol, double dt_max, bool adjust) {

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
      // std::cout << "theta0: " << theta0 << std::endl;
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
      // Initial guess for right bounce point
      // if (i < bouncel.size()-1) {
      //   zetar = bouncel[i+1];
      // } else {
      //   zetar = bouncel[i] + 2.*M_PI*ntransitmax/nfp;
      // }
      zetamax = zetar;
      adjusted = false;
      if (jpar) {
        x[0] = 0;
        // auto stepper = make_dense_output( tol, tol, runge_kutta_dopri5< state_type> () );
        // stepper.initialize(x, zetal, step_size);
        // while ( ( stepper.current_time() < zetar )) {
        //     factor_observer( stepper.current_state(), stepper.current_time() );
        //     stepper.do_step(jparf);
        //     if (stepper.current_time_step() > dt_max ) {
        //       stepper.initialize(stepper.current_state(), stepper.current_time(), dt_max);
        //     }
        // }
        // x = stepper.current_state();
        boost::numeric::odeint::integrate_adaptive(make_controlled< error_stepper_type >( tol , tol ), jparf, x, zetal, zetar, step_size, factor_observer);
        // if (adjusted) {
        integrals(i,0) = x[0];
        zetar = zetamax;
        // }
      }
      if (psidot) {
        x[0] = 0;
        boost::numeric::odeint::integrate_adaptive(make_controlled< error_stepper_type >( tol , tol ), psidotf, x, zetal, zetar, step_size, factor_observer);
        // if (adjusted) {
        integrals(i,1) = x[0];
        zetar = zetamax;
        // }
      }
      if (alphadot) {
        x[0] = 0;
        boost::numeric::odeint::integrate_adaptive(make_controlled< error_stepper_type >( tol , tol ), alphadotf, x, zetal, zetar, step_size, factor_observer);
        // if (adjusted) {
        integrals(i,2) = x[0];
        zetar = zetamax;
        // }
      }
      if (ihat) {
        x[0] = 0;
        boost::numeric::odeint::integrate_adaptive(make_controlled< error_stepper_type >( tol , tol ), ihatf, x, zetal, zetar, step_size, factor_observer);
        // if (adjusted) {
        integrals(i,3) = x[0];
        zetar = zetamax;
        // }
      }
      // if (khat) {
      //   x[0] = 0;
      //   boost::numeric::odeint::integrate_adaptive(make_controlled< error_stepper_type >( tol , tol ), khatf, x, zetal, zetar, step_size, factor_observer);
      //   if (adjusted) {
      //     integrals(i,4) = x[0];
      //     zetar = zetamax;
      //   }
      // }
      if (dkhatdalpha) {
        x[0] = 0;
        boost::numeric::odeint::integrate_adaptive(make_controlled< error_stepper_type >( tol , tol ), dkhatdalphaf, x, zetal, zetar, step_size, factor_observer);
        // if (adjusted) {
        integrals(i,5) = x[0];
        zetar = zetamax;
        // }
      }
      // if (tau) {
      //   x[0] = 0;
      //   boost::numeric::odeint::integrate_adaptive(make_controlled< error_stepper_type >( tol , tol ), tauf, x, zetal, zetar, step_size, factor_observer);
      //   if (adjusted) {
      //     integrals(i,6) = x[0];
      //     zetar = zetamax;
      //   }
      // }
      integrals(i,7) = bouncel[i];
      integrals(i,8) = zetar;
    }
    return integrals;
}


    // typedef rosenbrock4< double > error_stepper_type;
    // typedef runge_kutta_fehlberg78< state_type > error_stepper_type;
    // typedef controlled_runge_kutta< error_stepper_type > controlled_stepper_type;
    // controlled_stepper_type controlled_stepper;
    // integrate_adaptive( controlled_stepper , harmonic_oscillator , x , 0.0 , 10.0 , 0.01 );

    // typename BoozerMagneticField<T>::Tensor2 points0({{s, 0, 0}});
    // field->set_points(points0);


    // Vec zeta(nzeta, 0.);
    // for (int i = 0; i < nzeta; ++i) {
    //   zeta[i] = i*2*M_PI*nmax/(nfp*(nzeta-1));
    // }
    //
    // typename BoozerMagneticField<T>::Tensor2 points = xt::zeros<double>({nzeta, 3});
    // for (int i = 0; i < nzeta; ++i) {
    //   points(i, 0) = s;
    //   points(i, 1) = theta0 + iota * zeta[i];
    //   points(i, 2) = zeta[i];
    // }

    // field->set_points(points);
    // auto modB = field->modB();

    // Find points such that modB brackets 1/lam on either side and
    // left point has larger modB, right points has smaller modB
    // This yields the set of potential left bounce points.
    // Perform analogous operation for right bounce points.
    // std::vector<int> bouncel, bouncer;
    // for (int i = 0; i < nzeta-1; ++i) {
    //   if (modB(i,0) > 1/lam && modB(i+1,0) < 1/lam) {
    //     bouncel.push_back(i);
    //   }
    //   if (modB(i,0) < 1/lam && modB(i+1,0) > 1/lam) {
    //     bouncer.push_back(i);
    //   }
    // }



    // std::function<double(double)> rootf = [modBf,lam](double zeta) mutable {
    //   return modBf(zeta) - 1/lam;
    // };

    // uintmax_t rootmaxit = 200;
    // typedef typename boost::numeric::odeint::result_of::make_dense_output<runge_kutta_dopri5<State>>::type dense_stepper_type;

    // dense_stepper_type dense = make_dense_output(, tol, step_size, runge_kutta_dopri5<State>());
    // double t = 0;
    // double dt = 1e-3 * dtmax; // initial guess for first timestep, will be adjusted by adaptive timestepper
    // dense.initialize(y, t, dt);

    // std::function<std::tuple<double,double>(double)> rootf = [modBf,modBprimef,lam](double zeta) mutable {
    //   return std::make_tuple(modBf(zeta) - 1/lam,modBprimef(zeta));
    // };

    // Iterate over points bracketing potential left points. Peform root solve
    // to find left bounce points.
    // std::vector<double> indexl_try;
    // for (int ir = 0; ir < bouncel.size(); ++ir)  {
    //   double zetal = zeta[bouncel[ir]];
    //   double zetar = zeta[bouncel[ir]+1];
    //   if ((modBf(zetal)-1/lam < 0) || (modBf(zetar)-1/lam > 0) || (zetal > zetar)) {
    //     std::cout << "incorrect" << std::endl;
    //   }
    //   auto guess = 0.5*(zetal + zetar);
    //   // auto result = boost::math::tools::bisect(rootf, zetal, zetar, roottol);
    //   // auto root = 0.5*(result.first + result.second);
    //   auto root = boost::math::tools::newton_raphson_iterate(rootf, guess, zetal, zetar, digits);
    //   if ((root > zetar) || (root < zetal)) {
    //     std::cout << "incorrect2" << std::endl;
    //   }
    //   // Check that field is decreasing along field line
    //   // Only consider left bounce points
    //   if (modBprimef(root) <= 0) {
    //     indexl_try.push_back(root);
    //   }
    // }

    // // Perform same operation for right bounce points
    // std::vector<double> indexr_try;
    // for (int ir = 0; ir < bouncer.size(); ++ir)  {
    //   double zetal = zeta[bouncer[ir]];
    //   double zetar = zeta[bouncer[ir]+1];
    //   if ((modBf(zetal)-1/lam > 0) || (modBf(zetar)-1/lam < 0) || (zetal > zetar)) {
    //     std::cout << "incorrect" << std::endl;
    //   }
    //   auto guess = 0.5*(zetal + zetar);
    //   // auto result = boost::math::tools::bisect(rootf, zetal, zetar, roottol);
    //   // auto root = 0.5 * (result.first + result.second);
    //   auto root = boost::math::tools::newton_raphson_iterate(rootf, guess, zetal, zetar, digits);
    //   if ((root > zetar) || (root < zetal)) {
    //     std::cout << "incorrect2" << std::endl;
    //   }
    //   // Check that field is increasing along field line
    //   // Only consider right bounce points to the right of first left bounce
    //   // point
    //   if (modBprimef(root) >= 0 && indexl_try.size() > 0) {
    //     if (root > indexl_try[0]) {
    //       indexr_try.push_back(root);
    //     }
    //   }
    // }
    // // Remove last left bounce point if to right of last right bounce point
    // if (indexl_try.size() > 0 && indexr_try.size() > 0) {
    //   if (indexl_try.back() >= indexr_try.back()) {
    //     indexl_try.pop_back();
    //   }
    // }
    //
    // if (indexr_try.size() == 0) {
    //   indexl_try.clear();
    // }

  //   std::vector<double> indexr, indexl;
  //   while (indexl_try.size() > 0) {
  //     if (indexr_try.size() > 0) {
  //       if (indexr_try[0] > indexl_try[0]) {
  //         if (indexl_try.size() > 1) {
  //           // Remove extra left bounce point
  //           if (indexr_try[0] >= indexl_try[1]) {
  //             indexl_try.erase(indexl_try.begin());
  //           } else {
  //             // Correct order found. Add to list.
  //             indexl.push_back(indexl_try[0]);
  //             indexl_try.erase(indexl_try.begin());
  //             indexr.push_back(indexr_try[0]);
  //             indexr_try.erase(indexr_try.begin());
  //           }
  //       } else {
  //         // Correct order found. Add to list.
  //         indexl.push_back(indexl_try[0]);
  //         indexl_try.erase(indexl_try.begin());
  //         indexr.push_back(indexr_try[0]);
  //         indexr_try.erase(indexr_try.begin());
  //       }
  //     } else {
  //       // Remove extra right bounce point
  //       indexr_try.erase(indexr_try.begin());
  //     }
  //   } else {
  //     // No right bounce point so remove left bounce point
  //     indexl_try.erase(indexl_try.begin());
  //   }
  // }

    // The two vectors should now be the same size
    // if (indexl.size() != indexr.size()) {
    //   std::cout << "incorrect!" << std::endl;
    //   std::cout << indexl.size() << std::endl;
    //   std::cout << indexr.size() << std::endl;
    // } else {
    //   // And in the correct order
    //   for (int i = 0; i < indexl.size(); ++i) {
    //     if (indexr[i] <= indexl[i]) {
    //       std::cout << "incorrect!" << std::endl;
    //     }
    //     if (i + 1 < indexl.size()) {
    //       if (indexr[i] >= indexl[i+1]) {
    //         std::cout << "incorrect!" << std::endl;
    //       }
    //     }
    //   }
    // }

    /* The type of container used to hold the state vector */

    // std::function<void(const vector_type&, vector_type&, const double)> ihatf = [modBf,lam,jacfac](const vector_type &x , vector_type &dxdt , const double t) mutable {
    //     if (1 > lam*modBf(t)) {
    //       dxdt[0] = std::sqrt(1 - lam*modBf(t))*jacfac/(modBf(t)*modBf(t));
    //     } else {
    //       dxdt[0] = 0;
    //     }
    // };
    //
    // std::function<void(const vector_type&, matrix_type&, const double&, vector_type&)> ihatjacf = [modBf,lam,jacfac](const vector_type &x , matrix_type &J , const double &t, vector_type &dfdt) mutable {
    //     if (1 > lam*modBf(t)) {
    //       J( 0, 0 ) = - lam/(2*modBf(t)*modBf(t)*std::sqrt(1 - lam*modBf(t))) \
    //         - 2*std::sqrt(1 - lam * modBf(t))/(modBf(t)*modBf(t)*modBf(t));
    //     } else {
    //       J( 0, 0) = 0;
    //     }
    //     dfdt[0] = 0;
    // };

    // factor to set to zero if we have past the bounce point
    // double factor = 1.;
    // double zetamax, indexl;
    // bool adjusted;
    //
    // std::function<void(const state_type&, const double)> factor_observer = [modBf,modBprimef,lam,&indexl,&zetamax,&adjusted](const state_type &x , const double t) mutable {
    //   if ((1 <= lam*modBf(t)) && (t <= zetamax) && (t > indexl)) {
    //       // std::cout << "indexl: " << indexl << std::endl;
    //       // std::cout << "t: " << t <<  std::endl;
    //       // std::cout << "t - indexl: " << (t - indexl) << std::endl;
    //       // std::cout << "lam*modBf(t) - 1: " << (lam*modBf(t) - 1) << std::endl;
    //       // std::cout << "modBprime(indexl): " << modBprimef(indexl) << std::endl;
    //       // std::cout << "modBprime(t): " << modBprimef(t) << std::endl;
    //       // Store smallest value of right bounce point
    //       zetamax = t;
    //       // Remember that we have reached a bounce point
    //       adjusted = true;
    //     };
    // };
    //
    // std::function<void(const state_type&, state_type&, const double)> ihatf = [modBf,lam,jacfac,&zetamax](const state_type &x , state_type &dxdt , const double t) mutable {
    //     // std::cout << "t <= zetamax: " << (t <= zetamax) << std::endl;
    //     // std::cout << "1 >= lam*modBf(t): " << (1 >= lam*modBf(t)) << std::endl;
    //     if ((1 >= lam*modBf(t)) && (t <= zetamax)) {
    //       // std::cout << "inside ihat" << std::endl;
    //       // std::cout << "inside dkhatdalpha" << std::endl;
    //       dxdt[0] = std::sqrt(1 - lam*modBf(t))*jacfac/(modBf(t)*modBf(t));
    //     } else {
    //       dxdt[0] = 0;
    //     }
    // };
    //
    // std::function<void(const state_type&, state_type&, const double)> jparf = [modBf,lam,jacfac,&zetamax](const state_type &x , state_type &dxdt , const double t) mutable {
    //   if ((1 >= lam*modBf(t)) && (t <= zetamax)) {
    //       dxdt[0] = std::sqrt(1 - lam*modBf(t))*jacfac/modBf(t);
    //     } else {
    //       dxdt[0] = 0;
    //     }
    // };
    //
    // std::function<void(const state_type&, state_type&, const double)> dkhatdalphaf = [modBf,dmodBdthetaf,lam,jacfac,&zetamax](const state_type &x , state_type &dxdt , const double t) mutable {
    //   // std::cout << "t <= zetamax: " << (t <= zetamax) << std::endl;
    //   // std::cout << "1 >= lam*modBf(t): " << (1 >= lam*modBf(t)) << std::endl;
    //   if ((1 >= lam*modBf(t)) && (t <= zetamax)) {
    //       // std::cout << "inside dkhatdalpha" << std::endl;
    //       dxdt[0] = std::sqrt(1 - lam*modBf(t))*dmodBdthetaf(t)*(-1.5*lam - 2*(1 - lam*modBf(t))/modBf(t))*jacfac/(modBf(t)*modBf(t));
    //     } else {
    //       dxdt[0] = 0;
    //     }
    // };
    //
    // std::function<void(const state_type&, state_type&, const double)> alphadotf = [modBf,dmodBdthetaf,dmodBdzetaf,dmodBdpsif,Kf,dKdthetaf,dKdzetaf,G,I,dGdpsi,dIdpsi,diotadpsi,lam,jacfac,iota,&zetamax](const state_type &x , state_type &dxdt , const double t) mutable {
    //     auto modB = modBf(t);
    //     auto dmodBdtheta = dmodBdthetaf(t);
    //     auto dmodBdzeta = dmodBdzetaf(t);
    //     auto dmodBdpsi = dmodBdpsif(t);
    //     auto K = Kf(t);
    //     auto dKdtheta = dKdthetaf(t);
    //     auto dKdzeta = dKdzetaf(t);
    //     auto fac1 = K * (-iota*dmodBdtheta - dmodBdzeta) \
    //         + I * (-diotadpsi*t*dmodBdzeta + iota*dmodBdpsi) \
    //         + G * (dmodBdpsi + diotadpsi*t*dmodBdtheta);
    //     auto fac2 = -dIdpsi*iota - dGdpsi + dKdzeta + iota*dKdtheta;
    //     if ((1 >= lam*modBf(t)) && (t <= zetamax)) {
    //       dxdt[0] = (2 - lam*modB)*fac1/(std::sqrt(1 - lam*modB)*2*modB*modB) \
    //           + std::sqrt(1 - lam*modB)*fac2/modB;
    //     } else {
    //       dxdt[0] = 0;
    //     }
    // };
    //
    // std::function<void(const state_type&, state_type&, const double)> psidotf = [modBf,dmodBdthetaf,dmodBdzetaf,G,I,lam,jacfac,&zetamax](const state_type &x , state_type &dxdt , const double t) mutable {
    //     auto modB = modBf(t);
    //     auto dmodBdtheta = dmodBdthetaf(t);
    //     auto dmodBdzeta = dmodBdzetaf(t);
    //     if ((1 >= lam*modBf(t)) && (t <= zetamax)) {
    //       dxdt[0] = (2 - lam*modB)*(I*dmodBdzeta-G*dmodBdtheta)/(std::sqrt(1 - lam*modB)*2*modB*modB);
    //     } else {
    //       dxdt[0] = 0;
    //     }
    // };
    //
    // // vector_type x(1);
    // // state_type xs(1);
    // state_type x(1);
    // // Now perform bounce integral for each pair of bounce points
    // xt::xtensor<double, 2>::shape_type my_shape = {indexl_try.size(), 9};
    // xt::xarray<double> integrals = xt::zeros<double>(my_shape);
    // // std::cout << "indexl.size(): " << indexl.size() << std::endl;
    // double indexr;
    // for (int i = 0; i < indexl_try.size(); ++i) {
    //   integrals(i,7) = indexl_try[i];
    //   indexl = indexl_try[i];
    //   if (i < indexl_try.size()-1) {
    //     indexr = indexl_try[i+1];
    //   } else {
    //     indexr = indexl_try[i] + 2.*M_PI*ntransitmax/nfp;
    //   }
    //   // std::cout << "i: " << i << std::endl;
    //   // std::cout << "indexr start: " << indexr << std::endl;
    //   zetamax = indexr;
    //   adjusted = false;
    //   if (jpar) {
    //     x[0] = 0;
    //     // tanh_sinh<double> integrator;
    //     // boost::numeric::odeint::integrate(jparf, x, indexl[i], indexr[i], step_size);
    //     // integrate_adaptive(make_dense_output< rosenbrock4< double > >( 1.0e-6 , 1.0e-6 ), jparf, x, indexl[i], indexr[i], step_size);
    //     integrals(i,0) = x[0];
    //   }
    //   if (psidot) {
    //     x[0] = 0;
    //     // boost::numeric::odeint::integrate(psidotf, x, indexl[i], indexr[i], step_size);
    //     // integrate_adaptive(make_dense_output< rosenbrock4< double > >( 1.0e-6 , 1.0e-6 ), psidotf, x, indexl[i], indexr[i], step_size);
    //     // integrate_adaptive(make_controlled< error_stepper_type >( 1.0e-10 , 1.0e-6 ), psidotf, x, indexl[i], indexr[i], step_size);
    //     integrals(i,1) = x[0];
    //   }
    //   if (alphadot) {
    //     x[0] = 0;
    //     // boost::numeric::odeint::integrate(alphadotf, x, indexl[i], indexr[i], step_size);
    //     // integrate_adaptive(make_dense_output< rosenbrock4< double > >( 1.0e-6 , 1.0e-6 ), alphadotf, x, indexl[i], indexr[i], step_size);
    //     // boost::numeric::odeint::integrate_adaptive(make_controlled< error_stepper_type >( 1.0e-10 , 1.0e-6 ), alphadotf, x, indexl[i], indexr[i], step_size);
    //     integrals(i,2) = x[0];
    //   }
    //   if (ihat) {
    //     x[0] = 0;
    //     // boost::numeric::odeint::integrate(ihatf, x, indexl[i], indexr[i], step_size);
    //     // integrate_adaptive(make_dense_output< rosenbrock4< double > >( 1.0e-4 , 1.0e-4 ),
    //     //     make_pair(ihatf, ihatjacf), x, indexl[i], indexr[i], step_size);
    //     boost::numeric::odeint::integrate_adaptive(make_controlled< error_stepper_type >( tol , tol ), ihatf, x, indexl, indexr, step_size, factor_observer);
    //     if (adjusted) {
    //       integrals(i,3) = x[0];
    //       indexr = zetamax;
    //     }
    //
    //   }
    //   // if (khat) {
    //   //   boost::numeric::odeint::integrate(khatf, x, indexl[i], indexr[i], step_size);
    //   //   integrals(i,4) = x[0];
    //   // }
    //   if (dkhatdalpha) {
    //     x[0] = 0;
    //     // boost::numeric::odeint::integrate(dkhatdalphaf, x, indexl[i], indexr[i], step_size);
    //     // integrate_adaptive(make_dense_output< rosenbrock4< double > >( 1.0e-6 , 1.0e-6 ), dkhatdalphaf, x, indexl[i], indexr[i], step_size);
    //     boost::numeric::odeint::integrate_adaptive(make_controlled< error_stepper_type >( tol , tol ), dkhatdalphaf, x, indexl, indexr, step_size, factor_observer);
    //     // make sure the right bounce point was obtained
    //     if (adjusted) {
    //       integrals(i,5) = x[0];
    //       indexr = zetamax;
    //     }
    //   }
    //   // std::cout << "indexr after: " << indexr << std::endl;
    //   integrals(i,8) = indexr;
    //   // std::cout << "indexl: " << indexl << std::endl;
    //   // std::cout << "indexr: " << indexr << std::endl;
    //   // std::cout << "adjusted: " << adjusted << std::endl;
    //   // std::cout << "integrals(1,3): " << integrals(i,3) << std::endl;
    //   // std::cout << "integrals(1,5): " << integrals(i,5) << std::endl;
    //
    //   // if (tau) {
    //   //   boost::numeric::odeint::integrate(tauf, x, indexl[i], indexr[i], step_size);
    //   //   integrals(i,6) = x[0];
    //   // }
    // }
    // return integrals;
