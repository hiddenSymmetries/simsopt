#include "boozerradialinterpolant.h"
#include <math.h>
#include "xtensor-python/pyarray.hpp"
typedef xt::pyarray<double> Array;
#include <xtensor/xview.hpp>

Array compute_kmnc_kmns(Array& rmnc, Array& drmncds, Array& zmns, Array& dzmnsds,
    Array& numns, Array& dnumnsds, Array& bmnc,
    Array& rmns, Array& drmnsds, Array& zmnc, Array& dzmncds,
    Array& numnc, Array& dnumncds, Array& bmns,
    Array& iota, Array& G, Array& I, Array& xm, Array& xn, Array& thetas, Array& zetas) {

    int num_modes = rmnc.shape(0);
    int num_surf = rmnc.shape(1);
    int num_points = thetas.shape(0);

    Array kmnc_kmns = xt::zeros<double>({2,num_modes,num_surf});

    // Outer iteration over surface
    #pragma omp parallel for
    for (int isurf=0; isurf < num_surf; ++isurf) {
      for (int ip=0; ip < num_points; ++ip) {
        double B = 0.;
        double R = 0.;
        double dRdtheta = 0.;
        double dRdzeta = 0.;
        double dRds = 0.;
        double dZdtheta = 0.;
        double dZdzeta = 0.;
        double dZds = 0.;
        double nu = 0.;
        double dnuds = 0.;
        double dnudtheta = 0.;
        double dnudzeta = 0.;
        for (int im=0; im < num_modes; ++im) {
          double angle = xm(im)*thetas(ip)-xn(im)*zetas(ip);
          B += bmnc(im,isurf)*cos(angle) + bmns(im,isurf)*sin(angle);
          R += rmnc(im,isurf)*cos(angle) + rmns(im,isurf)*sin(angle);
          dRdtheta += -rmnc(im,isurf)*xm(im)*sin(angle) + rmns(im,isurf)*xm(im)*cos(angle);
          dRdzeta  +=  rmnc(im,isurf)*xn(im)*sin(angle) - rmns(im,isurf)*xn(im)*cos(angle);
          dRds += drmncds(im,isurf)*cos(angle) + drmnsds(im,isurf)*sin(angle);
          dZdtheta += zmns(im,isurf)*xm(im)*cos(angle) - zmnc(im,isurf)*xm(im)*sin(angle);
          dZdzeta += -zmns(im,isurf)*xn(im)*cos(angle) + zmnc(im,isurf)*xn(im)*sin(angle);
          dZds += dzmnsds(im,isurf)*sin(angle) + dzmncds(im,isurf)*cos(angle);
          nu   += numns(im,isurf)*sin(angle) + numnc(im,isurf)*cos(angle);
          dnuds += dnumnsds(im,isurf)*sin(angle) + dnumncds(im,isurf)*cos(angle);
          dnudtheta += numns(im,isurf)*xm(im)*cos(angle) - numnc(im,isurf)*xm(im)*sin(angle);
          dnudzeta += -numns(im,isurf)*xn(im)*cos(angle) + numnc(im,isurf)*xn(im)*sin(angle);
        }
        double phi = zetas(ip) - nu;
        double dphids = - dnuds;
        double dphidtheta = - dnudtheta;
        double dphidzeta = 1 - dnudzeta;
        double dXdtheta = dRdtheta * cos(phi) - R * sin(phi) * dphidtheta;
        double dYdtheta = dRdtheta * sin(phi) + R * cos(phi) * dphidtheta;
        double dXds   = dRds   * cos(phi) - R * sin(phi) * dphids;
        double dYds   = dRds   * sin(phi) + R * cos(phi) * dphids;
        double dXdzeta  = dRdzeta  * cos(phi) - R * sin(phi) * dphidzeta;
        double dYdzeta  = dRdzeta  * sin(phi) + R * cos(phi) * dphidzeta;
        double gstheta = dXdtheta * dXds + dYdtheta * dYds + dZdtheta * dZds;
        double gszeta  = dXdzeta  * dXds + dYdzeta  * dYds + dZdzeta  * dZds;
        double sqrtg = (G(isurf) + iota(isurf)*I(isurf))/(B*B);
        double K = (gszeta + iota(isurf)*gstheta)/sqrtg;

        for (int im=0; im < num_modes; ++im) {
          double angle = xm(im)*thetas(ip)-xn(im)*zetas(ip);
          if (im > 0) {
            kmnc_kmns(1,im,isurf) = kmnc_kmns(1,im,isurf) + K*sin(angle)/(2.*M_PI*M_PI);
            kmnc_kmns(0,im,isurf) = kmnc_kmns(0,im,isurf) + K*cos(angle)/(2.*M_PI*M_PI);
          } else {
            kmnc_kmns(0,im,isurf) = kmnc_kmns(0,im,isurf) + K*cos(angle)/(4.*M_PI*M_PI);
          }
        }
      }
    }
    return kmnc_kmns;
}

Array compute_kmns(Array& rmnc, Array& drmncds, Array& zmns, Array& dzmnsds, Array& numns, Array& dnumnsds, Array& bmnc, Array& iota, Array& G, Array& I, Array& xm, Array& xn, Array& thetas, Array& zetas) {

    int num_modes = rmnc.shape(0);
    int num_surf = rmnc.shape(1);
    int num_points = thetas.shape(0);

    Array kmns = xt::zeros<double>({num_modes,num_surf});

    // Outer iteration over surface
    #pragma omp parallel for
    for (int isurf=0; isurf < num_surf; ++isurf) {
      for (int ip=0; ip < num_points; ++ip) {
        double B = 0.;
        double R = 0.;
        double dRdtheta = 0.;
        double dRdzeta = 0.;
        double dRds = 0.;
        double dZdtheta = 0.;
        double dZdzeta = 0.;
        double dZds = 0.;
        double nu = 0.;
        double dnuds = 0.;
        double dnudtheta = 0.;
        double dnudzeta = 0.;
        for (int im=0; im < num_modes; ++im) {
          B += bmnc(im,isurf)*cos(xm(im)*thetas(ip)-xn(im)*zetas(ip));
          R += rmnc(im,isurf)*cos(xm(im)*thetas(ip)-xn(im)*zetas(ip));
          dRdtheta += -rmnc(im,isurf)*xm(im)*sin(xm(im)*thetas(ip)-xn(im)*zetas(ip));
          dRdzeta +=   rmnc(im,isurf)*xn(im)*sin(xm(im)*thetas(ip)-xn(im)*zetas(ip));
          dRds += drmncds(im,isurf)*cos(xm(im)*thetas(ip)-xn(im)*zetas(ip));
          dZdtheta += zmns(im,isurf)*xm(im)*cos(xm(im)*thetas(ip)-xn(im)*zetas(ip));
          dZdzeta += -zmns(im,isurf)*xn(im)*cos(xm(im)*thetas(ip)-xn(im)*zetas(ip));
          dZds += dzmnsds(im,isurf)*sin(xm(im)*thetas(ip)-xn(im)*zetas(ip));
          nu += numns(im,isurf)*sin(xm(im)*thetas(ip)-xn(im)*zetas(ip));
          dnuds += dnumnsds(im,isurf)*sin(xm(im)*thetas(ip)-xn(im)*zetas(ip));
          dnudtheta += numns(im,isurf)*xm(im)*cos(xm(im)*thetas(ip)-xn(im)*zetas(ip));
          dnudzeta += -numns(im,isurf)*xn(im)*cos(xm(im)*thetas(ip)-xn(im)*zetas(ip));
        }
        double phi = zetas(ip) - nu;
        double dphids = - dnuds;
        double dphidtheta = - dnudtheta;
        double dphidzeta = 1 - dnudzeta;
        double dXdtheta = dRdtheta * cos(phi) - R * sin(phi) * dphidtheta;
        double dYdtheta = dRdtheta * sin(phi) + R * cos(phi) * dphidtheta;
        double dXds   = dRds   * cos(phi) - R * sin(phi) * dphids;
        double dYds   = dRds   * sin(phi) + R * cos(phi) * dphids;
        double dXdzeta  = dRdzeta  * cos(phi) - R * sin(phi) * dphidzeta;
        double dYdzeta  = dRdzeta  * sin(phi) + R * cos(phi) * dphidzeta;
        double gstheta = dXdtheta * dXds + dYdtheta * dYds + dZdtheta * dZds;
        double gszeta  = dXdzeta  * dXds + dYdzeta  * dYds + dZdzeta  * dZds;
        double sqrtg = (G(isurf) + iota(isurf)*I(isurf))/(B*B);
        double K = (gszeta + iota(isurf)*gstheta)/sqrtg;

        for (int im=1; im < num_modes; ++im) {
          kmns(im,isurf) = kmns(im,isurf) + K*sin(xm(im)*thetas(ip)-xn(im)*zetas(ip))/(2.*M_PI*M_PI);
        }
      }
    }
    return kmns;
}

Array fourier_transform_odd(Array& K, Array& xm, Array& xn, Array& thetas, Array& zetas) {

    int num_modes = xm.shape(0);
    int num_points = thetas.shape(0);
    Array kmns = xt::zeros<double>({num_modes});

    double norm;
    for (int im=1; im < num_modes; ++im) {
      norm = 0.;
      #pragma omp parallel for
      for (int ip=0; ip < num_points; ++ip) {
        kmns(im) += K(ip)*sin(xm(im)*thetas(ip)-xn(im)*zetas(ip));
        norm += pow(sin(xm(im)*thetas(ip)-xn(im)*zetas(ip)),2.);
      }
      kmns(im) = kmns(im)/norm;
    }
    return kmns;
}

Array fourier_transform_even(Array& K, Array& xm, Array& xn, Array& thetas, Array& zetas) {

    int num_modes = xm.shape(0);
    int num_points = thetas.shape(0);
    Array kmns = xt::zeros<double>({num_modes});

    double norm;
    for (int im=0; im < num_modes; ++im) {
      norm = 0.;
      #pragma omp parallel for
      for (int ip=0; ip < num_points; ++ip) {
        kmns(im) += K(ip)*cos(xm(im)*thetas(ip)-xn(im)*zetas(ip));
        norm += pow(cos(xm(im)*thetas(ip)-xn(im)*zetas(ip)),2.);
      }
      kmns(im) = kmns(im)/norm;
    }
    return kmns;
}

void inverse_fourier_transform_odd(Array& K, Array& kmns, Array& xm, Array& xn, Array& thetas, Array& zetas) {

    int num_modes = xm.shape(0);
    int num_points = thetas.shape(0);
    int dim = kmns.dimension();
    if (dim==2) {
      for (int im=1; im < num_modes; ++im) {
        #pragma omp parallel for
        for (int ip=0; ip < num_points; ++ip) {
          K(ip) += kmns(im,ip)*sin(xm(im)*thetas(ip)-xn(im)*zetas(ip));
        }
      }
    } else {
      for (int im=1; im < num_modes; ++im) {
        #pragma omp parallel for
        for (int ip=0; ip < num_points; ++ip) {
          K(ip) += kmns(im)*sin(xm(im)*thetas(ip)-xn(im)*zetas(ip));
        }
      }
    }
}

void inverse_fourier_transform_even(Array& K, Array& kmns, Array& xm, Array& xn, Array& thetas, Array& zetas) {

    int num_modes = xm.shape(0);
    int num_points = thetas.shape(0);
    int dim = kmns.dimension();
    if (dim==2) {
      for (int im=0; im < num_modes; ++im) {
        #pragma omp parallel for
        for (int ip=0; ip < num_points; ++ip) {
          K(ip) += kmns(im,ip)*cos(xm(im)*thetas(ip)-xn(im)*zetas(ip));
        }
      }
    } else {
      for (int im=0; im < num_modes; ++im) {
        #pragma omp parallel for
        for (int ip=0; ip < num_points; ++ip) {
          K(ip) += kmns(im)*cos(xm(im)*thetas(ip)-xn(im)*zetas(ip));
        }
      }
    }
}
