#include "boozerradialinterpolant.h"
#include <math.h>
#include "xtensor-python/pyarray.hpp"
typedef xt::pyarray<double> Array;
#include <xtensor/xview.hpp>

Array compute_kmnc_kmns(Array& rmnc, Array& drmncds, Array& zmns, Array& dzmnsds,\
    Array& numns, Array& dnumnsds, Array& bmnc,\
    Array& rmns, Array& drmnsds, Array& zmnc, Array& dzmncds,\
    Array& numnc, Array& dnumncds, Array& bmns,\
    double iota, double G, double I, Array& xm, Array& xn, Array& thetas, Array& zetas) {

    int num_modes = rmnc.shape(0);
    int num_points = thetas.shape(0);

    Array kmnc_kmns = xt::zeros<double>({2,num_modes});

    // Outer iteration over surface
    #pragma omp parallel for
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
          B += bmnc(im)*cos(angle) + bmns(im)*sin(angle);
          R += rmnc(im)*cos(angle) + rmns(im)*sin(angle);
          dRdtheta += -rmnc(im)*xm(im)*sin(angle) + rmns(im)*xm(im)*cos(angle);
          dRdzeta  +=  rmnc(im)*xn(im)*sin(angle) - rmns(im)*xn(im)*cos(angle);
          dRds += drmncds(im)*cos(angle) + drmnsds(im)*sin(angle);
          dZdtheta += zmns(im)*xm(im)*cos(angle) - zmnc(im)*xm(im)*sin(angle);
          dZdzeta += -zmns(im)*xn(im)*cos(angle) + zmnc(im)*xn(im)*sin(angle);
          dZds += dzmnsds(im)*sin(angle) + dzmncds(im)*cos(angle);
          nu   += numns(im)*sin(angle) + numnc(im)*cos(angle);
          dnuds += dnumnsds(im)*sin(angle) + dnumncds(im)*cos(angle);
          dnudtheta += numns(im)*xm(im)*cos(angle) - numnc(im)*xm(im)*sin(angle);
          dnudzeta += -numns(im)*xn(im)*cos(angle) + numnc(im)*xn(im)*sin(angle);
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
        double sqrtg = (G + iota*I)/(B*B);
        double K = (gszeta + iota*gstheta)/sqrtg;

        for (int im=0; im < num_modes; ++im) {
          double angle = xm(im)*thetas(ip)-xn(im)*zetas(ip);
          if (im > 0) {
            kmnc_kmns(1,im) = kmnc_kmns(1,im) + K*sin(angle)/(2.*M_PI*M_PI);
            kmnc_kmns(0,im) = kmnc_kmns(0,im) + K*cos(angle)/(2.*M_PI*M_PI);
          } else {
            kmnc_kmns(0,im) = kmnc_kmns(0,im) + K*cos(angle)/(4.*M_PI*M_PI);
          }
        }
    }
    return kmnc_kmns;
}

Array compute_kmns(Array& rmnc, Array& drmncds, Array& zmns, Array& dzmnsds,\
    Array& numns, Array& dnumnsds, Array& bmnc, double iota, double G, double I,\
    Array& xm, Array& xn, Array& thetas, Array& zetas) {

    int num_modes = rmnc.shape(0);
    int num_points = thetas.shape(0);

    Array kmns = xt::zeros<double>({num_modes});

    // Outer iteration over surface
    #pragma omp parallel for
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
            B += bmnc(im)*cos(xm(im)*thetas(ip)-xn(im)*zetas(ip));
            R += rmnc(im)*cos(xm(im)*thetas(ip)-xn(im)*zetas(ip));
            dRdtheta += -rmnc(im)*xm(im)*sin(xm(im)*thetas(ip)-xn(im)*zetas(ip));
            dRdzeta +=   rmnc(im)*xn(im)*sin(xm(im)*thetas(ip)-xn(im)*zetas(ip));
            dRds += drmncds(im)*cos(xm(im)*thetas(ip)-xn(im)*zetas(ip));
            dZdtheta += zmns(im)*xm(im)*cos(xm(im)*thetas(ip)-xn(im)*zetas(ip));
            dZdzeta += -zmns(im)*xn(im)*cos(xm(im)*thetas(ip)-xn(im)*zetas(ip));
            dZds += dzmnsds(im)*sin(xm(im)*thetas(ip)-xn(im)*zetas(ip));
            nu += numns(im)*sin(xm(im)*thetas(ip)-xn(im)*zetas(ip));
            dnuds += dnumnsds(im)*sin(xm(im)*thetas(ip)-xn(im)*zetas(ip));
            dnudtheta += numns(im)*xm(im)*cos(xm(im)*thetas(ip)-xn(im)*zetas(ip));
            dnudzeta += -numns(im)*xn(im)*cos(xm(im)*thetas(ip)-xn(im)*zetas(ip));
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
        double sqrtg = (G + iota*I)/(B*B);
        double K = (gszeta + iota*gstheta)/sqrtg;

        for (int im=1; im < num_modes; ++im) {
            kmns(im) = kmns(im) + K*sin(xm(im)*thetas(ip)-xn(im)*zetas(ip))/(2.*M_PI*M_PI);
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

void inverse_fourier_transform_odd(Array& K, Array& kmns, Array& xm, Array& xn,
    Array& thetas, Array& zetas) {

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

void inverse_fourier_transform_even(Array& K, Array& kmns, Array& xm, Array& xn,
    Array& thetas, Array& zetas) {

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
