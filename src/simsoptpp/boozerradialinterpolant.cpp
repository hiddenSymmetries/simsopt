#include "boozerradialinterpolant.h"
#include <math.h>
#include "xtensor-python/pyarray.hpp"
#include <omp.h>
typedef xt::pyarray<double> Array;
#include <xtensor/xview.hpp>

void compute_kmnc_kmns(Array& kmnc, Array& kmns, Array& rmnc, Array& drmncds, Array& zmns, Array& dzmnsds,\
    Array& numns, Array& dnumnsds, Array& bmnc,\
    Array& rmns, Array& drmnsds, Array& zmnc, Array& dzmncds,\
    Array& numnc, Array& dnumncds, Array& bmns,\
    Array& iota, Array& G, Array& I, Array& xm, Array& xn, Array& thetas, Array& zetas) {

    int num_modes = rmnc.shape(1);
    int num_surf = rmnc.shape(0);
    int num_points = thetas.shape(0);
    double sin_angles[num_modes];
    double cos_angles[num_modes];

    double* kmnc_array = kmnc.data(); 
    double* kmns_array = kmns.data(); 

    double* rmnc_array = rmnc.data(); 
    double* drmncds_array = drmncds.data(); 
    double* zmns_array = zmns.data(); 
    double* dzmnsds_array = dzmnsds.data();
    double* numns_array = numns.data(); 
    double* dnumnsds_array = dnumnsds.data();
    double* bmnc_array = bmnc.data();
    double* rmns_array = rmns.data(); 
    double* drmnsds_array = drmnsds.data(); 
    double* zmnc_array = zmnc.data(); 
    double* dzmncds_array = dzmncds.data();
    double* numnc_array = numnc.data();
    double* dnumncds_array = dnumncds.data();
    double* bmns_array = bmns.data();

    double* iota_array = iota.data(); 
    double* G_array = G.data(); 
    double* I_array = I.data(); 
    double* xm_array = xm.data();
    double* xn_array = xn.data();
    double* thetas_array = thetas.data(); 
    double* zetas_array = zetas.data();

    // Outer iteration over surface
    #pragma omp parallel for private(sin_angles, cos_angles) reduction(+:kmns_array[:num_surf*num_modes], kmnc_array[:num_surf*num_modes])
    for (int ip=0; ip < num_points; ++ip) {
        double theta = thetas_array[ip];
        double zeta = zetas_array[ip];

        # pragma omp simd
        for (int im=0; im < num_modes; ++im) {
            sin_angles[im] = sin(xm_array[im]*theta-xn_array[im]*zeta);
            cos_angles[im] = cos(xm_array[im]*theta-xn_array[im]*zeta);
        }

        for (int isurf=0; isurf < num_surf; ++isurf) {  
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
            #pragma omp simd
            for (int im=0; im < num_modes; ++im) {
                B += bmnc_array[isurf*num_modes+im]*cos_angles[im] + bmns_array[isurf*num_modes+im]*sin_angles[im];
                R += rmnc_array[isurf*num_modes+im]*cos_angles[im] + rmns_array[isurf*num_modes+im]*sin_angles[im];
                dRdtheta += -rmnc_array[isurf*num_modes+im]*xm_array[im]*sin_angles[im] + rmns_array[isurf*num_modes+im]*xm_array[im]*cos_angles[im];
                dRdzeta  +=  rmnc_array[isurf*num_modes+im]*xn_array[im]*sin_angles[im] - rmns_array[isurf*num_modes+im]*xn_array[im]*cos_angles[im];
                dRds += drmncds_array[isurf*num_modes+im]*cos_angles[im] + drmnsds_array[isurf*num_modes+im]*sin_angles[im];
                dZdtheta += zmns_array[isurf*num_modes+im]*xm_array[im]*cos_angles[im] - zmnc_array[isurf*num_modes+im]*xm_array[im]*sin_angles[im];
                dZdzeta += -zmns_array[isurf*num_modes+im]*xn_array[im]*cos_angles[im] + zmnc_array[isurf*num_modes+im]*xn_array[im]*sin_angles[im];
                dZds += dzmnsds_array[isurf*num_modes+im]*sin_angles[im] + dzmncds_array[isurf*num_modes+im]*cos_angles[im];
                nu   += numns_array[isurf*num_modes+im]*sin_angles[im] + numnc_array[isurf*num_modes+im]*cos_angles[im];
                dnuds += dnumnsds_array[isurf*num_modes+im]*sin_angles[im] + dnumncds_array[isurf*num_modes+im]*cos_angles[im];
                dnudtheta += numns_array[isurf*num_modes+im]*xm_array[im]*cos_angles[im] - numnc_array[isurf*num_modes+im]*xm_array[im]*sin_angles[im];
                dnudzeta += -numns_array[isurf*num_modes+im]*xn_array[im]*cos_angles[im] + numnc_array[isurf*num_modes+im]*xn_array[im]*sin_angles[im];
            }
            double phi = zeta - nu;
            double dphids = - dnuds;
            double dphidtheta = - dnudtheta;
            double dphidzeta = 1. - dnudzeta;
            double dXdtheta = dRdtheta * cos(phi) - R * sin(phi) * dphidtheta;
            double dYdtheta = dRdtheta * sin(phi) + R * cos(phi) * dphidtheta;
            double dXds   = dRds   * cos(phi) - R * sin(phi) * dphids;
            double dYds   = dRds   * sin(phi) + R * cos(phi) * dphids;
            double dXdzeta  = dRdzeta  * cos(phi) - R * sin(phi) * dphidzeta;
            double dYdzeta  = dRdzeta  * sin(phi) + R * cos(phi) * dphidzeta;
            double gstheta = dXdtheta * dXds + dYdtheta * dYds + dZdtheta * dZds;
            double gszeta  = dXdzeta  * dXds + dYdzeta  * dYds + dZdzeta  * dZds;
            double sqrtg = (G_array[isurf] + iota_array[isurf]*I_array[isurf])/(B*B);
            double K = (gszeta + iota_array[isurf]*gstheta)/sqrtg;

            kmnc_array[isurf*num_modes] = kmnc_array[isurf*num_modes] + K*cos_angles[0]/(4.*M_PI*M_PI);

            #pragma omp simd
            for (int im=1; im < num_modes; ++im) {
                kmns_array[isurf*num_modes+im] = kmns_array[isurf*num_modes+im] + K*sin_angles[im]/(2.*M_PI*M_PI);
                kmnc_array[isurf*num_modes+im] = kmnc_array[isurf*num_modes+im] + K*cos_angles[im]/(2.*M_PI*M_PI);
            }
        }
    }
}

void compute_kmns(Array& kmns, Array& rmnc, Array& drmncds, Array& zmns, Array& dzmnsds,\
    Array& numns, Array& dnumnsds, Array& bmnc, Array& iota, Array& G, Array& I,\
    Array& xm, Array& xn, Array& thetas, Array& zetas) {

    int num_modes = rmnc.shape(1);
    int num_surf = rmnc.shape(0);
    int num_points = thetas.shape(0);

    double sin_angles[num_modes];
    double cos_angles[num_modes];

    double* kmns_array = kmns.data(); 

    double* rmnc_array = rmnc.data();
    double* drmncds_array = drmncds.data();
    double* zmns_array = zmns.data();
    double* dzmnsds_array = dzmnsds.data();
    double* numns_array = numns.data();
    double* dnumnsds_array = dnumnsds.data();
    double* bmnc_array = bmnc.data();

    double* iota_array = iota.data(); 
    double* G_array = G.data(); 
    double* I_array = I.data(); 
    double* xm_array = xm.data();
    double* xn_array = xn.data();
    double* thetas_array = thetas.data(); 
    double* zetas_array = zetas.data();

    // Outer iteration over surface
    #pragma omp parallel for private(sin_angles, cos_angles) reduction(+:kmns_array[:num_surf*num_modes])
    for (int ip=0; ip < num_points; ++ip) {
        double theta = thetas_array[ip];
        double zeta = zetas_array[ip];

        #pragma omp simd
        for (int im=0; im < num_modes; ++im) {
            sin_angles[im] = sin(xm_array[im]*theta-xn_array[im]*zeta);
            cos_angles[im] = cos(xm_array[im]*theta-xn_array[im]*zeta);
        }

        for (int isurf=0; isurf < num_surf; ++isurf) {    
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

            #pragma omp simd
            for (int im=0; im < num_modes; ++im) {
              B += bmnc_array[isurf*num_modes+im]*cos_angles[im];
              R += rmnc_array[isurf*num_modes+im]*cos_angles[im];
              dRdtheta += -rmnc_array[isurf*num_modes+im]*xm_array[im]*sin_angles[im];
              dRdzeta +=   rmnc_array[isurf*num_modes+im]*xn_array[im]*sin_angles[im];
              dRds += drmncds_array[isurf*num_modes+im]*cos_angles[im];
              dZdtheta += zmns_array[isurf*num_modes+im]*xm_array[im]*cos_angles[im];
              dZdzeta += -zmns_array[isurf*num_modes+im]*xn_array[im]*cos_angles[im];
              dZds += dzmnsds_array[isurf*num_modes+im]*sin_angles[im];
              nu += numns_array[isurf*num_modes+im]*sin_angles[im];
              dnuds += dnumnsds_array[isurf*num_modes+im]*sin_angles[im];
              dnudtheta += numns_array[isurf*num_modes+im]*xm_array[im]*cos_angles[im];
              dnudzeta += -numns_array[isurf*num_modes+im]*xn_array[im]*cos_angles[im];
            }

            double phi = zeta - nu;
            double dphids = - dnuds;
            double dphidtheta = - dnudtheta;
            double dphidzeta = 1. - dnudzeta;
            double dXdtheta = dRdtheta * cos(phi) - R * sin(phi) * dphidtheta;
            double dYdtheta = dRdtheta * sin(phi) + R * cos(phi) * dphidtheta;
            double dXds   = dRds   * cos(phi) - R * sin(phi) * dphids;
            double dYds   = dRds   * sin(phi) + R * cos(phi) * dphids;
            double dXdzeta  = dRdzeta  * cos(phi) - R * sin(phi) * dphidzeta;
            double dYdzeta  = dRdzeta  * sin(phi) + R * cos(phi) * dphidzeta;
            double gstheta = dXdtheta * dXds + dYdtheta * dYds + dZdtheta * dZds;
            double gszeta  = dXdzeta  * dXds + dYdzeta  * dYds + dZdzeta  * dZds;
            double sqrtg = (G_array[isurf] + iota_array[isurf]*I_array[isurf])/(B*B);
            double K = (gszeta + iota_array[isurf]*gstheta)/sqrtg;

            #pragma omp simd
            for (int im=1; im < num_modes; ++im) {
                kmns_array[isurf*num_modes+im] = kmns_array[isurf*num_modes+im] + K*sin_angles[im]/(2.*M_PI*M_PI);
            }
        }
    }
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

int omp_num_threads() {
   int num;
   #pragma omp parallel
  {
	#pragma omp single
	num = omp_get_num_threads();
  }
  return num;
}
