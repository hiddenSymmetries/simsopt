#include "boozerradialinterpolant.h"
#include <math.h>
#include "xtensor-python/pyarray.hpp"
typedef xt::pyarray<double> Array;
#include <xtensor/xview.hpp>
#include "simdhelpers.h"
#include <cstdio>
#include <iostream>
#include <string>
#include <xsimd/xsimd.hpp>

namespace xs = xsimd;
#define ANGLE_RECOMPUTE 5

void compute_kmnc_kmns(Array& kmnc, Array& kmns, Array& rmnc, Array& drmncds, Array& zmns, Array& dzmnsds,\
    Array& numns, Array& dnumnsds, Array& bmnc,\
    Array& rmns, Array& drmnsds, Array& zmnc, Array& dzmncds,\
    Array& numnc, Array& dnumncds, Array& bmns,\
    Array& iota, Array& G, Array& I, Array& xm, Array& xn, Array& thetas, Array& zetas) {

    std::size_t num_modes = rmnc.shape(1);
    std::size_t num_surf = rmnc.shape(0);
    std::size_t num_points = thetas.shape(0);

    constexpr std::size_t simd_size = xsimd::simd_type<double>::size;

    AlignedPaddedVec sin_angles(num_modes, 0.);
    AlignedPaddedVec cos_angles(num_modes, 0.);

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

    for (std::size_t ip=0; ip < num_points; ++ip) {
        double n_zeta = zetas(ip);
        simd_t theta(thetas(ip));
        simd_t zeta(n_zeta);
        

        for (std::size_t im=0; im < num_modes; im+=simd_size) {
            xs::batch<double, simd_size> b_sin, b_cos, b_xm, b_xn;

            b_xm = xs::load_aligned(&xm_array[im]);
            b_xn = xs::load_aligned(&xn_array[im]);

            sincos(xs::fms(b_xm, theta, b_xn*zeta), b_sin, b_cos);
            // sin_angles[im] = sin(xm_array[im]*theta-xn_array[im]*zeta);
            // cos_angles[im] = cos(xm_array[im]*theta-xn_array[im]*zeta);

            b_sin.store_aligned(&sin_angles[im]);
            b_cos.store_aligned(&cos_angles[im]);
        }

        for (std::size_t isurf=0; isurf < num_surf; ++isurf) {  
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
            double phi = n_zeta - nu;
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
            simd_t b_K(K);
            simd_t b_coe(2.*M_PI*M_PI);

            for (std::size_t im=0; im < num_modes; im+=simd_size) {
                xs::batch<double, simd_size> b_sin,b_cos, b_kmns, b_kmnc; 

                b_sin = xs::load_aligned(&sin_angles[im]);
                b_cos = xs::load_aligned(&cos_angles[im]);
                b_kmns = xs::load_aligned(&kmns_array[isurf*num_modes+im]);
                b_kmnc = xs::load_aligned(&kmnc_array[isurf*num_modes+im]);
                
                b_kmns = xs::fma(b_K, b_sin/b_coe, b_kmns);
                b_kmnc = xs::fma(b_K, b_cos/b_coe, b_kmnc);

                b_kmns.store_aligned(&kmns_array[isurf*num_modes+im]);
                b_kmnc.store_aligned(&kmnc_array[isurf*num_modes+im]);
            }
            kmnc_array[isurf*num_modes] = kmnc_array[isurf*num_modes] - K*cos_angles[0]/(4.*M_PI*M_PI);
        }
    }
}

void compute_kmns(Array& kmns, Array& rmnc, Array& drmncds, Array& zmns, Array& dzmnsds,\
    Array& numns, Array& dnumnsds, Array& bmnc, Array& iota, Array& G, Array& I,\
    Array& xm, Array& xn, Array& thetas, Array& zetas) {

    std::size_t num_modes = rmnc.shape(1);
    std::size_t num_surf = rmnc.shape(0);
    std::size_t num_points = thetas.shape(0);

    constexpr std::size_t simd_size = xsimd::simd_type<double>::size;

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

    AlignedPaddedVec sin_angles(num_modes, 0.);
    AlignedPaddedVec cos_angles(num_modes, 0.);

    for (std::size_t ip=0; ip < num_points; ++ip) {
        double n_zeta = zetas(ip);
        simd_t theta(thetas(ip));
        simd_t zeta(n_zeta);

        for (std::size_t im=0; im < num_modes; im+=simd_size) {
            xs::batch<double, simd_size> b_sin, b_cos, b_xm, b_xn;

            b_xm = xs::load_aligned(&xm_array[im]);
            b_xn = xs::load_aligned(&xn_array[im]);

            sincos(xs::fms(b_xm, theta, b_xn*zeta), b_sin, b_cos);

            b_sin.store_aligned(&sin_angles[im]);
            b_cos.store_aligned(&cos_angles[im]);
        }

        for (int isurf=0; isurf < num_surf; ++isurf) {    
            double B = 0.;
            double R = 0.;
            double dRdtheta = 0.;
            double dRdzeta = 0.;
            double dRds = 0.;
            double dZdtheta =  0.;
            double dZdzeta =  0.;
            double dZds =  0.;
            double nu = 0.;
            double dnuds = 0.;
            double dnudtheta = 0.;
            double dnudzeta =  0.;
            #pragma omp simd
            for (std::size_t im=0; im < num_modes; ++im) {
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
            double phi = n_zeta - nu;
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

            simd_t b_K(K);
            simd_t b_coe(2.*M_PI*M_PI);

            for (std::size_t im=0; im < num_modes; im+=simd_size) {
                xs::batch<double, simd_size> b_sin, b_kmns ; 

                b_sin = xs::load_aligned(&sin_angles[im]);
                b_kmns = xs::load_aligned(&kmns_array[isurf*num_modes+im]);
                
                b_kmns = xs::fma(b_K, b_sin/b_coe, b_kmns);

                b_kmns.store_aligned(&kmns_array[isurf*num_modes+im]);
                // kmns_array[isurf*num_modes+im] = kmns_array[isurf*num_modes+im] + K*sin_angles[im]/(2.*M_PI*M_PI); 
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
    Array& thetas, Array& zetas, int ntor, int nfp) {
    // K(ip) += kmns(im,ip)*sin(xm(im)*thetas(ip)-xn(im)*zetas(ip));
    std::size_t num_modes = xm.shape(0);
    std::size_t num_points = thetas.shape(0);
    std::size_t dim = kmns.dimension();

    constexpr std::size_t simd_size = xsimd::simd_type<double>::size;

    double* K_array = K.data();

    double* kmns_array = kmns.data();
    double* thetas_array = thetas.data();
    double* zetas_array = zetas.data();

    // std::size_t modes_s = (xm(0) || xn(0)) ? 0 : 1;

    if (num_points > 1) {
        #pragma omp parallel for
        for (int ip=0; ip < num_points; ip += simd_size){
            xs::batch<double, simd_size> b_kmns, b_thetas, b_zetas, b_K;
            b_thetas = xs::load_aligned(&thetas_array[ip]);
            b_zetas = xs::load_aligned(&zetas_array[ip]);
            b_K = xs::load_aligned(&K_array[ip]);
            
            simd_t sin_nfpzetas, cos_nfpzetas;
            simd_t sinterm, costerm;
            xs::sincos(-nfp*b_zetas, sin_nfpzetas, cos_nfpzetas);
            
            int num_ntor = ntor + 1;
            int m = xm(0);
            int n = xn(0);
            int i = 0;
            for (int im=0; im < num_modes; ++im) {
                b_kmns = xs::load_aligned(&kmns_array[im*num_points+ip]);
                
                // recompute the angle from scratch every so often, to
                // avoid accumulating floating point error
                if(i % ANGLE_RECOMPUTE == 0)
                    xs::sincos(m*b_thetas-n*b_zetas, sinterm, costerm);
                    
                b_K = xs::fma(b_kmns, sinterm, b_K);

                if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
                    simd_t sinterm_old = sinterm;
                    simd_t costerm_old = costerm;
                    sinterm = cos_nfpzetas * sinterm_old + costerm_old * sin_nfpzetas;
                    costerm = costerm_old * cos_nfpzetas - sinterm_old * sin_nfpzetas;
                }

                n += nfp;
                ++i;
                if (n > ntor * nfp) {
                    n = - ntor * nfp;
                    ++m;
                    i=0;
                }
            }
            b_K.store_aligned(&K_array[ip]);
        }
        // for (std::size_t im=modes_s; im < num_modes; ++im) {
        //     simd_t b_xm(xm(im));
        //     simd_t b_xn(xn(im));
        //     for (std::size_t ip=0; ip < num_points; ip += simd_size) {
        //         xs::batch<double, simd_size> b_kmns, b_thetas, b_zetas, b_K;
        //         b_thetas = xs::load_aligned(&thetas_array[ip]);
        //         b_zetas = xs::load_aligned(&zetas_array[ip]);
        //         b_K = xs::load_aligned(&K_array[ip]);
        //         b_kmns = xs::load_aligned(&kmns_array[im*num_points+ip]);

        //         b_K = xs::fma(b_kmns, sin(xs::fms(b_xm, b_thetas, b_xn*b_zetas)), b_K);
        //         b_K.store_aligned(&K_array[ip]);
        //     }
        // }
    } else {
        double* xm_array = xm.data();
        double* xn_array = xn.data();
        simd_t b_theta(thetas_array[0]);
        simd_t b_zeta(zetas_array[0]);
        simd_t b_K(0.);
        double res = 0;
        // there are some weird bugs associated with adding this
        // won't be able to get correct result if even and odd are
        // called consectively 
        // #pragma omp parallel private(b_K) reduction(+:res)
        {
            // #pragma omp for
            for (std::size_t im=0; im < num_modes; im += simd_size) {
                xs::batch<double, simd_size> b_xm, b_xn, b_kmns;
                b_xm = xs::load_aligned(&xm_array[im]);
                b_xn = xs::load_aligned(&xn_array[im]);
                b_kmns = xs::load_aligned(&kmns_array[im]);

                b_K = xs::fma(b_kmns, sin(xs::fms(b_xm, b_theta, b_xn*b_zeta)), b_K);
            }
            res = xs::hadd(b_K);
        }
        K_array[0] += res;
    }
}

void inverse_fourier_transform_even(Array& K, Array& kmns, Array& xm, Array& xn,
    Array& thetas, Array& zetas, int ntor, int nfp) {
    // K(ip) += kmns(im,ip)*cos(xm(im)*thetas(ip)-xn(im)*zetas(ip));
    int num_modes = xm.shape(0);
    int num_points = thetas.shape(0);
    int dim = kmns.dimension();

    constexpr std::size_t simd_size = xs::simd_type<double>::size;

    double* K_array = K.data();

    double* kmns_array = kmns.data();
    double* thetas_array = thetas.data();
    double* zetas_array = zetas.data();

    if (num_points > 1){
        #pragma omp parallel for
        for (int ip=0; ip < num_points; ip += simd_size){
            xs::batch<double, simd_size> b_kmns, b_thetas, b_zetas, b_K;
            b_thetas = xs::load_aligned(&thetas_array[ip]);
            b_zetas = xs::load_aligned(&zetas_array[ip]);
            b_K = xs::load_aligned(&K_array[ip]);
            
            simd_t sin_nfpzetas, cos_nfpzetas;
            simd_t sinterm, costerm;
            xs::sincos(-nfp*b_zetas, sin_nfpzetas, cos_nfpzetas);
            
            int num_ntor = ntor + 1;
            int m = xm(0);
            int n = xn(0);
            int i = 0;
            for (int im=0; im < num_modes; ++im) {
                b_kmns = xs::load_aligned(&kmns_array[im*num_points+ip]);
                
                // recompute the angle from scratch every so often, to
                // avoid accumulating floating point error
                if(i % ANGLE_RECOMPUTE == 0)
                    xs::sincos(m*b_thetas-n*b_zetas, sinterm, costerm);

                b_K = xs::fma(b_kmns, costerm, b_K);

                if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
                    simd_t sinterm_old = sinterm;
                    simd_t costerm_old = costerm;
                    sinterm = cos_nfpzetas * sinterm_old + costerm_old * sin_nfpzetas;
                    costerm = costerm_old * cos_nfpzetas - sinterm_old * sin_nfpzetas;
                }

                n += nfp;
                ++i;
                if (n > ntor * nfp) {
                    n = - ntor * nfp;
                    ++m;
                    i=0;
                }
            }
            b_K.store_aligned(&K_array[ip]);
        }
        // for (std::size_t im=0; im < num_modes; ++im) {
        //     simd_t b_xm(xm(im));
        //     simd_t b_xn(xn(im));
        //     for (std::size_t ip=0; ip < num_points; ip += simd_size) {
        //         xs::batch<double, simd_size> b_kmns, b_thetas, b_zetas, b_K;
        //         b_thetas = xs::load_aligned(&thetas_array[ip]);
        //         b_zetas = xs::load_aligned(&zetas_array[ip]);
        //         b_K = xs::load_aligned(&K_array[ip]);
        //         b_kmns = xs::load_aligned(&kmns_array[im*num_points+ip]);
                
        //         b_K = xs::fma(b_kmns, cos(xs::fms(b_xm, b_thetas, b_xn*b_zetas)), b_K);
        //         b_K.store_aligned(&K_array[ip]);
        //     }
        // }
    } else {
        double* xm_array = xm.data();
        double* xn_array = xn.data();
        simd_t b_theta(thetas_array[0]);
        simd_t b_zeta(zetas_array[0]);
        simd_t b_K(0.);
        double res = 0;
        // #pragma omp parallel private(b_K) reduction(+:res)
        {
            // #pragma omp for
            for (std::size_t im=0; im < num_modes; im += simd_size) {
                xs::batch<double, simd_size> b_xm, b_xn, b_kmns;
                b_xm = xs::load_aligned(&xm_array[im]);
                b_xn = xs::load_aligned(&xn_array[im]);
                b_kmns = xs::load_aligned(&kmns_array[im]);
                b_K = xs::fma(b_kmns, cos(xs::fms(b_xm, b_theta, b_xn*b_zeta)), b_K);
            }
            res = xs::hadd(b_K);
        }

        K_array[0] += res;
    }
}

int simd_alignment() {
    int alignment = xs::simd_type<double>::size * 8;
    return alignment;
}
