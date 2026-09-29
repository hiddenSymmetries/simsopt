#include "surfacerzfourier.h"
#include "simdhelpers.h"

// Optimization notes:
// We use two "tricks" in this part of the code to speed up some of the functions.
// 1) We use SIMD instructions to parallelise across the angle theta.
// 2) This parametrization requires the evaluation of
//          sin(m*theta-n*nfp*phi) and cos(m*theta-n*nfp*phi)
//    for many values of n and m. Since trigonometric functions are expensive,
//    we want to avoid lots of calls to sin and cos. Instead, we use the rules
//          sin(a + b) = cos(b) sin(a) + cos(a) sin(b)
//          cos(a + b) = cos(a) cos(b) - sin(a) sin(b)
//    to write
//          sin(m*theta-(n+1)*nfp*phi) = cos(-nfp*phi) * sin(m*theta-n*nfp*phi) + cos(m*theta-n*nfp*phi) * sin(-nfp*phi)
//          cos(m*theta-(n+1)*nfp*phi) = cos(m*theta-n*nfp*phi) * cos(-nfp*phi) + sin(m*theta-n*nfp*phi) * sin(-nfp*phi)
//    In our code we loop over n. So we start with n=-ntor, and then we always
//    just increase the angle by -nfp*phi.

#define ANGLE_RECOMPUTE 5

#if defined(USE_XSIMD)

template<class Array>
void SurfaceRZFourier<Array>::gamma_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    int numquadpoints_phi = quadpoints_phi.size();
    int numquadpoints_theta = quadpoints_theta.size();
    constexpr int simd_size = xsimd::simd_type<double>::size;
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for(int k2 = 0; k2 < numquadpoints_theta; k2 += simd_size) {
            // batches have no per-lane operator[] anymore; gather into a small
            // contiguous buffer first, then load it into a batch in one shot.
            alignas(xs::default_arch::alignment()) double theta_arr[simd_size] = {};
            for (int l = 0; l < simd_size; ++l) {
                if(k2 + l >= numquadpoints_theta)
                    break;
                theta_arr[l] = 2*M_PI * quadpoints_theta[k2+l];
            }
            simd_t theta = xs::load_aligned(theta_arr);
            simd_t r(0.);
            simd_t z(0.);
            double sin_nfpphi = sin(-nfp*phi);
            double cos_nfpphi = cos(-nfp*phi);
            for (int m = 0; m <= mpol; ++m) {
                simd_t sinterm, costerm;
                for (int i = 0; i < 2*ntor+1; ++i) {
                    int n  = i - ntor;
                    // recompute the angle from scratch every so often, to
                    // avoid accumulating floating point error
                    if(i % ANGLE_RECOMPUTE == 0)
                        { auto sincos_result = xsimd::sincos(m*theta-n*nfp*phi); sinterm = sincos_result.first; costerm = sincos_result.second; }
                    r += rc(m, i) * costerm;
                    if(!stellsym) {
                        r += rs(m, i) * sinterm;
                        z += zc(m, i) * costerm;
                    }
                    z += zs(m, i) * sinterm;
                    if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
                        simd_t sinterm_old = sinterm;
                        simd_t costerm_old = costerm;
                        sinterm = cos_nfpphi * sinterm_old + costerm_old * sin_nfpphi;
                        costerm = costerm_old * cos_nfpphi - sinterm_old * sin_nfpphi;
                    }
                }
            }
            auto x = r * cos(phi);
            auto y = r * sin(phi);
            alignas(xs::default_arch::alignment()) double x_arr[simd_size], y_arr[simd_size], z_arr[simd_size];
            x.store_aligned(x_arr);
            y.store_aligned(y_arr);
            z.store_aligned(z_arr);
            for (int l = 0; l < simd_size; ++l) {
                if(k2 + l >= numquadpoints_theta)
                    break;
                data(k1, k2+l, 0) = x_arr[l];
                data(k1, k2+l, 1) = y_arr[l];
                data(k1, k2+l, 2) = z_arr[l];
            }
        }
    }
}

#else

template<class Array>
void SurfaceRZFourier<Array>::gamma_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    int numquadpoints_phi = quadpoints_phi.size();
    int numquadpoints_theta = quadpoints_theta.size();
    constexpr int simd_size = 1;
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        auto phi  = 2*M_PI*quadpoints_phi[k1];
        for(int k2 = 0; k2 < numquadpoints_theta; k2 += simd_size) {
            auto theta = 2*M_PI * quadpoints_theta[k2];
            double r(0.);
            double z(0.);
            double sin_nfpphi = sin(-nfp*phi);
            double cos_nfpphi = cos(-nfp*phi);
            for (int m = 0; m <= mpol; ++m) {
                double sinterm, costerm;
                for (int i = 0; i < 2*ntor+1; ++i) {
                    int n  = i - ntor;
                    // recompute the angle from scratch every so often, to
                    // avoid accumulating floating point error
                    if(i % ANGLE_RECOMPUTE == 0) {
                        sinterm = sin(m*theta - n*nfp*phi);
                        costerm = cos(m*theta - n*nfp*phi);
                    }
                    r += rc(m, i) * costerm;
                    if(!stellsym) {
                        r += rs(m, i) * sinterm;
                        z += zc(m, i) * costerm;
                    }
                    z += zs(m, i) * sinterm;
                    if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
                        auto sinterm_old = sinterm;
                        auto costerm_old = costerm;
                        sinterm = cos_nfpphi * sinterm_old + costerm_old * sin_nfpphi;
                        costerm = costerm_old * cos_nfpphi - sinterm_old * sin_nfpphi;
                    }
                }
            }
            auto x = r * cos(phi);
            auto y = r * sin(phi);
            data(k1, k2, 0) = x;
            data(k1, k2, 1) = y;
            data(k1, k2, 2) = z;
        }
    }
}

#endif

template<class Array>
void SurfaceRZFourier<Array>::gamma_lin(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    int numquadpoints = quadpoints_phi.size();

#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        double theta  = 2*M_PI*quadpoints_theta[k1];
        double r = 0;
        double z = 0;
        for (int m = 0; m <= mpol; ++m) {
            for (int i = 0; i < 2*ntor+1; ++i) {
                int n  = i - ntor;
                r += rc(m, i) * cos(m*theta-n*nfp*phi);
                if(!stellsym) {
                    r += rs(m, i) * sin(m*theta-n*nfp*phi);
                    z += zc(m, i) * cos(m*theta-n*nfp*phi);
                }
                z += zs(m, i) * sin(m*theta-n*nfp*phi);
            }
        }
        data(k1, 0) = r * cos(phi);
        data(k1, 1) = r * sin(phi);
        data(k1, 2) = z;
    }
}

#if defined(USE_XSIMD)

template<class Array>
void SurfaceRZFourier<Array>::gammadash1_impl(Array& data) {
    constexpr int simd_size = xsimd::simd_type<double>::size;
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for(int k2 = 0; k2 < numquadpoints_theta; k2 += simd_size) {
            // batches have no per-lane operator[] anymore; gather into a small
            // contiguous buffer first, then load it into a batch in one shot.
            alignas(xs::default_arch::alignment()) double theta_arr[simd_size] = {};
            for (int l = 0; l < simd_size; ++l) {
                if(k2 + l >= numquadpoints_theta)
                    break;
                theta_arr[l] = 2*M_PI * quadpoints_theta[k2+l];
            }
            simd_t theta = xs::load_aligned(theta_arr);
            simd_t r(0.);
            simd_t rd(0.);
            simd_t zd(0.);
            double sin_nfpphi = sin(-nfp*phi);
            double cos_nfpphi = cos(-nfp*phi);
            for (int m = 0; m <= mpol; ++m) {
                simd_t sinterm, costerm;
                for (int i = 0; i < 2*ntor+1; ++i) {
                    int n  = i - ntor;
                     // recompute the angle from scratch every so often, to
                     // avoid accumulating floating point error
                    if(i % ANGLE_RECOMPUTE == 0)
                        { auto sincos_result = xsimd::sincos(m*theta-n*nfp*phi); sinterm = sincos_result.first; costerm = sincos_result.second; }
                    r  += rc(m, i) * costerm;
                    rd += rc(m, i) * (n*nfp) * sinterm;
                    if(!stellsym) {
                        r  += rs(m, i) * sinterm;
                        rd += rs(m, i) * (-n*nfp)*costerm;
                        zd += zc(m, i) * (n*nfp)*sinterm;
                    }
                    zd += zs(m, i) * (-n*nfp)*costerm;
                    if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
                        simd_t sinterm_old = sinterm;
                        simd_t costerm_old = costerm;
                        sinterm = cos_nfpphi * sinterm_old + costerm_old * sin_nfpphi;
                        costerm = costerm_old * cos_nfpphi - sinterm_old * sin_nfpphi;
                    }
                }
            }
            auto xd = 2*M_PI*(rd * cos(phi) - r * sin(phi));
            auto yd = 2*M_PI*(rd * sin(phi) + r * cos(phi));
            zd *= 2*M_PI;
            alignas(xs::default_arch::alignment()) double xd_arr[simd_size], yd_arr[simd_size], zd_arr[simd_size];
            xd.store_aligned(xd_arr);
            yd.store_aligned(yd_arr);
            zd.store_aligned(zd_arr);
            for (int l = 0; l < simd_size; ++l) {
                if(k2 + l >= numquadpoints_theta)
                    break;
                data(k1, k2+l, 0) = xd_arr[l];
                data(k1, k2+l, 1) = yd_arr[l];
                data(k1, k2+l, 2) = zd_arr[l];
            }
        }
    }
}

#else

template<class Array>
void SurfaceRZFourier<Array>::gammadash1_impl(Array& data) {
    constexpr int simd_size = 1;
    #pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for(int k2 = 0; k2 < numquadpoints_theta; k2 += simd_size) {
            double theta = 2*M_PI * quadpoints_theta[k2];
            double r(0.);
            double rd(0.);
            double zd(0.);
            double sin_nfpphi = sin(-nfp*phi);
            double cos_nfpphi = cos(-nfp*phi);
            for (int m = 0; m <= mpol; ++m) {
                double sinterm, costerm;
                for (int i = 0; i < 2*ntor+1; ++i) {
                    int n  = i - ntor;
                     // recompute the angle from scratch every so often, to
                     // avoid accumulating floating point error
                    if(i % ANGLE_RECOMPUTE == 0){
                        sinterm = sin(m*theta-n*nfp*phi);
                        costerm = cos(m*theta-n*nfp*phi);
                    }
                    r  += rc(m, i) * costerm;
                    rd += rc(m, i) * (n*nfp) * sinterm;
                    if(!stellsym) {
                        r  += rs(m, i) * sinterm;
                        rd += rs(m, i) * (-n*nfp)*costerm;
                        zd += zc(m, i) * (n*nfp)*sinterm;
                    }
                    zd += zs(m, i) * (-n*nfp)*costerm;
                    if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
                        auto sinterm_old = sinterm;
                        auto costerm_old = costerm;
                        sinterm = cos_nfpphi * sinterm_old + costerm_old * sin_nfpphi;
                        costerm = costerm_old * cos_nfpphi - sinterm_old * sin_nfpphi;
                    }
                }
            }
            auto xd = 2*M_PI*(rd * cos(phi) - r * sin(phi));
            auto yd = 2*M_PI*(rd * sin(phi) + r * cos(phi));
            zd *= 2*M_PI;
            data(k1, k2, 0) = xd;
            data(k1, k2, 1) = yd;
            data(k1, k2, 2) = zd;
        }
    }
}

#endif

template<class Array>
void SurfaceRZFourier<Array>::gammadash1dash1_impl(Array& data) {
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            double r = 0;
            double rd = 0;
            double rdd = 0;
            double zdd = 0;
            for (int i = 0; i < 2*ntor+1; ++i) {
                int n  = i - ntor;
                for (int m = 0; m <= mpol; ++m) {
                    r  += rc(m, i) * cos(m*theta-n*nfp*phi);
                    rd += -rc(m, i) * (-n*nfp) * sin(m*theta-n*nfp*phi);
                    rdd += -rc(m, i) * (-n*nfp) * (-n*nfp) * cos(m*theta-n*nfp*phi);
                    if(!stellsym) {
                        r  += rs(m, i) * sin(m*theta-n*nfp*phi);
                        rd += rs(m, i) * (-n*nfp)*cos(m*theta-n*nfp*phi);
                        rdd += -rs(m, i) * (-n*nfp)*(-n*nfp)*sin(m*theta-n*nfp*phi);
                        zdd += -zc(m, i) * (-n*nfp)*(-n*nfp)*cos(m*theta-n*nfp*phi);
                    }
                    zdd += -zs(m, i) * (-n*nfp)*(-n*nfp)*sin(m*theta-n*nfp*phi);
                }
            }
            data(k1, k2, 0) = 4*M_PI*M_PI*(rdd * cos(phi) - 2 * rd * sin(phi) - r * cos(phi));
            data(k1, k2, 1) = 4*M_PI*M_PI*(rdd * sin(phi) + 2 * rd * cos(phi) - r * sin(phi));
            data(k1, k2, 2) = 4*M_PI*M_PI*zdd;
        }
    }
}

template<class Array>
void SurfaceRZFourier<Array>::gammadash1dash2_impl(Array& data) {
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            double rd2 = 0;
            double rd1d2 = 0;
            double zd1d2 = 0;
            for (int i = 0; i < 2*ntor+1; ++i) {
                int n  = i - ntor;
                for (int m = 0; m <= mpol; ++m) {
                    rd2 += -rc(m, i) * (m) * sin(m*theta-n*nfp*phi);
                    rd1d2 += -rc(m, i) * (-n*nfp) * (m) * cos(m*theta-n*nfp*phi);
                    if(!stellsym) {
                        rd2 += rs(m, i) * (m) * cos(m*theta-n*nfp*phi);
                        rd1d2 += - rs(m, i) * (-n*nfp) * (m) * sin(m*theta-n*nfp*phi);
                        zd1d2 += -zc(m, i) * (-n*nfp) * m * cos(m*theta-n*nfp*phi);
                    }
                    zd1d2 += -zs(m, i) * (-n*nfp) * (m) * sin(m*theta-n*nfp*phi);
                }
            }
            data(k1, k2, 0) = 4*M_PI*M_PI*(rd1d2 * cos(phi) - rd2 * sin(phi));
            data(k1, k2, 1) = 4*M_PI*M_PI*(rd1d2 * sin(phi) + rd2 * cos(phi));
            data(k1, k2, 2) = 4*M_PI*M_PI*zd1d2;
        }
    }
}

template<class Array>
void SurfaceRZFourier<Array>::gammadash2dash2_impl(Array& data) {
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            double rd = 0;
            double zd = 0;
            double rdd = 0;
            double zdd = 0;
            for (int i = 0; i < 2*ntor+1; ++i) {
                int n  = i - ntor;
                for (int m = 0; m <= mpol; ++m) {
                    rd += -rc(m, i) * (m) * sin(m*theta-n*nfp*phi);
                    rdd += -rc(m, i) * (m) * (m) * cos(m*theta-n*nfp*phi);
                    if(!stellsym) {
                        rd +=    rs(m, i) * m * cos(m*theta-n*nfp*phi);
                        rdd +=  -rs(m, i) * m * m * sin(m*theta-n*nfp*phi);
                        zd +=  -zc(m, i) * m * sin(m*theta-n*nfp*phi);
                        zdd += -zc(m, i) * m * m * cos(m*theta-n*nfp*phi);
                    }
                    zd += zs(m, i) * m * cos(m*theta-n*nfp*phi);
                    zdd += -zs(m, i) * m * m * sin(m*theta-n*nfp*phi);
                }
            }
            data(k1, k2, 0) = 4*M_PI*M_PI*rdd*cos(phi);
            data(k1, k2, 1) = 4*M_PI*M_PI*rdd*sin(phi);
            data(k1, k2, 2) = 4*M_PI*M_PI*zdd;
        }
    }
}

#if defined(USE_XSIMD)

template<class Array>
void SurfaceRZFourier<Array>::gammadash2_impl(Array& data) {
    constexpr int simd_size = xsimd::simd_type<double>::size;
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for(int k2 = 0; k2 < numquadpoints_theta; k2 += simd_size) {
            // batches have no per-lane operator[] anymore; gather into a small
            // contiguous buffer first, then load it into a batch in one shot.
            alignas(xs::default_arch::alignment()) double theta_arr[simd_size] = {};
            for (int l = 0; l < simd_size; ++l) {
                if(k2 + l >= numquadpoints_theta)
                    break;
                theta_arr[l] = 2*M_PI * quadpoints_theta[k2+l];
            }
            simd_t theta = xs::load_aligned(theta_arr);
            simd_t rd(0.);
            simd_t zd(0.);
            double sin_nfpphi = sin(-nfp*phi);
            double cos_nfpphi = cos(-nfp*phi);
            for (int m = 0; m <= mpol; ++m) {
                simd_t sinterm, costerm;
                for (int i = 0; i < 2*ntor+1; ++i) {
                    int n  = i - ntor;
                     // recompute the angle from scratch every so often, to
                     // avoid accumulating floating point error
                    if(i % ANGLE_RECOMPUTE == 0)
                        { auto sincos_result = xsimd::sincos(m*theta-n*nfp*phi); sinterm = sincos_result.first; costerm = sincos_result.second; }
                    rd += rc(m, i) * (-m) * sinterm;
                    if(!stellsym) {
                        rd += rs(m, i) * m * costerm;
                        zd += zc(m, i) * (-m) * sinterm;
                    }
                    zd += zs(m, i) * m * costerm;
                    if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
                        simd_t sinterm_old = sinterm;
                        simd_t costerm_old = costerm;
                        sinterm = cos_nfpphi * sinterm_old + costerm_old * sin_nfpphi;
                        costerm = costerm_old * cos_nfpphi - sinterm_old * sin_nfpphi;
                    }
                }
            }
            auto xd = 2*M_PI*rd*cos(phi);
            auto yd = 2*M_PI*rd*sin(phi);
            zd *= 2*M_PI;
            alignas(xs::default_arch::alignment()) double xd_arr[simd_size], yd_arr[simd_size], zd_arr[simd_size];
            xd.store_aligned(xd_arr);
            yd.store_aligned(yd_arr);
            zd.store_aligned(zd_arr);
            for (int l = 0; l < simd_size; ++l) {
                if(k2 + l >= numquadpoints_theta)
                    break;
                data(k1, k2+l, 0) = xd_arr[l];
                data(k1, k2+l, 1) = yd_arr[l];
                data(k1, k2+l, 2) = zd_arr[l];
            }
        }
    }
}

#else

template<class Array>
void SurfaceRZFourier<Array>::gammadash2_impl(Array& data) {
    constexpr int simd_size = 1;
    #pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for(int k2 = 0; k2 < numquadpoints_theta; k2 += simd_size) {
            double theta = 2*M_PI * quadpoints_theta[k2];
            double rd(0.);
            double zd(0.);
            double sin_nfpphi = sin(-nfp*phi);
            double cos_nfpphi = cos(-nfp*phi);
            for (int m = 0; m <= mpol; ++m) {
                double sinterm, costerm;
                for (int i = 0; i < 2*ntor+1; ++i) {
                    int n  = i - ntor;
                     // recompute the angle from scratch every so often, to
                     // avoid accumulating floating point error
                    if(i % ANGLE_RECOMPUTE == 0) {
                        sinterm = sin(m*theta-n*nfp*phi);
                        costerm = cos(m*theta-n*nfp*phi);
                    }
                    rd += rc(m, i) * (-m) * sinterm;
                    if(!stellsym) {
                        rd += rs(m, i) * m * costerm;
                        zd += zc(m, i) * (-m) * sinterm;
                    }
                    zd += zs(m, i) * m * costerm;
                    if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
                        auto sinterm_old = sinterm;
                        auto costerm_old = costerm;
                        sinterm = cos_nfpphi * sinterm_old + costerm_old * sin_nfpphi;
                        costerm = costerm_old * cos_nfpphi - sinterm_old * sin_nfpphi;
                    }
                }
            }
            auto xd = 2*M_PI*rd*cos(phi);
            auto yd = 2*M_PI*rd*sin(phi);
            zd *= 2*M_PI;
            data(k1, k2, 0) = xd;
            data(k1, k2, 1) = yd;
            data(k1, k2, 2) = zd;
        }
    }
}

#endif

#if defined(USE_XSIMD)
template<class Array>
Array SurfaceRZFourier<Array>::dgamma_by_dcoeff_vjp(Array& v) {
    Array res = xt::zeros<double>({num_dofs()});
    constexpr int simd_size = xsimd::simd_type<double>::size;
    auto resptr = &(res(0));
#pragma omp parallel
    {
        double* resptr_private = new double[num_dofs()];
        for (int i = 0; i < num_dofs(); ++i) {
            resptr_private[i] = 0.;
        }
#pragma omp for
        for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
            double phi  = 2*M_PI*quadpoints_phi[k1];
            double sinphi = sin(phi);
            double cosphi = cos(phi);

            for(int i = 0; i < numquadpoints_theta; i += simd_size) {
                // batches have no per-lane operator[] anymore; gather into small
                // contiguous buffers first, then load each into a batch in one shot.
                alignas(xs::default_arch::alignment()) double theta_arr[simd_size] = {};
                alignas(xs::default_arch::alignment()) double v0_arr[simd_size] = {}, v1_arr[simd_size] = {}, v2_arr[simd_size] = {};
                for (int l = 0; l < simd_size; ++l) {
                    if(i + l >= numquadpoints_theta)
                        break;
                    v0_arr[l] = v(k1, i+l, 0);
                    v1_arr[l] = v(k1, i+l, 1);
                    v2_arr[l] = v(k1, i+l, 2);
                    theta_arr[l] = 2*M_PI * quadpoints_theta[i+l];
                }
                simd_t theta = xs::load_aligned(theta_arr);
                simd_t v0 = xs::load_aligned(v0_arr);
                simd_t v1 = xs::load_aligned(v1_arr);
                simd_t v2 = xs::load_aligned(v2_arr);
                int counter = 0;
                int shift0 = -ntor;
                int shift1 = !stellsym ? shift0 + (mpol+1) * (2*ntor+1) - ntor - 1 : shift0;
                int shift2 = !stellsym ? shift1 + (mpol+1) * (2*ntor+1) - ntor : shift1;
                int shift3 = shift2 + (mpol+1)*(2*ntor+1) - ntor - 1;
                double sin_nfpphi = sin(-nfp*phi);
                double cos_nfpphi = cos(-nfp*phi);
                for (int m = 0; m <= mpol; ++m) {
                    simd_t sinterm, costerm;
                    for (int n = -ntor; n <= ntor; ++n) {
                        int i = n + ntor;
                         // recompute the angle from scratch every so often, to
                         // avoid accumulating floating point error
                        if(i % ANGLE_RECOMPUTE == 0)
                            { auto sincos_result = xsimd::sincos(m*theta-n*nfp*phi); sinterm = sincos_result.first; costerm = sincos_result.second; }
                        if(!(m==0 && n<0)){
                            resptr_private[counter+shift0] += cosphi * xsimd::reduce_add(costerm * v0);
                            resptr_private[counter+shift0] += sinphi * xsimd::reduce_add(costerm * v1);
                        }
                        if(!(stellsym) && !(m==0 && n<=0)){
                            resptr_private[counter+shift1] += cosphi * xsimd::reduce_add(sinterm * v0);
                            resptr_private[counter+shift1] += sinphi * xsimd::reduce_add(sinterm * v1);
                        }
                        if(!(stellsym) && !(m==0 && n<0)){
                            resptr_private[counter+shift2] += xsimd::reduce_add(costerm * v2);
                        }
                        if(!(m==0 && n<=0)){
                            resptr_private[counter+shift3] += xsimd::reduce_add(sinterm * v2);
                        }
                        counter++;
                        if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
                            simd_t sinterm_old = sinterm;
                            simd_t costerm_old = costerm;
                            sinterm = cos_nfpphi * sinterm_old + costerm_old * sin_nfpphi;
                            costerm = costerm_old * cos_nfpphi - sinterm_old * sin_nfpphi;
                        }
                    }
                }
            }
        }
#pragma omp critical
        {
            for(int i=0; i<num_dofs(); ++i) {
                resptr[i] += resptr_private[i];
            }
        }
    }
    return res;
}

#else
template<class Array>
Array SurfaceRZFourier<Array>::dgamma_by_dcoeff_vjp(Array& v) {
    Array res = xt::zeros<double>({num_dofs()});
    constexpr int simd_size = 1;
    auto resptr = &(res(0));
    #pragma omp parallel
    {
        double* resptr_private = new double[num_dofs()];
        for (int i = 0; i < num_dofs(); ++i) {
            resptr_private[i] = 0.;
        }
#pragma omp for
        for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
            double phi  = 2*M_PI*quadpoints_phi[k1];
            double sinphi = sin(phi);
            double cosphi = cos(phi);

            for(int i = 0; i < numquadpoints_theta; i += simd_size) {
                double theta = 2*M_PI * quadpoints_theta[i];
                double v0 = v(k1, i, 0);
                double v1 = v(k1, i, 1);
                double v2 = v(k1, i, 2);
                int counter = 0;
                int shift0 = -ntor;
                int shift1 = !stellsym ? shift0 + (mpol+1) * (2*ntor+1) - ntor - 1 : shift0;
                int shift2 = !stellsym ? shift1 + (mpol+1) * (2*ntor+1) - ntor : shift1;
                int shift3 = shift2 + (mpol+1)*(2*ntor+1) - ntor - 1;
                double sin_nfpphi = sin(-nfp*phi);
                double cos_nfpphi = cos(-nfp*phi);
                for (int m = 0; m <= mpol; ++m) {
                    double sinterm, costerm;
                    for (int n = -ntor; n <= ntor; ++n) {
                        int i = n + ntor;
                         // recompute the angle from scratch every so often, to
                         // avoid accumulating floating point error
                        if(i % ANGLE_RECOMPUTE == 0)
                            sinterm = sin(m*theta-n*nfp*phi);
                            costerm = cos(m*theta-n*nfp*phi);
                        if(!(m==0 && n<0)){
                            resptr_private[counter+shift0] += cosphi * costerm * v0;
                            resptr_private[counter+shift0] += sinphi * costerm * v1;
                        }
                        if(!(stellsym) && !(m==0 && n<=0)){
                            resptr_private[counter+shift1] += cosphi * sinterm * v0;
                            resptr_private[counter+shift1] += sinphi * sinterm * v1;
                        }
                        if(!(stellsym) && !(m==0 && n<0)){
                            resptr_private[counter+shift2] += costerm * v2;
                        }
                        if(!(m==0 && n<=0)){
                            resptr_private[counter+shift3] += sinterm * v2;
                        }
                        counter++;
                        if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
                            auto sinterm_old = sinterm;
                            auto costerm_old = costerm;
                            sinterm = cos_nfpphi * sinterm_old + costerm_old * sin_nfpphi;
                            costerm = costerm_old * cos_nfpphi - sinterm_old * sin_nfpphi;
                        }
                    }
                }
            }
        }
#pragma omp critical
        {
            for(int i=0; i<num_dofs(); ++i) {
                resptr[i] += resptr_private[i];
            }
        }
    }
    return res;
}

#endif

template<class Array>
void SurfaceRZFourier<Array>::dgamma_by_dcoeff_impl(Array& data) {
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            int counter = 0;
            for (int m = 0; m <= mpol; ++m) {
                for (int n = -ntor; n <= ntor; ++n) {
                    if(m==0 && n<0) continue;
                    data(k1, k2, 0, counter) = cos(m*theta-n*nfp*phi) * cos(phi);
                    data(k1, k2, 1, counter) = cos(m*theta-n*nfp*phi) * sin(phi);
                    data(k1, k2, 2, counter) = 0;
                    counter++;
                }
            }
            if(!stellsym) {
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<=0) continue;
                        data(k1, k2, 0, counter) = sin(m*theta-n*nfp*phi) * cos(phi);
                        data(k1, k2, 1, counter) = sin(m*theta-n*nfp*phi) * sin(phi);
                        data(k1, k2, 2, counter) = 0;
                        counter++;
                    }
                }
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<0) continue;
                        data(k1, k2, 0, counter) = 0;
                        data(k1, k2, 1, counter) = 0;
                        data(k1, k2, 2, counter) = cos(m*theta-n*nfp*phi);
                        counter++;
                    }
                }
            }
            for (int m = 0; m <= mpol; ++m) {
                for (int n = -ntor; n <= ntor; ++n) {
                    if(m==0 && n<=0) continue;
                    data(k1, k2, 0, counter) = 0;
                    data(k1, k2, 1, counter) = 0;
                    data(k1, k2, 2, counter) = sin(m*theta-n*nfp*phi);
                    counter++;
                }
            }
        }
    }
}

#if defined(USE_XSIMD)

template<class Array>
Array SurfaceRZFourier<Array>::dgammadash1_by_dcoeff_vjp(Array& v) {
    Array res = xt::zeros<double>({num_dofs()});
    constexpr int simd_size = xsimd::simd_type<double>::size;
    auto resptr = &(res(0));
#pragma omp parallel
    {
        double* resptr_private = new double[num_dofs()];
        for (int i = 0; i < num_dofs(); ++i) {
            resptr_private[i] = 0.;
        }
#pragma omp for
        for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
            double phi  = 2*M_PI*quadpoints_phi[k1];
            double sinphi = sin(phi);
            double cosphi = cos(phi);

            for(int i = 0; i < numquadpoints_theta; i += simd_size) {
                // batches have no per-lane operator[] anymore; gather into small
                // contiguous buffers first, then load each into a batch in one shot.
                alignas(xs::default_arch::alignment()) double theta_arr[simd_size] = {};
                alignas(xs::default_arch::alignment()) double v0_arr[simd_size] = {}, v1_arr[simd_size] = {}, v2_arr[simd_size] = {};
                for (int l = 0; l < simd_size; ++l) {
                    if(i + l >= numquadpoints_theta)
                        break;
                    v0_arr[l] = v(k1, i+l, 0);
                    v1_arr[l] = v(k1, i+l, 1);
                    v2_arr[l] = v(k1, i+l, 2);
                    theta_arr[l] = 2*M_PI * quadpoints_theta[i+l];
                }
                simd_t theta = xs::load_aligned(theta_arr);
                simd_t v0 = xs::load_aligned(v0_arr);
                simd_t v1 = xs::load_aligned(v1_arr);
                simd_t v2 = xs::load_aligned(v2_arr);
                int counter = 0;
                int shift0 = -ntor;
                int shift1 = !stellsym ? shift0 + (mpol+1) * (2*ntor+1) - ntor - 1 : shift0;
                int shift2 = !stellsym ? shift1 + (mpol+1) * (2*ntor+1) - ntor : shift1;
                int shift3 = shift2 + (mpol+1)*(2*ntor+1) - ntor - 1;

                double sin_nfpphi = sin(-nfp*phi);
                double cos_nfpphi = cos(-nfp*phi);
                for (int m = 0; m <= mpol; ++m) {
                    simd_t sinterm, costerm;
                    for (int n = -ntor; n <= ntor; ++n) {
                        int i = n + ntor;
                        // recompute the angle from scratch every so often, to
                        // avoid accumulating floating point error
                        if(i % ANGLE_RECOMPUTE == 0)
                            { auto sincos_result = xsimd::sincos(m*theta-n*nfp*phi); sinterm = sincos_result.first; costerm = sincos_result.second; }
                        if(!(m==0 && n<0)){
                            resptr_private[counter+shift0] += xsimd::reduce_add((sinterm * ((n*nfp) * cosphi) - costerm * sinphi) * v0);
                            resptr_private[counter+shift0] += xsimd::reduce_add((sinterm * ((n*nfp) * sinphi) + costerm * cosphi) * v1);
                        }
                        if(!(stellsym) && !(m==0 && n<=0)){
                            resptr_private[counter+shift1] += xsimd::reduce_add((costerm * ((-n*nfp)*cosphi) - sinterm * sinphi) * v0);
                            resptr_private[counter+shift1] += xsimd::reduce_add((costerm * ((-n*nfp)*sinphi) + sinterm * cosphi) * v1);
                        }
                        if(!(stellsym) && !(m==0 && n<0)){
                            resptr_private[counter+shift2] += xsimd::reduce_add((n*nfp)*sinterm * v2);
                        }
                        if(!(m==0 && n<=0)){
                            resptr_private[counter+shift3] += xsimd::reduce_add((-n*nfp)*costerm * v2);
                        }
                        counter++;
                        if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
                            simd_t sinterm_old = sinterm;
                            simd_t costerm_old = costerm;
                            sinterm = cos_nfpphi * sinterm_old + costerm_old * sin_nfpphi;
                            costerm = costerm_old * cos_nfpphi - sinterm_old * sin_nfpphi;
                        }
                    }
                }
            }
        }
#pragma omp critical
        {
            for(int i=0; i<num_dofs(); ++i) {
                resptr[i] += resptr_private[i];
            }
        }
    }
    res *= 2*M_PI;
    return res;
}

#else

template<class Array>
Array SurfaceRZFourier<Array>::dgammadash1_by_dcoeff_vjp(Array& v) {
    Array res = xt::zeros<double>({num_dofs()});
    constexpr int simd_size = 1;
    auto resptr = &(res(0));
    #pragma omp parallel
    {
        double* resptr_private = new double[num_dofs()];
        for (int i = 0; i < num_dofs(); ++i) {
            resptr_private[i] = 0.;
        }
        #pragma omp for
        for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
            double phi  = 2*M_PI*quadpoints_phi[k1];
            double sinphi = sin(phi);
            double cosphi = cos(phi);

            for(int i = 0; i < numquadpoints_theta; i += simd_size) {
                double theta = 2*M_PI * quadpoints_theta[i];
                double v0 = v(k1, i, 0);
                double v1 = v(k1, i, 1);
                double v2 = v(k1, i, 2);

                int counter = 0;
                int shift0 = -ntor;
                int shift1 = !stellsym ? shift0 + (mpol+1) * (2*ntor+1) - ntor - 1 : shift0;
                int shift2 = !stellsym ? shift1 + (mpol+1) * (2*ntor+1) - ntor : shift1;
                int shift3 = shift2 + (mpol+1)*(2*ntor+1) - ntor - 1;

                double sin_nfpphi = sin(-nfp*phi);
                double cos_nfpphi = cos(-nfp*phi);
                for (int m = 0; m <= mpol; ++m) {
                    double sinterm, costerm;
                    for (int n = -ntor; n <= ntor; ++n) {
                        int i = n + ntor;
                        // recompute the angle from scratch every so often, to
                        // avoid accumulating floating point error
                        if(i % ANGLE_RECOMPUTE == 0)
                            sinterm = sin(m*theta-n*nfp*phi);
                            costerm = cos(m*theta-n*nfp*phi);
                        if(!(m==0 && n<0)){
                            resptr_private[counter+shift0] += ((sinterm * ((n*nfp) * cosphi) - costerm * sinphi) * v0);
                            resptr_private[counter+shift0] += ((sinterm * ((n*nfp) * sinphi) + costerm * cosphi) * v1);
                        }
                        if(!(stellsym) && !(m==0 && n<=0)){
                            resptr_private[counter+shift1] += ((costerm * ((-n*nfp)*cosphi) - sinterm * sinphi) * v0);
                            resptr_private[counter+shift1] += ((costerm * ((-n*nfp)*sinphi) + sinterm * cosphi) * v1);
                        }
                        if(!(stellsym) && !(m==0 && n<0)){
                            resptr_private[counter+shift2] += ((n*nfp)*sinterm * v2);
                        }
                        if(!(m==0 && n<=0)){
                            resptr_private[counter+shift3] += ((-n*nfp)*costerm * v2);
                        }
                        counter++;
                        if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
                            auto sinterm_old = sinterm;
                            auto costerm_old = costerm;
                            sinterm = cos_nfpphi * sinterm_old + costerm_old * sin_nfpphi;
                            costerm = costerm_old * cos_nfpphi - sinterm_old * sin_nfpphi;
                        }
                    }
                }
            }
        }
#pragma omp critical
        {
            for(int i=0; i<num_dofs(); ++i) {
                resptr[i] += resptr_private[i];
            }
        }
    }
    res *= 2*M_PI;
    return res;
}

#endif

template<class Array>
void SurfaceRZFourier<Array>::dgammadash1_by_dcoeff_impl(Array& data) {
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            int counter = 0;
            for (int m = 0; m <= mpol; ++m) {
                for (int n = -ntor; n <= ntor; ++n) {
                    if(m==0 && n<0) continue;
                    data(k1, k2, 0, counter) = 2*M_PI*((n*nfp) * sin(m*theta-n*nfp*phi) * cos(phi) - cos(m*theta-n*nfp*phi) * sin(phi));
                    data(k1, k2, 1, counter) = 2*M_PI*((n*nfp) * sin(m*theta-n*nfp*phi) * sin(phi) + cos(m*theta-n*nfp*phi) * cos(phi));
                    data(k1, k2, 2, counter) = 0;
                    counter++;
                }
            }
            if(!stellsym) {
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<=0) continue;
                        data(k1, k2, 0, counter) = 2*M_PI*((-n*nfp)*cos(m*theta-n*nfp*phi) * cos(phi) - sin(m*theta-n*nfp*phi) * sin(phi));
                        data(k1, k2, 1, counter) = 2*M_PI*((-n*nfp)*cos(m*theta-n*nfp*phi) * sin(phi) + sin(m*theta-n*nfp*phi) * cos(phi));
                        data(k1, k2, 2, counter) = 0;
                        counter++;
                    }
                }
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<0) continue;
                        data(k1, k2, 0, counter) = 0;
                        data(k1, k2, 1, counter) = 0;
                        data(k1, k2, 2, counter) = 2*M_PI*(n*nfp)*sin(m*theta-n*nfp*phi);
                        counter++;
                    }
                }
            }
            for (int m = 0; m <= mpol; ++m) {
                for (int n = -ntor; n <= ntor; ++n) {
                    if(m==0 && n<=0) continue;
                    data(k1, k2, 0, counter) = 0;
                    data(k1, k2, 1, counter) = 0;
                    data(k1, k2, 2, counter) = 2*M_PI*(-n*nfp)*cos(m*theta-n*nfp*phi);
                    counter++;
                }
            }
        }
    }
}

template<class Array>
void SurfaceRZFourier<Array>::dgammadash1dash2_by_dcoeff_impl(Array& data) {
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            int counter = 0;
            for (int m = 0; m <= mpol; ++m) {
                for (int n = -ntor; n <= ntor; ++n) {
                    if(m==0 && n<0) continue;
                    data(k1, k2, 0, counter) = 4*M_PI*M_PI*(m * (n*nfp) * cos(m*theta-n*nfp*phi) * cos(phi) \
                                                            + m * sin(m*theta-n*nfp*phi) * sin(phi));
                    data(k1, k2, 1, counter) = 4*M_PI*M_PI*(   m * (n*nfp) * cos(m*theta-n*nfp*phi) * sin(phi) \
                                                             - m * sin(m*theta-n*nfp*phi) * cos(phi));
                    data(k1, k2, 2, counter) = 0;
                    counter++;
                }
            }
            if(!stellsym) {
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<=0) continue;
                        data(k1, k2, 0, counter) = 4*M_PI*M_PI*(-(-n*nfp)*m*sin(m*theta-n*nfp*phi)*cos(phi) \
                                                                - m*cos(m*theta-n*nfp*phi)*sin(phi));
                        data(k1, k2, 1, counter) = 4*M_PI*M_PI*(-(-n*nfp)*m*sin(m*theta-n*nfp*phi)*sin(phi) \
                                                                + m*cos(m*theta-n*nfp*phi)*cos(phi));
                        data(k1, k2, 2, counter) = 0;
                        counter++;
                    }
                }
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<0) continue;
                        data(k1, k2, 0, counter) = 0;
                        data(k1, k2, 1, counter) = 0;
                        data(k1, k2, 2, counter) = 4*M_PI*M_PI*n*nfp*m*cos(m*theta-n*nfp*phi);
                        counter++;
                    }
                }
            }
            for (int m = 0; m <= mpol; ++m) {
                for (int n = -ntor; n <= ntor; ++n) {
                    if(m==0 && n<=0) continue;
                    data(k1, k2, 0, counter) = 0;
                    data(k1, k2, 1, counter) = 0;
                    data(k1, k2, 2, counter) = -4*M_PI*M_PI*(-n*nfp)*m*sin(m*theta-n*nfp*phi);
                    counter++;
                }
            }
        }
    }
}

template<class Array>
void SurfaceRZFourier<Array>::dgammadash1dash1_by_dcoeff_impl(Array& data) {
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            int counter = 0;
            for (int m = 0; m <= mpol; ++m) {
                for (int n = -ntor; n <= ntor; ++n) {
                    if(m==0 && n<0) continue;
                    data(k1, k2, 0, counter) = 4*M_PI*M_PI*(- (-n*nfp) * (-n*nfp) * cos(m*theta-n*nfp*phi) * cos(phi) \
                                                            + 2 * (-n*nfp) * sin(m*theta-n*nfp*phi) * sin(phi) \
                                                            - cos(m*theta-n*nfp*phi) * cos(phi));
                    data(k1, k2, 1, counter) = 4*M_PI*M_PI*(- (-n*nfp) * (-n*nfp) * cos(m*theta-n*nfp*phi) * sin(phi) \
                                                            - 2 * (-n*nfp) * sin(m*theta-n*nfp*phi) * cos(phi) \
                                                            - cos(m*theta-n*nfp*phi) * sin(phi));
                    data(k1, k2, 2, counter) = 0;
                    counter++;
                }
            }
            if(!stellsym) {
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<=0) continue;
                        data(k1, k2, 0, counter) = 4*M_PI*M_PI*(-(-n*nfp)*(-n*nfp)*sin(m*theta-n*nfp*phi) * cos(phi) \
                                                                - 2*(-n*nfp)*cos(m*theta-n*nfp*phi) * sin(phi) \
                                                                - sin(m*theta-n*nfp*phi) * cos(phi));
                        data(k1, k2, 1, counter) = 4*M_PI*M_PI*(-(-n*nfp)*(-n*nfp)*sin(m*theta-n*nfp*phi) * sin(phi) \
                                                                + 2*(-n*nfp)*cos(m*theta-n*nfp*phi) * cos(phi) \
                                                                - sin(m*theta-n*nfp*phi) * sin(phi));
                        data(k1, k2, 2, counter) = 0;
                        counter++;
                    }
                }
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<0) continue;
                        data(k1, k2, 0, counter) = 0;
                        data(k1, k2, 1, counter) = 0;
                        data(k1, k2, 2, counter) = -4*M_PI*M_PI*(-n*nfp)*(-n*nfp)*cos(m*theta-n*nfp*phi);
                        counter++;
                    }
                }
            }
            for (int m = 0; m <= mpol; ++m) {
                for (int n = -ntor; n <= ntor; ++n) {
                    if(m==0 && n<=0) continue;
                    data(k1, k2, 0, counter) = 0;
                    data(k1, k2, 1, counter) = 0;
                    data(k1, k2, 2, counter) = -4*M_PI*M_PI*(-n*nfp)*(-n*nfp)*sin(m*theta-n*nfp*phi);
                    counter++;
                }
            }
        }
    }
}

#if defined(USE_XSIMD)

template<class Array>
Array SurfaceRZFourier<Array>::dgammadash2_by_dcoeff_vjp(Array& v) {
    Array res = xt::zeros<double>({num_dofs()});
    constexpr int simd_size = xsimd::simd_type<double>::size;
    auto resptr = &(res(0));
#pragma omp parallel
    {
        double* resptr_private = new double[num_dofs()];
        for (int i = 0; i < num_dofs(); ++i) {
            resptr_private[i] = 0.;
        }
#pragma omp for
        for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
            double phi  = 2*M_PI*quadpoints_phi[k1];
            double sinphi = sin(phi);
            double cosphi = cos(phi);

            for(int i = 0; i < numquadpoints_theta; i += simd_size) {
                // batches have no per-lane operator[] anymore; gather into small
                // contiguous buffers first, then load each into a batch in one shot.
                alignas(xs::default_arch::alignment()) double theta_arr[simd_size] = {};
                alignas(xs::default_arch::alignment()) double v0_arr[simd_size] = {}, v1_arr[simd_size] = {}, v2_arr[simd_size] = {};
                for (int l = 0; l < simd_size; ++l) {
                    if(i + l >= numquadpoints_theta)
                        break;
                    v0_arr[l] = v(k1, i+l, 0);
                    v1_arr[l] = v(k1, i+l, 1);
                    v2_arr[l] = v(k1, i+l, 2);
                    theta_arr[l] = 2*M_PI * quadpoints_theta[i+l];
                }
                simd_t theta = xs::load_aligned(theta_arr);
                simd_t v0 = xs::load_aligned(v0_arr);
                simd_t v1 = xs::load_aligned(v1_arr);
                simd_t v2 = xs::load_aligned(v2_arr);
                int counter = 0;
                int shift0 = -ntor;
                int shift1 = !stellsym ? shift0 + (mpol+1) * (2*ntor+1) - ntor - 1 : shift0;
                int shift2 = !stellsym ? shift1 + (mpol+1) * (2*ntor+1) - ntor : shift1;
                int shift3 = shift2 + (mpol+1)*(2*ntor+1) - ntor - 1;

                double sin_nfpphi = sin(-nfp*phi);
                double cos_nfpphi = cos(-nfp*phi);
                for (int m = 0; m <= mpol; ++m) {
                    simd_t sinterm, costerm;
                    for (int n = -ntor; n <= ntor; ++n) {
                        int i = n + ntor;
                        // recompute the angle from scratch every so often, to
                        // avoid accumulating floating point error
                        if(i % ANGLE_RECOMPUTE == 0)
                            { auto sincos_result = xsimd::sincos(m*theta-n*nfp*phi); sinterm = sincos_result.first; costerm = sincos_result.second; }
                        if(!(m==0 && n<0)){
                            resptr_private[counter+shift0] -= (cosphi * m) * xsimd::reduce_add(sinterm * v0);
                            resptr_private[counter+shift0] -= (sinphi * m) * xsimd::reduce_add(sinterm * v1);
                        }
                        if(!(stellsym) && !(m==0 && n<=0)){
                            resptr_private[counter+shift1] += (cosphi * m) * xsimd::reduce_add(costerm * v0);
                            resptr_private[counter+shift1] += (sinphi * m) * xsimd::reduce_add(costerm * v1);
                        }
                        if(!(stellsym) && !(m==0 && n<0)){
                            resptr_private[counter+shift2] -= m * xsimd::reduce_add(sinterm * v2);
                        }
                        if(!(m==0 && n<=0)){
                            resptr_private[counter+shift3] += m * xsimd::reduce_add(costerm * v2);
                        }
                        counter++;
                        if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
                            simd_t sinterm_old = sinterm;
                            simd_t costerm_old = costerm;
                            sinterm = cos_nfpphi * sinterm_old + costerm_old * sin_nfpphi;
                            costerm = costerm_old * cos_nfpphi - sinterm_old * sin_nfpphi;
                        }
                    }
                }
            }
        }
#pragma omp critical
        {
            for(int i=0; i<num_dofs(); ++i) {
                resptr[i] += resptr_private[i];
            }
        }
    }
    res *= 2*M_PI;
    return res;
}

#else

template<class Array>
Array SurfaceRZFourier<Array>::dgammadash2_by_dcoeff_vjp(Array& v) {
    Array res = xt::zeros<double>({num_dofs()});
    constexpr int simd_size = 1;
    auto resptr = &(res(0));
    #pragma omp parallel
    {
        double* resptr_private = new double[num_dofs()];
        for (int i = 0; i < num_dofs(); ++i) {
            resptr_private[i] = 0.;
        }
        #pragma omp for
        for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
            double phi  = 2*M_PI*quadpoints_phi[k1];
            double sinphi = sin(phi);
            double cosphi = cos(phi);

            for(int i = 0; i < numquadpoints_theta; i += simd_size) {
                double theta = 2*M_PI * quadpoints_theta[i];
                double v0 = v(k1, i, 0);
                double v1 = v(k1, i, 1);
                double v2 = v(k1, i, 2);

                int counter = 0;
                int shift0 = -ntor;
                int shift1 = !stellsym ? shift0 + (mpol+1) * (2*ntor+1) - ntor - 1 : shift0;
                int shift2 = !stellsym ? shift1 + (mpol+1) * (2*ntor+1) - ntor : shift1;
                int shift3 = shift2 + (mpol+1)*(2*ntor+1) - ntor - 1;

                double sin_nfpphi = sin(-nfp*phi);
                double cos_nfpphi = cos(-nfp*phi);
                for (int m = 0; m <= mpol; ++m) {
                    double sinterm, costerm;
                    for (int n = -ntor; n <= ntor; ++n) {
                        int i = n + ntor;
                        // recompute the angle from scratch every so often, to
                        // avoid accumulating floating point error
                        if(i % ANGLE_RECOMPUTE == 0)
                            sinterm = sin(m*theta-n*nfp*phi);
                            costerm = cos(m*theta-n*nfp*phi);
                        if(!(m==0 && n<0)){
                            resptr_private[counter+shift0] -= (cosphi * m) * (sinterm * v0);
                            resptr_private[counter+shift0] -= (sinphi * m) * (sinterm * v1);
                        }
                        if(!(stellsym) && !(m==0 && n<=0)){
                            resptr_private[counter+shift1] += (cosphi * m) * (costerm * v0);
                            resptr_private[counter+shift1] += (sinphi * m) * (costerm * v1);
                        }
                        if(!(stellsym) && !(m==0 && n<0)){
                            resptr_private[counter+shift2] -= m * (sinterm * v2);
                        }
                        if(!(m==0 && n<=0)){
                            resptr_private[counter+shift3] += m * (costerm * v2);
                        }
                        counter++;
                        if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
                            auto sinterm_old = sinterm;
                            auto costerm_old = costerm;
                            sinterm = cos_nfpphi * sinterm_old + costerm_old * sin_nfpphi;
                            costerm = costerm_old * cos_nfpphi - sinterm_old * sin_nfpphi;
                        }
                    }
                }
            }
        }
        #pragma omp critical
        {
            for(int i=0; i<num_dofs(); ++i) {
                resptr[i] += resptr_private[i];
            }
        }
    }
    res *= 2*M_PI;
    return res;
}

#endif

template<class Array>
void SurfaceRZFourier<Array>::dgammadash2_by_dcoeff_impl(Array& data) {
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            int counter = 0;
            for (int m = 0; m <= mpol; ++m) {
                for (int n = -ntor; n <= ntor; ++n) {
                    if(m==0 && n<0) continue;
                    data(k1, k2, 0, counter) = 2*M_PI*(-m) * sin(m*theta-n*nfp*phi)*cos(phi);
                    data(k1, k2, 1, counter) = 2*M_PI*(-m) * sin(m*theta-n*nfp*phi)*sin(phi);
                    data(k1, k2, 2, counter) = 0;
                    counter++;
                }
            }
            if(!stellsym) {
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<=0) continue;
                        data(k1, k2, 0, counter) = 2*M_PI*m * cos(m*theta-n*nfp*phi)*cos(phi);
                        data(k1, k2, 1, counter) = 2*M_PI*m * cos(m*theta-n*nfp*phi)*sin(phi);
                        data(k1, k2, 2, counter) = 0;
                        counter++;
                    }
                }
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<0) continue;
                        data(k1, k2, 0, counter) = 0;
                        data(k1, k2, 1, counter) = 0;
                        data(k1, k2, 2, counter) = 2*M_PI*(-m) * sin(m*theta-n*nfp*phi);
                        counter++;
                    }
                }
            }
            for (int m = 0; m <= mpol; ++m) {
                for (int n = -ntor; n <= ntor; ++n) {
                    if(m==0 && n<=0) continue;
                    data(k1, k2, 0, counter) = 0;
                    data(k1, k2, 1, counter) = 0;
                    data(k1, k2, 2, counter) = 2*M_PI*m * cos(m*theta-n*nfp*phi);
                    counter++;
                }
            }
        }
    }
}

template<class Array>
void SurfaceRZFourier<Array>::dgammadash2dash2_by_dcoeff_impl(Array& data) {
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            int counter = 0;
            for (int m = 0; m <= mpol; ++m) {
                for (int n = -ntor; n <= ntor; ++n) {
                    if(m==0 && n<0) continue;
                    data(k1, k2, 0, counter) = -4*M_PI*M_PI * m * m * cos(m*theta-n*nfp*phi)*cos(phi);
                    data(k1, k2, 1, counter) = -4*M_PI*M_PI* m * m * cos(m*theta-n*nfp*phi)*sin(phi);
                    data(k1, k2, 2, counter) = 0;
                    counter++;
                }
            }
            if(!stellsym) {
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<=0) continue;
                        data(k1, k2, 0, counter) = -4*M_PI*M_PI*m*m*sin(m*theta-n*nfp*phi)*cos(phi);
                        data(k1, k2, 1, counter) = -4*M_PI*M_PI*m*m*sin(m*theta-n*nfp*phi)*sin(phi);
                        data(k1, k2, 2, counter) = 0;
                        counter++;
                    }
                }
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<0) continue;
                        data(k1, k2, 0, counter) = 0;
                        data(k1, k2, 1, counter) = 0;
                        data(k1, k2, 2, counter) = -4*M_PI*M_PI*m*m*cos(m*theta-n*nfp*phi);
                        counter++;
                    }
                }
            }
            for (int m = 0; m <= mpol; ++m) {
                for (int n = -ntor; n <= ntor; ++n) {
                    if(m==0 && n<=0) continue;
                    data(k1, k2, 0, counter) = 0;
                    data(k1, k2, 1, counter) = 0;
                    data(k1, k2, 2, counter) = -4*M_PI*M_PI*m*m*sin(m*theta-n*nfp*phi);
                    counter++;
                }
            }
        }
    }
}

#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
template class SurfaceRZFourier<Array>;
