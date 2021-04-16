#pragma once
#include <iostream>

#include <vector>
using std::vector;

#include "xsimd/xsimd.hpp"
namespace xs = xsimd;

#include "xsimd/xsimd.hpp"

// xsimd provides the 'aligned_allocator' which makes sure that objects are
// aligned properly.  we extend this operator to align a bit more memory than
// required, so that we definitely always have a multiple of the simd vector
// size. that way we can use simd operations for the entire vector, and don't
// have to special case the last few entries. this is used in biot_savart_kernel
template <size_t Align>
class aligned_padded_allocator : public xs::aligned_allocator<double, Align> {
    public:
        double* allocate(size_t n, const void* hint = 0) {
            int simdcount = Align/sizeof(double);
            int nn = (n + simdcount) - (n % simdcount); // round to next highest multiple of simdcount
            double* res = reinterpret_cast<double*>(xsimd::detail::xaligned_malloc(sizeof(double) * nn, Align));
            if (res == nullptr)
                throw std::bad_alloc();
            return res;
        }
};


using vector_type = std::vector<double, aligned_padded_allocator<XSIMD_DEFAULT_ALIGNMENT>>;
using simd_t = xs::simd_type<double>;

#include <Eigen/Core>
#include <Eigen/Dense>
typedef Eigen::Vector3d Vec3d;

struct Vec3dSimd {
    simd_t x;
    simd_t y;
    simd_t z;

    Vec3dSimd() : x(0.), y(0.), z(0.){
    }

    Vec3dSimd(double x_, double y_, double z_) : x(x_), y(y_), z(z_){
    }

    Vec3dSimd(Vec3d xyz) : x(xyz[0]), y(xyz[1]), z(xyz[2]){
    }

    Vec3dSimd(const simd_t& x_, const simd_t& y_, const simd_t& z_) : x(x_), y(y_), z(z_) {
    }

    Vec3dSimd(double* xptr, double* yptr, double *zptr){
        x = xs::load_aligned(xptr);
        y = xs::load_aligned(yptr);
        z = xs::load_aligned(zptr);
    }

    void store_aligned(double* xptr, double* yptr, double *zptr){
        x.store_aligned(xptr);
        y.store_aligned(yptr);
        z.store_aligned(zptr);
    }

    simd_t& operator[] (int i){
        if(i==0) {
            return x;
        }else if(i==1){
            return y;
        } else{
            return z;
        }
    }

    friend Vec3dSimd operator+(Vec3dSimd lhs, const Vec3d& rhs) {
        lhs.x += rhs[0];
        lhs.y += rhs[1];
        lhs.z += rhs[2];
        return lhs;
    }

    friend Vec3dSimd operator+(Vec3dSimd lhs, const Vec3dSimd& rhs) {
        lhs.x += rhs.x;
        lhs.y += rhs.y;
        lhs.z += rhs.z;
        return lhs;
    }

    Vec3dSimd& operator+=(const Vec3dSimd& rhs) {
        this->x += rhs.x;
        this->y += rhs.y;
        this->z += rhs.z;
        return *this;
    }

    Vec3dSimd& operator-=(const Vec3dSimd& rhs) {
        this->x -= rhs.x;
        this->y -= rhs.y;
        this->z -= rhs.z;
        return *this;
    }

    Vec3dSimd& operator*=(const double& rhs) {
        this->x *= rhs;
        this->y *= rhs;
        this->z *= rhs;
        return *this;
    }

    friend Vec3dSimd operator-(Vec3dSimd lhs, const Vec3d& rhs) {
        lhs.x -= rhs[0];
        lhs.y -= rhs[1];
        lhs.z -= rhs[2];
        return lhs;
    }

    friend Vec3dSimd operator-(Vec3dSimd lhs, const Vec3dSimd& rhs) {
        lhs.x -= rhs.x;
        lhs.y -= rhs.y;
        lhs.z -= rhs.z;
        return lhs;
    }

    friend Vec3dSimd operator*(Vec3dSimd lhs, const simd_t& rhs) {
        lhs.x *= rhs;
        lhs.y *= rhs;
        lhs.z *= rhs;
        return lhs;
    }
};


inline simd_t inner(const Vec3dSimd& a, const Vec3dSimd& b){
    return xsimd::fma(a.x, b.x, xsimd::fma(a.y, b.y, a.z*b.z));
}

inline simd_t inner(const Vec3d& b, const Vec3dSimd& a){
    return xsimd::fma(a.x, simd_t(b[0]), xsimd::fma(a.y, simd_t(b[1]), a.z*b[2]));
}

inline simd_t inner(const Vec3dSimd& a, const Vec3d& b){
    return xsimd::fma(a.x, simd_t(b[0]), xsimd::fma(a.y, simd_t(b[1]), a.z*b[2]));
}

inline simd_t inner(int i, Vec3dSimd& a){
    if(i==0)
        return a.x;
    else if(i==1)
        return a.y;
    else
        return a.z;
}

inline double inner(const Vec3d& a, const Vec3d& b){
    return a.dot(b);
}

inline Vec3d cross(const Vec3d& a, const Vec3d& b){
    return a.cross(b);
}

inline double norm(const Vec3d& a){
    return a.norm();
}

inline Vec3dSimd cross(Vec3dSimd& a, Vec3dSimd& b){
    return Vec3dSimd(
            xsimd::fms(a.y, b.z, a.z * b.y),
            xsimd::fms(a.z, b.x, a.x * b.z),
            xsimd::fms(a.x, b.y, a.y * b.x)
            );
}

inline Vec3dSimd cross(Vec3dSimd& a, Vec3d& b){
    return Vec3dSimd(
            xsimd::fms(a.y, simd_t(b[2]), a.z * b[1]),
            xsimd::fms(a.z, simd_t(b[0]), a.x * b[2]),
            xsimd::fms(a.x, simd_t(b[1]), a.y * b[0])
            );
}

inline Vec3dSimd cross(Vec3d& a, Vec3dSimd& b){
    return Vec3dSimd(
            xsimd::fms(simd_t(a[1]), b.z, a[2] * b.y),
            xsimd::fms(simd_t(a[2]), b.x, a[0] * b.z),
            xsimd::fms(simd_t(a[0]), b.y, a[1] * b.x)
            );
}

inline Vec3dSimd cross(Vec3dSimd& a, int i){
    if(i==0)
        return Vec3dSimd(simd_t(0.), a.z, -a.y);
    else if(i == 1)
        return Vec3dSimd(-a.z, simd_t(0.), a.x);
    else
        return Vec3dSimd(a.y, -a.x, simd_t(0.));
}

inline Vec3dSimd cross(int i, Vec3dSimd& b){
    if(i==0)
        return Vec3dSimd(simd_t(0.), -b.z, b.y);
    else if(i == 1)
        return Vec3dSimd(b.z, simd_t(0.), -b.x);
    else
        return Vec3dSimd(-b.y, b.x, simd_t(0.));
}

inline Vec3d cross(int i, Vec3d& b){
    if(i==0)
        return Vec3d{0., -b.coeff(2), b.coeff(1)};
    else if(i == 1)
        return Vec3d{b.coeff(2), 0., -b.coeff(0)};
    else
        return Vec3d{-b.coeff(1), b.coeff(0), 0.};
}

inline Vec3d cross(Vec3d& a, int i){
    if(i==0)
        return Vec3d{0., a.coeff(2), -a.coeff(1)};
    else if(i == 1)
        return Vec3d{-a.coeff(2), 0., a.coeff(0)};
    else
        return Vec3d{a.coeff(1), -a.coeff(0), 0.};
}

inline simd_t normsq(Vec3dSimd& a){
    return xsimd::fma(a.x, a.x, xsimd::fma(a.y, a.y, a.z*a.z));
}

#if __AVX512F__ 
// On skylake _mm512_sqrt_pd takes 24 CPI and _mm512_div_pd takes 16 CPI, so
// 1/sqrt(vec) takes 40 CPI. Instead we can use the approximate inverse square
// root _mm512_rsqrt14_pd which takes 2 CPI (but only gives 4 digits or so) and
// then refine that result using two iterations of Newton's method, which is
// fairly cheap.
inline void rsqrt_newton_intrin(simd_t& rinv, const simd_t& r2){
  //rinv = rinv*(1.5-r2*rinv*rinv);
  rinv = xsimd::fnma(rinv*rinv*rinv, r2, rinv*1.5);
}
inline simd_t rsqrt(simd_t r2){
  simd_t rinv = _mm512_rsqrt14_pd(r2);
  r2 *= 0.5;
  rsqrt_newton_intrin(rinv, r2);
  rsqrt_newton_intrin(rinv, r2);
  //rsqrt_newton_intrin(rinv, r2);
  return rinv;
}
#else
inline simd_t rsqrt(const simd_t& r2){
    //On my avx2 machine, computing the sqrt and then the inverse is actually a
    //bit faster. just keeping this line here to remind myself how to compute
    //the approximate inverse square root in that case.
    //simd_t rinv = _mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(r2)));
    return 1./sqrt(r2);
}
#endif
