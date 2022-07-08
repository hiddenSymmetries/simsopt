#pragma once
#include <iostream>

#include <vector>
using std::vector;

#include "xsimd/xsimd.hpp"
namespace xs = xsimd;

// xsimd provides the 'aligned_allocator' which makes sure that objects are
// aligned properly.  we extend this operator to align a bit more memory than
// required, so that we definitely always have a multiple of the simd vector
// size. that way we can use simd operations for the entire vector, and don't
// have to special case the last few entries. this is used in biot_savart_kernel
template <class T, size_t Align>
class aligned_padded_allocator : public xs::aligned_allocator<T, Align> {
    public:
        template <class U>
        struct rebind
        {
            using other = aligned_padded_allocator<U, Align>;
        };

        inline aligned_padded_allocator() noexcept { }

        inline aligned_padded_allocator(const aligned_padded_allocator&) noexcept { } 

        template <class U>
        inline aligned_padded_allocator(const aligned_padded_allocator<U, Align>&) noexcept { } 

        T* allocate(size_t n, const void* hint = 0) {
            int simdcount = Align/sizeof(T);
            int nn = (n + simdcount) - (n % simdcount); // round to next highest multiple of simdcount
            T* res = reinterpret_cast<T*>(xsimd::detail::xaligned_malloc(sizeof(T) * nn, Align));
            if (res == nullptr)
                throw std::bad_alloc();
            return res;
        }
};


using AlignedPaddedVec = std::vector<double, aligned_padded_allocator<double, XSIMD_DEFAULT_ALIGNMENT>>;
using simd_t = xs::simd_type<double>;

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
