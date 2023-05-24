#pragma once
#include <iostream>
#include <limits>
#include <cmath>
#include <new>
#include <vector>
#include "config.h"

using std::vector;

#if !defined(NO_XSIMD) && (defined(__x86_64__) || defined(__aarch64__))
#define USE_XSIMD
#endif


#if defined(USE_XSIMD)

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

#else
/*
We will implement a custom allocator that mimics aligned_padded_vector using C++17, but without xsimd
Credits: https://stackoverflow.com/questions/60169819/modern-approach-to-making-stdvector-allocate-aligned-memory
Credits: https://github.com/Twon/Alignment
*/
/**
 * Returns aligned pointers when allocations are requested. Default alignment
 * is 64B = 512b, sufficient for AVX-512 and most cache line sizes.
 *
 * @tparam ALIGNMENT_IN_BYTES Must be a positive power of 2.
 */
template<typename T, std::size_t ALIGNMENT_IN_BYTES = 32>
class AlignedPaddedAllocator
{
private:
    static_assert(
        ALIGNMENT_IN_BYTES >= alignof(T),
        "Beware that types like int have minimum alignment requirements "
        "or access will result in crashes."
    );

public:
    using value_type = T;
    static std::align_val_t constexpr ALIGNMENT{ALIGNMENT_IN_BYTES};

    /**
     * This is only necessary because AlignedAllocator has a second template
     * argument for the alignment that will make the default
     * std::allocator_traits implementation fail during compilation.
     * @see https://stackoverflow.com/a/48062758/2191065
     */
    template<class U>
    struct rebind
    {
        using other = AlignedPaddedAllocator<U, ALIGNMENT_IN_BYTES>;
    };

public:
    constexpr AlignedPaddedAllocator() noexcept = default;

    constexpr AlignedPaddedAllocator(const AlignedPaddedAllocator&) noexcept = default;

    template<typename U>
    constexpr AlignedPaddedAllocator(AlignedPaddedAllocator<U, ALIGNMENT_IN_BYTES> const&) noexcept
    {}

    [[nodiscard]] T* allocate(std::size_t n)
    {
        int simdcount = ALIGNMENT_IN_BYTES/sizeof(T);
        int nn = (n + simdcount) - (n % simdcount);
        if (nn > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }

        auto const nBytesToAllocate = nn * sizeof(T);
        T* res = reinterpret_cast<T*>(::operator new[](nBytesToAllocate, ALIGNMENT));
        if (res == nullptr)
            throw std::bad_alloc();
        return res;
    }

    void deallocate(T* ptr, [[maybe_unused]] std::size_t  nBytesAllocated)
    {
        /* According to the C++20 draft n4868 § 17.6.3.3, the delete operator
         * must be called with the same alignment argument as the new expression.
         * The size argument can be omitted but if present must also be equal to
         * the one used in new. */
        ::operator delete[](ptr, ALIGNMENT);
    }
};

using AlignedPaddedVec = std::vector<double, AlignedPaddedAllocator<double>>;

#endif

#if defined(USE_XSIMD)
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
#endif

inline double rsqrt(const double& r2){
    return 1./std::sqrt(r2);
}
