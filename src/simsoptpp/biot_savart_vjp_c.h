#pragma once
#include "simdhelpers.h"

#if defined(USE_XSIMD)

template<class T, int derivs>
void biot_savart_vjp_kernel(AlignedPaddedVec& pointsx, AlignedPaddedVec& pointsy, AlignedPaddedVec& pointsz, T& gamma, T& dgamma_by_dphi, T& v, T& res_gamma, T& res_dgamma_by_dphi, T& vgrad, T& res_grad_gamma, T& res_grad_dgamma_by_dphi);

#else

template<class T, int derivs>
void biot_savart_vjp_kernel(AlignedPaddedVecPortable& pointsx, AlignedPaddedVecPortable& pointsy, AlignedPaddedVecPortable& pointsz, T& gamma, T& dgamma_by_dphi, T& v, T& res_gamma, T& res_dgamma_by_dphi, T& vgrad, T& res_grad_gamma, T& res_grad_dgamma_by_dphi);

#endif
