#pragma once
#include "simdhelpers.h"

#if __x86_64__

template<class T, int derivs>
void biot_savart_kernel(AlignedPaddedVec& pointsx, AlignedPaddedVec& pointsy, AlignedPaddedVec& pointsz, T& gamma, T& dgamma_by_dphi, T& B, T& dB_by_dX, T& d2B_by_dXdX);

#else

template<class T, int derivs>
void biot_savart_kernel(AlignedPaddedVecPortable& pointsx, AlignedPaddedVecPortable& pointsy, AlignedPaddedVecPortable& pointsz, T& gamma, T& dgamma_by_dphi, T& B, T& dB_by_dX, T& d2B_by_dXdX);

#endif