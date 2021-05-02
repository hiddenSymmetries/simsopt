#pragma once
#include "simdhelpers.h"

template<class T, int derivs>
void biot_savart_kernel(vector_type& pointsx, vector_type& pointsy, vector_type& pointsz, T& gamma, T& dgamma_by_dphi, T& B, T& dB_by_dX, T& d2B_by_dXdX);
