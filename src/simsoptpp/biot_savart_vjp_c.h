#pragma once
#include "simdhelpers.h"

template<class T, int derivs>
void biot_savart_vjp_kernel(
    AlignedPaddedVec &pointsx,
    AlignedPaddedVec &pointsy,
    AlignedPaddedVec &pointsz,
    T &gamma,
    T &dgamma_by_dphi,
    T &v,
    T &res_gamma,
    T &res_dgamma_by_dphi,
    T &vgrad,
    T &res_grad_gamma,
    T &res_grad_dgamma_by_dphi,
    T &vhess,
    T &res_hess_gamma,
    T &res_hess_dgamma_by_dphi
);
