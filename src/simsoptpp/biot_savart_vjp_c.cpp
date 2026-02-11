#include "biot_savart_vjp_impl.h"
#include "biot_savart_vjp_c.h"
#include "xtensor/xarray.hpp"

template void biot_savart_vjp_kernel<xt::xarray<double>, 0>(
    AlignedPaddedVec &,
    AlignedPaddedVec &,
    AlignedPaddedVec &,
    xt::xarray<double> &gamma,
    xt::xarray<double> &dgamma_by_dphi,
    xt::xarray<double> &v,
    xt::xarray<double> &res_gamma,
    xt::xarray<double> &res_dgamma_by_dphi,
    xt::xarray<double> &vgrad,
    xt::xarray<double> &res_grad_gamma,
    xt::xarray<double> &res_grad_dgamma_by_dphi,
    xt::xarray<double> &vhess,
    xt::xarray<double> &res_hess_gamma,
    xt::xarray<double> &res_hess_dgamma_by_dphi
);

template void biot_savart_vjp_kernel<xt::xarray<double>, 1>(
    AlignedPaddedVec &,
    AlignedPaddedVec &,
    AlignedPaddedVec &,
    xt::xarray<double> &gamma,
    xt::xarray<double> &dgamma_by_dphi,
    xt::xarray<double> &v,
    xt::xarray<double> &res_gamma,
    xt::xarray<double> &res_dgamma_by_dphi,
    xt::xarray<double> &vgrad,
    xt::xarray<double> &res_grad_gamma,
    xt::xarray<double> &res_grad_dgamma_by_dphi,
    xt::xarray<double> &vhess,
    xt::xarray<double> &res_hess_gamma,
    xt::xarray<double> &res_hess_dgamma_by_dphi
);
