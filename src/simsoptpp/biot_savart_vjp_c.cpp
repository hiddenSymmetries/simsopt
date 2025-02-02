#include "biot_savart_vjp_impl.h"
#include "biot_savart_vjp_c.h"
#include "xtensor/xarray.hpp"
#include "operators.h"

template void biot_savart_vjp_kernel<xt::xarray<std::complex<double>>, 0>(AlignedPaddedVec&, AlignedPaddedVec&, AlignedPaddedVec&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&);
template void biot_savart_vjp_kernel<xt::xarray<std::complex<double>>, 1>(AlignedPaddedVec&, AlignedPaddedVec&, AlignedPaddedVec&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&);
