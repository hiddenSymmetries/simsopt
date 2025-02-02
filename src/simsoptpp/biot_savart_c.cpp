#include "biot_savart_impl.h"
#include "vec3dsimd.h"
#include "xtensor/xarray.hpp"
#include "operators.h"

template void biot_savart_kernel<xt::xarray<std::complex<double>>, 0>(AlignedPaddedVec&, AlignedPaddedVec&, AlignedPaddedVec&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&);
template void biot_savart_kernel<xt::xarray<std::complex<double>>, 1>(AlignedPaddedVec&, AlignedPaddedVec&, AlignedPaddedVec&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&);
template void biot_savart_kernel<xt::xarray<std::complex<double>>, 2>(AlignedPaddedVec&, AlignedPaddedVec&, AlignedPaddedVec&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&, xt::xarray<std::complex<double>>&);
