#include "biot_savart_vjp_impl.h"
#include "biot_savart_vjp_c.h"
#include "xtensor/xarray.hpp"

#if __x86_64__

template void biot_savart_vjp_kernel<xt::xarray<double>, 0>(AlignedPaddedVec&, AlignedPaddedVec&, AlignedPaddedVec&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&);
template void biot_savart_vjp_kernel<xt::xarray<double>, 1>(AlignedPaddedVec&, AlignedPaddedVec&, AlignedPaddedVec&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&);

#else

template void biot_savart_vjp_kernel<xt::xarray<double>, 0>(AlignedPaddedVecPortable&, AlignedPaddedVecPortable&, AlignedPaddedVecPortable&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&);
template void biot_savart_vjp_kernel<xt::xarray<double>, 1>(AlignedPaddedVecPortable&, AlignedPaddedVecPortable&, AlignedPaddedVecPortable&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&);

#endif
