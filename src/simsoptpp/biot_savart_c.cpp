#include "biot_savart_impl.h"
//#include "biot_savart_c.h"
#include "vec3dsimd.h"
#include "xtensor/xarray.hpp"

#if __x86_64__

template void biot_savart_kernel<xt::xarray<double>, 0>(AlignedPaddedVec&, AlignedPaddedVec&, AlignedPaddedVec&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&);
template void biot_savart_kernel<xt::xarray<double>, 1>(AlignedPaddedVec&, AlignedPaddedVec&, AlignedPaddedVec&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&);
template void biot_savart_kernel<xt::xarray<double>, 2>(AlignedPaddedVec&, AlignedPaddedVec&, AlignedPaddedVec&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&);

#else

template void biot_savart_kernel<xt::xarray<double>, 0>(AlignedPaddedVecPortable&, AlignedPaddedVecPortable&, AlignedPaddedVecPortable&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&);
template void biot_savart_kernel<xt::xarray<double>, 1>(AlignedPaddedVecPortable&, AlignedPaddedVecPortable&, AlignedPaddedVecPortable&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&);
template void biot_savart_kernel<xt::xarray<double>, 2>(AlignedPaddedVecPortable&, AlignedPaddedVecPortable&, AlignedPaddedVecPortable&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&);

template void biot_savart_kernel_nonsimd<xt::xarray<double>, 0>(AlignedPaddedVecPortable&, AlignedPaddedVecPortable&, AlignedPaddedVecPortable&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&);
template void biot_savart_kernel_nonsimd<xt::xarray<double>, 1>(AlignedPaddedVecPortable&, AlignedPaddedVecPortable&, AlignedPaddedVecPortable&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&);
template void biot_savart_kernel_nonsimd<xt::xarray<double>, 2>(AlignedPaddedVecPortable&, AlignedPaddedVecPortable&, AlignedPaddedVecPortable&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&);

#endif
