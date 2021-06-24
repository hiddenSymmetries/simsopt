#include "regular_grid_interpolant_3d_impl.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
typedef xt::xarray<double> Array;

template class RegularGridInterpolant3D<Array>;

template<class Type, std::size_t rank, xt::layout_type layout>
using DefaultTensor = xt::xtensor<Type, rank, layout, XTENSOR_DEFAULT_ALLOCATOR(double)>;
using Tensor2 = DefaultTensor<double, 2, xt::layout_type::row_major>;
template class RegularGridInterpolant3D<Tensor2>;
