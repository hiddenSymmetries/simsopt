#include "regular_grid_interpolant_3d_impl.h"
#include "xtensor/xarray.hpp"
typedef xt::xarray<double> Array;

template class RegularGridInterpolant3D<Array>;
