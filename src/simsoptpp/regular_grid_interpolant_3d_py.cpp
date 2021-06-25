#include "regular_grid_interpolant_3d_impl.h"
#include "xtensor/xlayout.hpp"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
#include "xtensor-python/pytensor.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;

template class RegularGridInterpolant3D<Array>;
template class RegularGridInterpolant3D<xt::pytensor<double, 2, xt::layout_type::row_major>>;
