#include "regular_grid_interpolant_3d_impl.h"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;

template class RegularGridInterpolant3D<Array>;
