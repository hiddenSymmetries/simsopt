#include <tuple>  // c++ tuples
#include "xtensor-python/pyarray.hpp"
typedef xt::pyarray<double> Array;

Array WindingSurfaceB(Array& points, Array& ws_points, Array& ws_normal, Array& K);
Array WindingSurfacedB(Array& points, Array& ws_points, Array& ws_normal, Array& K);
Array WindingSurfaceA(Array& points, Array& ws_points, Array& ws_normal, Array& K);
Array WindingSurfacedA(Array& points, Array& ws_points, Array& ws_normal, Array& K);
std::tuple<Array, Array> winding_surface_field_Bn(Array& points_plasma, Array& points_coil, Array& normal_plasma, Array& normal_coil, int nfp, int stellsym, Array& zeta_coil, Array& theta_coil, int ndofs, Array& m, Array& n);
Array winding_surface_field_Bn_GI(Array& points_plasma, Array& points_coil, Array& normal_plasma, int nfp, bool stellsym, Array& zeta_coil, Array& theta_coil, Array& G, Array& I, Array& dr_dtheta_coil, Array& dr_dzeta_coil);
