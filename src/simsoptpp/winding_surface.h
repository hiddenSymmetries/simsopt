#include <tuple>  // c++ tuples
#include "xtensor-python/pyarray.hpp"
typedef xt::pyarray<double> Array;

// Compute the Bnormal using the REGCOIL method.
Array WindingSurfaceBn_REGCOIL(Array& points, Array& ws_points, Array& ws_normal, Array& current_potential, Array& plasma_normal);

// Compute the B field using the Biot-Savart law from the winding surface current K.
Array WindingSurfaceB(Array& points, Array& ws_points, Array& ws_normal, Array& K);

// Compute the dB/dX field using the Biot-Savart law from the winding surface current K.
Array WindingSurfacedB(Array& points, Array& ws_points, Array& ws_normal, Array& K);

// Compute the A field using the Biot-Savart law from the winding surface current K.
Array WindingSurfaceA(Array& points, Array& ws_points, Array& ws_normal, Array& K);

// Compute the dA/dX field using the Biot-Savart law from the winding surface current K.
Array WindingSurfacedA(Array& points, Array& ws_points, Array& ws_normal, Array& K);

// Compute the Bn field using the winding surface current K.
std::tuple<Array, Array> winding_surface_field_Bn(Array& points_plasma, Array& points_coil, Array& normal_plasma, Array& normal_coil, bool stellsym, Array& zeta_coil, Array& theta_coil, int ndofs, Array& m, Array& n, int nfp);

// Compute part of the Bn field using the winding surface current K and the plasma surface normal.
Array winding_surface_field_Bn_GI(Array& points_plasma, Array& points_coil, Array& normal_plasma, Array& zeta_coil, Array& theta_coil, double G, double I, Array& gammadash1, Array& gammadash2); 

// Compute the K2 matrices using the winding surface current K and the plasma surface normal.
std::tuple<Array, Array> winding_surface_field_K2_matrices(Array& dr_dzeta_coil, Array& dr_dtheta_coil, Array& normal_coil, bool stellsym, Array& zeta_coil, Array& theta_coil, int ndofs, Array& m, Array& n, int nfp, double G, double I);
