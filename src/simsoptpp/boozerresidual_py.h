#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings

typedef xt::pyarray<double> Array;

double boozer_residual(double G, double iota, Array& xphi, Array& xtheta, Array& B, bool weight_inv_modB);
std::tuple<double, Array> boozer_residual_ds(double G, double iota, Array& B, Array& dB_dx, Array& xphi, Array& xtheta, Array& dx_ds, Array& dxphi_ds, Array& dxtheta_ds, bool weight_inv_modB);
std::tuple<double, Array, Array> boozer_residual_ds2(double G, double iota, Array& B, Array& dB_dx, Array& d2B_dx2, Array& xphi, Array& xtheta, Array& dx_ds, Array& dxphi_ds, Array& dxtheta_ds, bool weight_inv_modB);
