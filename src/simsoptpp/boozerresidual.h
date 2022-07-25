#pragma once

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
using std::vector;

double boozer_residual(double G, double iota, Array& xphi, Array& xtheta, Array& B);
std::tuple<double, Array> boozer_residual_ds(double G, double iota, Array& B, Array& dB_dx, Array& xphi, Array& xtheta, Array& dx_ds, Array& dxphi_ds, Array& dxtheta_ds);
std::tuple<double, Array, Array> boozer_residual_ds2(double G, double iota, Array& B, Array& dB_dx, Array& d2B_dx2, Array& xphi, Array& xtheta, Array& dx_ds, Array& dxphi_ds, Array& dxtheta_ds);
Array boozer_residual_dc(double G, Array& dB_dc, Array& B, Array& tang, Array& B2, Array& dxphi_dc, double iota, Array& dxtheta_dc);
