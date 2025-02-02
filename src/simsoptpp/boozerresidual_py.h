#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings

typedef xt::pyarray<std::complex<double>> Array;

std::complex<double> boozer_residual(std::complex<double> G, std::complex<double> iota, Array& xphi, Array& xtheta, Array& B, bool weight_inv_modB);
std::tuple<std::complex<double>, Array> boozer_residual_ds(std::complex<double> G, std::complex<double> iota, Array& B, Array& dB_dx, Array& xphi, Array& xtheta, Array& dx_ds, Array& dxphi_ds, Array& dxtheta_ds, bool weight_inv_modB);
std::tuple<std::complex<double>, Array, Array> boozer_residual_ds2(std::complex<double> G, std::complex<double> iota, Array& B, Array& dB_dx, Array& d2B_dx2, Array& xphi, Array& xtheta, Array& dx_ds, Array& dxphi_ds, Array& dxtheta_ds, bool weight_inv_modB);
