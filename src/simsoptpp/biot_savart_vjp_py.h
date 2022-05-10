#pragma once

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;

void biot_savart_vjp(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<double>& currents, Array& v, Array& vgrad, vector<Array>& dgamma_by_dcoeffs, vector<Array>& d2gamma_by_dphidcoeffs, vector<Array>& res_B, vector<Array>& res_dB);
void biot_savart_vjp_graph(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<double>& currents, Array& v, vector<Array>& res_gamma, vector<Array>& res_dgamma_by_dphi, Array& vgrad, vector<Array>& res_grad_gamma, vector<Array>& res_grad_dgamma_by_dphi);
void biot_savart_vector_potential_vjp_graph(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<double>& currents, Array& v, vector<Array>& res_gamma, vector<Array>& res_dgamma_by_dphi, Array& vgrad, vector<Array>& res_grad_gamma, vector<Array>& res_grad_dgamma_by_dphi);

