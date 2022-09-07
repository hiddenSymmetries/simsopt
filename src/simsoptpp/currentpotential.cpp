#include "currentpotential.h"

// template<template<class Array> class Surface, class Array>
template<class Array>
void CurrentPotential<Array>::K_impl_helper(Array& data, Array& dg1, Array& dg2, Array& normal) {
  auto dphid1 = this->Phidash1();
  auto dphid2 = this->Phidash2();
  // auto dg1 = this->winding_surface->gammadash1();
  // auto dg2 = this->winding_surface->gammadash2();
  // auto normal = this->winding_surface->normal();
  // K = n \times Phi
  // N \times \nabla \theta = - dr/dzeta
  // N \times \nabla \zeta = dr/dtheta
  // K = (- dPhidtheta dr/dzeta + dPhidzeta dr/dtheta)/N
  for (int i = 0; i < numquadpoints_phi; ++i) {
      for (int j = 0; j < numquadpoints_theta; ++j) {
          double normn = std::sqrt(normal(i, j, 0)*normal(i, j, 0) + normal(i, j, 1)*normal(i, j, 1) + normal(i, j, 2)*normal(i, j, 2));
          data(i, j, 0) = (- dg1(i,j,0) * (dphid2(i,j) + this->net_toroidal_current_amperes) + dg2(i,j,0) * (dphid1(i,j) + this->net_poloidal_current_amperes))/normn;
          data(i, j, 1) = (- dg1(i,j,1) * (dphid2(i,j) + this->net_toroidal_current_amperes) + dg2(i,j,1) * (dphid1(i,j) + this->net_poloidal_current_amperes))/normn;
          data(i, j, 2) = (- dg1(i,j,2) * (dphid2(i,j) + this->net_toroidal_current_amperes) + dg2(i,j,2) * (dphid1(i,j) + this->net_poloidal_current_amperes))/normn;
      }
  }
}

template<class Array>
void CurrentPotential<Array>::K_matrix_impl_helper(Array& data, Array& dg1, Array& dg2, Array& normal) {
  auto dphid1_dcoeff = this->dPhidash1_by_dcoeff();
  auto dphid2_dcoeff = this->dPhidash2_by_dcoeff();

  // Ak_i,j = fi cdot fj / N
  // K = (- dPhidtheta dr/dzeta + dPhidzeta dr/dtheta)/N = - Phi_j f_j /N
  // f_j =  dPhidtheta_dPhi_j dr/dzeta - dPhidzeta_dPhi_j dr/dtheta

  int ndofs = num_dofs();
  for (int i = 0; i < numquadpoints_phi; ++i) {
      for (int j = 0; j < numquadpoints_theta; ++j) {

          double* dphid1_dcoeff_ptr = &(dphid1_dcoeff(i, j, 0));
          double* dphid2_dcoeff_ptr = &(dphid2_dcoeff(i, j, 0));

          double gzz = dg1(i,j,0)*dg1(i,j,0) + dg1(i,j,1)*dg1(i,j,1) + dg1(i,j,2)*dg1(i,j,2);
          double gtt = dg2(i,j,0)*dg2(i,j,0) + dg2(i,j,1)*dg2(i,j,1) + dg2(i,j,2)*dg2(i,j,2);
          double gtz = dg1(i,j,0)*dg2(i,j,0) + dg1(i,j,1)*dg2(i,j,1) + dg1(i,j,2)*dg2(i,j,2);

          for (int m = 0; m < ndofs; ++m ) {
              for (int n = 0; n < ndofs; ++ n ) {
                  data(m,n) += (dphid1_dcoeff[m]*dphid1_dcoeff[n]*gzz + \
                             + dphid1_dcoeff[m]*dphid2_dcoeff[n]*gtz + \
                             + dphid2_dcoeff[m]*dphid2_dcoeff[n]*gtt)/normal(i,j);
              }
          }
      }
  }
};

  // auto dg1 = this->winding_surface->gammadash1();
  // auto dg2 = this->winding_surface->gammadash2();
  // auto normal = this->winding_surface->normal();
  // K = n \times Phi
  // N \times \nabla \theta = - dr/dzeta
  // N \times \nabla \zeta = dr/dtheta
  // K = (- dPhidtheta dr/dzeta + dPhidzeta dr/dtheta)/N
//   for (int i = 0; i < numquadpoints_phi; ++i) {
//       for (int j = 0; j < numquadpoints_theta; ++j) {
//           double normn = std::sqrt(normal(i, j, 0)*normal(i, j, 0) + normal(i, j, 1)*normal(i, j, 1) + normal(i, j, 2)*normal(i, j, 2));
//           data(i, j, 0) += (- dg1(i,j,0) * (dphid2(i,j) + this->net_toroidal_current_amperes) + dg2(i,j,0) * (dphid1(i,j) + this->net_poloidal_current_amperes))/normn;
//           data(i, j, 1) += (- dg1(i,j,1) * (dphid2(i,j) + this->net_toroidal_current_amperes) + dg2(i,j,1) * (dphid1(i,j) + this->net_poloidal_current_amperes))/normn;
//           data(i, j, 2) += (- dg1(i,j,2) * (dphid2(i,j) + this->net_toroidal_current_amperes) + dg2(i,j,2) * (dphid1(i,j) + this->net_poloidal_current_amperes))/normn;
//       }
//   }
// }
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
template class CurrentPotential<Array>;
