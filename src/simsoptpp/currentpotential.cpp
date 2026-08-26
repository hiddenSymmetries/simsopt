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
void CurrentPotential<Array>::K_GI_impl_helper(Array& data, Array& dg1, Array& dg2, Array& normal) {
  // K = n \times Phi
  // N \times \nabla \theta = - dr/dzeta
  // N \times \nabla \zeta = dr/dtheta
  // K = (- dPhidtheta dr/dzeta + dPhidzeta dr/dtheta)/N
  for (int i = 0; i < numquadpoints_phi; ++i) {
      for (int j = 0; j < numquadpoints_theta; ++j) {
          double normn = std::sqrt(normal(i, j, 0)*normal(i, j, 0) + normal(i, j, 1)*normal(i, j, 1) + normal(i, j, 2)*normal(i, j, 2));
          data(i, j, 0) = (- dg1(i,j,0) * this->net_toroidal_current_amperes + dg2(i,j,0) * this->net_poloidal_current_amperes)/normn;
          data(i, j, 1) = (- dg1(i,j,1) * this->net_toroidal_current_amperes + dg2(i,j,1) * this->net_poloidal_current_amperes)/normn;
          data(i, j, 2) = (- dg1(i,j,2) * this->net_toroidal_current_amperes + dg2(i,j,2) * this->net_poloidal_current_amperes)/normn;
      }
  }
}

template<class Array>
void CurrentPotential<Array>::K_by_dcoeff_impl_helper(Array& data, Array& dg1, Array& dg2, Array& normal) {
  auto dphid1_by_dcoeff = this->dPhidash1_by_dcoeff();
  auto dphid2_by_dcoeff = this->dPhidash2_by_dcoeff();
  // K = n \times Phi
  // N \times \nabla \theta = - dr/dzeta
  // N \times \nabla \zeta = dr/dtheta
  // K = (- dPhidtheta dr/dzeta + dPhidzeta dr/dtheta)/N
  int ndofs = num_dofs();
  for (int i = 0; i < numquadpoints_phi; ++i) {
      for (int j = 0; j < numquadpoints_theta; ++j) {
          double normn = std::sqrt(normal(i, j, 0)*normal(i, j, 0) + normal(i, j, 1)*normal(i, j, 1) + normal(i, j, 2)*normal(i, j, 2));
          for (int m = 0; m < ndofs; ++m ) {
              data(i, j, 0, m) = (- dg1(i,j,0) * dphid2_by_dcoeff(i,j,m) + dg2(i,j,0) * dphid1_by_dcoeff(i,j,m))/normn;
              data(i, j, 1, m) = (- dg1(i,j,1) * dphid2_by_dcoeff(i,j,m) + dg2(i,j,1) * dphid1_by_dcoeff(i,j,m))/normn;
              data(i, j, 2, m) = (- dg1(i,j,2) * dphid2_by_dcoeff(i,j,m) + dg2(i,j,2) * dphid1_by_dcoeff(i,j,m))/normn;
          }
      }
  }
}

template<class Array>
void CurrentPotential<Array>::K_rhs_impl_helper(Array& data, Array& dg1, Array& dg2, Array& normal) {
  int ndofs = num_dofs();
  Array K_by_dcoeff = xt::zeros<double>({numquadpoints_phi, numquadpoints_theta, 3, ndofs});
  this->K_by_dcoeff_impl_helper(K_by_dcoeff, dg1, dg2, normal);
  Array K_GI = xt::zeros<double>({numquadpoints_phi, numquadpoints_theta, 3});
  this->K_GI_impl_helper(K_GI, dg1, dg2, normal);
  for (int i = 0; i < numquadpoints_phi; ++i) {
      for (int j = 0; j < numquadpoints_theta; ++j) {
          double normn = std::sqrt(normal(i, j, 0)*normal(i, j, 0) + normal(i, j, 1)*normal(i, j, 1) + normal(i, j, 2)*normal(i, j, 2));
          for (int m = 0; m < ndofs; ++m) {
              double K_GI_dot_K_by_dcoeff = K_GI(i,j,0)*K_by_dcoeff(i,j,0,m) + K_GI(i,j,1)*K_by_dcoeff(i,j,1,m) + K_GI(i,j,2)*K_by_dcoeff(i,j,2,m);
              data(m) += -K_GI_dot_K_by_dcoeff*normn;
          }
      }
  }
};

template<class Array>
void CurrentPotential<Array>::K_matrix_impl_helper(Array& data, Array& dg1, Array& dg2, Array& normal) {
    int ndofs = num_dofs();
    Array K_by_dcoeff = xt::zeros<double>({numquadpoints_phi, numquadpoints_theta, 3, ndofs});
    this->K_by_dcoeff_impl_helper(K_by_dcoeff, dg1, dg2, normal);

  for (int i = 0; i < numquadpoints_phi; ++i) {
      for (int j = 0; j < numquadpoints_theta; ++j) {
          double normn = std::sqrt(normal(i, j, 0)*normal(i, j, 0) + normal(i, j, 1)*normal(i, j, 1) + normal(i, j, 2)*normal(i, j, 2));

          for (int m = 0; m < ndofs; ++m ) {
              for (int n = 0; n < ndofs; ++ n ) {
                  data(m,n) += (K_by_dcoeff(i,j,0,m)*K_by_dcoeff(i,j,0,n) + \
                              + K_by_dcoeff(i,j,1,m)*K_by_dcoeff(i,j,1,n) + \
                              + K_by_dcoeff(i,j,2,m)*K_by_dcoeff(i,j,2,n))*normn;
              }
          }
      }
  }
};

#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
template class CurrentPotential<Array>;
