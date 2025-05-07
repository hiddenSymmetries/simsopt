#include "surface.h"
#include <Eigen/Dense>
#include "simdhelpers.h"
#include "vec3dsimd.h"


template<class Array>
Array surface_vjp_contraction(const Array& mat, const Array& v){
    if(mat.layout() != xt::layout_type::row_major)
          throw std::runtime_error("mat needs to be in row-major storage order");
    if(v.layout() != xt::layout_type::row_major)
          throw std::runtime_error("v needs to be in row-major storage order");
    int numquadpoints_phi = mat.shape(0);
    int numquadpoints_theta = mat.shape(1);
    int numdofs = mat.shape(3);
    Array res = xt::zeros<double>({numdofs});

    // we have to compute the contraction below:
    //for (int j = 0; j < numquadpoints_phi; ++j) {
    //    for (int jj = 0; jj < numquadpoints_theta; ++jj) {
    //        for (int k = 0; k < 3; ++k) {
    //            for (int i = 0; i < numdofs; ++i) {
    //                res(i) += mat(j, jj, k, i) * v(j, jj, k);
    //            }
    //        }
    //    }
    //}
    // instead of worrying how to do this efficiently, we map to a eigen
    // matrix, and then write this as a matrix vector product.

    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_mat(const_cast<double*>(mat.data()), numquadpoints_phi*numquadpoints_theta*3, numdofs);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_v(const_cast<double*>(v.data()), 1, numquadpoints_phi*numquadpoints_theta*3);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_res(const_cast<double*>(res.data()), 1, numdofs);
    eigen_res = eigen_v*eigen_mat;
    //Eigen::MatrixXd eigen_res = eigen_v*eigen_mat;
    //for (int i = 0; i < numdofs; ++i) {
    //    res(i) = eigen_res(0, i);
    //}
    return res;
}

template<class Array>
void Surface<Array>::least_squares_fit(Array& target_values) {
    if(target_values.shape(0) != numquadpoints_phi)
        throw std::runtime_error("Wrong first dimension for target_values. Should match numquadpoints_phi.");
    if(target_values.shape(1) != numquadpoints_theta)
        throw std::runtime_error("Wrong second dimension for target_values. Should match numquadpoints_theta.");
    if(target_values.shape(2) != 3)
        throw std::runtime_error("Wrong third dimension for target_values. Should be 3.");

    if(!qr){
        auto dg_dc = this->dgamma_by_dcoeff();
        Eigen::MatrixXd A = Eigen::MatrixXd(numquadpoints_phi*numquadpoints_theta*3, num_dofs());
        int counter = 0;
        for (int i = 0; i < numquadpoints_phi; ++i) {
            for (int j = 0; j < numquadpoints_theta; ++j) {
                for (int d = 0; d < 3; ++d) {
                    for (int c = 0; c  < num_dofs(); ++c ) {
                        A(counter, c) = dg_dc(i, j, d, c);
                    }
                    counter++;
                }
            }
        }
        qr = std::make_unique<Eigen::FullPivHouseholderQR<Eigen::MatrixXd>>(A.fullPivHouseholderQr());
    }
    Eigen::VectorXd b = Eigen::VectorXd(numquadpoints_phi*numquadpoints_theta*3);
    int counter = 0;
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            for (int d = 0; d < 3; ++d) {
                b(counter++) = target_values(i, j, d);
            }
        }
    }
    Eigen::VectorXd x = qr->solve(b);
    vector<double> dofs(x.data(), x.data() + x.size());
    this->set_dofs(dofs);
}

template<class Array>
void Surface<Array>::fit_to_curve(Curve<Array>& curve, double radius, bool flip_theta) {
    Array curvexyz = xt::zeros<double>({numquadpoints_phi, 3});
    curve.gamma_impl(curvexyz, quadpoints_phi);
    Array target_values = xt::zeros<double>({numquadpoints_phi, numquadpoints_theta, 3});
    for (int i = 0; i < numquadpoints_phi; ++i) {
        double phi = 2*M_PI*quadpoints_phi[i];
        double R = sqrt(pow(curvexyz(i, 0), 2) + pow(curvexyz(i, 1), 2));
        double z = curvexyz(i, 2);
        for (int j = 0; j < numquadpoints_theta; ++j) {
            double theta = 2*M_PI*quadpoints_theta[j];
            if(flip_theta)
                theta *= -1;
            target_values(i, j, 0) = (R + radius * cos(theta))*cos(phi);
            target_values(i, j, 1) = (R + radius * cos(theta))*sin(phi);
            target_values(i, j, 2) = z + radius * sin(theta);
        }
    }
    this->least_squares_fit(target_values);
}

template<class Array>
void Surface<Array>::scale(double scale) {
    Array target_values = xt::zeros<double>({numquadpoints_phi, numquadpoints_theta, 3});
    auto gamma = this->gamma();
    for (int i = 0; i < numquadpoints_phi; ++i) {
        double phi = 2*M_PI*quadpoints_phi[i];
        double meanx = 0.;
        double meany = 0.;
        double meanz = 0.;
        for (int j = 0; j < numquadpoints_theta; ++j) {
            meanx += gamma(i, j, 0);
            meany += gamma(i, j, 1);
            meanz += gamma(i, j, 2);
        }
        meanx *= 1./numquadpoints_theta;
        meany *= 1./numquadpoints_theta;
        meanz *= 1./numquadpoints_theta;
        for (int j = 0; j < numquadpoints_theta; ++j) {
            target_values(i, j, 0) = meanx + scale * (gamma(i, j, 0) - meanx);
            target_values(i, j, 1) = meany + scale * (gamma(i, j, 1) - meany);
            target_values(i, j, 2) = meanz + scale * (gamma(i, j, 2) - meanz);
        }
    }
    this->least_squares_fit(target_values);
}

template<class Array>
void Surface<Array>::_extend_via_normal_for_nonuniform_phi(double distance) {
    Array target_values = xt::zeros<double>({numquadpoints_phi, numquadpoints_theta, 3});
    auto gamma = this->gamma();
    auto n = this->normal();
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            auto nij_norm = sqrt(n(i, j, 0)*n(i, j, 0) + n(i, j, 1)*n(i, j, 1) + n(i, j, 2)*n(i, j, 2));
            target_values(i, j, 0) = gamma(i, j, 0) + distance * n(i, j, 0) / nij_norm;
            target_values(i, j, 1) = gamma(i, j, 1) + distance * n(i, j, 1) / nij_norm;
            target_values(i, j, 2) = gamma(i, j, 2) + distance * n(i, j, 2) / nij_norm;
        }
    }
    this->least_squares_fit(target_values);
}

template<class Array>
void Surface<Array>::extend_via_projected_normal(double scale) {
    Array target_values_xyz = xt::zeros<double>({numquadpoints_phi, numquadpoints_theta, 3});
    Array target_values = xt::zeros<double>({numquadpoints_phi, numquadpoints_theta, 3});
    auto gamma = this->gamma();
    auto n = this->normal();
    auto phi = this->quadpoints_phi;
    // Loop over toroidal angle, calculate rotation matrix
    for (int i = 0; i < numquadpoints_phi; ++i) {
	double phi_i = 2 * M_PI * phi[i];
	double rotation[3][3] = {cos(phi_i), sin(phi_i), 0, -sin(phi_i), cos(phi_i), 0, 0, 0, 1};
	double rotation_inv[3][3] = {cos(phi_i), -sin(phi_i), 0, sin(phi_i), cos(phi_i), 0, 0, 0, 1};
        for (int j = 0; j < numquadpoints_theta; ++j) {
	    double nij[3] = {0.0, 0.0, 0.0};
	    double gammaij[3] = {0.0, 0.0, 0.0};
	    // convert nij and gammij from cartesian to cylindrical
            for (int k = 0; k < 3; ++k) {
                for (int kk = 0; kk < 3; ++kk) {
                    nij[k] += rotation[k][kk] * n(i, j, kk);
                    gammaij[k] += rotation[k][kk] * gamma(i, j, kk);
		}
	    }
            // keep toroidal direction constant
	    nij[1] = 0.0;
	    gammaij[1] = 0.0;
	    auto nij_norm = sqrt(nij[0]*nij[0] + nij[2]*nij[2]);
            target_values(i, j, 0) = gammaij[0] + scale * nij[0] / nij_norm;
            target_values(i, j, 2) = gammaij[2] + scale * nij[2] / nij_norm;
	    // now need to take target_values in (R, Z) and transform back to (X, Y, Z)
            for (int k = 0; k < 3; ++k) {
                for (int kk = 0; kk < 3; ++kk) {
                    target_values_xyz(i, j, k) += rotation_inv[k][kk] * target_values(i, j, kk);
		}
	    }
        }
    }
    this->least_squares_fit(target_values_xyz);
}

template<class Array>
void Surface<Array>::first_fund_form_impl(Array& data) {
  auto drd1 = this->gammadash1();
  auto drd2 = this->gammadash2();
  for (int i = 0; i < numquadpoints_phi; ++i) {
      for (int j = 0; j < numquadpoints_theta; ++j) {
          data(i, j, 0) = drd1(i,j,0) * drd1(i,j,0) + drd1(i,j,1) * drd1(i,j,1) + drd1(i,j,2) * drd1(i,j,2);
          data(i, j, 1) = drd1(i,j,0) * drd2(i,j,0) + drd1(i,j,1) * drd2(i,j,1) + drd1(i,j,2) * drd2(i,j,2);
          data(i, j, 2) = drd2(i,j,0) * drd2(i,j,0) + drd2(i,j,1) * drd2(i,j,1) + drd2(i,j,2) * drd2(i,j,2);
      }
  }
};

template<class Array>
void Surface<Array>::dfirst_fund_form_by_dcoeff_impl(Array& data) {
  auto drd1 = this->gammadash1();
  auto drd2 = this->gammadash2();
  auto drd1_dc = this->dgammadash1_by_dcoeff();
  auto drd2_dc = this->dgammadash2_by_dcoeff();
  int ndofs = num_dofs();
  for (int i = 0; i < numquadpoints_phi; ++i) {
      for (int j = 0; j < numquadpoints_theta; ++j) {
          double* data_0_ptr = &(data(i, j, 0, 0));
          double* data_1_ptr = &(data(i, j, 1, 0));
          double* data_2_ptr = &(data(i, j, 2, 0));

          double* drd1_0_dc_ptr = &(drd1_dc(i, j, 0, 0));
          double* drd1_1_dc_ptr = &(drd1_dc(i, j, 1, 0));
          double* drd1_2_dc_ptr = &(drd1_dc(i, j, 2, 0));
          double* drd2_0_dc_ptr = &(drd2_dc(i, j, 0, 0));
          double* drd2_1_dc_ptr = &(drd2_dc(i, j, 1, 0));
          double* drd2_2_dc_ptr = &(drd2_dc(i, j, 2, 0));

          for (int m = 0; m < ndofs; ++m ) {
              data_0_ptr[m] =  2*drd1_0_dc_ptr[m]*drd1(i,j,0) + 2*drd1_1_dc_ptr[m]*drd1(i,j,1) \
                             + 2*drd1_2_dc_ptr[m]*drd1(i,j,2);
          }
          for (int m = 0; m < ndofs; ++m ) {
              data_1_ptr[m] =  drd1_0_dc_ptr[m]*drd2(i,j,0) + drd2_0_dc_ptr[m]*drd1(i,j,0) \
                             + drd1_1_dc_ptr[m]*drd2(i,j,1) + drd2_1_dc_ptr[m]*drd1(i,j,1) \
                             + drd1_2_dc_ptr[m]*drd2(i,j,2) + drd2_2_dc_ptr[m]*drd1(i,j,2);
          }
          for (int m = 0; m < ndofs; ++m ) {
              data_2_ptr[m] =  2*drd2_0_dc_ptr[m]*drd2(i,j,0) + 2*drd2_1_dc_ptr[m]*drd2(i,j,1) \
                             + 2*drd2_2_dc_ptr[m]*drd2(i,j,2);
          }

      }
  }
};

template<class Array>
void Surface<Array>::second_fund_form_impl(Array& data) {
  auto drd1 = this->gammadash1();
  auto drd2 = this->gammadash2();
  auto d2rd1d1 = this->gammadash1dash1();
  auto d2rd1d2 = this->gammadash1dash2();
  auto d2rd2d2 = this->gammadash2dash2();
  auto unitnormal = this->unitnormal();
  for (int i = 0; i < numquadpoints_phi; ++i) {
      for (int j = 0; j < numquadpoints_theta; ++j) {
          data(i, j, 0) = unitnormal(i,j,0) * d2rd1d1(i,j,0) + unitnormal(i,j,1) * d2rd1d1(i,j,1) + unitnormal(i,j,2) * d2rd1d1(i,j,2);
          data(i, j, 1) = unitnormal(i,j,0) * d2rd1d2(i,j,0) + unitnormal(i,j,1) * d2rd1d2(i,j,1) + unitnormal(i,j,2) * d2rd1d2(i,j,2);
          data(i, j, 2) = unitnormal(i,j,0) * d2rd2d2(i,j,0) + unitnormal(i,j,1) * d2rd2d2(i,j,1) + unitnormal(i,j,2) * d2rd2d2(i,j,2);
      }
  }
};

template<class Array>
void Surface<Array>::dsecond_fund_form_by_dcoeff_impl(Array& data) {
  auto d2rd1d1 = this->gammadash1dash1();
  auto d2rd1d2 = this->gammadash1dash2();
  auto d2rd2d2 = this->gammadash2dash2();
  auto unitnormal = this->unitnormal();
  auto dd2rd1d1_dc = this->dgammadash1dash1_by_dcoeff();
  auto dd2rd1d2_dc = this->dgammadash1dash2_by_dcoeff();
  auto dd2rd2d2_dc = this->dgammadash2dash2_by_dcoeff();
  auto dunitnormal_dc = this->dunitnormal_by_dcoeff();

  int ndofs = num_dofs();
  for (int i = 0; i < numquadpoints_phi; ++i) {
      for (int j = 0; j < numquadpoints_theta; ++j) {

        double* data_0_ptr = &(data(i, j, 0, 0));
        double* data_1_ptr = &(data(i, j, 1, 0));
        double* data_2_ptr = &(data(i, j, 2, 0));

        double* dd2rd1d1_0_dc_ptr = &(dd2rd1d1_dc(i, j, 0, 0));
        double* dd2rd1d1_1_dc_ptr = &(dd2rd1d1_dc(i, j, 1, 0));
        double* dd2rd1d1_2_dc_ptr = &(dd2rd1d1_dc(i, j, 2, 0));
        double* dd2rd1d2_0_dc_ptr = &(dd2rd1d2_dc(i, j, 0, 0));
        double* dd2rd1d2_1_dc_ptr = &(dd2rd1d2_dc(i, j, 1, 0));
        double* dd2rd1d2_2_dc_ptr = &(dd2rd1d2_dc(i, j, 2, 0));
        double* dd2rd2d2_0_dc_ptr = &(dd2rd2d2_dc(i, j, 0, 0));
        double* dd2rd2d2_1_dc_ptr = &(dd2rd2d2_dc(i, j, 1, 0));
        double* dd2rd2d2_2_dc_ptr = &(dd2rd2d2_dc(i, j, 2, 0));
        double* dunitnormal_0_dc_ptr = &(dunitnormal_dc(i, j, 0, 0));
        double* dunitnormal_1_dc_ptr = &(dunitnormal_dc(i, j, 1, 0));
        double* dunitnormal_2_dc_ptr = &(dunitnormal_dc(i, j, 2, 0));

        for (int m = 0; m < ndofs; ++m ) {
            data_0_ptr[m] = dunitnormal_0_dc_ptr[m] * d2rd1d1(i,j,0) \
                          + unitnormal(i,j,0) * dd2rd1d1_0_dc_ptr[m] \
                          + dunitnormal_1_dc_ptr[m] * d2rd1d1(i,j,1) \
                          + unitnormal(i,j,1) * dd2rd1d1_1_dc_ptr[m] \
                          + dunitnormal_2_dc_ptr[m] * d2rd1d1(i,j,2) \
                          + unitnormal(i,j,2) * dd2rd1d1_2_dc_ptr[m];
            data_1_ptr[m] = dunitnormal_0_dc_ptr[m] * d2rd1d2(i,j,0) \
                          + unitnormal(i,j,0) * dd2rd1d2_0_dc_ptr[m] \
                          + dunitnormal_1_dc_ptr[m] * d2rd1d2(i,j,1) \
                          + unitnormal(i,j,1) * dd2rd1d2_1_dc_ptr[m] \
                          + dunitnormal_2_dc_ptr[m] * d2rd1d2(i,j,2) \
                          + unitnormal(i,j,2) * dd2rd1d2_2_dc_ptr[m];
            data_2_ptr[m] = dunitnormal_0_dc_ptr[m] * d2rd2d2(i,j,0) \
                          + unitnormal(i,j,0) * dd2rd2d2_0_dc_ptr[m] \
                          + dunitnormal_1_dc_ptr[m] * d2rd2d2(i,j,1) \
                          + unitnormal(i,j,1) * dd2rd2d2_1_dc_ptr[m] \
                          + dunitnormal_2_dc_ptr[m] * d2rd2d2(i,j,2) \
                          + unitnormal(i,j,2) * dd2rd2d2_2_dc_ptr[m];
        }
      }
  }
};

template<class Array>
void Surface<Array>::dsurface_curvatures_by_dcoeff_impl(Array& data) {
  auto first = this->first_fund_form();
  auto second = this->second_fund_form();
  auto dfirst_dc = this->dfirst_fund_form_by_dcoeff();
  auto dsecond_dc = this->dsecond_fund_form_by_dcoeff();
  int ndofs = num_dofs();
  for (int i = 0; i < numquadpoints_phi; ++i) {
      for (int j = 0; j < numquadpoints_theta; ++j) {

          double* data_0_ptr = &(data(i, j, 0, 0));
          double* data_1_ptr = &(data(i, j, 1, 0));
          double* data_2_ptr = &(data(i, j, 2, 0));
          double* data_3_ptr = &(data(i, j, 3, 0));

          double* dfirst_0_dc_ptr = &(dfirst_dc(i, j, 0, 0));
          double* dfirst_1_dc_ptr = &(dfirst_dc(i, j, 1, 0));
          double* dfirst_2_dc_ptr = &(dfirst_dc(i, j, 2, 0));
          double* dsecond_0_dc_ptr = &(dsecond_dc(i, j, 0, 0));
          double* dsecond_1_dc_ptr = &(dsecond_dc(i, j, 1, 0));
          double* dsecond_2_dc_ptr = &(dsecond_dc(i, j, 2, 0));

          auto denom = first(i,j,0)*first(i,j,2) - first(i,j,1)*first(i,j,1);
          auto data_0 = (second(i,j,0)*first(i,j,2) - 2*first(i,j,1)*second(i,j,1) + second(i,j,2)*first(i,j,0))/(2*denom);
          auto data_1 = (second(i,j,0)*second(i,j,2) - second(i,j,1)*second(i,j,1))/denom;
          auto term2 = std::sqrt(data_0*data_0 - data_1);
          for (int m = 0; m < ndofs; ++m ) {
              auto d_denom = dfirst_0_dc_ptr[m] * first(i,j,2) \
                           + first(i,j,0) * dfirst_2_dc_ptr[m] \
                           - 2*dfirst_1_dc_ptr[m] * first(i,j,1);
              data_0_ptr[m] = (dsecond_0_dc_ptr[m] * first(i,j,2) \
                            + second(i,j,0) * dfirst_2_dc_ptr[m] \
                            - 2*dfirst_1_dc_ptr[m] * second(i,j,1) \
                            - 2*first(i,j,1) * dsecond_1_dc_ptr[m] \
                            + dsecond_2_dc_ptr[m] * first(i,j,0) \
                            + second(i,j,2) * dfirst_0_dc_ptr[m])/(2*denom) \
                            - data_0*d_denom/denom;
              data_1_ptr[m] = (dsecond_0_dc_ptr[m]*second(i,j,2) + second(i,j,0)*dsecond_2_dc_ptr[m] - 2*dsecond_1_dc_ptr[m]*second(i,j,1))/denom \
                            - data_1*d_denom/denom;
              auto d_term2 = (data_0*data_0_ptr[m] - 0.5*data_1_ptr[m])/term2;

              data_2_ptr[m] = data_0_ptr[m] + d_term2;
              data_3_ptr[m] = data_0_ptr[m] - d_term2;
          }
      }
  }
};


template<class Array>
void Surface<Array>::surface_curvatures_impl(Array& data) {
  auto drd1 = this->gammadash1();
  auto drd2 = this->gammadash2();
  auto d2rd1d1 = this->gammadash1dash1();
  auto d2rd1d2 = this->gammadash1dash2();
  auto d2rd2d2 = this->gammadash2dash2();
  auto dg1_dc = this->dgammadash1_by_dcoeff();
  auto dg2_dc = this->dgammadash2_by_dcoeff();
  int ndofs = num_dofs();

  auto first = this->first_fund_form();
  auto second = this->second_fund_form();
  for (int i = 0; i < numquadpoints_phi; ++i) {
      for (int j = 0; j < numquadpoints_theta; ++j) {
          auto denom = first(i,j,0)*first(i,j,2) - first(i,j,1)*first(i,j,1);
          data(i, j, 0) = (second(i,j,0)*first(i,j,2) - 2*first(i,j,1)*second(i,j,1) + second(i,j,2)*first(i,j,0))/(2*denom); // H
          data(i, j, 1) = (second(i,j,0)*second(i,j,2) - second(i,j,1)*second(i,j,1))/denom; // K
          data(i, j, 2) = data(i, j, 0) + std::sqrt(data(i, j, 0)*data(i, j, 0) - data(i, j, 1));
          data(i, j, 3) = data(i, j, 0) - std::sqrt(data(i, j, 0)*data(i, j, 0) - data(i, j, 1));
      }
  }
};

template<class Array>
void Surface<Array>::normal_impl(Array& data)  {
    auto dg1 = this->gammadash1();
    auto dg2 = this->gammadash2();
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            data(i, j, 0) = dg1(i, j, 1)*dg2(i, j, 2) - dg1(i, j, 2)*dg2(i, j, 1);
            data(i, j, 1) = dg1(i, j, 2)*dg2(i, j, 0) - dg1(i, j, 0)*dg2(i, j, 2);
            data(i, j, 2) = dg1(i, j, 0)*dg2(i, j, 1) - dg1(i, j, 1)*dg2(i, j, 0);
        }
    }
};
template<class Array>
void Surface<Array>::dnormal_by_dcoeff_impl(Array& data)  {
    auto dg1 = this->gammadash1();
    auto dg2 = this->gammadash2();
    auto dg1_dc = this->dgammadash1_by_dcoeff();
    auto dg2_dc = this->dgammadash2_by_dcoeff();
    int ndofs = num_dofs();
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            double* data_0_ptr = &(data(i, j, 0, 0));
            double* data_1_ptr = &(data(i, j, 1, 0));
            double* data_2_ptr = &(data(i, j, 2, 0));
            double* dg1_0_dc_ptr = &(dg1_dc(i, j, 0, 0));
            double* dg1_1_dc_ptr = &(dg1_dc(i, j, 1, 0));
            double* dg1_2_dc_ptr = &(dg1_dc(i, j, 2, 0));
            double* dg2_0_dc_ptr = &(dg2_dc(i, j, 0, 0));
            double* dg2_1_dc_ptr = &(dg2_dc(i, j, 1, 0));
            double* dg2_2_dc_ptr = &(dg2_dc(i, j, 2, 0));
            for (int m = 0; m < ndofs; ++m ) {
                data_0_ptr[m] =  dg1_1_dc_ptr[m]*dg2(i, j, 2) - dg1_2_dc_ptr[m]*dg2(i, j, 1);
                data_0_ptr[m] += dg1(i, j, 1)*dg2_2_dc_ptr[m] - dg1(i, j, 2)*dg2_1_dc_ptr[m];
            }
            for (int m = 0; m < ndofs; ++m ) {
                data_1_ptr[m] =  dg1_2_dc_ptr[m]*dg2(i, j, 0) - dg1_0_dc_ptr[m]*dg2(i, j, 2);
                data_1_ptr[m] += dg1(i, j, 2)*dg2_0_dc_ptr[m] - dg1(i, j, 0)*dg2_2_dc_ptr[m];
            }
            for (int m = 0; m < ndofs; ++m ) {
                data_2_ptr[m] =  dg1_0_dc_ptr[m]*dg2(i, j, 1) - dg1_1_dc_ptr[m]*dg2(i, j, 0);
                data_2_ptr[m] += dg1(i, j, 0)*dg2_1_dc_ptr[m] - dg1(i, j, 1)*dg2_0_dc_ptr[m];
            }
        }
    }
};

template<class Array>
void Surface<Array>::d2normal_by_dcoeffdcoeff_impl(Array& data)  {
    auto dg1 = this->gammadash1();
    auto dg2 = this->gammadash2();
    auto dg1_dc = this->dgammadash1_by_dcoeff();
    auto dg2_dc = this->dgammadash2_by_dcoeff();
    int ndofs = num_dofs();
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            for (int m = 0; m < ndofs; ++m ) {
                for (int n = 0; n < ndofs; ++n ) {
                    data(i, j, 0, m, n) =  dg1_dc(i, j, 1, m)*dg2_dc(i, j, 2, n) - dg1_dc(i, j, 2, m)*dg2_dc(i, j, 1, n);
                    data(i, j, 0, m, n) += dg1_dc(i, j, 1, n)*dg2_dc(i, j, 2, m) - dg1_dc(i, j, 2, n)*dg2_dc(i, j, 1, m);
                    data(i, j, 1, m, n) =  dg1_dc(i, j, 2, m)*dg2_dc(i, j, 0, n) - dg1_dc(i, j, 0, m)*dg2_dc(i, j, 2, n);
                    data(i, j, 1, m, n) += dg1_dc(i, j, 2, n)*dg2_dc(i, j, 0, m) - dg1_dc(i, j, 0, n)*dg2_dc(i, j, 2, m);
                    data(i, j, 2, m, n) =  dg1_dc(i, j, 0, m)*dg2_dc(i, j, 1, n) - dg1_dc(i, j, 1, m)*dg2_dc(i, j, 0, n);
                    data(i, j, 2, m, n) += dg1_dc(i, j, 0, n)*dg2_dc(i, j, 1, m) - dg1_dc(i, j, 1, n)*dg2_dc(i, j, 0, m);
                }
            }
        }
    }
};

template<class Array>
void Surface<Array>::unitnormal_impl(Array& data)  {
    auto n = this->normal();
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            double normn = std::sqrt(n(i, j, 0)*n(i, j, 0) + n(i, j, 1)*n(i, j, 1) + n(i, j, 2)*n(i, j, 2));
            data(i, j, 0) = n(i, j, 0)/normn;
            data(i, j, 1) = n(i, j, 1)/normn;
            data(i, j, 2) = n(i, j, 2)/normn;
        }
    }
};

template<class Array>
void Surface<Array>::dunitnormal_by_dcoeff_impl(Array& data)  {
    auto n = this->normal();
    auto dn_dc = this->dnormal_by_dcoeff();
    int ndofs = num_dofs();
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
          double* data_0_ptr = &(data(i, j, 0, 0));
          double* data_1_ptr = &(data(i, j, 1, 0));
          double* data_2_ptr = &(data(i, j, 2, 0));
          double* dn_0_dc_ptr = &(dn_dc(i, j, 0, 0));
          double* dn_1_dc_ptr = &(dn_dc(i, j, 1, 0));
          double* dn_2_dc_ptr = &(dn_dc(i, j, 2, 0));
          auto normn = std::sqrt(n(i, j, 0)*n(i, j, 0) + n(i, j, 1)*n(i, j, 1) + n(i, j, 2)*n(i, j, 2));
          auto fac0 = -n(i,j,0)*n(i,j,0)/(normn*normn*normn) + 1/normn;
          auto fac1 = -n(i,j,1)*n(i,j,1)/(normn*normn*normn) + 1/normn;
          auto fac2 = -n(i,j,2)*n(i,j,2)/(normn*normn*normn) + 1/normn;
          for (int m = 0; m < ndofs; ++m ) {
              data_0_ptr[m] =  fac0 * dn_0_dc_ptr[m] - n(i,j,0)*n(i,j,1)*dn_1_dc_ptr[m]/(normn*normn*normn)
                                                     - n(i,j,0)*n(i,j,2)*dn_2_dc_ptr[m]/(normn*normn*normn);
              data_1_ptr[m] =  fac1 * dn_1_dc_ptr[m] - n(i,j,1)*n(i,j,0)*dn_0_dc_ptr[m]/(normn*normn*normn)
                                                     - n(i,j,1)*n(i,j,2)*dn_2_dc_ptr[m]/(normn*normn*normn);
              data_2_ptr[m] =  fac2 * dn_2_dc_ptr[m] - n(i,j,2)*n(i,j,0)*dn_0_dc_ptr[m]/(normn*normn*normn)
                                                     - n(i,j,2)*n(i,j,1)*dn_1_dc_ptr[m]/(normn*normn*normn);
          }
        }
    }
};

template<class Array>
double Surface<Array>::area() {
    double area = 0.;
    auto n = this->normal();
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            area += sqrt(n(i,j,0)*n(i,j,0) + n(i,j,1)*n(i,j,1) + n(i,j,2)*n(i,j,2));
        }
    }
    return area/(numquadpoints_phi*numquadpoints_theta);
}

template<class Array>
void Surface<Array>::darea_by_dcoeff_impl(Array& data) {
    auto n = this->normal();
    int ndofs = num_dofs();

    //auto dn_dc = this->dnormal_by_dcoeff();
    //data *= 0.;
    //for (int i = 0; i < numquadpoints_phi; ++i) {
    //    for (int j = 0; j < numquadpoints_theta; ++j) {
    //        double norm = sqrt(n(i,j,0)*n(i,j,0) + n(i,j,1)*n(i,j,1) + n(i,j,2)*n(i,j,2));
    //        for (int m = 0; m < ndofs; ++m) {
    //            data(m) += (dn_dc(i,j,0,m)*n(i,j,0) + dn_dc(i,j,1,m)*n(i,j,1) + dn_dc(i,j,2,m)*n(i,j,2)) / norm;
    //        }
    //    }
    //}
    //data *= 1./ (numquadpoints_phi*numquadpoints_theta);
    //return;


    Array darea_by_dn = xt::zeros<double>({numquadpoints_phi, numquadpoints_theta, 3});
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            double norm = sqrt(n(i,j,0)*n(i,j,0) + n(i,j,1)*n(i,j,1) + n(i,j,2)*n(i,j,2));
            darea_by_dn(i, j, 0) = n(i, j, 0)/norm;
            darea_by_dn(i, j, 1) = n(i, j, 1)/norm;
            darea_by_dn(i, j, 2) = n(i, j, 2)/norm;
        }
    }
    darea_by_dn *= 1./ (numquadpoints_phi*numquadpoints_theta);
    Array temp = dnormal_by_dcoeff_vjp(darea_by_dn);
    for (int m = 0; m < ndofs; ++m) {
        data(m) = temp(m);
    }
}

template<class Array>
Array Surface<Array>::dnormal_by_dcoeff_vjp(Array& v) {
    auto dg1 = this->gammadash1();
    auto dg2 = this->gammadash2();
    Array res_dgammadash1 = xt::zeros<double>({numquadpoints_phi, numquadpoints_theta, 3});
    Array res_dgammadash2 = xt::zeros<double>({numquadpoints_phi, numquadpoints_theta, 3});
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            for (int d = 0; d < 3; ++d) {
                double ed[3] = {0., 0., 0.};
                ed[d] = 1.0;
                double temp[3] = {0., 0., 0.};
                temp[0] = ed[1]*dg2(i, j, 2) - ed[2]*dg2(i, j, 1);
                temp[1] = ed[2]*dg2(i, j, 0) - ed[0]*dg2(i, j, 2);
                temp[2] = ed[0]*dg2(i, j, 1) - ed[1]*dg2(i, j, 0);
                res_dgammadash1(i, j, d) = temp[0] * v(i, j, 0) + temp[1] * v(i, j, 1) + temp[2] * v(i, j, 2);
                temp[0] = dg1(i, j, 1)*ed[2] - dg1(i, j, 2)*ed[1];
                temp[1] = dg1(i, j, 2)*ed[0] - dg1(i, j, 0)*ed[2];
                temp[2] = dg1(i, j, 0)*ed[1] - dg1(i, j, 1)*ed[0];
                res_dgammadash2(i, j, d) = temp[0] * v(i, j, 0) + temp[1] * v(i, j, 1) + temp[2] * v(i, j, 2);
            }
        }
    }
    return dgammadash1_by_dcoeff_vjp(res_dgammadash1) + dgammadash2_by_dcoeff_vjp(res_dgammadash2);
}

template<class Array>
void Surface<Array>::d2area_by_dcoeffdcoeff_impl(Array& data) {
    data *= 0.;
    double norm, dnorm_dcoeffn;
    auto nor = this->normal();
    auto dnor_dc = this->dnormal_by_dcoeff();
    auto d2nor_dcdc = this->d2normal_by_dcoeffdcoeff();
    int ndofs = num_dofs();
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            for (int m = 0; m < ndofs; ++m) {
                for (int n = 0; n < ndofs; ++n) {
                    norm = sqrt(nor(i,j,0)*nor(i,j,0)
                            + nor(i,j,1)*nor(i,j,1)
                            + nor(i,j,2)*nor(i,j,2));
                    dnorm_dcoeffn = (dnor_dc(i,j,0,n)*nor(i,j,0)
                            + dnor_dc(i,j,1,n)*nor(i,j,1)
                            + dnor_dc(i,j,2,n)*nor(i,j,2)) / norm;
                    data(m,n) +=  dnor_dc(i,j,0,m) * (dnor_dc(i,j,0,n) * norm - dnorm_dcoeffn * nor(i,j,0)) / (norm*norm)
                        + dnor_dc(i,j,1,m) * (dnor_dc(i,j,1,n) * norm - dnorm_dcoeffn * nor(i,j,1)) / (norm*norm)
                        + dnor_dc(i,j,2,m) * (dnor_dc(i,j,2,n) * norm - dnorm_dcoeffn * nor(i,j,2)) / (norm*norm)
                        + d2nor_dcdc(i,j,0,m,n) * nor(i,j,0) / norm
                        + d2nor_dcdc(i,j,1,m,n) * nor(i,j,1) / norm
                        + d2nor_dcdc(i,j,2,m,n) * nor(i,j,2) / norm;
                }
            }
        }
    }
    data *= 1./ (numquadpoints_phi*numquadpoints_theta);
}


template<class Array>
double Surface<Array>::volume() {
    double volume = 0.;
    auto n = this->normal();
    auto xyz = this->gamma();

    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            volume += (1./3) * (xyz(i, j, 0)*n(i,j,0)+xyz(i,j,1)*n(i,j,1)+xyz(i,j,2)*n(i,j,2));
        }
    }
    return volume/(numquadpoints_phi*numquadpoints_theta);

}

template<class Array>
void Surface<Array>::dvolume_by_dcoeff_impl(Array& data) {
    auto n = this->normal();
    auto xyz = this->gamma();
    int ndofs = num_dofs();
    //data *= 0.;
    //auto dn_dc = this->dnormal_by_dcoeff();
    //auto dxyz_dc = this->dgamma_by_dcoeff();
    //for (int i = 0; i < numquadpoints_phi; ++i) {
    //    for (int j = 0; j < numquadpoints_theta; ++j) {
    //        for (int m = 0; m < ndofs; ++m) {
    //            data(m) += (1./3) * (dxyz_dc(i,j,0,m)*n(i,j,0)+dxyz_dc(i,j,1,m)*n(i,j,1)+dxyz_dc(i,j,2,m)*n(i,j,2));
    //            data(m) += (1./3) * (xyz(i,j,0)*dn_dc(i,j,0,m)+xyz(i,j,1)*dn_dc(i,j,1,m)+xyz(i,j,2)*dn_dc(i,j,2,m));
    //        }
    //    }
    //}
    //data *= 1./ (numquadpoints_phi*numquadpoints_theta);

    Array dvolume_by_dn = xt::zeros<double>({numquadpoints_phi, numquadpoints_theta, 3});
    Array dvolume_by_dx = xt::zeros<double>({numquadpoints_phi, numquadpoints_theta, 3});
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            dvolume_by_dn(i, j, 0) = (1./3) * xyz(i, j, 0);
            dvolume_by_dn(i, j, 1) = (1./3) * xyz(i, j, 1);
            dvolume_by_dn(i, j, 2) = (1./3) * xyz(i, j, 2);
            dvolume_by_dx(i, j, 0) = (1./3) * n(i, j, 0);
            dvolume_by_dx(i, j, 1) = (1./3) * n(i, j, 1);
            dvolume_by_dx(i, j, 2) = (1./3) * n(i, j, 2);
        }
    }
    dvolume_by_dn *= 1./(numquadpoints_phi*numquadpoints_theta);
    dvolume_by_dx *= 1./(numquadpoints_phi*numquadpoints_theta);
    Array temp = dnormal_by_dcoeff_vjp(dvolume_by_dn) + dgamma_by_dcoeff_vjp(dvolume_by_dx);
    for (int m = 0; m < ndofs; ++m) {
        data(m) = temp(m);
    }
}

#if defined(USE_XSIMD)

template<class Array>
void Surface<Array>::d2volume_by_dcoeffdcoeff_impl(Array& data) {
    // this vectorized version of d2volume_by_dcoeffdcoeff computes the second derivative of
    // the surface normal on the fly, which alleviates memory requirements.
    constexpr int simd_size = xsimd::simd_type<double>::size;
    data *= 0.;
    auto nor = this->normal();
    auto xyz = this->gamma();
    auto dxyz_dc = this->dgamma_by_dcoeff();
    auto dnor_dc = this->dnormal_by_dcoeff();
    auto dg1_dc = this->dgammadash1_by_dcoeff();
    auto dg2_dc = this->dgammadash2_by_dcoeff();
    int ndofs = num_dofs();
    
    auto dg1x_dc = AlignedPaddedVec(ndofs, 0);
    auto dg1y_dc = AlignedPaddedVec(ndofs, 0);
    auto dg1z_dc = AlignedPaddedVec(ndofs, 0);

    auto dg2x_dc = AlignedPaddedVec(ndofs, 0);
    auto dg2y_dc = AlignedPaddedVec(ndofs, 0);
    auto dg2z_dc = AlignedPaddedVec(ndofs, 0);

    auto dnorx_dc = AlignedPaddedVec(ndofs, 0);
    auto dnory_dc = AlignedPaddedVec(ndofs, 0);
    auto dnorz_dc = AlignedPaddedVec(ndofs, 0);

    auto dxyzx_dc = AlignedPaddedVec(ndofs, 0);
    auto dxyzy_dc = AlignedPaddedVec(ndofs, 0);
    auto dxyzz_dc = AlignedPaddedVec(ndofs, 0);

    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            simd_t xyzij0(xyz(i,j,0));
            simd_t xyzij1(xyz(i,j,1));
            simd_t xyzij2(xyz(i,j,2));
            
            // load the tangents, normals, and derivatives wrt to surface coeffs into aligned and padded memory
            for (int n = 0; n < ndofs; ++n) {
                dg1x_dc[n] = dg1_dc(i, j, 0, n);
                dg1y_dc[n] = dg1_dc(i, j, 1, n);
                dg1z_dc[n] = dg1_dc(i, j, 2, n);

                dg2x_dc[n] = dg2_dc(i, j, 0, n);
                dg2y_dc[n] = dg2_dc(i, j, 1, n);
                dg2z_dc[n] = dg2_dc(i, j, 2, n);

                dnorx_dc[n] = dnor_dc(i, j, 0, n);
                dnory_dc[n] = dnor_dc(i, j, 1, n);
                dnorz_dc[n] = dnor_dc(i, j, 2, n);
 
                dxyzx_dc[n] = dxyz_dc(i, j, 0, n);
                dxyzy_dc[n] = dxyz_dc(i, j, 1, n);
                dxyzz_dc[n] = dxyz_dc(i, j, 2, n);
            }

            for (int m = 0; m < ndofs; ++m){ 
                simd_t dg1_dc_ij0m(dg1_dc(i, j, 0, m));
                simd_t dg1_dc_ij1m(dg1_dc(i, j, 1, m));
                simd_t dg1_dc_ij2m(dg1_dc(i, j, 2, m));
                
                simd_t dg2_dc_ij0m(dg2_dc(i, j, 0, m));
                simd_t dg2_dc_ij1m(dg2_dc(i, j, 1, m));
                simd_t dg2_dc_ij2m(dg2_dc(i, j, 2, m));
                
                simd_t dxyz_dc_ij0m(dxyz_dc(i, j, 0, m));
                simd_t dxyz_dc_ij1m(dxyz_dc(i, j, 1, m));
                simd_t dxyz_dc_ij2m(dxyz_dc(i, j, 2, m));
                
                simd_t dnor_dc_ij0m(dnor_dc(i, j, 0, m));
                simd_t dnor_dc_ij1m(dnor_dc(i, j, 1, m));
                simd_t dnor_dc_ij2m(dnor_dc(i, j, 2, m));

                for (int n = 0; n < ndofs; n+=simd_size){ 
                    // load into aligned and padded memory into batches
                    simd_t dg1_dc_ij0n = xs::load_aligned(&dg1x_dc[n]);
                    simd_t dg1_dc_ij1n = xs::load_aligned(&dg1y_dc[n]);
                    simd_t dg1_dc_ij2n = xs::load_aligned(&dg1z_dc[n]);
                    
                    simd_t dg2_dc_ij0n = xs::load_aligned(&dg2x_dc[n]);
                    simd_t dg2_dc_ij1n = xs::load_aligned(&dg2y_dc[n]);
                    simd_t dg2_dc_ij2n = xs::load_aligned(&dg2z_dc[n]);
                    
                    simd_t dnor_dc_ij0n = xs::load_aligned(&dnorx_dc[n]);
                    simd_t dnor_dc_ij1n = xs::load_aligned(&dnory_dc[n]);
                    simd_t dnor_dc_ij2n = xs::load_aligned(&dnorz_dc[n]);

                    simd_t dxyz_dc_ij0n = xs::load_aligned(&dxyzx_dc[n]);
                    simd_t dxyz_dc_ij1n = xs::load_aligned(&dxyzy_dc[n]);
                    simd_t dxyz_dc_ij2n = xs::load_aligned(&dxyzz_dc[n]);

                    // compute d2nor_dcdc on the fly
                    //data(i, j, 0, m, n) =  dg1_dc(i, j, 1, m)*dg2_dc(i, j, 2, n) - dg1_dc(i, j, 2, m)*dg2_dc(i, j, 1, n);
                    //data(i, j, 0, m, n) += dg1_dc(i, j, 1, n)*dg2_dc(i, j, 2, m) - dg1_dc(i, j, 2, n)*dg2_dc(i, j, 1, m);
                    //data(i, j, 1, m, n) =  dg1_dc(i, j, 2, m)*dg2_dc(i, j, 0, n) - dg1_dc(i, j, 0, m)*dg2_dc(i, j, 2, n);
                    //data(i, j, 1, m, n) += dg1_dc(i, j, 2, n)*dg2_dc(i, j, 0, m) - dg1_dc(i, j, 0, n)*dg2_dc(i, j, 2, m);
                    //data(i, j, 2, m, n) =  dg1_dc(i, j, 0, m)*dg2_dc(i, j, 1, n) - dg1_dc(i, j, 1, m)*dg2_dc(i, j, 0, n);
                    //data(i, j, 2, m, n) += dg1_dc(i, j, 0, n)*dg2_dc(i, j, 1, m) - dg1_dc(i, j, 1, n)*dg2_dc(i, j, 0, m);
                    auto d2nor_dcdc_ij0mn =  xsimd::fms(dg1_dc_ij1m, dg2_dc_ij2n,  dg1_dc_ij2m*dg2_dc_ij1n);
                         d2nor_dcdc_ij0mn += xsimd::fms(dg1_dc_ij1n, dg2_dc_ij2m,  dg1_dc_ij2n*dg2_dc_ij1m);
                    auto d2nor_dcdc_ij1mn =  xsimd::fms(dg1_dc_ij2m, dg2_dc_ij0n,  dg1_dc_ij0m*dg2_dc_ij2n);
                         d2nor_dcdc_ij1mn += xsimd::fms(dg1_dc_ij2n, dg2_dc_ij0m,  dg1_dc_ij0n*dg2_dc_ij2m);
                    auto d2nor_dcdc_ij2mn =  xsimd::fms(dg1_dc_ij0m, dg2_dc_ij1n,  dg1_dc_ij1m*dg2_dc_ij0n);
                         d2nor_dcdc_ij2mn += xsimd::fms(dg1_dc_ij0n, dg2_dc_ij1m,  dg1_dc_ij1n*dg2_dc_ij0m);
                    
                    // now compute d2volume_by_dcoeffdcoeff
                    //data(m,n) += (1./3) * (dxyz_dc(i,j,0,m)*dnor_dc(i,j,0,n)
                    //        +dxyz_dc(i,j,1,m)*dnor_dc(i,j,1,n)
                    //        +dxyz_dc(i,j,2,m)*dnor_dc(i,j,2,n));
                    //data(m,n) += (1./3) * (xyz(i,j,0)*d2nor_dcdc(i,j,0,m,n) + dxyz_dc(i,j,0,n) * dnor_dc(i,j,0,m)
                    //        +xyz(i,j,1)*d2nor_dcdc(i,j,1,m,n) + dxyz_dc(i,j,1,n) * dnor_dc(i,j,1,m)
                    //        +xyz(i,j,2)*d2nor_dcdc(i,j,2,m,n) + dxyz_dc(i,j,2,n) * dnor_dc(i,j,2,m));
                    auto temp = xsimd::fma(dxyz_dc_ij0m, dnor_dc_ij0n, dxyz_dc_ij1m*dnor_dc_ij1n);
                    auto data1  = (1./3) * xsimd::fma(dxyz_dc_ij2m, dnor_dc_ij2n, temp);
                    auto data2  = (1./3) * (xsimd::fma(xyzij0, d2nor_dcdc_ij0mn , dxyz_dc_ij0n * dnor_dc_ij0m)
                                           +xsimd::fma(xyzij1, d2nor_dcdc_ij1mn , dxyz_dc_ij1n * dnor_dc_ij1m)
                                           +xsimd::fma(xyzij2, d2nor_dcdc_ij2mn , dxyz_dc_ij2n * dnor_dc_ij2m) );

                    int jjlimit = std::min(simd_size, ndofs-n);
                    for(int jj=0; jj<jjlimit; jj++){
                        data(m, n+jj) += data1[jj]+data2[jj];
                    }
                }
            }
        }
    }
    data *= 1./ (numquadpoints_phi*numquadpoints_theta);
}

#else

#include "xsimd/xsimd.hpp"
#include "simdhelpers.h"
#include "vec3dsimd.h"

template<class Array>
void Surface<Array>::d2volume_by_dcoeffdcoeff_impl(Array& data) {
    data *= 0.;
    auto nor = this->normal();
    auto dnor_dc = this->dnormal_by_dcoeff();
    auto d2nor_dcdc = this->d2normal_by_dcoeffdcoeff(); // uses a lot of memory for moderate surface complexity
    auto xyz = this->gamma();
    auto dxyz_dc = this->dgamma_by_dcoeff();
    int ndofs = num_dofs();
    for (int i = 0; i < numquadpoints_phi; ++i) {
        for (int j = 0; j < numquadpoints_theta; ++j) {
            for (int m = 0; m < ndofs; ++m){
                for (int n = 0; n < ndofs; ++n){
                    data(m,n) += (1./3) * (dxyz_dc(i,j,0,m)*dnor_dc(i,j,0,n)
                            +dxyz_dc(i,j,1,m)*dnor_dc(i,j,1,n)
                            +dxyz_dc(i,j,2,m)*dnor_dc(i,j,2,n));
                    data(m,n) += (1./3) * (xyz(i,j,0)*d2nor_dcdc(i,j,0,m,n) + dxyz_dc(i,j,0,n) * dnor_dc(i,j,0,m)
                            +xyz(i,j,1)*d2nor_dcdc(i,j,1,m,n) + dxyz_dc(i,j,1,n) * dnor_dc(i,j,1,m)
                            +xyz(i,j,2)*d2nor_dcdc(i,j,2,m,n) + dxyz_dc(i,j,2,n) * dnor_dc(i,j,2,m));
                }
            }
        }
    }
    data *= 1./ (numquadpoints_phi*numquadpoints_theta);
}

#endif

#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
template class Surface<Array>;
