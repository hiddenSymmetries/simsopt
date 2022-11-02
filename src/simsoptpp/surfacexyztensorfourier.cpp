#include "surfacexyztensorfourier.h"

template<class Array>
SurfaceXYZTensorFourier<Array>::SurfaceXYZTensorFourier(int _mpol, int _ntor, int _nfp, bool _stellsym, std::vector<bool> _clamped_dims, vector<double> _quadpoints_phi, vector<double> _quadpoints_theta)
    : Surface<Array>(_quadpoints_phi, _quadpoints_theta), mpol(_mpol), ntor(_ntor), nfp(_nfp), stellsym(_stellsym), clamped_dims(_clamped_dims) {
        x = xt::zeros<double>({2*mpol+1, 2*ntor+1});
        y = xt::zeros<double>({2*mpol+1, 2*ntor+1});
        z = xt::zeros<double>({2*mpol+1, 2*ntor+1});
        build_cache(quadpoints_phi, quadpoints_theta);
}

template<class Array>
int SurfaceXYZTensorFourier<Array>::num_dofs() {
    if(stellsym)
        return (ntor+1)*(mpol+1)+ ntor*mpol + 2*(ntor+1)*mpol + 2*ntor*(mpol+1);
    else
        return 3 * (2*mpol+1) * (2*ntor+1);
}

template<class Array>
void SurfaceXYZTensorFourier<Array>::set_dofs_impl(const vector<double>& dofs) {
    int counter = 0;
    for (int m = 0; m <= 2*mpol; ++m) {
        for (int n = 0; n <= 2*ntor; ++n) {
            if(skip(0, m, n)) continue;
            x(m, n) = dofs[counter++];
        }
    }
    for (int m = 0; m <= 2*mpol; ++m) {
        for (int n = 0; n <= 2*ntor; ++n) {
            if(skip(1, m, n)) continue;
            y(m, n) = dofs[counter++];
        }
    }
    for (int m = 0; m <= 2*mpol; ++m) {
        for (int n = 0; n <= 2*ntor; ++n) {
            if(skip(2, m, n)) continue;
            z(m, n) = dofs[counter++];
        }
    }
}

template<class Array>
vector<double> SurfaceXYZTensorFourier<Array>::get_dofs() {
    auto res = vector<double>(num_dofs(), 0.);
    int counter = 0;
    for (int m = 0; m <= 2*mpol; ++m) {
        for (int n = 0; n <= 2*ntor; ++n) {
            if(skip(0, m, n)) continue;
            res[counter++] = x(m, n);
        }
    }
    for (int m = 0; m <= 2*mpol; ++m) {
        for (int n = 0; n <= 2*ntor; ++n) {
            if(skip(1, m, n)) continue;
            res[counter++] = y(m, n);
        }
    }
    for (int m = 0; m <= 2*mpol; ++m) {
        for (int n = 0; n <= 2*ntor; ++n) {
            if(skip(2, m, n)) continue;
            res[counter++] = z(m, n);
        }
    }
    return res;
}

template<class Array>
void SurfaceXYZTensorFourier<Array>::gamma_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    rebuild_cache(quadpoints_phi, quadpoints_theta);
    auto numquadpoints_phi = quadpoints_phi.size();
    auto numquadpoints_theta = quadpoints_theta.size();
    data *= 0.;
#pragma omp parallel for
    for (auto k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            double xhat = 0;
            double yhat = 0;
            double z = 0;
            for (int m = 0; m <= 2*mpol; ++m) {
                for (int n = 0; n <= 2*ntor; ++n) {
                    xhat += get_coeff(0, m, n) * basis_fun(0, n, k1, m, k2);
                    yhat += get_coeff(1, m, n) * basis_fun(1, n, k1, m, k2);
                    z += get_coeff(2, m, n) * basis_fun(2, n, k1, m, k2);
                    //xhat += get_coeff(0, m, n) * basis_fun(0, n, phi, m, theta);
                    //yhat += get_coeff(1, m, n) * basis_fun(1, n, phi, m, theta);
                    //z += get_coeff(2, m, n) * basis_fun(2, n, phi, m, theta);
                }
            }
            data(k1, k2, 0) = xhat * cos(phi) - yhat * sin(phi);
            data(k1, k2, 1) = xhat * sin(phi) + yhat * cos(phi);
            data(k1, k2, 2) = z;
        }
    }
}

template<class Array>
void SurfaceXYZTensorFourier<Array>::gamma_lin(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    rebuild_cache(quadpoints_phi, quadpoints_theta);
    int numquadpoints_phi = quadpoints_phi.size();
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        double theta  = 2*M_PI*quadpoints_theta[k1];
        double xhat = 0;
        double yhat = 0;
        double z = 0;
        for (int m = 0; m <= 2*mpol; ++m) {
            for (int n = 0; n <= 2*ntor; ++n) {
                xhat += get_coeff(0, m, n) * basis_fun(0, n, phi, m, theta);
                yhat += get_coeff(1, m, n) * basis_fun(1, n, phi, m, theta);
                z += get_coeff(2, m, n) * basis_fun(2, n, phi, m, theta);
            }
        }
        data(k1, 0) = xhat * cos(phi) - yhat * sin(phi);
        data(k1, 1) = xhat * sin(phi) + yhat * cos(phi);
        data(k1, 2) = z;
    }
}

template<class Array>
void SurfaceXYZTensorFourier<Array>::gammadash1_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    rebuild_cache(quadpoints_phi, quadpoints_theta);
    int numquadpoints_phi = quadpoints_phi.size();
    int numquadpoints_theta = quadpoints_theta.size();
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        double sinphi = sin(phi);
        double cosphi = cos(phi);
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            double xhat = 0;
            double yhat = 0;
            double xhatdash = 0;
            double yhatdash = 0;
            double zdash = 0;
            for (int m = 0; m <= 2*mpol; ++m) {
                for (int n = 0; n <= 2*ntor; ++n) {
                    xhat += get_coeff(0, m, n) * basis_fun(0, n, k1, m, k2);
                    yhat += get_coeff(1, m, n) * basis_fun(1, n, k1, m, k2);
                    xhatdash += get_coeff(0, m, n) * basis_fun_dphi(0, n, k1, m, k2);
                    yhatdash += get_coeff(1, m, n) * basis_fun_dphi(1, n, k1, m, k2);
                    zdash += get_coeff(2, m, n) * basis_fun_dphi(2, n, k1, m, k2);
                    //xhat += get_coeff(0, m, n) * basis_fun(0, n, phi, m, theta);
                    //yhat += get_coeff(1, m, n) * basis_fun(1, n, phi, m, theta);
                    //xhatdash += get_coeff(0, m, n) * basis_fun_dphi(0, n, phi, m, theta);
                    //yhatdash += get_coeff(1, m, n) * basis_fun_dphi(1, n, phi, m, theta);
                    //zdash += get_coeff(2, m, n) * basis_fun_dphi(2, n, phi, m, theta);
                }
            }
            double xdash = xhatdash * cosphi - yhatdash * sinphi - xhat * sinphi - yhat * cosphi;
            double ydash = xhatdash * sinphi + yhatdash * cosphi + xhat * cosphi - yhat * sinphi;
            data(k1, k2, 0) = 2*M_PI*xdash;
            data(k1, k2, 1) = 2*M_PI*ydash;
            data(k1, k2, 2) = 2*M_PI*zdash;
        }
    }
}

template<class Array>
void SurfaceXYZTensorFourier<Array>::gammadash2_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    rebuild_cache(quadpoints_phi, quadpoints_theta);
    int numquadpoints_phi = quadpoints_phi.size();
    int numquadpoints_theta = quadpoints_theta.size();
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        double sinphi = sin(phi);
        double cosphi = cos(phi);
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            double xhat = 0;
            double yhat = 0;
            double xhatdash = 0;
            double yhatdash = 0;
            double zdash = 0;
            for (int m = 0; m <= 2*mpol; ++m) {
                for (int n = 0; n <= 2*ntor; ++n) {
                    xhatdash += get_coeff(0, m, n) * basis_fun_dtheta(0, n, k1, m, k2);
                    yhatdash += get_coeff(1, m, n) * basis_fun_dtheta(1, n, k1, m, k2);
                    zdash += get_coeff(2, m, n) * basis_fun_dtheta(2, n, k1, m, k2);
                    //xhatdash += get_coeff(0, m, n) * basis_fun_dtheta(0, n, phi, m, theta);
                    //yhatdash += get_coeff(1, m, n) * basis_fun_dtheta(1, n, phi, m, theta);
                    //zdash += get_coeff(2, m, n) * basis_fun_dtheta(2, n, phi, m, theta);
                }
            }
            double xdash = xhatdash * cosphi - yhatdash * sinphi;
            double ydash = xhatdash * sinphi + yhatdash * cosphi;
            data(k1, k2, 0) = 2*M_PI*xdash;
            data(k1, k2, 1) = 2*M_PI*ydash;
            data(k1, k2, 2) = 2*M_PI*zdash;
        }
    }
}

template<class Array>
void SurfaceXYZTensorFourier<Array>::gammadash2dash2_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    rebuild_cache(quadpoints_phi, quadpoints_theta);
    int numquadpoints_phi = quadpoints_phi.size();
    int numquadpoints_theta = quadpoints_theta.size();
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        double sinphi = sin(phi);
        double cosphi = cos(phi);
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            double xhatdd = 0;
            double yhatdd = 0;
            double zdd = 0;
            for (int m = 0; m <= 2*mpol; ++m) {
                for (int n = 0; n <= 2*ntor; ++n) {
                    xhatdd += get_coeff(0, m, n) * basis_fun_dthetadtheta(0, n, k1, m, k2);
                    yhatdd += get_coeff(1, m, n) * basis_fun_dthetadtheta(1, n, k1, m, k2);
                    zdd += get_coeff(2, m, n) * basis_fun_dthetadtheta(2, n, k1, m, k2);
                }
            }
            double xdd = xhatdd * cosphi - yhatdd * sinphi;
            double ydd = xhatdd * sinphi + yhatdd * cosphi;
            data(k1, k2, 0) = 4*M_PI*M_PI*xdd;
            data(k1, k2, 1) = 4*M_PI*M_PI*ydd;
            data(k1, k2, 2) = 4*M_PI*M_PI*zdd;
        }
    }
}

template<class Array>
void SurfaceXYZTensorFourier<Array>::gammadash1dash2_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    rebuild_cache(quadpoints_phi, quadpoints_theta);
    int numquadpoints_phi = quadpoints_phi.size();
    int numquadpoints_theta = quadpoints_theta.size();
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        double sinphi = sin(phi);
        double cosphi = cos(phi);
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            double xhatd2 = 0;
            double yhatd2 = 0;
            double xhatd1d2 = 0;
            double yhatd1d2 = 0;
            double zd1d2 = 0;
            for (int m = 0; m <= 2*mpol; ++m) {
                for (int n = 0; n <= 2*ntor; ++n) {
                    xhatd2 += get_coeff(0, m, n) * basis_fun_dtheta(0, n, k1, m, k2);
                    yhatd2 += get_coeff(1, m, n) * basis_fun_dtheta(1, n, k1, m, k2);
                    xhatd1d2 += get_coeff(0, m, n) * basis_fun_dthetadphi(0, n, k1, m, k2);
                    yhatd1d2 += get_coeff(1, m, n) * basis_fun_dthetadphi(1, n, k1, m, k2);
                    zd1d2 += get_coeff(2, m, n) * basis_fun_dthetadphi(2, n, k1, m, k2);
                }
            }
            double xd1d2 = xhatd1d2 * cosphi - xhatd2 * sinphi - yhatd1d2 * sinphi  \
                        -  yhatd2 * cosphi;
            double yd1d2 = xhatd1d2 * sinphi + xhatd2 * cosphi
                        +  yhatd1d2 * cosphi - yhatd2 * sinphi;
            data(k1, k2, 0) = 4*M_PI*M_PI*xd1d2;
            data(k1, k2, 1) = 4*M_PI*M_PI*yd1d2;
            data(k1, k2, 2) = 4*M_PI*M_PI*zd1d2;
        }
    }
}

template<class Array>
void SurfaceXYZTensorFourier<Array>::gammadash1dash1_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    rebuild_cache(quadpoints_phi, quadpoints_theta);
    int numquadpoints_phi = quadpoints_phi.size();
    int numquadpoints_theta = quadpoints_theta.size();
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        double sinphi = sin(phi);
        double cosphi = cos(phi);
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            double xhat = 0;
            double yhat = 0;
            double xhatd = 0;
            double yhatd = 0;
            double xhatdd = 0;
            double yhatdd = 0;
            double zdd = 0;
            for (int m = 0; m <= 2*mpol; ++m) {
                for (int n = 0; n <= 2*ntor; ++n) {
                    xhat += get_coeff(0, m, n) * basis_fun(0, n, k1, m, k2);
                    yhat += get_coeff(1, m, n) * basis_fun(1, n, k1, m, k2);
                    xhatd += get_coeff(0, m, n) * basis_fun_dphi(0, n, k1, m, k2);
                    yhatd += get_coeff(1, m, n) * basis_fun_dphi(1, n, k1, m, k2);
                    xhatdd += get_coeff(0, m, n) * basis_fun_dphidphi(0, n, k1, m, k2);
                    yhatdd += get_coeff(1, m, n) * basis_fun_dphidphi(1, n, k1, m, k2);
                    zdd += get_coeff(2, m, n) * basis_fun_dphidphi(2, n, k1, m, k2);
                }
            }
            double xdd = xhatdd * cosphi - 2 * xhatd * sinphi - xhat * cosphi \
                      -  yhatdd * sinphi - 2 * yhatd * cosphi + yhat * sinphi;
            double ydd = xhatdd * sinphi + 2 * xhatd * cosphi - xhat * sinphi \
                      +  yhatdd * cosphi - 2 * yhatd * sinphi - yhat * cosphi;
            data(k1, k2, 0) = 4*M_PI*M_PI*xdd;
            data(k1, k2, 1) = 4*M_PI*M_PI*ydd;
            data(k1, k2, 2) = 4*M_PI*M_PI*zdd;
        }
    }
}

template<class Array>
void SurfaceXYZTensorFourier<Array>::dgammadash1dash1_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    rebuild_cache(quadpoints_phi, quadpoints_theta);
    int numquadpoints_phi = quadpoints_phi.size();
    int numquadpoints_theta = quadpoints_theta.size();
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            int counter = 0;
            for (int d = 0; d < 3; ++d) {
                for (int m = 0; m <= 2*mpol; ++m) {
                    for (int n = 0; n <= 2*ntor; ++n) {
                        if(skip(d, m, n)) continue;
                        double wivj = basis_fun(d, n, phi, m, theta);
                        double wivjd = basis_fun_dphi(d, n, phi, m, theta);
                        double wivjdd = basis_fun_dphidphi(d, n, phi, m, theta);
                        if(d==0) {
                            double dxhat = wivj;
                            double dxhatd = wivjd;
                            double dxhatdd = wivjdd;
                            double dyhat = 0.;
                            double dyhatd = 0.;
                            double dyhatdd = 0.;
                            double dxdd = dxhatdd * cos(phi) - 2*dxhatd * sin(phi) - dxhat * cos(phi) \
                                        - dyhatdd * sin(phi) - 2*dyhatd * cos(phi) + dyhat * sin(phi);
                            double dydd = dxhatdd * sin(phi) + 2 * dxhatd * cos(phi) - dxhat * sin(phi) \
                                        + dyhatdd * cos(phi) - 2 * dyhatd * sin(phi) - dyhat * cos(phi);
                            data(k1, k2, 0, counter) = 4*M_PI*M_PI*dxdd;
                            data(k1, k2, 1, counter) = 4*M_PI*M_PI*dydd;
                        } else if(d==1) {
                            double dxhat = 0.;
                            double dxhatd = 0.;
                            double dxhatdd = 0.;
                            double dyhat = wivj;
                            double dyhatd = wivjd;
                            double dyhatdd = wivjdd;
                            double dxdd = dxhatdd * cos(phi) - 2*dxhatd * sin(phi) - dxhat * cos(phi) \
                                        - dyhatdd * sin(phi) - 2*dyhatd * cos(phi) + dyhat * sin(phi);
                            double dydd = dxhatdd * sin(phi) + 2 * dxhatd * cos(phi) - dxhat * sin(phi) \
                                        + dyhatdd * cos(phi) - 2 * dyhatd * sin(phi) - dyhat * cos(phi);
                            data(k1, k2, 0, counter) = 4*M_PI*M_PI*dxdd;
                            data(k1, k2, 1, counter) = 4*M_PI*M_PI*dydd;
                        }else {
                            double dzdd = wivjdd;
                            data(k1, k2, 2, counter) = 4*M_PI*M_PI*dzdd;
                        }
                        counter++;
                    }
                }
            }
        }
    }
}

template<class Array>
void SurfaceXYZTensorFourier<Array>::dgammadash1dash2_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    rebuild_cache(quadpoints_phi, quadpoints_theta);
    int numquadpoints_phi = quadpoints_phi.size();
    int numquadpoints_theta = quadpoints_theta.size();
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            int counter = 0;
            for (int d = 0; d < 3; ++d) {
                for (int m = 0; m <= 2*mpol; ++m) {
                    for (int n = 0; n <= 2*ntor; ++n) {
                        if(skip(d, m, n)) continue;
                        double wivjd2 = basis_fun_dtheta(d, n, phi, m, theta);
                        double wivjd1d2 = basis_fun_dthetadphi(d, n, phi, m, theta);
                        if(d==0) {
                            double dxhatd2 = wivjd2;
                            double dxhatd1d2 = wivjd1d2;
                            double dyhatd2 = 0.;
                            double dyhatd1d2 = 0.;
                            double dxd1d2 = dxhatd1d2 * cos(phi) - dxhatd2 * sin(phi) \
                                          - dyhatd1d2 * sin(phi) - dyhatd2 * cos(phi);
                            double dyd1d2 = dxhatd1d2 * sin(phi) + dxhatd2 * cos(phi) \
                                        + dyhatd1d2 * cos(phi) - dyhatd2 * sin(phi);
                            data(k1, k2, 0, counter) = 4*M_PI*M_PI*dxd1d2;
                            data(k1, k2, 1, counter) = 4*M_PI*M_PI*dyd1d2;
                        }else if(d==1) {
                            double dxhatd2 = 0.;
                            double dxhatd1d2 = 0.;
                            double dyhatd2 = wivjd2;
                            double dyhatd1d2 = wivjd1d2;
                            double dxd1d2 = dxhatd1d2 * cos(phi) - dxhatd2 * sin(phi) \
                                          - dyhatd1d2 * sin(phi) - dyhatd2 * cos(phi);
                            double dyd1d2 = dxhatd1d2 * sin(phi) + dxhatd2 * cos(phi) \
                                          + dyhatd1d2 * cos(phi) - dyhatd2 * sin(phi);
                            data(k1, k2, 0, counter) = 4*M_PI*M_PI*dxd1d2;
                            data(k1, k2, 1, counter) = 4*M_PI*M_PI*dyd1d2;
                        }else {
                            double dzd1d2 = wivjd1d2;
                            data(k1, k2, 2, counter) = 4*M_PI*M_PI*dzd1d2;
                        }
                        counter++;
                    }
                }
            }
        }
    }
}

template<class Array>
void SurfaceXYZTensorFourier<Array>::dgammadash2dash2_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    rebuild_cache(quadpoints_phi, quadpoints_theta);
    int numquadpoints_phi = quadpoints_phi.size();
    int numquadpoints_theta = quadpoints_theta.size();
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            int counter = 0;
            for (int d = 0; d < 3; ++d) {
                for (int m = 0; m <= 2*mpol; ++m) {
                    for (int n = 0; n <= 2*ntor; ++n) {
                        if(skip(d, m, n)) continue;
                        double wivj = basis_fun(d, n, phi, m, theta);
                        double wivjd = basis_fun_dtheta(d, n, phi, m, theta);
                        double wivjdd = basis_fun_dthetadtheta(d, n, phi, m, theta);
                        if(d==0) {
                            double dxhat = wivj;
                            double dyhat = 0.;
                            double dxhatd = wivjd;
                            double dyhatd = 0.;
                            double dxhatdd = wivjdd;
                            double dyhatdd = 0.;
                            double dxdd = dxhatdd * cos(phi) - dyhatdd * sin(phi);
                            double dydd = dxhatdd * sin(phi) + dyhatdd * cos(phi);
                            data(k1, k2, 0, counter) = 4*M_PI*M_PI*dxdd;
                            data(k1, k2, 1, counter) = 4*M_PI*M_PI*dydd;
                        }else if(d==1) {
                            double dxhat = 0.;
                            double dyhat = wivj;
                            double dxhatd = 0.;
                            double dyhatd = wivjd;
                            double dxhatdd = 0.;
                            double dyhatdd = wivjdd;
                            double dxdd = dxhatdd * cos(phi) - dyhatdd * sin(phi);
                            double dydd = dxhatdd * sin(phi) + dyhatdd * cos(phi);
                            data(k1, k2, 0, counter) = 4*M_PI*M_PI*dxdd;
                            data(k1, k2, 1, counter) = 4*M_PI*M_PI*dydd;
                        }else {
                            double dzdd = wivjdd;
                            data(k1, k2, 2, counter) = 4*M_PI*M_PI*dzdd;;
                        }
                        counter++;
                    }
                }
            }
        }
    }
}

template<class Array>
void SurfaceXYZTensorFourier<Array>::dgamma_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    rebuild_cache(quadpoints_phi, quadpoints_theta);
    int numquadpoints_phi = quadpoints_phi.size();
    int numquadpoints_theta = quadpoints_theta.size();
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            int counter = 0;
            for (int d = 0; d < 3; ++d) {
                for (int m = 0; m <= 2*mpol; ++m) {
                    for (int n = 0; n <= 2*ntor; ++n) {
                        if(skip(d, m, n)) continue;
                        double wivj = basis_fun(d, n, phi, m, theta);
                        if(d==0) {
                            double dxhat = wivj;
                            double dyhat = 0;
                            double dx = dxhat * cos(phi) - dyhat * sin(phi);
                            double dy = dxhat * sin(phi) + dyhat * cos(phi);
                            data(k1, k2, 0, counter) = dx;
                            data(k1, k2, 1, counter) = dy;
                        }else if(d==1) {
                            double dxhat = 0;
                            double dyhat = wivj;
                            double dx = dxhat * cos(phi) - dyhat * sin(phi);
                            double dy = dxhat * sin(phi) + dyhat * cos(phi);
                            data(k1, k2, 0, counter) = dx;
                            data(k1, k2, 1, counter) = dy;
                        }else {
                            double dz = wivj;
                            data(k1, k2, 2, counter) = dz;
                        }
                        counter++;
                    }
                }
            }
        }
    }
}

template<class Array>
void SurfaceXYZTensorFourier<Array>::dgammadash1_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    rebuild_cache(quadpoints_phi, quadpoints_theta);
    int numquadpoints_phi = quadpoints_phi.size();
    int numquadpoints_theta = quadpoints_theta.size();
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            int counter = 0;
            for (int d = 0; d < 3; ++d) {
                for (int m = 0; m <= 2*mpol; ++m) {
                    for (int n = 0; n <= 2*ntor; ++n) {
                        if(skip(d, m, n)) continue;
                        double wivj = basis_fun(d, n, phi, m, theta);
                        double wivjdash = basis_fun_dphi(d, n, phi, m, theta);
                        if(d==0) {
                            double dxhat = wivj;
                            double dyhat = 0.;
                            double dxhatdash = wivjdash;
                            double dyhatdash = 0.;
                            double dxdash = dxhatdash * cos(phi) - dyhatdash * sin(phi) - dxhat * sin(phi) - dyhat * cos(phi);
                            double dydash = dxhatdash * sin(phi) + dyhatdash * cos(phi) + dxhat * cos(phi) - dyhat * sin(phi);
                            data(k1, k2, 0, counter) = 2*M_PI*dxdash;
                            data(k1, k2, 1, counter) = 2*M_PI*dydash;
                        }else if(d==1) {
                            double dxhat = 0.;
                            double dyhat = wivj;
                            double dxhatdash = 0.;
                            double dyhatdash = wivjdash;
                            double dxdash = dxhatdash * cos(phi) - dyhatdash * sin(phi) - dxhat * sin(phi) - dyhat * cos(phi);
                            double dydash = dxhatdash * sin(phi) + dyhatdash * cos(phi) + dxhat * cos(phi) - dyhat * sin(phi);
                            data(k1, k2, 0, counter) = 2*M_PI*dxdash;
                            data(k1, k2, 1, counter) = 2*M_PI*dydash;
                        }else {
                            double dzdash = wivjdash;
                            data(k1, k2, 2, counter) = 2*M_PI*dzdash;;
                        }
                        counter++;
                    }
                }
            }
        }
    }
}

template<class Array>
void SurfaceXYZTensorFourier<Array>::dgammadash2_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    rebuild_cache(quadpoints_phi, quadpoints_theta);
    int numquadpoints_phi = quadpoints_phi.size();
    int numquadpoints_theta = quadpoints_theta.size();
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            int counter = 0;
            for (int d = 0; d < 3; ++d) {
                for (int m = 0; m <= 2*mpol; ++m) {
                    for (int n = 0; n <= 2*ntor; ++n) {
                        if(skip(d, m, n)) continue;
                        double wivj = basis_fun(d, n, phi, m, theta);
                        double wivjdash = basis_fun_dtheta(d, n, phi, m, theta);
                        if(d==0) {
                            double dxhat = wivj;
                            double dyhat = 0.;
                            double dxhatdash = wivjdash;
                            double dyhatdash = 0.;
                            double dxdash = dxhatdash * cos(phi) - dyhatdash * sin(phi);
                            double dydash = dxhatdash * sin(phi) + dyhatdash * cos(phi);
                            data(k1, k2, 0, counter) = 2*M_PI*dxdash;
                            data(k1, k2, 1, counter) = 2*M_PI*dydash;
                        }else if(d==1) {
                            double dxhat = 0.;
                            double dyhat = wivj;
                            double dxhatdash = 0.;
                            double dyhatdash = wivjdash;
                            double dxdash = dxhatdash * cos(phi) - dyhatdash * sin(phi);
                            double dydash = dxhatdash * sin(phi) + dyhatdash * cos(phi);
                            data(k1, k2, 0, counter) = 2*M_PI*dxdash;
                            data(k1, k2, 1, counter) = 2*M_PI*dydash;
                        }else {
                            double dzdash = wivjdash;
                            data(k1, k2, 2, counter) = 2*M_PI*dzdash;;
                        }
                        counter++;
                    }
                }
            }
        }
    }
}

template<class Array>
void SurfaceXYZTensorFourier<Array>::build_cache(Array& quadpoints_phi, Array& quadpoints_theta) {
    int numquadpoints_phi = quadpoints_phi.size();
    int numquadpoints_theta = quadpoints_theta.size();

    cache_quadpoints_phi = xt::zeros<double>({numquadpoints_phi});
    for (int i = 0; i < numquadpoints_phi; ++i) {
        cache_quadpoints_phi[i] = quadpoints_phi[i];
    }
    cache_quadpoints_theta = xt::zeros<double>({numquadpoints_theta});
    for (int i = 0; i < numquadpoints_theta; ++i) {
        cache_quadpoints_theta[i] = quadpoints_theta[i];
    }

    cache_basis_fun_phi = xt::zeros<double>({numquadpoints_phi, 2*ntor+1});
    cache_basis_fun_phi_dash = xt::zeros<double>({numquadpoints_phi, 2*ntor+1});
    cache_basis_fun_phi_dashdash = xt::zeros<double>({numquadpoints_phi, 2*ntor+1});
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*cache_quadpoints_phi[k1];
        for (int n = 0; n <= 2*ntor; ++n) {
            cache_basis_fun_phi(k1, n)= basis_fun_phi(n, phi);
            cache_basis_fun_phi_dash(k1, n) = basis_fun_phi_dash(n, phi);
            cache_basis_fun_phi_dashdash(k1, n) = basis_fun_phi_dashdash(n, phi);
        }
    }
    cache_basis_fun_theta = xt::zeros<double>({numquadpoints_theta, 2*mpol+1});
    cache_basis_fun_theta_dash = xt::zeros<double>({numquadpoints_theta, 2*mpol+1});
    cache_basis_fun_theta_dashdash = xt::zeros<double>({numquadpoints_theta, 2*mpol+1});
    for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
        double theta  = 2*M_PI*cache_quadpoints_theta[k2];
        for (int m = 0; m <= 2*mpol; ++m) {
            cache_basis_fun_theta(k2, m) = basis_fun_theta(m, theta);
            cache_basis_fun_theta_dash(k2, m) = basis_fun_theta_dash(m, theta);
            cache_basis_fun_theta_dashdash(k2, m) = basis_fun_theta_dashdash(m, theta);
        }
    }
    cache_enforcer = xt::zeros<double>({numquadpoints_phi, numquadpoints_theta});
    cache_enforcer_dphi = xt::zeros<double>({numquadpoints_phi, numquadpoints_theta});
    cache_enforcer_dtheta = xt::zeros<double>({numquadpoints_phi, numquadpoints_theta});
    cache_enforcer_dphidphi = xt::zeros<double>({numquadpoints_phi, numquadpoints_theta});
    cache_enforcer_dthetadtheta = xt::zeros<double>({numquadpoints_phi, numquadpoints_theta});
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*cache_quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*cache_quadpoints_theta[k2];
            cache_enforcer(k1, k2) = pow(sin(nfp*phi/2), 2) + pow(sin(theta/2), 2);
            cache_enforcer_dphi(k1, k2) = nfp*cos(nfp*phi/2)*sin(nfp*phi/2);
            cache_enforcer_dphidphi(k1, k2) = nfp*(nfp/2)*(pow(cos(nfp*phi/2),2) - pow(sin(nfp*phi/2),2));
            cache_enforcer_dtheta(k1, k2) = cos(theta/2)*sin(theta/2);
            cache_enforcer_dthetadtheta(k1, k2) = (1/2)*(pow(cos(theta/2),2) - pow(sin(theta/2),2));
        }
    }

}

template<class Array>
void SurfaceXYZTensorFourier<Array>::rebuild_cache(Array& quadpoints_phi, Array& quadpoints_theta) {
    if ((quadpoints_phi.size() != cache_quadpoints_phi.size()) ||
        (quadpoints_theta.size() != cache_quadpoints_theta.size()) ||
        (quadpoints_phi[0] != cache_quadpoints_phi[0]) ||
        (quadpoints_theta[0] != cache_quadpoints_theta[0])
    )
        build_cache(quadpoints_phi, quadpoints_theta);

}


#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
template class SurfaceXYZTensorFourier<Array>;
