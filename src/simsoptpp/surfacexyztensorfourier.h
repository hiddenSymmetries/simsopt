#pragma once

#include "surface.h"

template<class Array>
class SurfaceXYZTensorFourier : public Surface<Array> {
    /*
       SurfaceXYZTensorFourier is a surface that is represented in cartesian
       coordinates using the following Fourier series:

       \hat x(theta, phi) =
           \sum_{i=0}^{2*mpol} \sum_{j=0}^{2*ntor} x_{ij} w_i(\theta)*v_j(\phi)
       \hat y(theta, phi) =
           \sum_{i=0}^{2*mpol} \sum_{j=0}^{2*ntor} y_{ij} w_i(\theta)*v_j(\phi)

       x = \hat x * \cos(\phi) - \hat y * \sin(\phi)
       y = \hat x * \sin(\phi) + \hat y * \cos(\phi)

       z(theta, phi) =
           \sum_{i=0}^{2*mpol} \sum_{j=0}^{2*ntor} z_{ij} w_i(\theta)*v_j(\phi)

       where the basis functions {v_j} are given by

        {1, cos(1*nfp*\phi), ..., cos(ntor*nfp*phi), sin(1*nfp*phi), ..., sin(ntor*nfp*phi)}

       and {w_i} are given by

        {1, cos(1*\theta), ..., cos(ntor*theta), sin(1*theta), ..., sin(ntor*theta)}

       When enforcing stellarator symmetry, ...
       */

    public:
        using Surface<Array>::quadpoints_phi;
        using Surface<Array>::quadpoints_theta;
        using Surface<Array>::numquadpoints_phi;
        using Surface<Array>::numquadpoints_theta;
        Array x;
        Array y;
        Array z;
        Array cache_basis_fun_phi;
        Array cache_basis_fun_phi_dash;
        Array cache_basis_fun_phi_dashdash;
        Array cache_basis_fun_theta;
        Array cache_basis_fun_theta_dash;
        Array cache_basis_fun_theta_dashdash;
        Array cache_enforcer;
        Array cache_enforcer_dphi;
        Array cache_enforcer_dtheta;
        Array cache_enforcer_dphidphi;
        Array cache_enforcer_dthetadtheta;
        int nfp;
        int mpol;
        int ntor;
        bool stellsym;
        std::vector<bool> clamped_dims;

        SurfaceXYZTensorFourier(int _mpol, int _ntor, int _nfp, bool _stellsym, std::vector<bool> _clamped_dims, vector<double> _quadpoints_phi, vector<double> _quadpoints_theta)
            : Surface<Array>(_quadpoints_phi, _quadpoints_theta), mpol(_mpol), ntor(_ntor), nfp(_nfp), stellsym(_stellsym), clamped_dims(_clamped_dims) {
                x = xt::zeros<double>({2*mpol+1, 2*ntor+1});
                y = xt::zeros<double>({2*mpol+1, 2*ntor+1});
                z = xt::zeros<double>({2*mpol+1, 2*ntor+1});
                build_cache();
            }

        int num_dofs() override {
            if(stellsym)
                return (ntor+1)*(mpol+1)+ ntor*mpol + 2*(ntor+1)*mpol + 2*ntor*(mpol+1);
            else
                return 3 * (2*mpol+1) * (2*ntor+1);
        }

        void set_dofs_impl(const vector<double>& dofs) override {
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

        vector<double> get_dofs() override {
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

        void gamma_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override {
            int numquadpoints_phi = quadpoints_phi.size();
            int numquadpoints_theta = quadpoints_theta.size();
            data *= 0.;
#pragma omp parallel for
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
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
        void gamma_lin(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override {
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


        void gammadash1_impl(Array& data) override {
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

        void gammadash2_impl(Array& data) override {
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

        void gammadash2dash2_impl(Array& data) override {
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

        void gammadash1dash2_impl(Array& data) override {
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

        void gammadash1dash1_impl(Array& data) override {
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

        void dgammadash1dash1_by_dcoeff_impl(Array& data) override {
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

        void dgammadash1dash2_by_dcoeff_impl(Array& data) override {
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

        void dgammadash2dash2_by_dcoeff_impl(Array& data) override {
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

        void dgamma_by_dcoeff_impl(Array& data) override {
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

        void dgammadash1_by_dcoeff_impl(Array& data) override {
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

        void dgammadash2_by_dcoeff_impl(Array& data) override {
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

    private:

        void build_cache() {
            cache_basis_fun_phi = xt::zeros<double>({numquadpoints_phi, 2*ntor+1});
            cache_basis_fun_phi_dash = xt::zeros<double>({numquadpoints_phi, 2*ntor+1});
            cache_basis_fun_phi_dashdash = xt::zeros<double>({numquadpoints_phi, 2*ntor+1});
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
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
                double theta  = 2*M_PI*quadpoints_theta[k2];
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
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    cache_enforcer(k1, k2) = pow(sin(nfp*phi/2), 2) + pow(sin(theta/2), 2);
                    cache_enforcer_dphi(k1, k2) = nfp*cos(nfp*phi/2)*sin(nfp*phi/2);
                    cache_enforcer_dphidphi(k1, k2) = nfp*(nfp/2)*(pow(cos(nfp*phi/2),2) - pow(sin(nfp*phi/2),2));
                    cache_enforcer_dtheta(k1, k2) = cos(theta/2)*sin(theta/2);
                    cache_enforcer_dthetadtheta(k1, k2) = (1/2)*(pow(cos(theta/2),2) - pow(sin(theta/2),2));
                }
            }

        }

        inline bool apply_bc_enforcer(int dim, int n, int m) {
            return (clamped_dims[dim] && n<=ntor && m<=mpol);
        }

        inline double bc_enforcer_fun(int dim, int n, double phi, int m, double theta){
            if(apply_bc_enforcer(dim, n, m))
                return pow(sin(nfp*phi/2), 2) + pow(sin(theta/2), 2);
            else
                return 1;
        }

        inline double bc_enforcer_dphi_fun(int dim, int n, double phi, int m, double theta){
            if(apply_bc_enforcer(dim, n, m))
                return nfp*cos(nfp*phi/2)*sin(nfp*phi/2);
            else
                return 0;
        }

        inline double bc_enforcer_dphidphi_fun(int dim, int n, double phi, int m, double theta){
            if(apply_bc_enforcer(dim, n, m))
                return (nfp*nfp/2)*(pow(cos(nfp*phi/2),2) - pow(sin(nfp*phi/2),2));
            else
                return 0;
        }

        inline double bc_enforcer_dtheta_fun(int dim, int n, double phi, int m, double theta){
            if(apply_bc_enforcer(dim, n, m))
                return cos(theta/2)*sin(theta/2);
            else
                return 0;
        }

        inline double bc_enforcer_dthetadtheta_fun(int dim, int n, double phi, int m, double theta){
            if(apply_bc_enforcer(dim, n, m))
                return (1/2)*(pow(cos(theta/2),2) - pow(sin(theta/2),2));
            else
                return 0;
        }

        inline double basis_fun(int dim, int n, int phiidx, int m, int thetaidx){
            double fun = cache_basis_fun_phi(phiidx, n)*cache_basis_fun_theta(thetaidx, m);
            if(apply_bc_enforcer(dim, n, m))
                fun *= cache_enforcer(phiidx, thetaidx);
            return fun;
        }

        inline double basis_fun_dphi(int dim, int n, int phiidx, int m, int thetaidx){
            double fun_dphi = cache_basis_fun_phi_dash(phiidx, n)*cache_basis_fun_theta(thetaidx, m);
            if(apply_bc_enforcer(dim, n, m)){
                double fun = cache_basis_fun_phi(phiidx, n)*cache_basis_fun_theta(thetaidx, m);
                fun_dphi = fun_dphi*cache_enforcer(phiidx, thetaidx) + fun*cache_enforcer_dphi(phiidx, thetaidx);
            }
            return fun_dphi;
        }

        inline double basis_fun_dphidphi(int dim, int n, int phiidx, int m, int thetaidx){
            double fun_dphidphi = cache_basis_fun_phi_dashdash(phiidx, n)*cache_basis_fun_theta(thetaidx, m);
            if(apply_bc_enforcer(dim, n, m)){
                double fun_dphi = cache_basis_fun_phi_dash(phiidx, n)*cache_basis_fun_theta(thetaidx, m);
                double fun = cache_basis_fun_phi(phiidx, n)*cache_basis_fun_theta(thetaidx, m);
                fun_dphidphi = fun_dphidphi*cache_enforcer(phiidx, thetaidx) + 2*fun_dphi*cache_enforcer_dphi(phiidx, thetaidx) \
                            +  fun*cache_enforcer_dphidphi(phiidx, thetaidx);
            }
            return fun_dphidphi;
        }

        inline double basis_fun_dtheta(int dim, int n, int phiidx, int m, int thetaidx){
            double fun_dtheta = cache_basis_fun_phi(phiidx, n)*cache_basis_fun_theta_dash(thetaidx, m);
            if(apply_bc_enforcer(dim, n, m)){
                double fun = cache_basis_fun_phi(phiidx, n)*cache_basis_fun_theta(thetaidx, m);
                fun_dtheta = fun_dtheta*cache_enforcer(phiidx, thetaidx) + fun*cache_enforcer_dtheta(phiidx, thetaidx);
            }
            return fun_dtheta;
        }

        inline double basis_fun_dthetadphi(int dim, int n, int phiidx, int m, int thetaidx){
            double fun_dthetadphi = cache_basis_fun_phi_dash(phiidx, n)*cache_basis_fun_theta_dash(thetaidx, m);
            if(apply_bc_enforcer(dim, n, m)){
                double fun_dtheta = cache_basis_fun_phi(phiidx, n)*cache_basis_fun_theta_dash(thetaidx, m);
                double fun_dphi = cache_basis_fun_phi_dash(phiidx, n)*cache_basis_fun_theta(thetaidx, m);
                fun_dthetadphi = fun_dthetadphi*cache_enforcer(phiidx, thetaidx) \
                                + fun_dtheta*cache_enforcer_dphi(phiidx, thetaidx) \
                                + fun_dphi*cache_enforcer_dtheta(phiidx, thetaidx);
            }
            return fun_dthetadphi;
        }

        inline double basis_fun_dthetadtheta(int dim, int n, int phiidx, int m, int thetaidx){
          double fun_dthetadtheta = cache_basis_fun_phi(phiidx, n)*cache_basis_fun_theta_dashdash(thetaidx, m);
          if(apply_bc_enforcer(dim, n, m)){
              double fun = cache_basis_fun_phi(phiidx, n)*cache_basis_fun_theta(thetaidx, m);
              double fun_dtheta = cache_basis_fun_phi(phiidx, n)*cache_basis_fun_theta_dash(thetaidx, m);
              fun_dthetadtheta = fun_dthetadtheta*cache_enforcer(phiidx, thetaidx) \
                              + 2*fun_dtheta*cache_enforcer_dtheta(phiidx, thetaidx) \
                              + fun*cache_enforcer_dthetadtheta(phiidx, thetaidx);
          }
          return fun_dthetadtheta;
        }

        inline double basis_fun(int dim, int n, double phi, int m, double theta){
            double bc_enforcer = bc_enforcer_fun(dim, n, phi, m, theta);
            return basis_fun_phi(n, phi) * basis_fun_theta(m, theta) * bc_enforcer;
        }

        inline double basis_fun_dphi(int dim, int n, double phi, int m, double theta){
            double bc_enforcer =  bc_enforcer_fun(dim, n, phi, m, theta);
            double bc_enforcer_dphi = bc_enforcer_dphi_fun(dim, n, phi, m, theta);
            return basis_fun_phi_dash(n, phi) * basis_fun_theta(m, theta) * bc_enforcer + basis_fun_phi(n, phi) * basis_fun_theta(m, theta) * bc_enforcer_dphi;
        }

        inline double basis_fun_dtheta(int dim, int n, double phi, int m, double theta){
            double bc_enforcer =  bc_enforcer_fun(dim, n, phi, m, theta);
            double bc_enforcer_dtheta = bc_enforcer_dtheta_fun(dim, n, phi, m, theta);
            return basis_fun_phi(n, phi) * basis_fun_theta_dash(m, theta) * bc_enforcer + basis_fun_phi(n, phi) * basis_fun_theta(m, theta) * bc_enforcer_dtheta;
        }

        inline double basis_fun_dthetadtheta(int dim, int n, double phi, int m, double theta){
            double bc_enforcer =  bc_enforcer_fun(dim, n, phi, m, theta);
            double bc_enforcer_dtheta = bc_enforcer_dtheta_fun(dim, n, phi, m, theta);
            double bc_enforcer_dthetadtheta = bc_enforcer_dthetadtheta_fun(dim, n, phi, m, theta);
            return basis_fun_phi(n, phi) * (basis_fun_theta_dashdash(m, theta) * bc_enforcer \
                 + 2* basis_fun_theta_dash(m, theta) * bc_enforcer_dtheta \
                 +    basis_fun_theta(m, theta) * bc_enforcer_dthetadtheta);
        }

        inline double basis_fun_dphidphi(int dim, int n, double phi, int m, double theta){
            double bc_enforcer =  bc_enforcer_fun(dim, n, phi, m, theta);
            double bc_enforcer_dphi = bc_enforcer_dphi_fun(dim, n, phi, m, theta);
            double bc_enforcer_dphidphi = bc_enforcer_dphidphi_fun(dim, n, phi, m, theta);
            return basis_fun_theta(m, theta) * (basis_fun_phi_dashdash(n, phi) * bc_enforcer \
                  + 2*basis_fun_phi_dash(n, phi)*bc_enforcer_dphi \
                  + basis_fun_phi(n,phi) * bc_enforcer_dphidphi);
        }

        inline double basis_fun_dthetadphi(int dim, int n, double phi, int m, double theta){
            double bc_enforcer =  bc_enforcer_fun(dim, n, phi, m, theta);
            double bc_enforcer_dtheta = bc_enforcer_dtheta_fun(dim, n, phi, m, theta);
            double bc_enforcer_dphi = bc_enforcer_dphi_fun(dim, n, phi, m, theta);
            return basis_fun_phi_dash(n, phi) * basis_fun_theta_dash(m, theta) * bc_enforcer \
                 +  basis_fun_phi(n, phi) * basis_fun_theta_dash(m, theta) * bc_enforcer_dphi \
                 +  basis_fun_phi_dash(n, phi) * basis_fun_theta(m, theta) * bc_enforcer_dtheta;
        }

        inline double basis_fun_phi(int n, double phi){
            if(n <= ntor)
                return cos(nfp*n*phi);
            else
                return sin(nfp*(n-ntor)*phi);
        }

        inline double basis_fun_phi_dash(int n, double phi){
            if(n <= ntor)
                return -nfp*n*sin(nfp*n*phi);
            else
                return nfp*(n-ntor)*cos(nfp*(n-ntor)*phi);
        }

        inline double basis_fun_phi_dashdash(int n, double phi){
            if(n <= ntor)
                return -nfp*n*nfp*n*cos(nfp*n*phi);
            else
                return -nfp*(n-ntor)*nfp*(n-ntor)*sin(nfp*(n-ntor)*phi);
        }

        inline double basis_fun_theta(int m, double theta){
            if(m <= mpol)
                return cos(m*theta);
            else
                return sin((m-mpol)*theta);
        }

        inline double basis_fun_theta_dash(int m, double theta){
            if(m <= mpol)
                return -m*sin(m*theta);
            else
                return (m-mpol)*cos((m-mpol)*theta);
        }

        inline double basis_fun_theta_dashdash(int m, double theta){
            if(m <= mpol)
                return -m*m*cos(m*theta);
            else
                return -(m-mpol)*(m-mpol)*sin((m-mpol)*theta);
        }

        inline bool skip(int dim, int m, int n){
            if (!stellsym)
                return false;
            if (dim == 0)
                return (n <= ntor && m >  mpol) || (n >  ntor && m <= mpol);
            else if(dim == 1)
                return (n <= ntor && m <= mpol) || (n >  ntor && m >  mpol);
            else
                return (n <= ntor && m <= mpol) || (n >  ntor && m >  mpol);
        }

        inline double get_coeff(int dim, int m, int n) {
            if(skip(dim, m, n))
                return 0.;
            if (dim == 0){
                return this->x(m, n);
            }
            else if (dim == 1) {
                return this->y(m, n);
            }
            else {
                return this->z(m, n);
            }
        }
};
