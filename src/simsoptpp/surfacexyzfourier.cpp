#include "surfacexyzfourier.h"

template<class Array>
void SurfaceXYZFourier<Array>::gamma_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    int numquadpoints_phi = quadpoints_phi.size();
    int numquadpoints_theta = quadpoints_theta.size();
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            double x = 0;
            double y = 0;
            double z = 0;
            for (int m = 0; m <= mpol; ++m) {
                for (int i = 0; i < 2*ntor+1; ++i) {
                    int n  = i - ntor;
                    double xhat = get_coeff(0, true, m, i) * cos(m*theta-n*nfp*phi) + get_coeff(0, false, m, i) * sin(m*theta-n*nfp*phi);
                    double yhat = get_coeff(1, true, m, i) * cos(m*theta-n*nfp*phi) + get_coeff(1, false, m, i) * sin(m*theta-n*nfp*phi);
                    x += xhat * cos(phi) - yhat * sin(phi);
                    y += xhat * sin(phi) + yhat * cos(phi);
                    z += get_coeff(2, true , m, i) * cos(m*theta-n*nfp*phi) + get_coeff(2, false, m, i) * sin(m*theta-n*nfp*phi);
                }
            }
            data(k1, k2, 0) = x;
            data(k1, k2, 1) = y;
            data(k1, k2, 2) = z;
        }
    }
}

template<class Array>
void SurfaceXYZFourier<Array>::gamma_lin(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    int numquadpoints = quadpoints_phi.size();
    data *= 0.;
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        double theta  = 2*M_PI*quadpoints_theta[k1];
        double x = 0;
        double y = 0;
        double z = 0;
        for (int m = 0; m <= mpol; ++m) {
            for (int i = 0; i < 2*ntor+1; ++i) {
                int n  = i - ntor;
                double xhat = get_coeff(0, true, m, i) * cos(m*theta-n*nfp*phi) + get_coeff(0, false, m, i) * sin(m*theta-n*nfp*phi);
                double yhat = get_coeff(1, true, m, i) * cos(m*theta-n*nfp*phi) + get_coeff(1, false, m, i) * sin(m*theta-n*nfp*phi);
                x += xhat * cos(phi) - yhat * sin(phi);
                y += xhat * sin(phi) + yhat * cos(phi);
                z += get_coeff(2, true , m, i) * cos(m*theta-n*nfp*phi) + get_coeff(2, false, m, i) * sin(m*theta-n*nfp*phi);
            }
        }
        data(k1, 0) = x;
        data(k1, 1) = y;
        data(k1, 2) = z;
    }
}


template<class Array>
void SurfaceXYZFourier<Array>::gammadash1_impl(Array& data) {
    data *= 0.;
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            double xdash = 0;
            double ydash = 0;
            double zdash = 0;
            for (int m = 0; m <= mpol; ++m) {
                for (int i = 0; i < 2*ntor+1; ++i) {
                    int n  = i - ntor;
                    double xhat = get_coeff(0, true, m, i) * cos(m*theta-n*nfp*phi) + get_coeff(0, false, m, i) * sin(m*theta-n*nfp*phi);
                    double yhat = get_coeff(1, true, m, i) * cos(m*theta-n*nfp*phi) + get_coeff(1, false, m, i) * sin(m*theta-n*nfp*phi);
                    double xhatdash = get_coeff(0, true, m, i) * (n*nfp)*sin(m*theta-n*nfp*phi) + get_coeff(0, false, m, i) * (-n*nfp)*cos(m*theta-n*nfp*phi);
                    double yhatdash = get_coeff(1, true, m, i) * (n*nfp)*sin(m*theta-n*nfp*phi) + get_coeff(1, false, m, i) * (-n*nfp)*cos(m*theta-n*nfp*phi);
                    xdash += xhatdash * cos(phi) - yhatdash * sin(phi) - xhat * sin(phi) - yhat * cos(phi);
                    ydash += xhatdash * sin(phi) + yhatdash * cos(phi) + xhat * cos(phi) - yhat * sin(phi);
                    zdash += get_coeff(2, true , m, i) * (n*nfp)*sin(m*theta-n*nfp*phi) + get_coeff(2, false, m, i) * (-n*nfp)*cos(m*theta-n*nfp*phi);
                }
            }
            data(k1, k2, 0) = 2*M_PI*xdash;
            data(k1, k2, 1) = 2*M_PI*ydash;
            data(k1, k2, 2) = 2*M_PI*zdash;
        }
    }
}

template<class Array>
void SurfaceXYZFourier<Array>::gammadash2_impl(Array& data) {
    data *= 0.;
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            double xdash = 0;
            double ydash = 0;
            double zdash = 0;
            for (int m = 0; m <= mpol; ++m) {
                for (int i = 0; i < 2*ntor+1; ++i) {
                    int n  = i - ntor;
                    double xhatdash = get_coeff(0, true, m, i) * (-m)* sin(m*theta-n*nfp*phi) + get_coeff(0, false, m, i) * m * cos(m*theta-n*nfp*phi);
                    double yhatdash = get_coeff(1, true, m, i) * (-m)* sin(m*theta-n*nfp*phi) + get_coeff(1, false, m, i) * m * cos(m*theta-n*nfp*phi);
                    xdash += xhatdash * cos(phi) - yhatdash * sin(phi);
                    ydash += xhatdash * sin(phi) + yhatdash * cos(phi);
                    zdash += get_coeff(2, true , m, i) * (-m) * sin(m*theta-n*nfp*phi) + get_coeff(2, false, m, i) * m * cos(m*theta-n*nfp*phi);
                }
            }
            data(k1, k2, 0) = 2*M_PI*xdash;
            data(k1, k2, 1) = 2*M_PI*ydash;
            data(k1, k2, 2) = 2*M_PI*zdash;
        }
    }
}

template<class Array>
void SurfaceXYZFourier<Array>::gammadash1dash1_impl(Array& data) {
    data *= 0.;
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            double xdd = 0;
            double ydd = 0;
            double zdd = 0;
            for (int m = 0; m <= mpol; ++m) {
                for (int i = 0; i < 2*ntor+1; ++i) {
                    int n  = i - ntor;
                    double xhat = get_coeff(0, true, m, i) * cos(m*theta-n*nfp*phi) + get_coeff(0, false, m, i) * sin(m*theta-n*nfp*phi);
                    double yhat = get_coeff(1, true, m, i) * cos(m*theta-n*nfp*phi) + get_coeff(1, false, m, i) * sin(m*theta-n*nfp*phi);
                    double xhatd = -get_coeff(0, true, m, i) * (-n*nfp)*sin(m*theta-n*nfp*phi) + get_coeff(0, false, m, i) * (-n*nfp)*cos(m*theta-n*nfp*phi);
                    double yhatd = -get_coeff(1, true, m, i) * (-n*nfp)*sin(m*theta-n*nfp*phi) + get_coeff(1, false, m, i) * (-n*nfp)*cos(m*theta-n*nfp*phi);
                    double xhatdd = -get_coeff(0, true, m, i) * (-n*nfp)*(-n*nfp)*cos(m*theta-n*nfp*phi) - get_coeff(0, false, m, i) * (-n*nfp) * (-n*nfp)*sin(m*theta-n*nfp*phi);
                    double yhatdd = -get_coeff(1, true, m, i) * (-n*nfp)*(-n*nfp)*cos(m*theta-n*nfp*phi) - get_coeff(1, false, m, i) * (-n*nfp) * (-n*nfp)*sin(m*theta-n*nfp*phi);

                    xdd += xhatdd * cos(phi) - 2*xhatd * sin(phi) - xhat * cos(phi)
                         - yhatdd * sin(phi) - 2*yhatd * cos(phi) + yhat * sin(phi);
                    ydd += xhatdd * sin(phi) + 2*xhatd * cos(phi) - xhat * sin(phi)
                         + yhatdd * cos(phi) - 2*yhatd * sin(phi) - yhat * cos(phi);
                    zdd += -get_coeff(2, true , m, i) * (-n*nfp) * (-n*nfp) * cos(m*theta-n*nfp*phi) - get_coeff(2, false, m, i) * (-n*nfp) * (-n*nfp) * sin(m*theta-n*nfp*phi);
                }
            }
            data(k1, k2, 0) = 4*M_PI*M_PI*xdd;
            data(k1, k2, 1) = 4*M_PI*M_PI*ydd;
            data(k1, k2, 2) = 4*M_PI*M_PI*zdd;
        }
    }
}

template<class Array>
void SurfaceXYZFourier<Array>::gammadash1dash2_impl(Array& data) {
    data *= 0.;
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            double xd1d2 = 0;
            double yd1d2 = 0;
            double zd1d2 = 0;
            for (int m = 0; m <= mpol; ++m) {
                for (int i = 0; i < 2*ntor+1; ++i) {
                    int n  = i - ntor;
                    double xhat = get_coeff(0, true, m, i) * cos(m*theta-n*nfp*phi) + get_coeff(0, false, m, i) * sin(m*theta-n*nfp*phi);
                    double yhat = get_coeff(1, true, m, i) * cos(m*theta-n*nfp*phi) + get_coeff(1, false, m, i) * sin(m*theta-n*nfp*phi);
                    double xhatd2 = -get_coeff(0, true, m, i) * (m)*sin(m*theta-n*nfp*phi) + get_coeff(0, false, m, i) * (m)*cos(m*theta-n*nfp*phi);
                    double yhatd2 = -get_coeff(1, true, m, i) * (m)*sin(m*theta-n*nfp*phi) + get_coeff(1, false, m, i) * (m)*cos(m*theta-n*nfp*phi);
                    double xhatd1d2 = -get_coeff(0, true, m, i) * (-n*nfp)*(m)*cos(m*theta-n*nfp*phi) - get_coeff(0, false, m, i) * (-n*nfp) * (m)*sin(m*theta-n*nfp*phi);
                    double yhatd1d2 = -get_coeff(1, true, m, i) * (-n*nfp)*(m)*cos(m*theta-n*nfp*phi) - get_coeff(1, false, m, i) * (-n*nfp) * (m)*sin(m*theta-n*nfp*phi);

                    xd1d2 += xhatd1d2 * cos(phi) - xhatd2 * sin(phi)
                           - yhatd1d2 * sin(phi) - yhatd2 * cos(phi);
                    yd1d2 += xhatd1d2 * sin(phi) + xhatd2 * cos(phi)
                           + yhatd1d2 * cos(phi) - yhatd2 * sin(phi);
                    zd1d2 += -get_coeff(2, true , m, i) * (-n*nfp) * (m) * cos(m*theta-n*nfp*phi) - get_coeff(2, false, m, i) * (-n*nfp) * (m) * sin(m*theta-n*nfp*phi);
                }
            }
            data(k1, k2, 0) = 4*M_PI*M_PI*xd1d2;
            data(k1, k2, 1) = 4*M_PI*M_PI*yd1d2;
            data(k1, k2, 2) = 4*M_PI*M_PI*zd1d2;
        }
    }
}

template<class Array>
void SurfaceXYZFourier<Array>::gammadash2dash2_impl(Array& data) {
    data *= 0.;
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            double xdd = 0;
            double ydd = 0;
            double zdd = 0;
            for (int m = 0; m <= mpol; ++m) {
                for (int i = 0; i < 2*ntor+1; ++i) {
                    int n  = i - ntor;
                    double xhatdd = -get_coeff(0, true, m, i) * (m) * (m) * cos(m*theta-n*nfp*phi) - get_coeff(0, false, m, i) * m * m * sin(m*theta-n*nfp*phi);
                    double yhatdd = -get_coeff(1, true, m, i) * (m) * (m) * cos(m*theta-n*nfp*phi) - get_coeff(1, false, m, i) * m * m * sin(m*theta-n*nfp*phi);
                    xdd += xhatdd * cos(phi) - yhatdd * sin(phi);
                    ydd += xhatdd * sin(phi) + yhatdd * cos(phi);
                    zdd += -get_coeff(2, true , m, i) * (m) * (m) * cos(m*theta-n*nfp*phi) - get_coeff(2, false, m, i) * m * m * sin(m*theta-n*nfp*phi);
                }
            }
            data(k1, k2, 0) = 4*M_PI*M_PI*xdd;
            data(k1, k2, 1) = 4*M_PI*M_PI*ydd;
            data(k1, k2, 2) = 4*M_PI*M_PI*zdd;
        }
    }
}


template<class Array>
void SurfaceXYZFourier<Array>::dgamma_by_dcoeff_impl(Array& data) {
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            int counter = 0;
            for (int d = 0; d < 3; ++d) {
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<0) continue;
                        if(d == 0) {
                            data(k1, k2, 0, counter) = cos(m*theta-n*nfp*phi) * cos(phi);
                            data(k1, k2, 1, counter) = cos(m*theta-n*nfp*phi) * sin(phi);
                        }else if(d == 1) {
                            if(stellsym)
                                continue;
                            data(k1, k2, 0, counter) = -cos(m*theta-n*nfp*phi) * sin(phi);
                            data(k1, k2, 1, counter) =  cos(m*theta-n*nfp*phi) * cos(phi);
                        }
                        else if(d == 2) {
                            if(stellsym)
                                continue;
                            data(k1, k2, 2, counter) =  cos(m*theta-n*nfp*phi);
                        }
                        counter++;
                    }
                }
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<=0) continue;
                        if(d == 0) {
                            if(stellsym)
                                continue;
                            data(k1, k2, 0, counter) = sin(m*theta-n*nfp*phi) * cos(phi);
                            data(k1, k2, 1, counter) = sin(m*theta-n*nfp*phi) * sin(phi);
                        }else if(d == 1) {
                            data(k1, k2, 0, counter) = -sin(m*theta-n*nfp*phi) * sin(phi);
                            data(k1, k2, 1, counter) =  sin(m*theta-n*nfp*phi) * cos(phi);
                        }
                        else if(d == 2) {
                            data(k1, k2, 2, counter) =  sin(m*theta-n*nfp*phi);
                        }
                        counter++;
                    }
                }
            }
        }
    }
}

template<class Array>
void SurfaceXYZFourier<Array>::dgammadash1_by_dcoeff_impl(Array& data) {
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            int counter = 0;
            for (int d = 0; d < 3; ++d) {
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<0) continue;
                        if(d == 0) {
                            data(k1, k2, 0, counter) = (n*nfp)*sin(m*theta-n*nfp*phi) * cos(phi) - cos(m*theta-n*nfp*phi) * sin(phi);
                            data(k1, k2, 1, counter) = (n*nfp)*sin(m*theta-n*nfp*phi) * sin(phi) + cos(m*theta-n*nfp*phi) * cos(phi);
                        } else if(d == 1) {
                            if(stellsym)
                                continue;
                            data(k1, k2, 0, counter) = -(n*nfp)*sin(m*theta-n*nfp*phi) * sin(phi) - cos(m*theta-n*nfp*phi) * cos(phi);
                            data(k1, k2, 1, counter) =  (n*nfp)*sin(m*theta-n*nfp*phi) * cos(phi) - cos(m*theta-n*nfp*phi) * sin(phi);
                        }
                        else if(d == 2) {
                            if(stellsym)
                                continue;
                            data(k1, k2, 2, counter) =  (n*nfp)*sin(m*theta-n*nfp*phi);
                        }
                        counter++;
                    }
                }
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<=0) continue;
                        if(d == 0) {
                            if(stellsym)
                                continue;
                            data(k1, k2, 0, counter) = -(n*nfp)*cos(m*theta-n*nfp*phi) * cos(phi) - sin(m*theta-n*nfp*phi) * sin(phi);
                            data(k1, k2, 1, counter) = -(n*nfp)*cos(m*theta-n*nfp*phi) * sin(phi) + sin(m*theta-n*nfp*phi) * cos(phi);
                        }else if(d == 1) {
                            data(k1, k2, 0, counter) = (n*nfp)*cos(m*theta-n*nfp*phi) * sin(phi)  - sin(m*theta-n*nfp*phi) * cos(phi);
                            data(k1, k2, 1, counter) = (-n*nfp)*cos(m*theta-n*nfp*phi) * cos(phi) - sin(m*theta-n*nfp*phi) * sin(phi);
                        }
                        else if(d == 2) {
                            data(k1, k2, 2, counter) = (-n*nfp)*cos(m*theta-n*nfp*phi);
                        }
                        counter++;
                    }
                }
            }
        }
    }
    data *= 2 * M_PI;
}

template<class Array>
void SurfaceXYZFourier<Array>::dgammadash2_by_dcoeff_impl(Array& data) {
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            int counter = 0;
            for (int d = 0; d < 3; ++d) {
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<0) continue;
                        if(d == 0) {
                            data(k1, k2, 0, counter) = (-m)* sin(m*theta-n*nfp*phi) * cos(phi);
                            data(k1, k2, 1, counter) = (-m)* sin(m*theta-n*nfp*phi) * sin(phi);
                        }else if(d == 1) {
                            if(stellsym)
                                continue;
                            data(k1, k2, 0, counter) = (-m)* sin(m*theta-n*nfp*phi) * (-1) * sin(phi);
                            data(k1, k2, 1, counter) = (-m)* sin(m*theta-n*nfp*phi) * cos(phi);
                        }
                        else if(d == 2) {
                            if(stellsym)
                                continue;
                            data(k1, k2, 2, counter) = (-m) * sin(m*theta-n*nfp*phi);
                        }
                        counter++;
                    }
                }
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<=0) continue;
                        if(d == 0) {
                            if(stellsym)
                                continue;
                            data(k1, k2, 0, counter) = m * cos(m*theta-n*nfp*phi) * cos(phi);
                            data(k1, k2, 1, counter) = m * cos(m*theta-n*nfp*phi) * sin(phi);
                        }else if(d == 1) {
                            data(k1, k2, 0, counter) = m * cos(m*theta-n*nfp*phi) * (-1) * sin(phi);
                            data(k1, k2, 1, counter) = m * cos(m*theta-n*nfp*phi) * cos(phi);
                        }
                        else if(d == 2) {
                            data(k1, k2, 2, counter) = m * cos(m*theta-n*nfp*phi);
                        }
                        counter++;
                    }
                }
            }
        }
    }
    data *= 2*M_PI;
}

template<class Array>
void SurfaceXYZFourier<Array>::dgammadash1dash1_by_dcoeff_impl(Array& data) {
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            int counter = 0;
            for (int d = 0; d < 3; ++d) {
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<0) continue;
                        if(d == 0) {
                            data(k1, k2, 0, counter) = (n*nfp) * (-n*nfp) * cos(m*theta-n*nfp*phi) * cos(phi) \
                                                      - 2*(n*nfp)*sin(m*theta-n*nfp*phi) * sin(phi) \
                                                      - cos(m*theta-n*nfp*phi) * cos(phi);
                            data(k1, k2, 1, counter) = (n*nfp)*(-n*nfp)*cos(m*theta-n*nfp*phi)*sin(phi) \
                                                     + 2*(n*nfp)*sin(m*theta-n*nfp*phi) * cos(phi) \
                                                     - cos(m*theta-n*nfp*phi) * sin(phi);
                        } else if(d == 1) {
                            if(stellsym)
                                continue;
                            data(k1, k2, 0, counter) = -(n*nfp)*(-n*nfp)*cos(m*theta-n*nfp*phi) * sin(phi) \
                                                       - 2*(n*nfp)*      sin(m*theta-n*nfp*phi) * cos(phi) \
                                                                 +       cos(m*theta-n*nfp*phi) * sin(phi);
                            data(k1, k2, 1, counter) = (n*nfp)*(-n*nfp)*cos(m*theta-n*nfp*phi)*cos(phi) \
                                                     - 2*(n*nfp)*sin(m*theta-n*nfp*phi) * sin(phi) \
                                                     - cos(m*theta-n*nfp*phi) * cos(phi);
                        }
                        else if (d == 2) {
                            if(stellsym)
                                continue;
                            data(k1, k2, 2, counter) = (n*nfp)*(-n*nfp)*cos(m*theta-n*nfp*phi);
                        }
                        counter++;
                    }
                }
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<=0) continue;
                        if(d == 0) {
                            if(stellsym)
                                continue;
                            data(k1, k2, 0, counter) = (n*nfp)*(-n*nfp)*sin(m*theta-n*nfp*phi) * cos(phi) \
                                                      +       2*(n*nfp)*cos(m*theta-n*nfp*phi) * sin(phi) \
                                                      -                 sin(m*theta-n*nfp*phi) * cos(phi);
                            data(k1, k2, 1, counter) = (n*nfp)*(-n*nfp)*sin(m*theta-n*nfp*phi) * sin(phi) \
                                                      - 2*(n*nfp)*cos(m*theta-n*nfp*phi) * cos(phi) \
                                                      - sin(m*theta-n*nfp*phi) * sin(phi);
                        } else if(d == 1) {
                            data(k1, k2, 0, counter) = -(n*nfp)*(-n*nfp)*sin(m*theta-n*nfp*phi) * sin(phi) \
                                                       + 2*(n*nfp)*cos(m*theta-n*nfp*phi) * cos(phi) \
                                                       + sin(m*theta-n*nfp*phi) * sin(phi);
                            data(k1, k2, 1, counter) = -(-n*nfp)*(-n*nfp)*sin(m*theta-n*nfp*phi) * cos(phi) \
                                                       - 2*(-n*nfp)*cos(m*theta-n*nfp*phi) * sin(phi) \
                                                       - sin(m*theta-n*nfp*phi) * cos(phi);
                        }
                        else if(d == 2) {
                            data(k1, k2, 2, counter) = -(-n*nfp)*(-n*nfp)*sin(m*theta-n*nfp*phi);
                        }
                        counter++;
                    }
                }
            }
        }
    }
    data *= 4 * M_PI * M_PI;
}

template<class Array>
void SurfaceXYZFourier<Array>::dgammadash1dash2_by_dcoeff_impl(Array& data) {
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            int counter = 0;
            for (int d = 0; d < 3; ++d) {
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<0) continue;
                        if(d == 0) {
                            data(k1, k2, 0, counter) = (n*nfp) * m * cos(m*theta-n*nfp*phi) * cos(phi) \
                                                      + m * sin(m*theta-n*nfp*phi) * sin(phi);
                            data(k1, k2, 1, counter) = (n*nfp)*m*cos(m*theta-n*nfp*phi) * sin(phi) \
                                                      - m * sin(m*theta-n*nfp*phi) * cos(phi);
                        }else if(d == 1) {
                            if(stellsym)
                                continue;
                            data(k1, k2, 0, counter) = -(n*nfp)*m*cos(m*theta-n*nfp*phi) * sin(phi) \
                                                       + m * sin(m*theta-n*nfp*phi) * cos(phi);
                            data(k1, k2, 1, counter) = (n*nfp)*m*cos(m*theta-n*nfp*phi) * cos(phi) \
                                                       + m * sin(m*theta-n*nfp*phi) * sin(phi);
                        }
                        else if(d == 2) {
                            if(stellsym)
                                continue;
                            data(k1, k2, 2, counter) = (n*nfp)*m*cos(m*theta-n*nfp*phi);
                        }
                        counter++;
                    }
                }
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<=0) continue;
                        if(d == 0) {
                            if(stellsym)
                                continue;
                            data(k1, k2, 0, counter) = (n*nfp)*m*sin(m*theta-n*nfp*phi) * cos(phi) \
                                                      - m * cos(m*theta-n*nfp*phi) * sin(phi);
                            data(k1, k2, 1, counter) = (n*nfp) * m * sin(m*theta-n*nfp*phi) * sin(phi) \
                                                      + m * cos(m*theta-n*nfp*phi) * cos(phi);
                        }else if(d == 1) {
                            data(k1, k2, 0, counter) = -(n*nfp)*m*sin(m*theta-n*nfp*phi) * sin(phi) \
                                                       - m * cos(m*theta-n*nfp*phi) * cos(phi);
                            data(k1, k2, 1, counter) = -(-n*nfp)*m*sin(m*theta-n*nfp*phi) * cos(phi) \
                                                       - m * cos(m*theta-n*nfp*phi) * sin(phi);
                        }
                        else if(d == 2) {
                            data(k1, k2, 2, counter) = -(-n*nfp)*(m)*sin(m*theta-n*nfp*phi);
                        }
                        counter++;
                    }
                }
            }
        }
    }
    data *= 4 * M_PI * M_PI;
}

template<class Array>
void SurfaceXYZFourier<Array>::dgammadash2dash2_by_dcoeff_impl(Array& data) {
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            int counter = 0;
            for (int d = 0; d < 3; ++d) {
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<0) continue;
                        if(d == 0) {
                            data(k1, k2, 0, counter) = (-m) * m * cos(m*theta-n*nfp*phi) * cos(phi);
                            data(k1, k2, 1, counter) = (-m) * m * cos(m*theta-n*nfp*phi) * sin(phi);
                        }else if(d == 1) {
                            if(stellsym)
                                continue;
                            data(k1, k2, 0, counter) = (-m)* m * cos(m*theta-n*nfp*phi) * (-1) * sin(phi);
                            data(k1, k2, 1, counter) = (-m) *m * cos(m*theta-n*nfp*phi) * cos(phi);
                        }
                        else if(d == 2) {
                            if(stellsym)
                                continue;
                            data(k1, k2, 2, counter) = (-m) * m * cos(m*theta-n*nfp*phi);
                        }
                        counter++;
                    }
                }
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<=0) continue;
                        if(d == 0) {
                            if(stellsym)
                                continue;
                            data(k1, k2, 0, counter) = - m * m * sin(m*theta-n*nfp*phi) * cos(phi);
                            data(k1, k2, 1, counter) = - m * m * sin(m*theta-n*nfp*phi) * sin(phi);
                        }else if(d == 1) {
                            data(k1, k2, 0, counter) = - m * m * sin(m*theta-n*nfp*phi) * (-1) * sin(phi);
                            data(k1, k2, 1, counter) = - m * m * sin(m*theta-n*nfp*phi) * cos(phi);
                        }
                        else if(d == 2) {
                            data(k1, k2, 2, counter) = - m * m * sin(m*theta-n*nfp*phi);
                        }
                        counter++;
                    }
                }
            }
        }
    }
    data *= 4*M_PI*M_PI;
}

#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
template class SurfaceXYZFourier<Array>;
