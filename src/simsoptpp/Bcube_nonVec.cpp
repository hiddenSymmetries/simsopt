
#include "Bcube_nonVec.h"
#include <cmath>

double mu0 = 4 * M_PI * 1e-7;

double heaviside(double x1, double x2) {
    if (x1 < 0) {
        return 0;
    } else if (x1 > 0) {
        return 1;
    } else {
        return x2;
    };
}

Array Pd(double phi, double theta) {
    double cosT = std::cos(theta);
    double sinT = std::sin(theta);
    double cosP = std::cos(phi);
    double sinP = std::sin(phi);
    Array P = xt::zeros<double>({3, 3});
    P(0, 0) = cosT * cosP;
    P(0, 1) = -cosT * sinP;
    P(0, 2) = sinT;
    P(1, 0) = sinP;
    P(1, 1) = cosP;
    P(1, 2) = 0.0;
    P(2, 0) = -sinT * cosP;
    P(2, 1) = sinT * sinP;
    P(2, 2) = cosT;
    return P;
}

Array iterate_over_corners(Array corner, Array x, Array y, Array z) {
    int i = corner[0], j = corner[1], k = corner[2];
    double summa = std::pow(-1, i+j+k);
    double rijk = std::sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k]);

    double atan_xy = std::atan2(y[j]*x[i],z[k]*rijk);
    double atan_xz = std::atan2(z[k]*x[i],y[j]*rijk);
    double atan_yz = std::atan2(y[j]*z[k],x[i]*rijk);
    double log_x = std::log(x[i] + rijk);
    double log_y = std::log(y[j] + rijk);
    double log_z = std::log(z[k] + rijk);
    
    Array h = xt::zeros<double>({3, 3});
    h(0,0) = atan_xy + atan_xz;
    h(0,1) = log_z;
    h(0,2) = log_y;
    h(1,0) = log_z;
    h(1,1) = atan_xy + atan_yz;
    h(1,2) = log_x;
    h(2,0) = log_y;
    h(2,1) = log_x;
    h(2,2) = atan_xz + atan_yz;
    return summa * h;
}

Array Hd_i_prime(Array r, Array dims) {
    
    Array H = xt::zeros<double>({3, 3});

    double xp = r[0], yp = r[1], zp = r[2];

    Array X = xt::zeros<double>({2});
    X[0] = xp + dims[0]/2;
    X[1] = xp - dims[0]/2;
    Array Y = xt::zeros<double>({2});
    Y[0] = yp + dims[1]/2;
    Y[1] = yp - dims[1]/2;
    Array Z = xt::zeros<double>({2});
    Z[0] = zp + dims[2]/2;
    Z[1] = zp - dims[2]/2;

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                Array corner = xt::zeros<double>({3});
                corner[0] = i;
                corner[1] = j;
                corner[2] = k;

                H += iterate_over_corners(corner, X, Y, Z) / (4.0 * M_PI);
            }
        }
    }
    return H;
}


Array B_direct(Array points, Array magPos, Array M, Array dims, Array phiThetas) {
    int N = points.shape(0);
    int D = M.shape(0);

    Array B = xt::zeros<double>({N, 3});
    for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
            Array P = Pd(phiThetas(d,0), phiThetas(d,1));
            Array r = xt::zeros<double>({3});
            r[0] = points(n,0) - magPos(d,0);
            r[1] = points(n,1) - magPos(d,1);
            r[2] = points(n,2) - magPos(d,2);
            Array r_loc = xt::zeros<double>({3});
            Array M_loc = xt::zeros<double>({3});
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    r_loc[i] += P(i,j) * r[j];
                    M_loc[i] += P(i,j) * M[j];
                }
            }

            Array H = Hd_i_prime(r_loc, dims);

            double tx = heaviside(dims[0]/2 - std::abs(r_loc[0]),0.5);
            double ty = heaviside(dims[1]/2 - std::abs(r_loc[1]),0.5);
            double tz = heaviside(dims[2]/2 - std::abs(r_loc[2]),0.5);    
            double tm = 2*tx*ty*tz;

            Array B_loc = xt::zeros<double>({3});
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    B_loc(n,i) += mu0 * (H(i,j) * M_loc[j] + tm * M_loc[i]);
                }                   
            }
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    B(n,i) += P(j,i) * B_loc[j];
                }
            }
        }
    }
    return B;
}

Array Bn_direct(Array points, Array magPos, Array M, Array norms, Array dims, Array phiThetas) {
    int N = points.shape(0);
    
    Array B = B_direct(points, magPos, M, dims, phiThetas);
    Array Bn = xt::zeros<double>({N});
    
    for (int n = 0; n < N; ++n) {
        for (int i = 0; i < 3; ++ i) {
            Bn[n] += B(n,i) * norms(n,i);
        }
    }
    return Bn;
}


Array gd_i(Array r_loc, Array n_i_loc, Array dims) {
    double tx = heaviside(dims[0]/2 - std::abs(r_loc[0]),0.5);
    double ty = heaviside(dims[1]/2 - std::abs(r_loc[1]),0.5);
    double tz = heaviside(dims[2]/2 - std::abs(r_loc[2]),0.5);      
    double tm = 2*tx*ty*tz;

    // Hd.T = Hd, symmetric matrix
    Array g_loc = xt::zeros<double>({3});
    Array tmEye = xt::zeros<double>({3,3});
    tmEye(0,0) = tm;
    tmEye(1,1) = tm;
    tmEye(2,2) = tm;
    Array H_tmEye_sum = Hd_i_prime(r_loc,dims) + tmEye;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            g_loc[i] += H_tmEye_sum(i,j) * n_i_loc[j];
        }
    }

    return mu0 * g_loc;
}

Array Acube(Array points, Array magPos, Array norms, Array dims, Array phiThetas) {
    int N = points.shape(0);
    int D = magPos.shape(0);
    double magVol = dims[0] * dims[1] * dims[2];
    
    Array A = xt::zeros<double>({N,3*D});
    for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
            Array P = Pd(phiThetas(d,0), phiThetas(d,1));
            Array r = xt::zeros<double>({3});
            r[0] = points(n,0) - magPos(d,0);
            r[1] = points(n,1) - magPos(d,1);
            r[2] = points(n,2) - magPos(d,2);
            Array r_loc = xt::zeros<double>({3});
            Array n_loc = xt::zeros<double>({3});
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    r_loc[i] += P(i,j) * r[j];
                    n_loc[i] += P(i,j) * norms(n,j);
                }
            }
            Array g_loc = gd_i(r_loc, n_loc, dims);      
            Array g = xt::zeros<double>({3});
            // double magVol = dims(d,0) * dims(d,1) * dims(d,2);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    g[i] += (P(j,i) * g_loc[j]) / magVol;
                }
            }
            A(n,3*d) = g[0];
            A(n,3*d + 1) = g[1];
            A(n,3*d + 2) = g[2];
        }
    }
    return A;
}

Array Bn_fromMat(Array points, Array magPos, Array M, Array norms, Array dims, Array phiThetas) {
    int N = points.shape(0);
    int D = M.shape(0);
    Array A = Acube(points, magPos, norms, dims, phiThetas);
    Array Ms = xt::zeros<double>({3*D});
    for (int d = 0; d < D; ++d) {
        for (int i = 0; i < 3; ++i) {
            Ms(3*d + i) = M(d,i);
        }
    }
    Array Bn = xt::zeros<double>({N});
    for (int n = 0; n < N; ++n) {
        for (int d = 0; d < 3*D; ++d) {
            Bn[n] += A(n,d) * Ms[d];
        }
    }
    return Bn;
}

