
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

Array iterate_over_corners(Array& corner, Array& x, Array& y, Array& z) {
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

Array Hd_i_prime(double rx_loc, double ry_loc, double rz_loc, Array& dims) {
    
    Array H = xt::zeros<double>({3, 3});

    Array X = xt::zeros<double>({2});
    X[0] = rx_loc + dims[0]/2;
    X[1] = rx_loc - dims[0]/2;
    Array Y = xt::zeros<double>({2});
    Y[0] = ry_loc + dims[1]/2;
    Y[1] = ry_loc - dims[1]/2;
    Array Z = xt::zeros<double>({2});
    Z[0] = rz_loc + dims[2]/2;
    Z[1] = rz_loc - dims[2]/2;

    for (int i = 0; i < 2; ++i) {
        Array corner = xt::zeros<double>({3});
        corner[0] = i;
        for (int j = 0; j < 2; ++j) {
            corner[1] = j;
            for (int k = 0; k < 2; ++k) {
                corner[2] = k;

                H += iterate_over_corners(corner, X, Y, Z) / (4.0 * M_PI);
            }
        }
    }
    return H;
}


Array B_direct(Array& points, Array& magPos, Array& M, Array& dims, Array& phiThetas) {
    int N = points.shape(0);
    int D = M.shape(0);

    Array B = xt::zeros<double>({N, 3});
    #pragma omp parallel for
    for (int n = 0; n < N; ++n) {
        double x = points(n, 0);
        double y = points(n, 1);
        double z = points(n, 2);
        for (int d = 0; d < D; ++d) {
            Array P = Pd(phiThetas(d,0), phiThetas(d,1));

            double rx_glob = x - magPos(d,0);
            double ry_glob = y - magPos(d,1);
            double rz_glob = z - magPos(d,2);

            double mx_glob = M(d, 0);
            double my_glob = M(d, 1);
            double mz_glob = M(d, 2);

            double rx_loc = P(0, 0) * rx_glob + P(0, 1) * ry_glob + P(0, 2) * rz_glob;
            double ry_loc = P(1, 0) * rx_glob + P(1, 1) * ry_glob + P(1, 2) * rz_glob;
            double rz_loc = P(2, 0) * rx_glob + P(2, 1) * ry_glob + P(2, 2) * rz_glob;
            
            double Mx_loc = P(0, 0) * mx_glob + P(0, 1) * my_glob + P(0, 2) * mz_glob;
            double My_loc = P(1, 0) * mx_glob + P(1, 1) * my_glob + P(1, 2) * mz_glob;
            double Mz_loc = P(2, 0) * mx_glob + P(2, 1) * my_glob + P(2, 2) * mz_glob;

            Array H = Hd_i_prime(rx_loc, ry_loc, rz_loc, dims);

            double tx = heaviside(dims[0]/2 - std::abs(rx_loc), 0.5);
            double ty = heaviside(dims[1]/2 - std::abs(ry_loc), 0.5);
            double tz = heaviside(dims[2]/2 - std::abs(rz_loc), 0.5);    
            double tm = 2*tx*ty*tz;

            double Bx_loc = mu0 * (H(0, 0) * Mx_loc + tm * Mx_loc + H(0, 1) * My_loc + tm * Mx_loc + H(0, 2) * Mz_loc + tm * Mx_loc);
            double By_loc = mu0 * (H(1, 0) * Mx_loc + tm * My_loc + H(0, 1) * My_loc + tm * My_loc + H(0, 2) * Mz_loc + tm * My_loc);
            double Bz_loc = mu0 * (H(2, 0) * Mx_loc + tm * Mz_loc + H(0, 1) * My_loc + tm * Mz_loc + H(0, 2) * Mz_loc + tm * Mz_loc);

            B(n, 0) = P(0, 0) * Bx_loc + P(0, 1) * By_loc + P(0, 2) * Bz_loc;
            B(n, 1) = P(1, 0) * Bx_loc + P(1, 1) * By_loc + P(1, 2) * Bz_loc;
            B(n, 2) = P(2, 0) * Bx_loc + P(2, 1) * By_loc + P(2, 2) * Bz_loc;
        }
    }
    return B;
}

Array Bn_direct(Array& points, Array& magPos, Array& M, Array& norms, Array& dims, Array& phiThetas) {
    int N = points.shape(0);
    
    Array B = B_direct(points, magPos, M, dims, phiThetas);
    Array Bn = xt::zeros<double>({N});
    
    #pragma omp parallel for
    for (int n = 0; n < N; ++n) {
        Bn[n] = B(n, 0) * norms(n, 0) + B(n, 1) * norms(n, 1) + B(n, 2) * norms(n, 2);
    }
    return Bn;
}


Array gd_i(double rx_loc, double ry_loc, double rz_loc, double nx_loc, double ny_loc, double nz_loc, Array& dims) {
    double tx = heaviside(dims[0]/2 - std::abs(rx_loc),0.5);
    double ty = heaviside(dims[1]/2 - std::abs(ry_loc),0.5);
    double tz = heaviside(dims[2]/2 - std::abs(rz_loc),0.5);      
    double tm = 2*tx*ty*tz;

    // Hd.T = Hd, symmetric matrix
    Array g_loc = xt::zeros<double>({3});
    Array tmEye = xt::zeros<double>({3,3});
    tmEye(0,0) = tm;
    tmEye(1,1) = tm;
    tmEye(2,2) = tm;
    Array H_tmEye_sum = Hd_i_prime(rx_loc, ry_loc, rz_loc, dims) + tmEye;

    g_loc[0] = H_tmEye_sum(0, 0) * nx_loc + H_tmEye_sum(0, 1) * ny_loc + H_tmEye_sum(0, 2) * nz_loc;
    g_loc[1] = H_tmEye_sum(1, 0) * nx_loc + H_tmEye_sum(1, 1) * ny_loc + H_tmEye_sum(1, 2) * nz_loc;
    g_loc[2] = H_tmEye_sum(2, 0) * nx_loc + H_tmEye_sum(2, 1) * ny_loc + H_tmEye_sum(2, 2) * nz_loc;

    return mu0 * g_loc;
}

Array Acube(Array& points, Array& magPos, Array& norms, Array& dims, Array& phiThetas) {
    int N = points.shape(0);
    int D = magPos.shape(0);
    double magVol = dims[0] * dims[1] * dims[2];
    
    Array A = xt::zeros<double>({N, 3*D});
    #pragma omp parallel for
    for (int n = 0; n < N; ++n) {
        double x = points(n, 0);
        double y = points(n, 1);
        double z = points(n, 2);

        double nx_glob = norms(n, 0);
        double ny_glob = norms(n, 1);
        double nz_glob = norms(n, 2);
        for (int d = 0; d < D; ++d) {
            Array P = Pd(phiThetas(d,0), phiThetas(d,1));

            double rx_glob = x - magPos(d,0);
            double ry_glob = y - magPos(d,1);
            double rz_glob = z - magPos(d,2);

            double rx_loc = P(0, 0) * rx_glob + P(0, 1) * ry_glob + P(0, 2) * rz_glob;
            double ry_loc = P(1, 0) * rx_glob + P(1, 1) * ry_glob + P(1, 2) * rz_glob;
            double rz_loc = P(2, 0) * rx_glob + P(2, 1) * ry_glob + P(2, 2) * rz_glob;
            
            double nx_loc = P(0, 0) * nx_glob + P(0, 1) * ny_glob + P(0, 2) * nz_glob;
            double ny_loc = P(1, 0) * nx_glob + P(1, 1) * ny_glob + P(1, 2) * nz_glob;
            double nz_loc = P(2, 0) * nx_glob + P(2, 1) * ny_glob + P(2, 2) * nz_glob;

            Array g_loc = gd_i(rx_loc, ry_loc, rz_loc, nx_loc, ny_loc, nz_loc, dims);      

            gx_glob = (P(0, 0) * g_loc[0] + P(0, 1) * g_loc[1] + P(0, 2) * g_loc[2]) / magVol;
            gy_glob = (P(1, 0) * g_loc[0] + P(1, 1) * g_loc[1] + P(1, 2) * g_loc[2]) / magVol;
            gz_glob = (P(2, 0) * g_loc[0] + P(2, 1) * g_loc[1] + P(2, 2) * g_loc[2]) / magVol;

            A(n, 3*d) = gx_glob;
            A(n, 3*d + 1) = gy_glob;
            A(n, 3*d + 2) = gz_glob;
        }
    }
    return A;
}

Array Bn_fromMat(Array& points, Array& magPos, Array& M, Array& norms, Array& dims, Array& phiThetas) {
    int N = points.shape(0);
    int D = M.shape(0);
    Array A = Acube(points, magPos, norms, dims, phiThetas);
    Array Ms = xt::zeros<double>({3*D});
    #pragma omp parallel for
    for (int d = 0; d < D; ++d) {
        for (int i = 0; i < 3; ++i) {
            Ms(3*d + i) = M(d,i);
        }
    }
    Array Bn = xt::zeros<double>({N});
    #pragma omp parallel for
    for (int n = 0; n < N; ++n) {
        for (int d = 0; d < 3*D; ++d) {
            Bn[n] += A(n,d) * Ms[d];
        }
    }
    return Bn;
}

