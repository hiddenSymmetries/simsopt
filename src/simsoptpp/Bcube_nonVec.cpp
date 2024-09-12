
#include "Bcube_nonVec.h"

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
    int i = corner(0), j = corner(1), k = corner(2);
    double summa = std::pow(-1, i+j+k);
    double rijk = std::sqrt(x(i)*x(i) + y(j)*y(j) + z(k)*z(k));

    double atan_xy = std::atan2(y(j)*x(i),z(k)*rijk);
    double atan_xz = std::atan2(z(k)*x(i),y(j)*rijk);
    double atan_yz = std::atan2(y(j)*z(k),x(i)*rijk);
    double log_x = std::log(x(i) + rijk);
    double log_y = std::log(y(j) + rijk);
    double log_z = std::log(z(k) + rijk);
    
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
    X(0) = rx_loc + dims(0)/2;
    X(1) = rx_loc - dims(0)/2;
    Array Y = xt::zeros<double>({2});
    Y(0) = ry_loc + dims(1)/2;
    Y(1) = ry_loc - dims(1)/2;
    Array Z = xt::zeros<double>({2});
    Z(0) = rz_loc + dims(2)/2;
    Z(1) = rz_loc - dims(2)/2;

    for (int i = 0; i < 2; ++i) {
        Array corner = xt::zeros<double>({3});
        corner(0) = i;
        for (int j = 0; j < 2; ++j) {
            corner(1) = j;
            for (int k = 0; k < 2; ++k) {
                corner(2) = k;

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
    Array B_loc = xt::zeros<double>({3});
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

            double Mx_glob = M(d, 0);
            double My_glob = M(d, 1);
            double Mz_glob = M(d, 2);

            double rx_loc = P(0, 0) * rx_glob + P(0, 1) * ry_glob + P(0, 2) * rz_glob;
            double ry_loc = P(1, 0) * rx_glob + P(1, 1) * ry_glob + P(1, 2) * rz_glob;
            double rz_loc = P(2, 0) * rx_glob + P(2, 1) * ry_glob + P(2, 2) * rz_glob;
            
            double Mx_loc = P(0, 0) * Mx_glob + P(0, 1) * My_glob + P(0, 2) * Mz_glob;
            double My_loc = P(1, 0) * Mx_glob + P(1, 1) * My_glob + P(1, 2) * Mz_glob;
            double Mz_loc = P(2, 0) * Mx_glob + P(2, 1) * My_glob + P(2, 2) * Mz_glob;

            Array H = Hd_i_prime(rx_loc, ry_loc, rz_loc, dims);

            double tx = heaviside(dims(0)/2 - std::abs(rx_loc), 0.5);
            double ty = heaviside(dims(1)/2 - std::abs(ry_loc), 0.5);
            double tz = heaviside(dims(2)/2 - std::abs(rz_loc), 0.5);    
            double tm = 2*tx*ty*tz;

            double Bx_loc = mu0 * (H(0, 0) * Mx_loc + tm * Mx_loc + H(0, 1) * My_loc + tm * Mx_loc + H(0, 2) * Mz_loc + tm * Mx_loc);
            double By_loc = mu0 * (H(1, 0) * Mx_loc + tm * My_loc + H(1, 1) * My_loc + tm * My_loc + H(1, 2) * Mz_loc + tm * My_loc);
            double Bz_loc = mu0 * (H(2, 0) * Mx_loc + tm * Mz_loc + H(2, 1) * My_loc + tm * Mz_loc + H(2, 2) * Mz_loc + tm * Mz_loc);

            B_loc(0) = Bx_loc;
            B_loc(1) = By_loc;
            B_loc(2) = Bz_loc;

            B(n, 0) += P(0, 0) * Bx_loc + P(1, 0) * By_loc + P(2, 0) * Bz_loc;
            B(n, 1) += P(0, 1) * Bx_loc + P(1, 1) * By_loc + P(2, 1) * Bz_loc;
            B(n, 2) += P(0, 2) * Bx_loc + P(1, 2) * By_loc + P(2, 2) * Bz_loc;
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
        Bn(n) = B(n, 0) * norms(n, 0) + B(n, 1) * norms(n, 1) + B(n, 2) * norms(n, 2);
    }
    return Bn;
}


Array gd_i(double rx_loc, double ry_loc, double rz_loc, double nx_loc, double ny_loc, double nz_loc, Array& dims) {
    double tx = heaviside(dims(0)/2 - std::abs(rx_loc),0.5);
    double ty = heaviside(dims(1)/2 - std::abs(ry_loc),0.5);
    double tz = heaviside(dims(2)/2 - std::abs(rz_loc),0.5);      
    double tm = 2*tx*ty*tz;

    // Hd.T = Hd, symmetric matrix
    Array g_loc = xt::zeros<double>({3});
    Array tmEye = xt::zeros<double>({3,3});
    tmEye(0,0) = tm;
    tmEye(1,1) = tm;
    tmEye(2,2) = tm;
    Array H_tmEye_sum = Hd_i_prime(rx_loc, ry_loc, rz_loc, dims) + tmEye;

    g_loc(0) = H_tmEye_sum(0, 0) * nx_loc + H_tmEye_sum(0, 1) * ny_loc + H_tmEye_sum(0, 2) * nz_loc;
    g_loc(1) = H_tmEye_sum(1, 0) * nx_loc + H_tmEye_sum(1, 1) * ny_loc + H_tmEye_sum(1, 2) * nz_loc;
    g_loc(2) = H_tmEye_sum(2, 0) * nx_loc + H_tmEye_sum(2, 1) * ny_loc + H_tmEye_sum(2, 2) * nz_loc;

    return mu0 * g_loc;
}

Array Acube(Array& points, Array& magPos, Array& norms, Array& dims, Array& phiThetas) {
    int N = points.shape(0);
    int D = magPos.shape(0);
    // double magVol = dims(0) * dims(1) * dims(2);
    double* points_ptr = &(points(0, 0));
    double* magPos_ptr = &(magPos(0, 0));
    double* phiThetas_ptr = &(phiThetas(0, 0));
    double* norms_ptr = &(norms(0, 0));
    
    Array A = xt::zeros<double>({N, 3*D});
    double* A_ptr = &(A(0, 0));
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(4); // Use 4 threads for all consecutive parallel regions
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        double x = points_ptr[n];
        double y = points_ptr[n + 1];
        double z = points_ptr[n + 2];
        std::cout << "threads=" << omp_get_num_threads() << std::endl;
        double nx_glob = norms_ptr[n];
        double ny_glob = norms_ptr[n + 1];
        double nz_glob = norms_ptr[n + 2];
        for (int d = 0; d < D; ++d) {
            // Array P = Pd(phiThetas(d, 0), phiThetas(d, 1));
            double phi = phiThetas_ptr[d];
            double theta = phiThetas_ptr[d + 1];
            double cosT = std::cos(theta);
            double sinT = std::sin(theta);
            double cosP = std::cos(phi);
            double sinP = std::sin(phi);
            double P00 = cosT * cosP;
            double P01 = -cosT * sinP;
            double P02 = sinT;
            double P10 = sinP;
            double P11 = cosP;
            double P12 = 0.0;
            double P20 = -sinT * cosP;
            double P21 = sinT * sinP;
            double P22 = cosT;
            double rx_glob = x - magPos_ptr[d];
            double ry_glob = y - magPos_ptr[d + 1];
            double rz_glob = z - magPos_ptr[d + 2];
            double rx_loc = P00 * rx_glob + P01 * ry_glob + P02 * rz_glob;
            double ry_loc = P10 * rx_glob + P11 * ry_glob + P12 * rz_glob;
            double rz_loc = P20 * rx_glob + P21 * ry_glob + P22 * rz_glob;
            double nx_loc = P00 * nx_glob + P01 * ny_glob + P02 * nz_glob;
            double ny_loc = P10 * nx_glob + P11 * ny_glob + P12 * nz_glob;
            double nz_loc = P20 * nx_glob + P21 * ny_glob + P22 * nz_glob;

            Array g_loc = gd_i(rx_loc, ry_loc, rz_loc, nx_loc, ny_loc, nz_loc, dims);      

            // got rid of / magVol factor here so need to divide the matrix 
            // in the Python code, after the call to this c++ function
            A_ptr[(n * D + d) * 3] = (P00 * g_loc(0) + P10 * g_loc(1) + P20 * g_loc(2));
            A_ptr[(n * D + d) * 3 + 1] = (P00 * g_loc(0) + P10 * g_loc(1) + P20 * g_loc(2));
            A_ptr[(n * D + d) * 3 + 2] = (P00 * g_loc(0) + P10 * g_loc(1) + P20 * g_loc(2));
            // A(n, 3*d) = (P00 * g_loc[0] + P10 * g_loc[1] + P20 * g_loc[2]);
            // A(n, 3*d + 1) = (P01 * g_loc[0] + P11 * g_loc[1] + P21 * g_loc[2]);
            // A(n, 3*d + 2) = (P02 * g_loc[0] + P12 * g_loc[1] + P22 * g_loc[2]);
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
            Bn(n) += A(n,d) * Ms(d);
        }
    }
    return Bn;
}

