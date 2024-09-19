
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

std::tuple<double, double, double, double, double, double, double, double, double> \
    Pd(double phi, double theta) {
    double cosT = std::cos(theta);
    double sinT = std::sin(theta);
    double cosP = std::cos(phi);
    double sinP = std::sin(phi);
    double p00 = cosT * cosP;
    double p01 = -cosT * sinP;
    double p02 = sinT;
    double p10 = sinP;
    double p11 = cosP;
    double p12 = 0.0;
    double p20 = -sinT * cosP;
    double p21 = sinT * sinP;
    double p22 = cosT;
    return std::make_tuple(p00, p01, p02, p10, p11, p12, p20, p21, p22);
}

void iterate_over_corners(int i, int j, int k, \
    double x, double y, double z, \
    double& h00, double& h01, double& h02, double& h10, double& h11, \
    double& h12, double& h20, double& h21, double& h22) {
    //    
    double summa = std::pow(-1, i+j+k);
    double rijk = std::sqrt(x * x + y * y + z * z);

    double epsx = 0.0;
    double epsy = 0.0;
    double epsz = 0.0;
    if (rijk == std::abs(x) && x < 0) {
        epsx = 1e-20;
    }
    if (rijk == std::abs(y) && x < 0) {
        epsx = 1e-20;
    }
    if (rijk == std::abs(z) && x < 0) {
        epsz = 1e-20;
    }

    double atan_xy = summa * std::atan2(y * x, z * rijk);
    double atan_xz = summa * std::atan2(z * x, y * rijk);
    double atan_yz = summa * std::atan2(y * z, x * rijk);
    double log_x = summa * std::log(x + rijk + epsx);
    double log_y = summa * std::log(y + rijk + epsy);
    double log_z = summa * std::log(z + rijk + epsz);
    h00 += atan_xy + atan_xz;
    h01 += log_z;
    h02 += log_y;
    h10 += log_z;
    h11 += atan_xy + atan_yz;
    h12 += log_x;
    h20 += log_y;
    h21 += log_x;
    h22 += atan_xz + atan_yz;
    return;
}

std::tuple<double, double, double, double, double, double, double, double, double> \
    Hd_i_prime(double rx_loc, double ry_loc, double rz_loc, double dimx, double dimy, double dimz) {
    //
    double X0 = rx_loc + dimx / 2.0;
    double X1 = rx_loc - dimx / 2.0;
    double Y0 = ry_loc + dimy / 2.0;
    double Y1 = ry_loc - dimy / 2.0;
    double Z0 = rz_loc + dimz / 2.0;
    double Z1 = rz_loc - dimz / 2.0;
    double H00 = 0.0;
    double H01 = 0.0;
    double H02 = 0.0;
    double H10 = 0.0;
    double H11 = 0.0;
    double H12 = 0.0;
    double H20 = 0.0;
    double H21 = 0.0;
    double H22 = 0.0;
    iterate_over_corners(0, 0, 0, X0, Y0, Z0, H00, H01, H02, H10, H11, H12, H20, H21, H22);
    iterate_over_corners(0, 0, 1, X0, Y0, Z1, H00, H01, H02, H10, H11, H12, H20, H21, H22);
    iterate_over_corners(0, 1, 0, X0, Y1, Z0, H00, H01, H02, H10, H11, H12, H20, H21, H22);
    iterate_over_corners(1, 0, 0, X1, Y0, Z0, H00, H01, H02, H10, H11, H12, H20, H21, H22);
    iterate_over_corners(1, 1, 0, X1, Y1, Z0, H00, H01, H02, H10, H11, H12, H20, H21, H22);
    iterate_over_corners(0, 1, 1, X0, Y1, Z1, H00, H01, H02, H10, H11, H12, H20, H21, H22);
    iterate_over_corners(1, 0, 1, X1, Y0, Z1, H00, H01, H02, H10, H11, H12, H20, H21, H22);
    iterate_over_corners(1, 1, 1, X1, Y1, Z1, H00, H01, H02, H10, H11, H12, H20, H21, H22);
    H00 *= 1.0 / (4.0 * M_PI);
    H01 *= 1.0 / (4.0 * M_PI);
    H02 *= 1.0 / (4.0 * M_PI);
    H10 *= 1.0 / (4.0 * M_PI);
    H11 *= 1.0 / (4.0 * M_PI);
    H12 *= 1.0 / (4.0 * M_PI);
    H20 *= 1.0 / (4.0 * M_PI);
    H21 *= 1.0 / (4.0 * M_PI);
    H22 *= 1.0 / (4.0 * M_PI);
    return std::make_tuple(H00, H01, H02, H10, H11, H12, H20, H21, H22);
}

Array B_direct(Array& points, Array& magPos, Array& M, Array& dims, Array& phiThetas) {
    int N = points.shape(0);
    int D = M.shape(0);
    double dimx = dims(0);
    double dimy = dims(1);
    double dimz = dims(2);
    double* points_ptr = &(points(0, 0));
    double* magPos_ptr = &(magPos(0, 0));
    double* phiThetas_ptr = &(phiThetas(0, 0));
    double* M_ptr = &(M(0, 0));
    Array B = xt::zeros<double>({N, 3});
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        double x = points_ptr[3 * n];
        double y = points_ptr[3 * n + 1];
        double z = points_ptr[3 * n + 2];
        for (int d = 0; d < D; ++d) {
            double p00, p01, p02, p10, p11, p12, p20, p21, p22;
            std::tie(p00, p01, p02, p10, p11, p12, p20, p21, p22) = Pd( \
                phiThetas_ptr[2 * d], phiThetas_ptr[2 * d + 1]);
            double rx_glob = x - magPos_ptr[3 * d];
            double ry_glob = y - magPos_ptr[3 * d + 1];
            double rz_glob = z - magPos_ptr[3 * d + 2];
            double Mx_glob = M_ptr[3 * d];
            double My_glob = M_ptr[3 * d + 1];
            double Mz_glob = M_ptr[3 * d + 2];
            double rx_loc = p00 * rx_glob + p01 * ry_glob + p02 * rz_glob;
            double ry_loc = p10 * rx_glob + p11 * ry_glob + p12 * rz_glob;
            double rz_loc = p20 * rx_glob + p21 * ry_glob + p22 * rz_glob;
            double Mx_loc = p00 * Mx_glob + p01 * My_glob + p02 * Mz_glob;
            double My_loc = p10 * Mx_glob + p11 * My_glob + p12 * Mz_glob;
            double Mz_loc = p20 * Mx_glob + p21 * My_glob + p22 * Mz_glob;
            double h00, h01, h02, h10, h11, h12, h20, h21, h22;
            std::tie(h00, h01, h02, h10, h11, h12, h20, h21, h22) = Hd_i_prime( \
                rx_loc, ry_loc, rz_loc, dimx, dimy, dimz);

            double tx = heaviside(dimx / 2.0 - std::abs(rx_loc), 0.5);
            double ty = heaviside(dimy / 2.0 - std::abs(ry_loc), 0.5);
            double tz = heaviside(dimz / 2.0 - std::abs(rz_loc), 0.5);    
            double tm = 2.0 * tx * ty * tz;
            double Bx_loc = (h00 * Mx_loc + h01 * My_loc + h02 * Mz_loc) + tm * Mx_loc;
            double By_loc = (h10 * Mx_loc + h11 * My_loc + h12 * Mz_loc) + tm * My_loc;
            double Bz_loc = (h20 * Mx_loc + h21 * My_loc + h22 * Mz_loc) + tm * Mz_loc;
            B(n, 0) += p00 * Bx_loc + p10 * By_loc + p20 * Bz_loc;
            B(n, 1) += p01 * Bx_loc + p11 * By_loc + p21 * Bz_loc;
            B(n, 2) += p02 * Bx_loc + p12 * By_loc + p22 * Bz_loc;
        }
    }
    double vol = dimx * dimy * dimz;
    return B * mu0 / vol;
}

Array Bn_direct(Array& points, Array& magPos, Array& M, Array& norms, Array& dims, Array& phiThetas) {
    int N = points.shape(0);
    Array B = B_direct(points, magPos, M, dims, phiThetas);
    Array Bn = xt::zeros<double>({N});
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        Bn(n) = B(n, 0) * norms(n, 0) + B(n, 1) * norms(n, 1) + B(n, 2) * norms(n, 2);
    }
    return Bn;
}

std::tuple<double, double, double> gd_i(double rx_loc, double ry_loc, double rz_loc, \
    double nx_loc, double ny_loc, double nz_loc, double dimx, double dimy, double dimz) {
    // 
    double gx_loc, gy_loc, gz_loc, h00, h01, h02, h10, h11, h12, h20, h21, h22;
    double tx = heaviside(dimx / 2.0 - std::abs(rx_loc), 0.5);
    double ty = heaviside(dimy / 2.0 - std::abs(ry_loc), 0.5);
    double tz = heaviside(dimz / 2.0 - std::abs(rz_loc), 0.5);   
    double tm = 2 * tx * ty * tz;
    std::tie(h00, h01, h02, h10, h11, h12, h20, h21, h22) = Hd_i_prime(rx_loc, ry_loc, rz_loc, dimx, dimy, dimz);
    h00 += tm;
    h11 += tm;
    h22 += tm;
    gx_loc = h00 * nx_loc + h10 * ny_loc + h20 * nz_loc;
    gy_loc = h01 * nx_loc + h11 * ny_loc + h21 * nz_loc;
    gz_loc = h02 * nx_loc + h12 * ny_loc + h22 * nz_loc;
    return std::make_tuple(gx_loc, gy_loc, gz_loc);
}

Array Acube(Array& points, Array& magPos, Array& norms, Array& dims, \ 
    Array& phiThetas, int nfp, int stellsym) {
    int N = points.shape(0);
    int D = magPos.shape(0);
    double* points_ptr = &(points(0, 0));
    double* magPos_ptr = &(magPos(0, 0));
    double* phiThetas_ptr = &(phiThetas(0, 0));
    double* norms_ptr = &(norms(0, 0));
    double dimx = dims(0);
    double dimy = dims(1);
    double dimz = dims(2);
    
    Array A = xt::zeros<double>({N, 3 * D});
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        double x = points_ptr[3 * n];
        double y = points_ptr[3 * n + 1];
        double z = points_ptr[3 * n + 2];
        double nx_glob = norms_ptr[3 * n];
        double ny_glob = norms_ptr[3 * n + 1];
        double nz_glob = norms_ptr[3 * n + 2];
        for (int d = 0; d < D; ++d) {
            double phi = phiThetas_ptr[2 * d];
            double theta = phiThetas_ptr[2 * d + 1];
            double cosT = std::cos(theta);
            double sinT = std::sin(theta);
            double cosP = std::cos(phi);
            double sinP = std::sin(phi);
            double P00, P01, P02, P10, P11, P12, P20, P21, P22;
            std::tie(P00, P01, P02, P10, P11, P12, P20, P21, P22) = Pd(phi, theta);
            double mpx = magPos_ptr[3 * d];
            double mpy = magPos_ptr[3 * d + 1];
            double mpz = magPos_ptr[3 * d + 2];
            for(int fp = 0; fp < nfp; ++fp) {
                double phi0 = (2 * M_PI / ((double) nfp)) * fp;
                auto sphi0 = std::sin(phi0);
                auto cphi0 = std::cos(phi0);
                for (int stell = 0; stell < (stellsym + 1); ++stell) {
                    // Calculate new dipole location after accounting for the symmetries
                    // reflect the y and z-components and then rotate by phi0
                    auto mp_x_new = mpx * cphi0 - mpy * sphi0 * pow(-1, stell);
                    auto mp_y_new = mpx * sphi0 + mpy * cphi0 * pow(-1, stell);
                    auto mp_z_new = mpz * pow(-1, stell);
                    double rx_glob = x - mp_x_new;
                    double ry_glob = y - mp_y_new;
                    double rz_glob = z - mp_z_new;
                    double rx_loc = P00 * rx_glob + P01 * ry_glob + P02 * rz_glob;
                    double ry_loc = P10 * rx_glob + P11 * ry_glob + P12 * rz_glob;
                    double rz_loc = P20 * rx_glob + P21 * ry_glob + P22 * rz_glob;
                    double nx_loc = P00 * nx_glob + P01 * ny_glob + P02 * nz_glob;
                    double ny_loc = P10 * nx_glob + P11 * ny_glob + P12 * nz_glob;
                    double nz_loc = P20 * nx_glob + P21 * ny_glob + P22 * nz_glob;
                    double gx_loc, gy_loc, gz_loc;
                    std::tie(gx_loc, gy_loc, gz_loc) = gd_i( \
                        rx_loc, ry_loc, rz_loc, nx_loc, \
                        ny_loc, nz_loc, dimx, dimy, dimz);
                    auto Gx = (P00 * gx_loc + P10 * gy_loc + P20 * gz_loc);
                    auto Gy = (P01 * gx_loc + P11 * gy_loc + P21 * gz_loc);
                    auto Gz = (P02 * gx_loc + P12 * gy_loc + P22 * gz_loc);
                    
                    // rotate by -phi0 and then flip x component
                    // This should be the reverse of what is done to the m vector and the dipole grid
                    // because A * m = A * R^T * R * m and R is an orthogonal matrix both
                    // for a reflection and a rotation.
                    A(n, 3 * d) += (Gx * cphi0 + Gy * sphi0) * pow(-1, stell);
                    A(n, 3 * d + 1) += (-Gx * sphi0 + Gy * cphi0);
                    A(n, 3 * d + 2) += Gz;
                }
            }
        }
    }
    double vol = dimx * dimy * dimz;
    return mu0 * A / vol;
}

Array Bn_fromMat(Array& points, Array& magPos, Array& M, Array& norms, Array& dims, \
    Array& phiThetas, int nfp, int stellsym) {
    int N = points.shape(0);
    int D = M.shape(0);
    Array A = Acube(points, magPos, norms, dims, phiThetas, nfp, stellsym);
    Array Bn = xt::zeros<double>({N});
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
            Bn(n) += A(n, 3 * d) * M(d, 0) + A(n, 3 * d + 1) * M(d, 1) + A(n, 3 * d + 2) * M(d, 2);
        }
    }
    return Bn;
}

