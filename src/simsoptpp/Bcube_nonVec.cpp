
#include <cmath>
#include <iostream>


double mu0 = 4 * M_PI * 1e-7;

double heaviside(double& x1, double& x2) {
    if (x1 < 0) {
        return 0;
    };
    else if (x1 > 0) {
        return 1;
    };
    else {
        return x2;
    };
}


double Pd(double& phi, double& theta) {
    const int rows = 3;
    const int cols = 3;
    
    double cosT = std::cos(theta);
    double sinT = std::sin(theta);
    double cosP = std::cos(phi);
    double sinP = std::sin(phi);
    
    double* P[rows][cols] = {
    {cosT * cosP, -cosT * sinP, sinT},
    {sinP, cosP, 0.0},
    {-sinT * cosP, sinT * sinP, cosT}
    };
    return P;
}


double iterate_over_corners(tuple& corner, double& x, double& y, double& z) {
    int i = corner[0], j = corner[1], k = corner[2];
    int summa = std::pow(-1, i+j+k);
    double rijk = std::sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k]);

    double atan_xy = std::atan2(y[j]*x[i],z[k]*rijk);
    atan_xz = std::atan2(z[k]*x[i],y[j]*rijk);
    atan_yz = std::atan2(y[j]*z[k],x[i]*rijk);
    log_x = std::log(x[i] + rijk);
    log_y = std::log(y[j] + rijk);
    log_z = std::log(z[k] + rijk);
    
    const int rows = 3;
    const int cols = 3;
    double h[rows][cols] = {
    {atan_xy + atan_xz, log_z, log_y},
    {log_z, atan_xy + atan_yz, log_x},
    {log_y, log_x, atan_xz + atan_yz}
    };

    return h;
}


double Hd_i_prime(double& r, double& dims) {
    const int rows = 3;
    const int cols = 3;
    
    double H[rows][cols] = {0};

    double xp = r[0], yp = r[1], zp = r[2];

    double x[2] = {xp + dims[0]/2, xp - dims[0]/2};
    double y[2] = {yp + dims[1]/2, yp - dims[1]/2};
    double z[2] = {zp + dims[2]/2, zp - dims[2]/2};
}


double B_direct(double& points, double& magPos, double& M, double& dims, double& phiThetas) {
    const int N = sizeof(points) / sizeof(points[0]);
    const int D = sizeof(M) / sizeof(M[0]);
    
    double B[N][3] = {0}; //find out how to do dot products, how to use multiple indeces
    for (n = 0; n < N; ++n) {
        for (d = 0; d < D; ++d) {
            double P = Pd(phiThetas[d][0], phiThetas[d][1]);
            double r_loc = P DOT (points[n] - magPos[d]);

            double tx = heaviside(dims[0]/2 - np.abs(r_loc[0]),0.5);
            double ty = heaviside(dims[1]/2 - np.abs(r_loc[1]),0.5);
            double tz = heaviside(dims[2]/2 - np.abs(r_loc[2]),0.5);    
            double tm = 2*tx*ty*tz;
            
            B[n] += mu0 * P.T DOT (Hd_i_prime(r_loc,dims) @ (P @ M[d]) + tm*P@M[d]); //how does transpose work?
                };
    };
    return B;
}


double Bn_direct(double& points, double& magPos, double& M, double& norms, double& dims, double& phiThetas) {
    const int N = sizeof(points) / sizeof(points[0]);
    
    double B = B_direct(points, magPos, M, dims, phiThetas);
    double Bn[N] = {0};
    
    for (n = 0; n < N; ++n) {
        Bn[n] = B[n] DOT norms[n];
    };
    return Bn;
}


double gd_i(r_loc, n_i_loc, dims) {
    double tx = heaviside(dims[0]/2 - np.abs(r_loc[0]),0.5);
    double ty = heaviside(dims[1]/2 - np.abs(r_loc[1]),0.5);
    double tz = heaviside(dims[2]/2 - np.abs(r_loc[2]),0.5);      
    double tm = 2*tx*ty*tz;

    return mu0 * (Hd_i_prime(r_loc,dims).T + tm*np.eye(3)) DOT n_i_loc; //shortcut for identity matrix?
}


double Acube(double& points, double& magPos, double& norms, double& dims, double& phiThetas) {
    const int N = sizeof(points) / sizeof(points[0]);
    const int D = sizeof(magPos) / sizeof(magPos[0]);
    
    double A[N][3*D] = {0};
    for (n = 0; n < N; ++n) {
        for (d = 0; d < D; ++d) {
            double P = Pd(phiThetas[d][0], phiThetas[d][1]);
            double r_loc = P DOT (points[n] - magPos[d]);
            double n_loc = P DOT norms[n];
            
            double g = P.T DOT gd_i(r_loc,n_loc,dims);
            A[n][3*d : 3*d + 3] = g; //can I still index this way?
            }
        };
    };

    if ((N,3*D) != A.shape)
        throw std::runtime_error("A shape altered during Acube"); //is there an assert function?
    return A;
}


double Bn_fromMat(double& points, double& magPos, double& M, double& norms, double& dims, double& phiThetas) {
    double A = Acube(points, magPos, norms, dims, phiThetas);
    Ms = np.concatenate(M); //equiv for concatenate?
    return A DOT Ms;
}

