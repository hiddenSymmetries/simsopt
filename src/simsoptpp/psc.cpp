#include "psc.h"
#include <cstdio>

// Calculate the inductance matrix needed for the PSC forward problem
Array L_matrix(Array& points, Array& alphas, Array& deltas, Array& phi, double R)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(alphas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("alphas normal needs to be in row-major storage order");
    if(deltas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("deltas needs to be in row-major storage order");
    if(phi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("phi needs to be in row-major storage order");
    
    // points shape should be (num_coils, 3)
    int num_coils = alphas.shape(0);  // shape should be (num_coils)
    int num_phi = phi.shape(0);  // shape should be (num_phi)
    Array L = xt::zeros<double>({num_coils, num_coils});
    
    // initialize pointers to the beginning of alphas, deltas, points
    double* points_ptr = &(points(0, 0));
    double* alphas_ptr = &(alphas(0));
    double* deltas_ptr = &(deltas(0));
    double* phi_ptr = &(phi(0));
//     m_ptr[3 * j + 0]
    
    // Loop through the the PSC array
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_coils; i++) {
        auto cai = cos(alphas_ptr[i]);
        auto sai = sin(alphas_ptr[i]);
        auto cdi = cos(deltas_ptr[i]);
        auto sdi = sin(deltas_ptr[i]);
        auto xi = points_ptr[3 * i];
        auto yi = points_ptr[3 * i + 1];
        auto zi = points_ptr[3 * i + 2];
    	// Loop through all j > i coils
        for (int j = (i + 1); j < num_coils; ++j) {
            auto xj = (points_ptr[3 * j] - xi) / R;
            auto yj = (points_ptr[3 * j + 1] - yi) / R;
            auto zj = (points_ptr[3 * j + 2] - zi) / R;
            auto caj = cos(alphas_ptr[j]);
            auto saj = sin(alphas_ptr[j]);
            auto cdj = cos(deltas_ptr[j]);
            auto sdj = sin(deltas_ptr[j]);
            auto integrand = 0.0;
            for (int k = 0; k < num_phi; ++k) {
                auto ck = cos(phi_ptr[k]);
                auto sk = sin(phi_ptr[k]);
                for (int kk = 0; kk < num_phi; ++kk) {
                    auto ckk = cos(phi_ptr[kk]);
                    auto skk = sin(phi_ptr[kk]);
                    auto dl2_x = (-sk * cdi + ck * sai * sdi) * (-skk * cdj + ckk * saj * sdj);
                    auto dl2_y = (ck * cai) * (ckk * caj);
                    auto dl2_z = (sk * sdi + ck * sai * cdi) * (skk * sdj + ckk * saj * cdj);
                    auto x2 = (xj + ckk * cdj + skk * saj * sdj - ck * cdi - sk * sai * sdi);
                    auto y2 = (yj + skk * caj - sk * cai);
                    auto z2 = (zj + skk * saj * cdj - ckk * sdj - sk * sai * cdi + ck * sdi);
                    integrand += (dl2_x + dl2_y + dl2_z) / sqrt(x2 * x2 + y2 * y2 + z2 * z2);
                }
            }
            L(i, j) = integrand;
        }
    }
    return L;
}

// Array TF_fluxes(Array& points, Array& alphas, Array& deltas, Array& rho, Array& phi, Array& I, Array& normal, double R)
Array flux_xyz(Array& points, Array& alphas, Array& deltas, Array& rho, Array& phi, Array& normal)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(alphas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("alphas normal needs to be in row-major storage order");
    if(deltas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("deltas needs to be in row-major storage order");
    if(rho.layout() != xt::layout_type::row_major)
          throw std::runtime_error("rho needs to be in row-major storage order");
    if(phi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("phi needs to be in row-major storage order");
//     if(I.layout() != xt::layout_type::row_major)
//           throw std::runtime_error("I needs to be in row-major storage order");
    if(normal.layout() != xt::layout_type::row_major)
          throw std::runtime_error("normal needs to be in row-major storage order");
          
    // points shape should be (num_coils, 3)
    // I shape should be (num_coils, 3)
    // normal shape should be (num_coils, 3)
    int num_coils = alphas.shape(0);  // shape should be (num_coils)
    int num_phi = phi.shape(0);  // shape should be (num_phi)
    int num_rho = rho.shape(0);  // shape should be (num_rho)
//     Array Psi = xt::zeros<double>({num_coils});
    Array XYZs = xt::zeros<double>({num_coils, num_rho, num_phi, 3});
//     double R2 = R * R;
//     double fac = 4e-7 * R2;
//     using namespace boost::math;
    
    #pragma omp parallel for schedule(static)
    for(int j = 0; j < num_coils; j++) {
        auto cdj = cos(deltas(j));
        auto caj = cos(alphas(j));
        auto sdj = sin(deltas(j));
        auto saj = sin(alphas(j));
        auto nx = normal(j, 0);
        auto ny = normal(j, 1);
        auto nz = normal(j, 2);
        auto xj = points(j, 0);
        auto yj = points(j, 1);
        auto zj = points(j, 2);
//         double integral = 0.0;
        for (int k = 0; k < num_rho; ++k) {
            auto xx = rho(k);
            for (int kk = 0; kk < num_phi; ++kk) {
//                 printf("%d %d %d\n", j, k, kk);
                auto yy = phi(kk);
                auto x0 = xx * cos(yy);
                auto y0 = xx * sin(yy);
                auto x = x0 * cdj + y0 * saj * sdj + xj;
                auto y = y0 * caj + yj;
                auto z = -x0 * sdj + y0 * saj * cdj + zj;
//                 auto rho2 = (x - xj) * (x - xj) + (y - yj) * (y - yj);
//                 auto r2 = rho2 + (z - zj) * (z - zj);
//                 auto rho = sqrt(rho2);
//                 auto R2_r2 = R2 + r2;
//                 auto gamma2 = R2_r2 + 2 * R * rho;
//                 auto beta2 = R2_r2 - 2 * R * rho;
//                 auto beta = sqrt(beta2);
//                 auto k2 = 1 - gamma2 / beta2;
//                 auto beta_gamma2 = beta * gamma2;
//                 auto k = sqrt(k2);
//                 auto ellipe = ellint_2(k);
//                 auto ellipk = ellint_1(k);
//                 auto Eplus = R2_r2 * ellipe - gamma2 * ellipk;
//                 auto Eminus = (R2 - r2) * ellipe + gamma2 * ellipk;
//                 auto Bx = (x - xj) * (z - zj) * Eplus / (rho2 * beta_gamma2);
//                 auto By = (y - yj) * (z - zj) * Eplus / (rho2 * beta_gamma2);
//                 auto Bz = Eminus / beta_gamma2;
//                 auto Bn = Bx * nx + By * ny + Bz * nz;
//                 integral += Bn * xx;
                XYZs(j, k, kk, 0) = x;
                XYZs(j, k, kk, 1) = y;
                XYZs(j, k, kk, 2) = z;
            }
        }
//         Psi(j) = integral * fac * I(j);
    }
//     return Psi;
    return XYZs;
}

Array flux_integration(Array& B, Array& rho, Array& normal)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(B.layout() != xt::layout_type::row_major)
          throw std::runtime_error("B needs to be in row-major storage order");
    if(rho.layout() != xt::layout_type::row_major)
          throw std::runtime_error("rho needs to be in row-major storage order");
    if(normal.layout() != xt::layout_type::row_major)
          throw std::runtime_error("normal needs to be in row-major storage order");
          
    // normal shape should be (num_coils, 3)
    int num_coils = B.shape(0);  // shape should be (num_coils, N, N, 3)
    int N = rho.shape(0);  // shape should be (N)
    Array Psi = xt::zeros<double>({num_coils}); 
    
    #pragma omp parallel for schedule(static)
    for(int j = 0; j < num_coils; j++) {
        auto nx = normal(j, 0);
        auto ny = normal(j, 1);
        auto nz = normal(j, 2);
        double integral = 0.0;
        for(int k = 0; k < N; k++) {
            auto xx = rho(k);
            for(int kk = 0; kk < N; kk++) {
                auto Bn = B(j, k, kk, 0) * nx + B(j, k, kk, 1) * ny + B(j, k, kk, 2) * nz; 
                integral += Bn * xx;
            }
        }
        Psi(j) = integral;
    }
    return Psi;
}

Array Bn_PSC(Array& points, Array& plasma_points, Array& alphas, Array& deltas, Array& plasma_normal, Array& coil_normal, double R)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(alphas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("alphas needs to be in row-major storage order");
    if(deltas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("deltas needs to be in row-major storage order");
    if(plasma_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("plasma_points needs to be in row-major storage order");
    if(plasma_normal.layout() != xt::layout_type::row_major)
          throw std::runtime_error("plasma_normal needs to be in row-major storage order");
          
    // points shape should be (num_coils, 3)
    // plasma_normal shape should be (num_plasma_points, 3)
    // plasma_points should be (num_plasma_points, 3)
    int num_coils = alphas.shape(0);  // shape should be (num_coils)
    int num_plasma_points = plasma_points.shape(0);
    
    // this variable is the A matrix in the least-squares term so A * I = Bn
    Array Bn_PSC = xt::zeros<double>({num_plasma_points, num_coils});
    double R2 = R * R;
    double fac = 2.0e-7;  // * R2;
    using namespace boost::math;
    
//     #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_plasma_points; ++i) {
        auto xp = plasma_points(i, 0);
        auto yp = plasma_points(i, 1);
        auto zp = plasma_points(i, 2);
        auto nxp = plasma_normal(i, 0);
        auto nyp = plasma_normal(i, 1);
        auto nzp = plasma_normal(i, 2);
        for(int j = 0; j < num_coils; j++) {
            auto cdj = cos(deltas(j));
            auto caj = cos(alphas(j));
            auto sdj = sin(deltas(j));
            auto saj = sin(alphas(j));
            auto x0 = (xp - points(j, 0));
            auto y0 = (yp - points(j, 1));
            auto z0 = (zp - points(j, 2));
            // get phi, theta from coil normals
            auto nx = coil_normal(j, 0);
            auto ny = coil_normal(j, 1);
            auto nz = coil_normal(j, 2);
            auto theta = atan2(ny, nx);
            auto phi = atan2(sqrt(nx * nx + ny * ny), nz);
            auto ct = cos(theta);
            auto st = sin(theta);
            auto cp = cos(phi);
            auto sp = sin(phi);
//             auto nxx = cp * ct * ct + st * st;
//             auto nxy = -sin(phi / 2.0) * sin(phi / 2.0) * sin(2.0 * theta);
//             auto nxz = ct * sp;
//             auto nyx = -sin(phi / 2.0) * sin(phi / 2.0) * sin(2.0 * theta);
//             auto nyy = ct * ct + cp * st * st;
//             auto nyz = sp * st;
//             auto nzx = -ct * sp;
//             auto nzy = -sp * st;
//             auto nzz = cp;
//  rotation_matrix[i, :, :] = np.array(
//      [[np.cos(deltas[i]), 
//        np.sin(deltas[i]) * np.sin(alphas[i]),
//        np.sin(deltas[i]) * np.cos(alphas[i])],
//        [0.0, np.cos(alphas[i]), -np.sin(alphas[i])],
//        [-np.sin(deltas[i]), 
//         np.cos(deltas[i]) * np.sin(alphas[i]),
//         np.cos(deltas[i]) * np.cos(alphas[i])]])
            auto nxx = cdj;
            auto nxy = sdj * saj;
            auto nxz = sdj * caj;
            auto nyx = 0.0;
            auto nyy = caj;
            auto nyz = -saj;
            auto nzx = -sdj;
            auto nzy = cdj * saj;
            auto nzz = cdj * caj;
            // apply transpose of the rotation matrix
            auto x = x0 * nxx + y0 * nyx + z0 * nzx;
            auto y = x0 * nxy + y0 * nyy + z0 * nzy;
            auto z = x0 * nxz + y0 * nyz + z0 * nzz;
//             printf("i j x y z = %d %d %f %f %f\n", i, j, x, y, z);
//             auto x = x0 * cdj - z0 * sdj;
//             auto y = x0 * saj * sdj + y0 * caj + z0 * saj * cdj;
//             auto z = x0 * sdj * caj - y0 * saj + z0 * caj * cdj;
            auto rho2 = x * x + y * y;
            auto r2 = rho2 + z * z;
            auto rho = sqrt(rho2);
            auto R2_r2 = R2 + r2;
            auto gamma2 = R2_r2 - 2.0 * R * rho;
            auto beta2 = R2_r2 + 2.0 * R * rho;
            auto beta = sqrt(beta2);
            auto k2 = 1.0 - gamma2 / beta2;
            auto beta_gamma2 = beta * gamma2;
            auto k = sqrt(k2);
            auto ellipe = ellint_2(k);
            auto ellipk = ellint_1(k);
            auto Eplus = R2_r2 * ellipe - gamma2 * ellipk;
            auto Eminus = (R2 - r2) * ellipe + gamma2 * ellipk;
            auto Bx = x * z * Eplus / (rho2 * beta_gamma2);
            auto By = y * z * Eplus / (rho2 * beta_gamma2);
            auto Bz = Eminus / beta_gamma2;
            // Need to rotate the vector
            auto Bx_rot = Bx * nxx + By * nxy + Bz * nxz;
            auto By_rot = Bx * nyx + By * nyy + Bz * nyz;
            auto Bz_rot = Bx * nzx + By * nzy + Bz * nzz;
//             printf("i j x y z = %d %d %f %f %f\n", i, j, Bx, By, Bz);

//             auto Bx_rot = Bx * cdj + By * saj * sdj + Bz * caj * sdj;
//             auto By_rot = By * caj - Bz * saj;
//             auto Bz_rot = -Bx * sdj + By * saj * cdj + Bz * caj * cdj;
            Bn_PSC(i, j) = Bx_rot * nxp + By_rot * nyp + Bz_rot * nzp;
        }
    }
    return Bn_PSC * fac;
}
