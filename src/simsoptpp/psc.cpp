#include "psc.h"
#include "simdhelpers.h"
#include "vec3dsimd.h"

#if defined(USE_XSIMD)
// Calculate the inductance matrix needed for the PSC forward problem
Array L_matrix(Array& points, Array& alphas, Array& deltas, Array& int_points, Array& int_weights)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(alphas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("alphas normal needs to be in row-major storage order");
    if(deltas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("deltas needs to be in row-major storage order");
    
    // points shape should be (num_coils, 3)
    int num_coils = alphas.shape(0);  // shape should be (num_coils)
    int num_quad = int_points.shape(0);  // shape should be (num_phi)
    Array L = xt::zeros<double>({num_coils, num_coils});
    
    // initialize pointers to the beginning of alphas, deltas, points
    double* points_ptr = &(points(0, 0));
    double* alphas_ptr = &(alphas(0));
    double* deltas_ptr = &(deltas(0));
    double* weight_ptr = &(int_weights(0));
    double* int_point_ptr = &(int_points(0));
    constexpr int simd_size = xsimd::simd_type<double>::size;
    
    // Loop through the the PSC array
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_coils; i++) {
        auto point_i = Vec3dSimd(points_ptr[3 * i], \
                                  points_ptr[3 * i + 1], \
                                  points_ptr[3 * i + 2]);   
        simd_t ai = ((simd_t) alphas_ptr[i]);   
        simd_t di = ((simd_t) deltas_ptr[i]);       
        simd_t cai = xsimd::cos(ai);
        simd_t sai = xsimd::sin(ai);
        simd_t cdi = xsimd::cos(di);
        simd_t sdi = xsimd::sin(di);
        simd_t sai_sdi = sai * sdi;
        simd_t sai_cdi = sai * cdi;
     	// Loop through all j > i coils
        for (int j = (i + 1); j < num_coils; j += simd_size) {
            auto point_j = Vec3dSimd();
            // check that j + k isn't bigger than num_points
            int klimit = std::min(simd_size, num_coils - j);
            simd_t integrand, aj, dj;
            for(int k = 0; k < klimit; k++){
                for (int d = 0; d < 3; ++d) {
                    point_j[d][k] = points_ptr[3 * (j + k) + d];
                }
                integrand[k] = 0.0;
                aj[k] = alphas_ptr[j + k];
                dj[k] = deltas_ptr[j + k];
            }
            point_j = point_j - point_i;
            simd_t caj = xsimd::cos(aj);
            simd_t saj = xsimd::sin(aj);
            simd_t cdj = xsimd::cos(dj);
            simd_t sdj = xsimd::sin(dj);
            simd_t saj_sdj = saj * sdj;
            simd_t saj_cdj = saj * cdj;
            for (int k = 0; k < num_quad; ++k) {
                simd_t pk = ((simd_t) int_point_ptr[k]);
                simd_t ck = xsimd::cos(pk);
                simd_t sk = xsimd::sin(pk);
                auto dli = Vec3dSimd(-sk * cdi + ck * sai_sdi, \
                                      ck * cai, \
                                      sk * sdi + ck * sai_cdi);
                auto r2_partial = Vec3dSimd(point_j.x - ck * cdi - sk * sai_sdi, \
                                            point_j.y - sk * cai, \
                                            point_j.z - sk * sai_cdi + ck * sdi);
                for (int kk = 0; kk < num_quad; ++kk) {
                    simd_t weight = ((simd_t) (weight_ptr[k] * weight_ptr[kk]));
                    simd_t pkk = ((simd_t) int_point_ptr[kk]);
                    simd_t ckk = xsimd::cos(pkk);
                    simd_t skk = xsimd::sin(pkk);
                    auto dlj = Vec3dSimd(-skk * cdj + ckk * saj_sdj, \
                                          ckk * caj, \
                                          skk * sdj + ckk * saj_cdj);
                    simd_t dl_dot_prod = inner(dli, dlj);
                    auto r_diff = Vec3dSimd(r2_partial.x + ckk * cdj + skk * saj_sdj, \
                                            r2_partial.y + skk * caj, \
                                            r2_partial.z + skk * saj_cdj - ckk * sdj);
                    simd_t rmag_2 = normsq(r_diff);
                    simd_t rmag_inv = rsqrt(rmag_2);
                    integrand += weight * dl_dot_prod * rmag_inv;
                }
            }
            for(int k = 0; k < klimit; k++){
                L(i, j + k) += integrand[k];
            }
        }
    }
    return L * M_PI * M_PI;
}

// Calculate the inductance matrix needed for the PSC forward problem
Array L_deriv(Array& points, Array& alphas, Array& deltas, Array& int_points, Array& int_weights)
{
    // points shape should be (num_coils, 3)
    int num_coils = alphas.shape(0);  // shape should be (num_coils)
    int num_quad = int_points.shape(0);
    Array L_deriv = xt::zeros<double>({2 * num_coils, num_coils, num_coils});
    
    // initialize pointers to the beginning of alphas, deltas, points
    double* points_ptr = &(points(0, 0));
    double* alphas_ptr = &(alphas(0));
    double* deltas_ptr = &(deltas(0));
    double* weight_ptr = &(int_weights(0));
    double* int_point_ptr = &(int_points(0));    
    constexpr int simd_size = xsimd::simd_type<double>::size;
    
    // Loop through the the PSC array
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_coils; i += simd_size) {
        int klimit = std::min(simd_size, num_coils - i);
        auto point_i = Vec3dSimd();
        simd_t ai, di;
        for(int k = 0; k < klimit; k++){
            for (int d = 0; d < 3; ++d) {
                point_i[d][k] = points_ptr[3 * (i + k) + d];
            }
            ai[k] = alphas_ptr[i + k];
            di[k] = deltas_ptr[i + k];
        }
        simd_t cai = xsimd::cos(ai);
        simd_t sai = xsimd::sin(ai);
        simd_t cdi = xsimd::cos(di);
        simd_t sdi = xsimd::sin(di);
    	// Loop through all j > i coils
        for (int j = 0; j < num_coils; ++j) {
            auto point_j = Vec3dSimd(points_ptr[3 * j] - point_i.x, \
                                     points_ptr[3 * j + 1] - point_i.y, \
                                     points_ptr[3 * j + 2] - point_i.z);
            simd_t aj = ((simd_t) alphas_ptr[j]);
            simd_t dj = ((simd_t) deltas_ptr[j]);
            simd_t caj = xsimd::cos(aj);
            simd_t saj = xsimd::sin(aj);
            simd_t cdj = xsimd::cos(dj);
            simd_t sdj = xsimd::sin(dj);
            simd_t integrand1 = ((simd_t) 0.0);
            simd_t integrand2 = ((simd_t) 0.0);
            for (int k = 0; k < num_quad; ++k) {
                simd_t pk = ((simd_t) int_point_ptr[k]);
                simd_t ck = xsimd::cos(pk);
                simd_t sk = xsimd::sin(pk);
                for (int kk = 0; kk < num_quad; ++kk) {
                    simd_t pkk = ((simd_t) int_point_ptr[kk]);
                    simd_t weight = ((simd_t) (weight_ptr[k] * weight_ptr[kk]));
                    simd_t ckk = xsimd::cos(pkk);
                    simd_t skk = xsimd::sin(pkk);
                    auto dl2 = Vec3dSimd((-sk * cdi + ck * sai * sdi) * (-skk * cdj + ckk * saj * sdj), \
                                         (ck * cai) * (ckk * caj), \
                                         (sk * sdi + ck * sai * cdi) * (skk * sdj + ckk * saj * cdj));
                    // dkappa of the numerator
                    auto dl2_dalpha = Vec3dSimd((ck * cai * sdi) * (-skk * cdj + ckk * saj * sdj), \
                                                -(ck * sai) * (ckk * caj), \
                                                (ck * cai * cdi) * (skk * sdj + ckk * saj * cdj));
                    auto dl2_ddelta = Vec3dSimd((sk * sdi + ck * sai * cdi) * (-skk * cdj + ckk * saj * sdj), \
                                                ((simd_t) 0.0), \
                                                (sk * cdi - ck * sai * sdi) * (skk * sdj + ckk * saj * cdj));
                    auto xxi = Vec3dSimd(ck * cdi + sk * sai * sdi, \
                                         sk * cai, \
                                         sk * sai * cdi - ck * sdi);
                    auto xxj = Vec3dSimd(point_j.x + ckk * cdj + skk * saj * sdj, \
                                         point_j.y + skk * caj, \
                                         point_j.z + skk * saj * cdj - ckk * sdj);
                    auto xxi_dalpha = Vec3dSimd(sk * cai * sdi, \
                                                -sk * sai, \
                                                sk * cai * cdi);
                    auto xxi_ddelta = Vec3dSimd(-ck * sdi + sk * sai * cdi, \
                                                ((simd_t) 0.0), \
                                                -sk * sai * sdi - ck * cdi);
                    Vec3dSimd p2 = xxi - xxj;
                    simd_t deriv_alpha = inner(xxi_dalpha, p2);
                    simd_t deriv_delta = inner(xxi_ddelta, p2);
                    simd_t rmag_2 = normsq(p2);
                    simd_t rmag_inv = rsqrt(rmag_2);
                    simd_t denom3 = rmag_inv * rmag_inv * rmag_inv;
                    // First term in the derivative
                    integrand1 += weight * (dl2_dalpha.x + dl2_dalpha.y + dl2_dalpha.z) * rmag_inv;
                    integrand2 += weight * (dl2_ddelta.x + dl2_ddelta.y + dl2_ddelta.z) * rmag_inv;
                    // Second term in the derivative
                    integrand1 -= weight * (dl2.x + dl2.y + dl2.z) * deriv_alpha * denom3;
                    integrand2 -= weight * (dl2.x + dl2.y + dl2.z) * deriv_delta * denom3;
                }
            }
            for(int k = 0; k < klimit; k++){
                if ((i + k) != j) {
                    L_deriv(i + k, i + k, j) = integrand1[k];
                    L_deriv(i + k + num_coils, i + k, j) = integrand2[k];
                }
            }
        }
    }
    return L_deriv * M_PI * M_PI;
}

Array dpsi_dkappa(Array& I_TF, Array& dl_TF, Array& gamma_TF, Array& PSC_points, Array& alphas, Array& deltas, Array& coil_normals, Array& rho, Array& phi, double R)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(gamma_TF.layout() != xt::layout_type::row_major)
          throw std::runtime_error("gamma_TF needs to be in row-major storage order");
    if(alphas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("alphas needs to be in row-major storage order");
    if(deltas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("deltas needs to be in row-major storage order");
    if(PSC_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("PSC_points needs to be in row-major storage order");
    if(coil_normals.layout() != xt::layout_type::row_major)
          throw std::runtime_error("coil_normals needs to be in row-major storage order");
          
    using namespace boost::math;
    // points shape should be (num_coils, 3)
    // plasma_normal shape should be (num_plasma_points, 3)
    // plasma_points should be (num_plasma_points, 3)
    int num_TF_coils = I_TF.shape(0);  // shape should be (num_coils)
    int num_PSC_coils = coil_normals.shape(0);  // shape should be (num_coils)
    int num_evaluation_points = PSC_points.shape(0);
    int num_phi_TF = gamma_TF.shape(1);
    int num_integration_points = rho.shape(0);
    
    // this variable is the A matrix in the least-squares term so A * I = Bn
    Array dpsi = xt::zeros<double>({num_PSC_coils * 2});
    constexpr int simd_size = xsimd::simd_type<double>::size;
    
    double* points_ptr = &(PSC_points(0, 0));
    double* normals_ptr = &(coil_normals(0, 0));
    double* alphas_ptr = &(alphas(0));
    double* deltas_ptr = &(deltas(0));
    // Shared pointers over indices other than kk can cause memory issues
    double* rho_ptr = &(rho(0));
    double* phi_ptr = &(phi(0));
    double* I_ptr = &(I_TF(0));
    double* gamma_ptr = &(gamma_TF(0, 0, 0));
    double* dl_ptr = &(dl_TF(0, 0, 0));
    double fac = 1.0e-7;
    
    // loop over all the PSC coils
    #pragma omp parallel for schedule(static)
    for (int kk = 0; kk < num_PSC_coils; kk += simd_size) {
        auto point_kk = Vec3dSimd();
        auto n_kk = Vec3dSimd();
        int klimit = std::min(simd_size, num_PSC_coils - kk);
        simd_t integrand, aj, dj;
        for(int k = 0; k < klimit; k++){
            for (int d = 0; d < 3; ++d) {
                point_kk[d][k] = points_ptr[3 * (kk + k) + d];
                n_kk[d][k] = normals_ptr[3 * (kk + k) + d];
            }
            integrand[k] = 0.0;
            aj[k] = alphas_ptr[kk + k];
            dj[k] = deltas_ptr[kk + k];
        }
        simd_t cdj = xsimd::cos(dj);
        simd_t caj = xsimd::cos(aj);
        simd_t sdj = xsimd::sin(dj);
        simd_t saj = xsimd::sin(aj);
        // same normal for all these evaluation points so need an extra loop over all the PSCs
        auto Rxx = cdj;
        auto Rxy = sdj * saj;
        auto Ryx = 0.0;
        auto Ryy = caj;
        auto Rzx = -sdj;
        auto Rzy = cdj * saj;
        auto dRxx_dalpha = 0.0;
        auto dRxy_dalpha = sdj * caj;
        auto dRxz_dalpha = -sdj * saj;
        auto dRyx_dalpha = 0.0;
        auto dRyy_dalpha = -saj;
        auto dRyz_dalpha = -caj;
        auto dRzx_dalpha = 0.0;
        auto dRzy_dalpha = cdj * caj;
        auto dRzz_dalpha = -cdj * saj;
        auto dRxx_ddelta = -sdj;
        auto dRxy_ddelta = cdj * saj;
        auto dRxz_ddelta = cdj * caj;
        auto dRyx_ddelta = 0.0;
        auto dRyy_ddelta = 0.0;
        auto dRyz_ddelta = 0.0;
        auto dRzx_ddelta = -cdj;
        auto dRzy_ddelta = -sdj * saj;
        auto dRzz_ddelta = -sdj * caj;
        auto B1 = Vec3dSimd();
        auto B2 = Vec3dSimd();
        auto B3 = Vec3dSimd();
        // Do the integral over the PSC cross section
        for (int i = 0; i < num_integration_points; ++i) {
            // evaluation points here should be the points on a PSC coil cross section
            simd_t rho_i = ((simd_t) rho_ptr[i]);  // needed for integrating over the disk
            simd_t phi_i = ((simd_t) phi_ptr[i]);
            simd_t x0 = rho_i * xsimd::cos(phi_i);
            simd_t y0 = rho_i * xsimd::sin(phi_i);
            auto xi = Vec3dSimd((Rxx * x0 + Rxy * y0) + point_kk.x, \
                                (Ryy * y0) + point_kk.y, \
                                (Rzx * x0 + Rzy * y0) + point_kk.z);
            // loop here is over all the TF coils
            for(int j = 0; j < num_TF_coils; j++) {
                simd_t I_j = ((simd_t) I_ptr[j]);
                simd_t int_fac = rho_i * I_j;
                // Do Biot Savart over each TF coil
                for (int k = 0; k < num_phi_TF; ++k) {
                    auto gamma_k = Vec3dSimd(gamma_ptr[(j * num_phi_TF + k) * 3], \
                                             gamma_ptr[(j * num_phi_TF + k) * 3 + 1], \
                                             gamma_ptr[(j * num_phi_TF + k) * 3 + 2]);
                    auto dl_k = Vec3dSimd(dl_ptr[(j * num_phi_TF + k) * 3], \
                                          dl_ptr[(j * num_phi_TF + k) * 3 + 1], \
                                          dl_ptr[(j * num_phi_TF + k) * 3 + 2]);
                    Vec3dSimd RTdiff = xi - gamma_k;
                    auto dl_cross_RTdiff = cross(dl_k, RTdiff);
                    simd_t rmag_2 = normsq(RTdiff);
                    simd_t rmag_inv = rsqrt(rmag_2);
                    simd_t denom3 = rmag_inv * rmag_inv * rmag_inv;
                    simd_t denom5 = denom3 * rmag_inv * rmag_inv;
                    // First derivative contribution of three
                    B1 += dl_cross_RTdiff * denom3 * int_fac;
                    // second derivative contribution (should be dR/dalpha)
                    auto dR_dalphaT = Vec3dSimd(dRxx_dalpha * x0 + dRxy_dalpha * y0, \
                                                dRyx_dalpha * x0 + dRyy_dalpha * y0, \
                                                dRzx_dalpha * x0 + dRzy_dalpha * y0);
                    auto dR_ddeltaT = Vec3dSimd(dRxx_ddelta * x0 + dRxy_ddelta * y0, \
                                                dRyx_ddelta * x0 + dRyy_ddelta * y0, \
                                                dRzx_ddelta * x0 + dRzy_ddelta * y0);
                    auto dl_cross_dR_dalphaT = cross(dl_k, dR_dalphaT);
                    auto dl_cross_dR_ddeltaT = cross(dl_k, dR_ddeltaT);
                    B2 += dl_cross_dR_dalphaT * denom3 * int_fac;
                    B3 += dl_cross_dR_ddeltaT * denom3 * int_fac;
                    // third derivative contribution
                    simd_t RTdiff_dot_dR_dalpha = inner(RTdiff, dR_dalphaT);
                    simd_t RTdiff_dot_dR_ddelta = inner(RTdiff, dR_ddeltaT);
                    B2 += dl_cross_RTdiff * RTdiff_dot_dR_dalpha * denom5 * (int_fac * -3.0);
                    B3 += dl_cross_RTdiff * RTdiff_dot_dR_ddelta * denom5 * (int_fac * -3.0);
                }
            }
        }
        simd_t inner_prod2 = inner(B2, n_kk);
        simd_t inner_prod3 = inner(B3, n_kk);
        for(int k = 0; k < klimit; k++){
            // rotate first contribution by dR/dalpha, then dot into zhat direction (not the normal!)
            dpsi(kk + k) = (dRxz_dalpha * B1.x + dRyz_dalpha * B1.y + dRzz_dalpha * B1.z)[k];
            // second contribution just gets dotted with the normal vector to the PSC loop
            dpsi(kk + k) += inner_prod2[k];
            // repeat for delta derivative
            dpsi(kk + k + num_PSC_coils) = (dRxz_ddelta * B1.x + dRyz_ddelta * B1.y + dRzz_ddelta * B1.z)[k];  // * nz;
            dpsi(kk + k + num_PSC_coils) += inner_prod3[k];
        }
    }
    return dpsi * fac;
}

#else

Array dpsi_dkappa(Array& I_TF, Array& dl_TF, Array& gamma_TF, Array& PSC_points, Array& alphas, Array& deltas, Array& coil_normals, Array& rho, Array& phi, double R)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(gamma_TF.layout() != xt::layout_type::row_major)
          throw std::runtime_error("gamma_TF needs to be in row-major storage order");
    if(alphas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("alphas needs to be in row-major storage order");
    if(deltas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("deltas needs to be in row-major storage order");
    if(PSC_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("PSC_points needs to be in row-major storage order");
    if(coil_normals.layout() != xt::layout_type::row_major)
          throw std::runtime_error("coil_normals needs to be in row-major storage order");
          
    using namespace boost::math;
    // points shape should be (num_coils, 3)
    // plasma_normal shape should be (num_plasma_points, 3)
    // plasma_points should be (num_plasma_points, 3)
    int num_TF_coils = I_TF.shape(0);  // shape should be (num_coils)
    int num_PSC_coils = coil_normals.shape(0);  // shape should be (num_coils)
    int num_evaluation_points = PSC_points.shape(0);
    int num_phi_TF = gamma_TF.shape(1);
    int num_integration_points = rho.shape(0);
    
    // this variable is the A matrix in the least-squares term so A * I = Bn
    Array dpsi = xt::zeros<double>({num_PSC_coils * 2});
    
    double* points_ptr = &(PSC_points(0, 0));
    double* normals_ptr = &(coil_normals(0, 0));
    double* alphas_ptr = &(alphas(0));
    double* deltas_ptr = &(deltas(0));
    // Shared pointers over indices other than kk can cause memory issues
    double* rho_ptr = &(rho(0));
    double* phi_ptr = &(phi(0));
    double* I_ptr = &(I_TF(0));
    double* gamma_ptr = &(gamma_TF(0, 0, 0));
    double* dl_ptr = &(dl_TF(0, 0, 0));
    double fac = 1.0e-7;
    
    // loop over all the PSC coils
    #pragma omp parallel for schedule(static)
    for (int kk = 0; kk < num_PSC_coils; ++kk) {
        auto xkk = points_ptr[3 * kk];
        auto ykk = points_ptr[3 * kk + 1];
        auto zkk = points_ptr[3 * kk + 2];
        auto cdj = cos(deltas_ptr[kk]);
        auto caj = cos(alphas_ptr[kk]);
        auto sdj = sin(deltas_ptr[kk]);
        auto saj = sin(alphas_ptr[kk]);
        // same normal for all these evaluation points so need an extra loop over all the PSCs
        auto nx = normals_ptr[3 * kk];
        auto ny = normals_ptr[3 * kk + 1];
        auto nz = normals_ptr[3 * kk + 2];
        auto Rxx = cdj;
        auto Rxy = sdj * saj;
        auto Ryx = 0.0;
        auto Ryy = caj;
        auto Rzx = -sdj;
        auto Rzy = cdj * saj;
        auto dRxx_dalpha = 0.0;
        auto dRxy_dalpha = sdj * caj;
        auto dRxz_dalpha = -sdj * saj;
        auto dRyx_dalpha = 0.0;
        auto dRyy_dalpha = -saj;
        auto dRyz_dalpha = -caj;
        auto dRzx_dalpha = 0.0;
        auto dRzy_dalpha = cdj * caj;
        auto dRzz_dalpha = -cdj * saj;
        auto dRxx_ddelta = -sdj;
        auto dRxy_ddelta = cdj * saj;
        auto dRxz_ddelta = cdj * caj;
        auto dRyx_ddelta = 0.0;
        auto dRyy_ddelta = 0.0;
        auto dRyz_ddelta = 0.0;
        auto dRzx_ddelta = -cdj;
        auto dRzy_ddelta = -sdj * saj;
        auto dRzz_ddelta = -sdj * caj;
        auto Bx1 = 0.0;
        auto By1 = 0.0;
        auto Bz1 = 0.0;
        auto Bx2 = 0.0;
        auto By2 = 0.0;
        auto Bz2 = 0.0;
        auto Bx3 = 0.0;
        auto By3 = 0.0;
        auto Bz3 = 0.0;
        // Do the integral over the PSC cross section
        for (int i = 0; i < num_integration_points; ++i) {
            // evaluation points here should be the points on a PSC coil cross section
            auto rho_i = rho_ptr[i];  // needed for integrating over the disk
            auto phi_i = phi_ptr[i];
            auto x0 = rho_i * cos(phi_i);
            auto y0 = rho_i * sin(phi_i);
            // z0 = 0 here
            auto xi = (Rxx * x0 + Rxy * y0) + xkk;
            auto yi = (Ryy * y0) + ykk;
            auto zi = (Rzx * x0 + Rzy * y0) + zkk;
            // loop here is over all the TF coils
            for(int j = 0; j < num_TF_coils; j++) {
                auto I_j = I_ptr[j];
                auto int_fac = rho_i * I_j;
                // Do Biot Savart over each TF coil
                for (int k = 0; k < num_phi_TF; ++k) {
                    auto xk = gamma_ptr[(j * num_phi_TF + k) * 3];
                    auto yk = gamma_ptr[(j * num_phi_TF + k) * 3 + 1];
                    auto zk = gamma_ptr[(j * num_phi_TF + k) * 3 + 2];
                    auto dlx = dl_ptr[(j * num_phi_TF + k) * 3];
                    auto dly = dl_ptr[(j * num_phi_TF + k) * 3 + 1];
                    auto dlz = dl_ptr[(j * num_phi_TF + k) * 3 + 2];
                    // multiply by R (not R^T!) and then subtract off coil coordinate
                    auto RTxdiff = xi - xk;
                    auto RTydiff = yi - yk;
                    auto RTzdiff = zi - zk;
                    auto dl_cross_RTdiff_x = dly * RTzdiff - dlz * RTydiff;
                    auto dl_cross_RTdiff_y = dlz * RTxdiff - dlx * RTzdiff;
                    auto dl_cross_RTdiff_z = dlx * RTydiff - dly * RTxdiff;
                    auto denom = sqrt(RTxdiff * RTxdiff + RTydiff * RTydiff + RTzdiff * RTzdiff);
                    auto denom3 = denom * denom * denom;
                    auto denom5 = denom3 * denom * denom;
                    // First derivative contribution of three
                    Bx1 += dl_cross_RTdiff_x / denom3 * int_fac;
                    By1 += dl_cross_RTdiff_y / denom3 * int_fac;
                    Bz1 += dl_cross_RTdiff_z / denom3 * int_fac;
                    // second derivative contribution (should be dR/dalpha)
                    auto dR_dalphaT_x = dRxx_dalpha * x0 + dRxy_dalpha * y0;
                    auto dR_dalphaT_y = dRyx_dalpha * x0 + dRyy_dalpha * y0;
                    auto dR_dalphaT_z = dRzx_dalpha * x0 + dRzy_dalpha * y0;
                    auto dR_ddeltaT_x = dRxx_ddelta * x0 + dRxy_ddelta * y0;
                    auto dR_ddeltaT_y = dRyx_ddelta * x0 + dRyy_ddelta * y0;
                    auto dR_ddeltaT_z = dRzx_ddelta * x0 + dRzy_ddelta * y0;
                    auto dl_cross_dR_dalphaT_x = dly * dR_dalphaT_z - dlz * dR_dalphaT_y;
                    auto dl_cross_dR_dalphaT_y = dlz * dR_dalphaT_x - dlx * dR_dalphaT_z;
                    auto dl_cross_dR_dalphaT_z = dlx * dR_dalphaT_y - dly * dR_dalphaT_x;
                    auto dl_cross_dR_ddeltaT_x = dly * dR_ddeltaT_z - dlz * dR_ddeltaT_y;
                    auto dl_cross_dR_ddeltaT_y = dlz * dR_ddeltaT_x - dlx * dR_ddeltaT_z;
                    auto dl_cross_dR_ddeltaT_z = dlx * dR_ddeltaT_y - dly * dR_ddeltaT_x;
                    Bx2 += dl_cross_dR_dalphaT_x / denom3 * int_fac;
                    By2 += dl_cross_dR_dalphaT_y / denom3 * int_fac;
                    Bz2 += dl_cross_dR_dalphaT_z / denom3 * int_fac;
                    Bx3 += dl_cross_dR_ddeltaT_x / denom3 * int_fac;
                    By3 += dl_cross_dR_ddeltaT_y / denom3 * int_fac;
                    Bz3 += dl_cross_dR_ddeltaT_z / denom3 * int_fac;
                    // third derivative contribution
                    auto RTxdiff_dot_dR_dalpha = RTxdiff * dR_dalphaT_x + RTydiff * dR_dalphaT_y + RTzdiff * dR_dalphaT_z;
                    auto RTxdiff_dot_dR_ddelta = RTxdiff * dR_ddeltaT_x + RTydiff * dR_ddeltaT_y + RTzdiff * dR_ddeltaT_z;
                    Bx2 += - 3.0 * dl_cross_RTdiff_x * RTxdiff_dot_dR_dalpha / denom5 * int_fac;
                    By2 += - 3.0 * dl_cross_RTdiff_y * RTxdiff_dot_dR_dalpha / denom5 * int_fac;
                    Bz2 += - 3.0 * dl_cross_RTdiff_z * RTxdiff_dot_dR_dalpha / denom5 * int_fac;
                    Bx3 += - 3.0 * dl_cross_RTdiff_x * RTxdiff_dot_dR_ddelta / denom5 * int_fac;
                    By3 += - 3.0 * dl_cross_RTdiff_y * RTxdiff_dot_dR_ddelta / denom5 * int_fac;
                    Bz3 += - 3.0 * dl_cross_RTdiff_z * RTxdiff_dot_dR_ddelta / denom5 * int_fac;
                }
            }
        }
        // rotate first contribution by dR/dalpha, then dot into zhat direction (not the normal!)
        dpsi(kk) = dRxz_dalpha * Bx1 + dRyz_dalpha * By1 + dRzz_dalpha * Bz1;
        // second contribution just gets dotted with the normal vector to the PSC loop
        dpsi(kk) += Bx2 * nx + By2 * ny + Bz2 * nz;
        // repeat for delta derivative
        dpsi(kk + num_PSC_coils) = dRxz_ddelta * Bx1 + dRyz_ddelta * By1 + dRzz_ddelta * Bz1;  // * nz;
        dpsi(kk + num_PSC_coils) += Bx3 * nx + By3 * ny + Bz3 * nz;
    }
    return dpsi * fac;
}

// Calculate the inductance matrix needed for the PSC forward problem
Array L_matrix(Array& points, Array& alphas, Array& deltas, Array& int_points, Array& int_weights)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(alphas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("alphas normal needs to be in row-major storage order");
    if(deltas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("deltas needs to be in row-major storage order");
    
    // points shape should be (num_coils, 3)
    int num_coils = alphas.shape(0);  // shape should be (num_coils)
    int num_quad = int_points.shape(0);
    Array L = xt::zeros<double>({num_coils, num_coils});
    
    // initialize pointers to the beginning of alphas, deltas, points
    double* points_ptr = &(points(0, 0));  // normalized by coil radius
    double* alphas_ptr = &(alphas(0));
    double* deltas_ptr = &(deltas(0));
    double* weight_ptr = &(int_weights(0));
    double* int_point_ptr = &(int_points(0));
    
    // Loop through the the PSC array
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_coils; i++) {
        auto cai = cos(alphas_ptr[i]);
        auto sai = sin(alphas_ptr[i]);
        auto cdi = cos(deltas_ptr[i]);
        auto sdi = sin(deltas_ptr[i]);
        auto sai_sdi = sai * sdi;
        auto sai_cdi = sai * cdi;
        auto xi = points_ptr[3 * i];
        auto yi = points_ptr[3 * i + 1];
        auto zi = points_ptr[3 * i + 2];
    	// Loop through all j > i coils
        for (int j = (i + 1); j < num_coils; ++j) {
            auto xj = points_ptr[3 * j] - xi;
            auto yj = points_ptr[3 * j + 1] - yi;
            auto zj = points_ptr[3 * j + 2] - zi;
            auto caj = cos(alphas_ptr[j]);
            auto saj = sin(alphas_ptr[j]);
            auto cdj = cos(deltas_ptr[j]);
            auto sdj = sin(deltas_ptr[j]);
            auto saj_sdj = saj * sdj;
            auto saj_cdj = saj * cdj;
            auto integrand = 0.0;
            for (int k = 0; k < num_quad; ++k) {
                auto pk = int_point_ptr[k];
                auto ck = cos(pk);
                auto sk = sin(pk);
                auto f1 = -sk * cdi + ck * sai_sdi;
                auto f2 = ck * cai;
                auto f3 = sk * sdi + ck * sai_cdi;
                auto x2_partial = xj - ck * cdi - sk * sai_sdi;
                auto y2_partial = yj - sk * cai;
                auto z2_partial = zj - sk * sai_cdi + ck * sdi;
                for (int kk = 0; kk < num_quad; ++kk) {
                    auto weight = weight_ptr[k] * weight_ptr[kk];
                    auto pkk = int_point_ptr[kk];
                    auto ckk = cos(pkk);
                    auto skk = sin(pkk);
                    auto x2 = x2_partial + ckk * cdj + skk * saj_sdj;
                    auto y2 = y2_partial + skk * caj;
                    auto z2 = z2_partial + skk * saj_cdj - ckk * sdj;
                    integrand += weight * (f1 * (-skk * cdj + ckk * saj_sdj) +  \
                        f2 * (ckk * caj) + f3 * (skk * sdj + ckk * saj_cdj)) \
                        / sqrt(x2 * x2 + y2 * y2 + z2 * z2);
                }
            }
            L(i, j) = integrand;
        }
    }
    return L * M_PI * M_PI; // M_PI ** 2 factor from Gauss Quadrature [-1, 1] to [0, 2*pi]
}

// Calculate the inductance matrix needed for the PSC forward problem
Array L_deriv(Array& points, Array& alphas, Array& deltas, Array& int_points, Array& int_weights)
{
    // points shape should be (num_coils, 3)
    int num_coils = alphas.shape(0);  // shape should be (num_coils)
    int num_quad = int_points.shape(0);
    Array L_deriv = xt::zeros<double>({2 * num_coils, num_coils, num_coils});
    
    // initialize pointers to the beginning of alphas, deltas, points
    double* points_ptr = &(points(0, 0));
    double* alphas_ptr = &(alphas(0));
    double* deltas_ptr = &(deltas(0));
    double* weight_ptr = &(int_weights(0));
    double* int_point_ptr = &(int_points(0));
    
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
        for (int j = 0; j < num_coils; ++j) {
            if (i != j) {
                auto xj = points_ptr[3 * j] - xi;
                auto yj = points_ptr[3 * j + 1] - yi;
                auto zj = points_ptr[3 * j + 2] - zi;
                auto caj = cos(alphas_ptr[j]);
                auto saj = sin(alphas_ptr[j]);
                auto cdj = cos(deltas_ptr[j]);
                auto sdj = sin(deltas_ptr[j]);
                auto integrand1 = 0.0;
                auto integrand2 = 0.0;
                for (int k = 0; k < num_quad; ++k) {
                    auto pk = int_point_ptr[k];
                    auto ck = cos(pk);
                    auto sk = sin(pk);
                    for (int kk = 0; kk < num_quad; ++kk) {
                        auto pkk = int_point_ptr[kk];
                        auto weight = weight_ptr[k] * weight_ptr[kk];
                        auto ckk = cos(pkk);
                        auto skk = sin(pkk);
                        auto dl2_x = (-sk * cdi + ck * sai * sdi) * (-skk * cdj + ckk * saj * sdj);
                        auto dl2_y = (ck * cai) * (ckk * caj);
                        auto dl2_z = (sk * sdi + ck * sai * cdi) * (skk * sdj + ckk * saj * cdj);
                        // dkappa of the numerator
                        auto dl2_x_dalpha = (ck * cai * sdi) * (-skk * cdj + ckk * saj * sdj);
                        auto dl2_y_dalpha = -(ck * sai) * (ckk * caj);
                        auto dl2_z_dalpha = (ck * cai * cdi) * (skk * sdj + ckk * saj * cdj);
                        auto dl2_x_ddelta = (sk * sdi + ck * sai * cdi) * (-skk * cdj + ckk * saj * sdj);
                        auto dl2_y_ddelta = 0.0;
                        auto dl2_z_ddelta = (sk * cdi - ck * sai * sdi) * (skk * sdj + ckk * saj * cdj);
                        auto xxi = ck * cdi + sk * sai * sdi;  // fixed minus sign on the last term here
                        auto xxj = xj + ckk * cdj + skk * saj * sdj;
                        auto yyi = sk * cai;
                        auto yyj = yj + skk * caj;
                        auto zzi = sk * sai * cdi - ck * sdi;
                        auto zzj = zj + skk * saj * cdj - ckk * sdj;
                        auto xxi_dalpha = sk * cai * sdi; //0.0;
                        auto yyi_dalpha = -sk * sai;
                        auto zzi_dalpha = sk * cai * cdi;
                        auto xxi_ddelta = -ck * sdi + sk * sai * cdi;
                        auto yyi_ddelta = 0.0;
                        auto zzi_ddelta = -sk * sai * sdi - ck * cdi;
                        auto deriv_alpha = xxi_dalpha * (xxi - xxj) + yyi_dalpha * (yyi - yyj) + zzi_dalpha * (zzi - zzj);
                        auto deriv_delta = xxi_ddelta * (xxi - xxj) + yyi_ddelta * (yyi - yyj) + zzi_ddelta * (zzi - zzj);
                        auto x2 = (xxj - xxi) * (xxj - xxi);
                        auto y2 = (yyj - yyi) * (yyj - yyi);
                        auto z2 = (zzj - zzi) * (zzj - zzi);
                        auto denom = sqrt(x2 + y2 + z2);
                        auto denom3 = denom * denom * denom;
                        // First term in the derivative
                        integrand1 += weight * (dl2_x_dalpha + dl2_y_dalpha + dl2_z_dalpha) / denom;
                        integrand2 += weight * (dl2_x_ddelta + dl2_y_ddelta + dl2_z_ddelta) / denom;
                        // Second term in the derivative
                        integrand1 -= weight * (dl2_x + dl2_y + dl2_z) * deriv_alpha / denom3;
                        integrand2 -= weight * (dl2_x + dl2_y + dl2_z) * deriv_delta / denom3;
                    }
                }
                L_deriv(i, i, j) = integrand1;
                L_deriv(i + num_coils, i, j) = integrand2;
            }
        }
    }
    return L_deriv * M_PI * M_PI;
}


#endif


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
    if(normal.layout() != xt::layout_type::row_major)
          throw std::runtime_error("normal needs to be in row-major storage order");
          
    // points shape should be (num_coils, 3)
    // normal shape should be (num_coils, 3)
    int num_coils = alphas.shape(0);  // shape should be (num_coils)
    int num_phi = phi.shape(0);  // shape should be (num_phi)
    int num_rho = rho.shape(0);  // shape should be (num_rho)
    Array XYZs = xt::zeros<double>({num_coils, num_rho, num_phi, 3});
    
    #pragma omp parallel for schedule(static)
    for(int j = 0; j < num_coils; j++) {
        auto cdj = cos(deltas(j));
        auto caj = cos(alphas(j));
        auto sdj = sin(deltas(j));
        auto saj = sin(alphas(j));
        auto xj = points(j, 0);
        auto yj = points(j, 1);
        auto zj = points(j, 2);
        // uses uniform quadrature since good Gaussian
        // quadrature on the unit disk is somewhat more involved
        for (int k = 0; k < num_rho; ++k) {
            auto xx = rho(k);
            for (int kk = 0; kk < num_phi; ++kk) {
                auto yy = phi(kk);
                auto x0 = xx * cos(yy);
                auto y0 = xx * sin(yy);
                auto x = x0 * cdj + y0 * saj * sdj + xj;
                auto y = y0 * caj + yj;
                auto z = -x0 * sdj + y0 * saj * cdj + zj;
                XYZs(j, k, kk, 0) = x;
                XYZs(j, k, kk, 1) = y;
                XYZs(j, k, kk, 2) = z;
            }
        }
    }
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
    int num_coils = B.shape(0);  // shape should be (num_coils, N, Nphi, 3)
    int N = rho.shape(0);  // shape should be (N)
    int Nphi = B.shape(2);
    Array Psi = xt::zeros<double>({num_coils}); 
    
    double* B_ptr = &(B(0, 0, 0, 0));
    
    #pragma omp parallel for schedule(static)
    for(int j = 0; j < num_coils; j++) {
        auto nx = normal(j, 0);
        auto ny = normal(j, 1);
        auto nz = normal(j, 2);
        double integral = 0.0;
        for(int k = 0; k < N; k++) {
            auto xx = rho(k);
            for(int kk = 0; kk < Nphi; kk++) {
                auto Bn = B(j, k, kk, 0) * nx + B(j, k, kk, 1) * ny + B(j, k, kk, 2) * nz; 
//                 auto Bn = B_ptr[192 * j + 24 * k + 3 * kk] * nx + \
//                           B_ptr[192 * j + 24 * k + 3 * kk + 1] * ny + \
//                           B_ptr[192 * j + 24 * k + 3 * kk + 2] * nz;
                integral += Bn * xx;
            }
        }
        Psi(j) = integral;
    }
    return Psi;
}

Array A_matrix(Array& points, Array& plasma_points, Array& alphas, Array& deltas, Array& plasma_normal, double R)
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
    Array A = xt::zeros<double>({num_plasma_points, num_coils});
    double R2 = R * R;
    double fac = 2.0e-7;
    using namespace boost::math;
    
    double* alphas_ptr = &(alphas(0));
    double* deltas_ptr = &(deltas(0));
    double* points_ptr = &(points(0, 0));
    double* plasma_points_ptr = &(plasma_points(0, 0));
    double* plasma_normal_ptr = &(plasma_normal(0, 0));
    
    #pragma omp parallel for schedule(static)
    for(int j = 0; j < num_coils; j++) {
        auto cdj = cos(deltas_ptr[j]);
        auto caj = cos(alphas_ptr[j]);
        auto sdj = sin(deltas_ptr[j]);
        auto saj = sin(alphas_ptr[j]);
        auto xj = points_ptr[3 * j];
        auto yj = points_ptr[3 * j + 1];
        auto zj = points_ptr[3 * j + 2];
        for (int i = 0; i < num_plasma_points; ++i) {
            auto xp = plasma_points_ptr[3 * i];
            auto yp = plasma_points_ptr[3 * i + 1];
            auto zp = plasma_points_ptr[3 * i + 2];
            auto nx = plasma_normal_ptr[3 * i];
            auto ny = plasma_normal_ptr[3 * i + 1];
            auto nz = plasma_normal_ptr[3 * i + 2];
            auto x0 = (xp - xj);
            auto y0 = (yp - yj);
            auto z0 = (zp - zj);
            auto nxx = cdj;
            auto nxy = sdj * saj;
            auto nxz = sdj * caj;
            auto nyx = 0.0;
            auto nyy = caj;
            auto nyz = -saj;
            auto nzx = -sdj;
            auto nzy = cdj * saj;
            auto nzz = cdj * caj;
            auto x = x0 * nxx + y0 * nyx + z0 * nzx;
            auto y = x0 * nxy + y0 * nyy + z0 * nzy;
            auto z = x0 * nxz + y0 * nyz + z0 * nzz;
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
            A(i, j) = Bx_rot * nx + By_rot * ny + Bz_rot * nz;
        }
    }
    return A * fac;
}

Array B_PSC(Array& points, Array& plasma_points, Array& alphas, Array& deltas, Array& psc_currents, double R)
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
    if(psc_currents.layout() != xt::layout_type::row_major)
          throw std::runtime_error("psc_currents needs to be in row-major storage order");
          
    // points shape should be (num_coils, 3)
    // plasma_normal shape should be (num_plasma_points, 3)
    // plasma_points should be (num_plasma_points, 3)
    int num_coils = alphas.shape(0);  // shape should be (num_coils)
    int num_plasma_points = plasma_points.shape(0);
    
    // this variable is the A matrix in the least-squares term so A * I = Bn
    Array B_PSC = xt::zeros<double>({num_plasma_points, 3});
    double R2 = R * R;
    double fac = 2.0e-7;
    using namespace boost::math;
    
    double* alphas_ptr = &(alphas(0));
    double* deltas_ptr = &(deltas(0));
    double* I_ptr = &(psc_currents(0));
    double* points_ptr = &(points(0, 0));
    double* plasma_points_ptr = &(plasma_points(0, 0));
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_plasma_points; ++i) {
        auto xp = plasma_points_ptr[3 * i];
        auto yp = plasma_points_ptr[3 * i + 1];
        auto zp = plasma_points_ptr[3 * i + 2];
        auto Bx_psc = 0.0;
        auto By_psc = 0.0;
        auto Bz_psc = 0.0;
        for(int j = 0; j < num_coils; j++) {
            auto current = I_ptr[j];
            auto cdj = cos(deltas_ptr[j]);
            auto caj = cos(alphas_ptr[j]);
            auto sdj = sin(deltas_ptr[j]);
            auto saj = sin(alphas_ptr[j]);
            auto x0 = (xp - points_ptr[3 * j]);
            auto y0 = (yp - points_ptr[3 * j + 1]);
            auto z0 = (zp - points_ptr[3 * j + 2]);
            auto nxx = cdj;
            auto nxy = sdj * saj;
            auto nxz = sdj * caj;
            auto nyx = 0.0;
            auto nyy = caj;
            auto nyz = -saj;
            auto nzx = -sdj;
            auto nzy = cdj * saj;
            auto nzz = cdj * caj;
            // multiply by R^T
            auto x = x0 * nxx + y0 * nyx + z0 * nzx;
            auto y = x0 * nxy + y0 * nyy + z0 * nzy;
            auto z = x0 * nxz + y0 * nyz + z0 * nzz;
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
            auto Bx = current * x * z * Eplus / (rho2 * beta_gamma2);
            auto By = current * y * z * Eplus / (rho2 * beta_gamma2);
            auto Bz = current * Eminus / beta_gamma2;
            // Need to rotate the vector
            Bx_psc += Bx * nxx + By * nxy + Bz * nxz;
            By_psc += Bx * nyx + By * nyy + Bz * nyz;
            Bz_psc += Bx * nzx + By * nzy + Bz * nzz;
        }
        B_PSC(i, 0) = Bx_psc;
        B_PSC(i, 1) = By_psc;
        B_PSC(i, 2) = Bz_psc;
    }
    return B_PSC * fac;
}


Array A_matrix_direct(Array& points, Array& plasma_points, Array& alphas, Array& deltas, Array& plasma_normal, Array& phi, double R)
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
   
    // points shape should be (num_coils, 3)
    // plasma_normal shape should be (num_plasma_points, 3)
    // plasma_points should be (num_plasma_points, 3)
    int num_coils = alphas.shape(0);  // shape should be (num_coils)
    int num_plasma_points = plasma_points.shape(0);
    int num_phi = phi.shape(0);  // shape should be (num_phi)
    
    // this variable is the A matrix in the least-squares term so A * I = Bn
    Array A = xt::zeros<double>({num_plasma_points, num_coils});
    double fac = 1.0e-7;
    using namespace boost::math;
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_plasma_points; ++i) {
        auto xp = plasma_points(i, 0);
        auto yp = plasma_points(i, 1);
        auto zp = plasma_points(i, 2);
        auto nx = plasma_normal(i, 0);
        auto ny = plasma_normal(i, 1);
        auto nz = plasma_normal(i, 2);
        for(int j = 0; j < num_coils; j++) {
            auto cdj = cos(deltas(j));
            auto caj = cos(alphas(j));
            auto sdj = sin(deltas(j));
            auto saj = sin(alphas(j));
            auto xk = points(j, 0);
            auto yk = points(j, 1);
            auto zk = points(j, 2);
            auto x0 = xp - xk;
            auto y0 = yp - yk;
            auto z0 = zp - zk;
            auto Rxx = cdj;
            auto Rxy = sdj * saj;
            auto Rxz = sdj * caj;
            auto Ryx = 0.0;
            auto Ryy = caj;
            auto Ryz = -saj;
            auto Rzx = -sdj;
            auto Rzy = cdj * saj;
            auto Rzz = cdj * caj;
            auto Bx = 0.0;
            auto By = 0.0;
            auto Bz = 0.0;
            for (int k = 0; k < num_phi; ++k) {
                auto dlx = -R * sin(phi(k));
                auto dly = R * cos(phi(k));
                auto dlz = 0.0;
                // multiply by R^T
                auto RTxdiff = (Rxx * x0 + Ryx * y0 + Rzx * z0) - R * cos(phi(k));
                auto RTydiff = (Rxy * x0 + Ryy * y0 + Rzy * z0) - R * sin(phi(k));
                auto RTzdiff = (Rxz * x0 + Ryz * y0 + Rzz * z0);
//                 auto RTxdiff = (cdj * xp - sdj * zp - xk);
//                 auto RTydiff = (sdj * saj * xp + caj * yp + cdj * saj * zp - yk);
//                 auto RTzdiff = (sdj * caj * xp - saj * yp + cdj * caj * zp - zk);
                auto dl_cross_RTdiff_x = dly * RTzdiff - dlz * RTydiff;
                auto dl_cross_RTdiff_y = dlz * RTxdiff - dlx * RTzdiff;
                auto dl_cross_RTdiff_z = dlx * RTydiff - dly * RTxdiff;
                auto denom = sqrt(RTxdiff * RTxdiff + RTydiff * RTydiff + RTzdiff * RTzdiff);
                auto denom3 = denom * denom * denom;
                Bx += dl_cross_RTdiff_x / denom3;
                By += dl_cross_RTdiff_y / denom3;
                Bz += dl_cross_RTdiff_z / denom3;
            }
            // rotate by R
            A(i, j) = (Rxx * Bx + Rxy * By + Rxz * Bz) * nx + (Ryx * Bx + Ryy * By + Ryz * Bz) * ny + (Rzx * Bx + Rzy * By + Rzz * Bz) * nz;
        }
    }
    return A * fac;
}



Array dA_dkappa(Array& points, Array& plasma_points, Array& alphas, Array& deltas, Array& plasma_normal, Array& phi, double R)
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
    int num_phi = phi.shape(0);  // shape should be (num_phi)
    
    // this variable is the A matrix in the least-squares term so A * I = Bn
    Array dA = xt::zeros<double>({num_coils * 2, num_plasma_points});
    double fac = 1.0e-7;
    using namespace boost::math;
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_plasma_points; ++i) {
        auto xp = plasma_points(i, 0);
        auto yp = plasma_points(i, 1);
        auto zp = plasma_points(i, 2);
        auto nx = plasma_normal(i, 0);
        auto ny = plasma_normal(i, 1);
        auto nz = plasma_normal(i, 2);
        for(int j = 0; j < num_coils; j++) {
            auto cdj = cos(deltas(j));
            auto caj = cos(alphas(j));
            auto sdj = sin(deltas(j));
            auto saj = sin(alphas(j));
            auto xk = points(j, 0);
            auto yk = points(j, 1);
            auto zk = points(j, 2);
            auto x0 = xp - xk;
            auto y0 = yp - yk;
            auto z0 = zp - zk;
            auto Rxx = cdj;
            auto Rxy = sdj * saj;
            auto Rxz = sdj * caj;
            auto Ryx = 0.0;
            auto Ryy = caj;
            auto Ryz = -saj;
            auto Rzx = -sdj;
            auto Rzy = cdj * saj;
            auto Rzz = cdj * caj;
            auto dRxx_dalpha = 0.0;
            auto dRxy_dalpha = sdj * caj;
            auto dRxz_dalpha = -sdj * saj;
            auto dRyx_dalpha = 0.0;
            auto dRyy_dalpha = -saj;
            auto dRyz_dalpha = -caj;
            auto dRzx_dalpha = 0.0;
            auto dRzy_dalpha = cdj * caj;
            auto dRzz_dalpha = -cdj * saj;
            auto dRxx_ddelta = -sdj;
            auto dRxy_ddelta = cdj * saj;
            auto dRxz_ddelta = cdj * caj;
            auto dRyx_ddelta = 0.0;
            auto dRyy_ddelta = 0.0;
            auto dRyz_ddelta = 0.0;
            auto dRzx_ddelta = -cdj;
            auto dRzy_ddelta = -sdj * saj;
            auto dRzz_ddelta = -sdj * caj;
            auto Bx1 = 0.0;
            auto By1 = 0.0;
            auto Bz1 = 0.0;
            auto Bx2 = 0.0;
            auto By2 = 0.0;
            auto Bz2 = 0.0;
            auto Bx3 = 0.0;
            auto By3 = 0.0;
            auto Bz3 = 0.0;
            auto Bx4 = 0.0;
            auto By4 = 0.0;
            auto Bz4 = 0.0;
            auto Bx5 = 0.0;
            auto By5 = 0.0;
            auto Bz5 = 0.0;
            for (int k = 0; k < num_phi; ++k) {
                auto dlx = -R * sin(phi(k));
                auto dly = R * cos(phi(k));
                auto dlz = 0.0;
                // multiply by R^T and then subtract off coil coordinate
                auto RTxdiff = (Rxx * x0 + Ryx * y0 + Rzx * z0) - R * cos(phi(k));
                auto RTydiff = (Rxy * x0 + Ryy * y0 + Rzy * z0) - R * sin(phi(k));
                auto RTzdiff = (Rxz * x0 + Ryz * y0 + Rzz * z0);
                auto dl_cross_RTdiff_x = dly * RTzdiff - dlz * RTydiff;
                auto dl_cross_RTdiff_y = dlz * RTxdiff - dlx * RTzdiff;
                auto dl_cross_RTdiff_z = dlx * RTydiff - dly * RTxdiff;
                auto denom = sqrt(RTxdiff * RTxdiff + RTydiff * RTydiff + RTzdiff * RTzdiff);
                auto denom3 = denom * denom * denom;
                auto denom5 = denom3 * denom * denom;
                // First derivative contribution of three
                Bx1 += dl_cross_RTdiff_x / denom3;
                By1 += dl_cross_RTdiff_y / denom3;
                Bz1 += dl_cross_RTdiff_z / denom3;
                // second derivative contribution (should be dRT/dalpha)
                auto dR_dalphaT_x = dRxx_dalpha * x0 + dRyx_dalpha * y0 + dRzx_dalpha * z0;
                auto dR_dalphaT_y = dRxy_dalpha * x0 + dRyy_dalpha * y0 + dRzy_dalpha * z0;
                auto dR_dalphaT_z = dRxz_dalpha * x0 + dRyz_dalpha * y0 + dRzz_dalpha * z0;
                auto dR_ddeltaT_x = dRxx_ddelta * x0 + dRyx_ddelta * y0 + dRzx_ddelta * z0;
                auto dR_ddeltaT_y = dRxy_ddelta * x0 + dRyy_ddelta * y0 + dRzy_ddelta * z0;
                auto dR_ddeltaT_z = dRxz_ddelta * x0 + dRyz_ddelta * y0 + dRzz_ddelta * z0;
                auto dl_cross_dR_dalphaT_x = dly * dR_dalphaT_z - dlz * dR_dalphaT_y;
                auto dl_cross_dR_dalphaT_y = dlz * dR_dalphaT_x - dlx * dR_dalphaT_z;
                auto dl_cross_dR_dalphaT_z = dlx * dR_dalphaT_y - dly * dR_dalphaT_x;
                auto dl_cross_dR_ddeltaT_x = dly * dR_ddeltaT_z - dlz * dR_ddeltaT_y;
                auto dl_cross_dR_ddeltaT_y = dlz * dR_ddeltaT_x - dlx * dR_ddeltaT_z;
                auto dl_cross_dR_ddeltaT_z = dlx * dR_ddeltaT_y - dly * dR_ddeltaT_x;
                Bx2 += dl_cross_dR_dalphaT_x / denom3;
                By2 += dl_cross_dR_dalphaT_y / denom3;
                Bz2 += dl_cross_dR_dalphaT_z / denom3;
                Bx4 += dl_cross_dR_ddeltaT_x / denom3;
                By4 += dl_cross_dR_ddeltaT_y / denom3;
                Bz4 += dl_cross_dR_ddeltaT_z / denom3;
                // third derivative contribution
                auto RTxdiff_dot_dR_dalpha = RTxdiff * dR_dalphaT_x + RTydiff * dR_dalphaT_y + RTzdiff * dR_dalphaT_z;
                auto RTxdiff_dot_dR_ddelta = RTxdiff * dR_ddeltaT_x + RTydiff * dR_ddeltaT_y + RTzdiff * dR_ddeltaT_z;
                Bx3 += - 3.0 * dl_cross_RTdiff_x * RTxdiff_dot_dR_dalpha / denom5;
                By3 += - 3.0 * dl_cross_RTdiff_y * RTxdiff_dot_dR_dalpha / denom5;
                Bz3 += - 3.0 * dl_cross_RTdiff_z * RTxdiff_dot_dR_dalpha / denom5;
                Bx5 += - 3.0 * dl_cross_RTdiff_x * RTxdiff_dot_dR_ddelta / denom5;
                By5 += - 3.0 * dl_cross_RTdiff_y * RTxdiff_dot_dR_ddelta / denom5;
                Bz5 += - 3.0 * dl_cross_RTdiff_z * RTxdiff_dot_dR_ddelta / denom5;
            }
            // rotate first contribution by dR/dalpha
            dA(j, i) += (dRxx_dalpha * Bx1 + dRxy_dalpha * By1 + dRxz_dalpha * Bz1) * nx;
            dA(j, i) += (dRyx_dalpha * Bx1 + dRyy_dalpha * By1 + dRyz_dalpha * Bz1) * ny;
            dA(j, i) += (dRzx_dalpha * Bx1 + dRzy_dalpha * By1 + dRzz_dalpha * Bz1) * nz;
            // rotate second and third contribution by R
            dA(j, i) += (Rxx * (Bx2 + Bx3) + Rxy * (By2 + By3) + Rxz * (Bz2 + Bz3)) * nx;
            dA(j, i) += (Ryx * (Bx2 + Bx3) + Ryy * (By2 + By3) + Ryz * (Bz2 + Bz3)) * ny;
            dA(j, i) += (Rzx * (Bx2 + Bx3) + Rzy * (By2 + By3) + Rzz * (Bz2 + Bz3)) * nz;
            // repeat for delta derivative
            dA(j + num_coils, i) += (dRxx_ddelta * Bx1 + dRxy_ddelta * By1 + dRxz_ddelta * Bz1) * nx;
            dA(j + num_coils, i) += (dRyx_ddelta * Bx1 + dRyy_ddelta * By1 + dRyz_ddelta * Bz1) * ny;
            dA(j + num_coils, i) += (dRzx_ddelta * Bx1 + dRzy_ddelta * By1 + dRzz_ddelta * Bz1) * nz;
            dA(j + num_coils, i) += (Rxx * (Bx4 + Bx5) + Rxy * (By4 + By5) + Rxz * (Bz4 + Bz5)) * nx;
            dA(j + num_coils, i) += (Ryx * (Bx4 + Bx5) + Ryy * (By4 + By5) + Ryz * (Bz4 + Bz5)) * ny;
            dA(j + num_coils, i) += (Rzx * (Bx4 + Bx5) + Rzy * (By4 + By5) + Rzz * (Bz4 + Bz5)) * nz;
        }
    }
    return dA * fac;
}


Array psi_check(Array& I_TF, Array& dl_TF, Array& gamma_TF, Array& PSC_points, Array& alphas, Array& deltas, Array& coil_normals, Array& rho, Array& phi, double R)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(gamma_TF.layout() != xt::layout_type::row_major)
          throw std::runtime_error("gamma_TF needs to be in row-major storage order");
    if(alphas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("alphas needs to be in row-major storage order");
    if(deltas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("deltas needs to be in row-major storage order");
    if(PSC_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("PSC_points needs to be in row-major storage order");
    if(coil_normals.layout() != xt::layout_type::row_major)
          throw std::runtime_error("coil_normals needs to be in row-major storage order");
          
    // points shape should be (num_coils, 3)
    // plasma_normal shape should be (num_plasma_points, 3)
    // plasma_points should be (num_plasma_points, 3)
    int num_TF_coils = I_TF.shape(0);  // shape should be (num_coils)
    int num_PSC_coils = coil_normals.shape(0);  // shape should be (num_coils)
    int num_evaluation_points = PSC_points.shape(0);
    int num_phi_TF = gamma_TF.shape(1);
    int num_integration_points = rho.shape(0);
    double* points_ptr = &(PSC_points(0, 0));
    double* normals_ptr = &(coil_normals(0, 0));
    double* alphas_ptr = &(alphas(0));
    double* deltas_ptr = &(deltas(0));
    // Shared pointers over indices other than kk can cause memory issues
    double* rho_ptr = &(rho(0));
    double* phi_ptr = &(phi(0));
    double* I_ptr = &(I_TF(0));
    double* gamma_ptr = &(gamma_TF(0, 0, 0));
    double* dl_ptr = &(dl_TF(0, 0, 0));
        
    // this variable is the A matrix in the least-squares term so A * I = Bn
    Array psi = xt::zeros<double>({num_PSC_coils});
    double fac = 1.0e-7;
    using namespace boost::math;
    
    // loop over all the PSC coils
    #pragma omp parallel for schedule(static)
    for (int kk = 0; kk < num_PSC_coils; ++kk) {
        auto xkk = points_ptr[3 * kk];
        auto ykk = points_ptr[3 * kk + 1];
        auto zkk = points_ptr[3 * kk + 2];
        auto cdj = cos(deltas_ptr[kk]);
        auto caj = cos(alphas_ptr[kk]);
        auto sdj = sin(deltas_ptr[kk]);
        auto saj = sin(alphas_ptr[kk]);
        // same normal for all these evaluation points so need an extra loop over all the PSCs
        auto nx = normals_ptr[3 * kk];
        auto ny = normals_ptr[3 * kk + 1];
        auto nz = normals_ptr[3 * kk + 2];
        auto Rxx = cdj;
        auto Rxy = sdj * saj;
        auto Ryy = caj;
        auto Rzx = -sdj;
        auto Rzy = cdj * saj;
        auto Bx = 0.0;
        auto By = 0.0;
        auto Bz = 0.0;
        // Do the integral over the PSC cross section
        for (int i = 0; i < num_integration_points; ++i) {
            // evaluation points here should be the points on a PSC coil cross section
            auto rho_i = rho_ptr[i];  // needed for integrating over the disk
            auto phi_i = phi_ptr[i];
            auto x0 = rho_i * cos(phi_i);
            auto y0 = rho_i * sin(phi_i);
            // z0 = 0 here
            auto xi = (Rxx * x0 + Rxy * y0) + xkk;
            auto yi = (Ryy * y0) + ykk;
            auto zi = (Rzx * x0 + Rzy * y0) + zkk;
            // loop here is over all the TF coils
            for(int j = 0; j < num_TF_coils; j++) {
                auto I_j = I_ptr[j];
                auto int_fac = rho_i * I_j;
                auto Bx_temp = 0.0;
                auto By_temp = 0.0;
                auto Bz_temp = 0.0;
                // Do Biot Savart over each TF coil - can probably downsample
                for (int k = 0; k < num_phi_TF; ++k) {
                    auto xk = gamma_ptr[(j * num_phi_TF + k) * 3];
                    auto yk = gamma_ptr[(j * num_phi_TF + k) * 3 + 1];
                    auto zk = gamma_ptr[(j * num_phi_TF + k) * 3 + 2];
                    auto dlx = dl_ptr[(j * num_phi_TF + k) * 3];
                    auto dly = dl_ptr[(j * num_phi_TF + k) * 3 + 1];
                    auto dlz = dl_ptr[(j * num_phi_TF + k) * 3 + 2];
                    // multiply by R (not R^T!) and then subtract off coil coordinate
                    auto RTxdiff = xi - xk; 
                    auto RTydiff = yi - yk;
                    auto RTzdiff = zi - zk;
                    auto dl_cross_RTdiff_x = dly * RTzdiff - dlz * RTydiff;
                    auto dl_cross_RTdiff_y = dlz * RTxdiff - dlx * RTzdiff;
                    auto dl_cross_RTdiff_z = dlx * RTydiff - dly * RTxdiff;
                    auto denom = sqrt(RTxdiff * RTxdiff + RTydiff * RTydiff + RTzdiff * RTzdiff);
                    auto denom3 = denom * denom * denom;
                    Bx_temp += dl_cross_RTdiff_x / denom3;
                    By_temp += dl_cross_RTdiff_y / denom3;
                    Bz_temp += dl_cross_RTdiff_z / denom3;
                }
                Bx += int_fac * Bx_temp;
                By += int_fac * By_temp;
                Bz += int_fac * Bz_temp;
            }
        }
        psi(kk) = Bx * nx + By * ny + Bz * nz;
    }
    return psi * fac;
}


Array B_TF(Array& I_TF, Array& dl_TF, Array& gamma_TF, Array& PSC_points)
{         
    // points shape should be (num_coils, 3)
    // plasma_normal shape should be (num_plasma_points, 3)
    // plasma_points should be (num_plasma_points, 3)
    int num_TF_coils = I_TF.shape(0);  // shape should be (num_coils)
    int num_evaluation_points = PSC_points.shape(0);
    int num_phi_TF = gamma_TF.shape(1);
    Array B_TF = xt::zeros<double>({num_evaluation_points, 3});
    double fac = 1.0e-7;
    using namespace boost::math;
    
    // loop over all the PSC coils
    #pragma omp parallel for schedule(static)
    for (int kk = 0; kk < num_evaluation_points; ++kk) {
        auto xkk = PSC_points(kk, 0);
        auto ykk = PSC_points(kk, 1);
        auto zkk = PSC_points(kk, 2);
        auto Bx = 0.0;
        auto By = 0.0;
        auto Bz = 0.0;
        // loop here is over all the TF coils
        for(int j = 0; j < num_TF_coils; j++) {
            auto I_j = I_TF(j);
            // Do Biot Savart over each TF coil
            for (int k = 0; k < num_phi_TF; ++k) {
                auto xk = gamma_TF(j, k, 0);
                auto yk = gamma_TF(j, k, 1);
                auto zk = gamma_TF(j, k, 2);
                auto dlx = dl_TF(j, k, 0);
                auto dly = dl_TF(j, k, 1);
                auto dlz = dl_TF(j, k, 2);
                // multiply by R (not R^T!) and then subtract off coil coordinate
                auto RTxdiff = xkk - xk;
                auto RTydiff = ykk - yk;
                auto RTzdiff = zkk - zk;
                auto dl_cross_RTdiff_x = dly * RTzdiff - dlz * RTydiff;
                auto dl_cross_RTdiff_y = dlz * RTxdiff - dlx * RTzdiff;
                auto dl_cross_RTdiff_z = dlx * RTydiff - dly * RTxdiff;
                auto denom = sqrt(RTxdiff * RTxdiff + RTydiff * RTydiff + RTzdiff * RTzdiff);
                auto denom3 = denom * denom * denom;
                Bx += I_j * dl_cross_RTdiff_x / denom3;
                By += I_j * dl_cross_RTdiff_y / denom3;
                Bz += I_j * dl_cross_RTdiff_z / denom3;
            }
        }
        B_TF(kk, 0) = Bx;
        B_TF(kk, 1) = By;
        B_TF(kk, 2) = Bz;
    }
    return B_TF * fac;
}
