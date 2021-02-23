#include "biot_savart.h"

template<class T>
void biot_savart_B_only_vjp_impl(vector_type& pointsx, vector_type& pointsy, vector_type& pointsz, T& gamma, T& dgamma_by_dphi, T& v, T& res_gamma, T& res_dgamma_by_dphi, T& vgrad, T& res_grad_gamma, T& res_grad_dgamma_by_dphi) {
    int num_points         = pointsx.size();
    int num_quad_points    = gamma.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    for(int i = 0; i < num_points-num_points%simd_size; i += simd_size) {
        Vec3dSimd point_i = Vec3dSimd(&(pointsx[i]), &(pointsy[i]), &(pointsz[i]));
        auto v_i   = Vec3dSimd();
        auto vgrad_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, XSIMD_DEFAULT_ALIGNMENT>>{
                Vec3dSimd(), Vec3dSimd(), Vec3dSimd()
            };
        for(int k=0; k<simd_size; k++){
            for (int d = 0; d < 3; ++d) {
                v_i[d][k] = v(i+k, d);
                for (int dd = 0; dd < 3; ++dd) {
                    vgrad_i[dd][d][k] = vgrad(i+k, dd, d);
                }
            }
        }
        for (int j = 0; j < num_quad_points; ++j) {
            auto gamma_j = Vec3d{ gamma(j, 0), gamma(j, 1), gamma(j, 2)};
            auto dgamma_j_by_dphi = Vec3d{ dgamma_by_dphi(j, 0), dgamma_by_dphi(j, 1), dgamma_by_dphi(j, 2)};
            auto diff = point_i - gamma_j;
            auto norm_diff_2 = normsq(diff);
            auto norm_diff = sqrt(norm_diff_2);
            auto norm_diff_3_inv = 1/(norm_diff_2*norm_diff);
            auto norm_diff_5_inv = norm_diff_3_inv/(norm_diff_2);
            auto norm_diff_5_inv_times_3 = 3.*norm_diff_5_inv;

            auto res_dgamma_by_dphi_add = cross(diff, v_i) * norm_diff_3_inv;
            res_dgamma_by_dphi(j, 0) += xsimd::hadd(res_dgamma_by_dphi_add.x);
            res_dgamma_by_dphi(j, 1) += xsimd::hadd(res_dgamma_by_dphi_add.y);
            res_dgamma_by_dphi(j, 2) += xsimd::hadd(res_dgamma_by_dphi_add.z);

            auto cross_dgamma_j_by_dphi_diff = cross(dgamma_j_by_dphi, diff);
            auto res_gamma_add = cross(dgamma_j_by_dphi, v_i) * norm_diff_3_inv;
            res_gamma_add += diff * inner(cross_dgamma_j_by_dphi_diff, v_i) * (norm_diff_5_inv_times_3);
            res_gamma(j, 0) += xsimd::hadd(res_gamma_add.x);
            res_gamma(j, 1) += xsimd::hadd(res_gamma_add.y);
            res_gamma(j, 2) += xsimd::hadd(res_gamma_add.z);

            auto norm_diff_7_inv = norm_diff_5_inv/(norm_diff_2);
            auto res_grad_dgamma_by_dphi_add = Vec3dSimd();
            auto res_grad_gamma_add = Vec3dSimd();

            for(int k=0; k<3; k++){
                auto eksimd = Vec3dSimd();
                eksimd[k] += 1.;
                Vec3d ek = Vec3d::Zero();
                ek[k] = 1.;
                res_grad_dgamma_by_dphi_add += cross(k, vgrad_i[k]) * norm_diff_3_inv;
                res_grad_dgamma_by_dphi_add -= cross(diff, vgrad_i[k]) * (diff[k] * norm_diff_5_inv_times_3);

                res_grad_gamma_add += diff * (inner(cross(dgamma_j_by_dphi, ek), vgrad_i[k]) * norm_diff_5_inv_times_3);
                res_grad_gamma_add += eksimd * (inner(cross_dgamma_j_by_dphi_diff, vgrad_i[k]) * norm_diff_5_inv_times_3);
                res_grad_gamma_add += cross(vgrad_i[k], dgamma_j_by_dphi) * (norm_diff_5_inv_times_3 * diff[k]);
                res_grad_gamma_add -= diff * (15. * diff[k] * inner(cross_dgamma_j_by_dphi_diff, vgrad_i[k]) * norm_diff_7_inv);
            }
            res_grad_dgamma_by_dphi(j, 0) += xsimd::hadd(res_grad_dgamma_by_dphi_add.x);
            res_grad_dgamma_by_dphi(j, 1) += xsimd::hadd(res_grad_dgamma_by_dphi_add.y);
            res_grad_dgamma_by_dphi(j, 2) += xsimd::hadd(res_grad_dgamma_by_dphi_add.z);
            res_grad_gamma(j, 0) += xsimd::hadd(res_grad_gamma_add.x);
            res_grad_gamma(j, 1) += xsimd::hadd(res_grad_gamma_add.y);
            res_grad_gamma(j, 2) += xsimd::hadd(res_grad_gamma_add.z);
        }
    }
    for (int i = num_points - num_points % simd_size; i < num_points; ++i) {
        auto point_i = Vec3d{pointsx[i], pointsy[i], pointsz[i]};
        Vec3d v_i   = Vec3d::Zero();
        auto vgrad_i = vector<Vec3d>{
            Vec3d::Zero(), Vec3d::Zero(), Vec3d::Zero()
            };
        for (int d = 0; d < 3; ++d) {
            v_i[d] = v(i, d);
            for (int dd = 0; dd < 3; ++dd) {
                vgrad_i[dd][d] = vgrad(i, dd, d);
            }
        }
        for (int j = 0; j < num_quad_points; ++j) {
            Vec3d gamma_j = Vec3d{ gamma(j, 0), gamma(j, 1), gamma(j, 2) };
            Vec3d dgamma_j_by_dphi = Vec3d{ dgamma_by_dphi(j, 0), dgamma_by_dphi(j, 1), dgamma_by_dphi(j, 2) };
            Vec3d diff = point_i - gamma_j;
            double norm_diff = norm(diff);
            double norm_diff_2 = norm_diff*norm_diff;
            double norm_diff_3_inv = 1/(norm_diff_2*norm_diff);
            double norm_diff_5_inv = norm_diff_3_inv/(norm_diff_2);
            double norm_diff_5_inv_times_3 = 3.*norm_diff_5_inv;

            Vec3d res_dgamma_by_dphi_add = cross(diff, v_i) * norm_diff_3_inv;
            res_dgamma_by_dphi(j, 0) += res_dgamma_by_dphi_add[0];
            res_dgamma_by_dphi(j, 1) += res_dgamma_by_dphi_add[1];
            res_dgamma_by_dphi(j, 2) += res_dgamma_by_dphi_add[2];

            Vec3d cross_dgamma_j_by_dphi_diff = cross(dgamma_j_by_dphi, diff);
            Vec3d res_gamma_add = cross(dgamma_j_by_dphi, v_i) * norm_diff_3_inv;
            res_gamma_add += diff * inner(cross_dgamma_j_by_dphi_diff, v_i) * (norm_diff_5_inv_times_3);
            res_gamma(j, 0) += res_gamma_add[0];
            res_gamma(j, 1) += res_gamma_add[1];
            res_gamma(j, 2) += res_gamma_add[2];

            double norm_diff_7_inv = norm_diff_5_inv/(norm_diff_2);
            Vec3d res_grad_dgamma_by_dphi_add = Vec3d::Zero();
            Vec3d res_grad_gamma_add = Vec3d::Zero();

            for(int k=0; k<3; k++){
                Vec3d ek = Vec3d::Zero();
                ek[k] = 1.;
                res_grad_dgamma_by_dphi_add += cross(ek, vgrad_i[k]) * norm_diff_3_inv;
                res_grad_dgamma_by_dphi_add -= cross(diff, vgrad_i[k]) * (diff[k] * norm_diff_5_inv_times_3);

                res_grad_gamma_add += diff * (inner(cross(dgamma_j_by_dphi, ek), vgrad_i[k]) * norm_diff_5_inv_times_3);
                res_grad_gamma_add += ek * (inner(cross_dgamma_j_by_dphi_diff, vgrad_i[k]) * norm_diff_5_inv_times_3);
                res_grad_gamma_add += cross(vgrad_i[k], dgamma_j_by_dphi) * (norm_diff_5_inv_times_3 * diff[k]);
                res_grad_gamma_add -= diff * (15. * diff[k] * inner(cross_dgamma_j_by_dphi_diff, vgrad_i[k]) * norm_diff_7_inv);
            }
            res_grad_dgamma_by_dphi(j, 0) += res_grad_dgamma_by_dphi_add[0];
            res_grad_dgamma_by_dphi(j, 1) += res_grad_dgamma_by_dphi_add[1];
            res_grad_dgamma_by_dphi(j, 2) += res_grad_dgamma_by_dphi_add[2];
            res_grad_gamma(j, 0) += res_grad_gamma_add[0];
            res_grad_gamma(j, 1) += res_grad_gamma_add[1];
            res_grad_gamma(j, 2) += res_grad_gamma_add[2];
        }
    }
}


template void biot_savart_B_only_vjp_impl<xt::xarray<double>>(vector_type&, vector_type&, vector_type&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&);

void biot_savart_by_dcoilcoeff_all_vjp(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<double>& currents, Array& v, vector<Array>& res_gamma, vector<Array>& res_dgamma_by_dphi, Array& vgrad, vector<Array>& res_grad_gamma, vector<Array>& res_grad_dgamma_by_dphi) {
    auto pointsx = vector_type(points.shape(0), 0);
    auto pointsy = vector_type(points.shape(0), 0);
    auto pointsz = vector_type(points.shape(0), 0);
    for (int i = 0; i < points.shape(0); ++i) {
        pointsx[i] = points(i, 0);
        pointsy[i] = points(i, 1);
        pointsz[i] = points(i, 2);
    }

    int num_coils  = gammas.size();

    #pragma omp parallel for
    for(int i=0; i<num_coils; i++) {
        biot_savart_B_only_vjp_impl<Array>(pointsx, pointsy, pointsz, gammas[i], dgamma_by_dphis[i],
                v, res_gamma[i], res_dgamma_by_dphi[i],
                vgrad, res_grad_gamma[i], res_grad_dgamma_by_dphi[i]);
        double fak = (currents[i] * 1e-7/gammas[i].shape(0));
        res_gamma[i] *= fak;
        res_dgamma_by_dphi[i] *= fak;
        res_grad_gamma[i] *= fak;
        res_grad_dgamma_by_dphi[i] *= fak;
    }
}

void biot_savart_by_dcoilcoeff_all_vjp_full(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<double>& currents, Array& v, Array& vgrad, vector<Array>& dgamma_by_dcoeffs, vector<Array>& d2gamma_by_dphidcoeffs, vector<Array>& res_B, vector<Array>& res_dB){
    auto pointsx = vector_type(points.shape(0), 0);
    auto pointsy = vector_type(points.shape(0), 0);
    auto pointsz = vector_type(points.shape(0), 0);
    for (int i = 0; i < points.shape(0); ++i) {
        pointsx[i] = points(i, 0);
        pointsy[i] = points(i, 1);
        pointsz[i] = points(i, 2);
    }

    int num_coils  = gammas.size();

    auto res_gamma = std::vector<Array>(num_coils, Array());
    auto res_dgamma_by_dphi = std::vector<Array>(num_coils, Array());
    auto res_grad_gamma = std::vector<Array>(num_coils, Array());
    auto res_grad_dgamma_by_dphi = std::vector<Array>(num_coils, Array());

    // Don't understand why, but in parallel this loop segfaults...
    for(int i=0; i<num_coils; i++) {
        int num_points = gammas[i].shape(0);
        res_gamma[i] = xt::zeros<double>({num_points, 3});
        res_dgamma_by_dphi[i] = xt::zeros<double>({num_points, 3});
        res_grad_gamma[i] = xt::zeros<double>({num_points, 3});
        res_grad_dgamma_by_dphi[i] = xt::zeros<double>({num_points, 3});
    }

    #pragma omp parallel for
    for(int i=0; i<num_coils; i++) {
        biot_savart_B_only_vjp_impl<Array>(pointsx, pointsy, pointsz, gammas[i], dgamma_by_dphis[i],
                v, res_gamma[i], res_dgamma_by_dphi[i],
                vgrad, res_grad_gamma[i], res_grad_dgamma_by_dphi[i]);
        int numcoeff = dgamma_by_dcoeffs[i].shape(2);
        for (int j = 0; j < dgamma_by_dcoeffs[i].shape(0); ++j) {
            for (int l = 0; l < 3; ++l) {
                auto t1 = res_gamma[i](j, l);
                auto t2 = res_dgamma_by_dphi[i](j, l);
                auto t3 = res_grad_gamma[i](j, l);
                auto t4 = res_grad_dgamma_by_dphi[i](j, l);
                for (int k = 0; k < numcoeff; ++k) {
                    res_B[i](k) += dgamma_by_dcoeffs[i](j, l, k) * t1 + d2gamma_by_dphidcoeffs[i](j, l, k) * t2;
                    res_dB[i](k) += dgamma_by_dcoeffs[i](j, l, k) * t3 + d2gamma_by_dphidcoeffs[i](j, l, k) * t4;
                }
            }
        }
        double fak = (currents[i] * 1e-7/gammas[i].shape(0));
        res_B[i] *= fak;
        res_dB[i] *= fak;
    }
}
