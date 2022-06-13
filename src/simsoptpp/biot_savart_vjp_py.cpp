#include "biot_savart_vjp_impl.h"
#include "biot_savart_vjp_py.h"

void biot_savart_vjp(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<double>& currents, Array& v, Array& vgrad, vector<Array>& dgamma_by_dcoeffs, vector<Array>& d2gamma_by_dphidcoeffs, vector<Array>& res_B, vector<Array>& res_dB){
    auto pointsx = AlignedPaddedVec(points.shape(0), 0);
    auto pointsy = AlignedPaddedVec(points.shape(0), 0);
    auto pointsz = AlignedPaddedVec(points.shape(0), 0);
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

    bool compute_dB = res_dB.size() > 0;
    // Don't understand why, but in parallel this loop segfaults...
    for(int i=0; i<num_coils; i++) {
        int num_points = gammas[i].shape(0);
        res_gamma[i] = xt::zeros<double>({num_points, 3});
        res_dgamma_by_dphi[i] = xt::zeros<double>({num_points, 3});
        if(compute_dB) {
            res_grad_gamma[i] = xt::zeros<double>({num_points, 3});
            res_grad_dgamma_by_dphi[i] = xt::zeros<double>({num_points, 3});
        }
    }
    Array dummy = Array();

    #pragma omp parallel for
    for(int i=0; i<num_coils; i++) {
        if(compute_dB)
            biot_savart_vjp_kernel<Array, 1>(pointsx, pointsy, pointsz, gammas[i], dgamma_by_dphis[i],
                    v, res_gamma[i], res_dgamma_by_dphi[i],
                    vgrad, res_grad_gamma[i], res_grad_dgamma_by_dphi[i]);
        else
            biot_savart_vjp_kernel<Array, 0>(pointsx, pointsy, pointsz, gammas[i], dgamma_by_dphis[i],
                    v, res_gamma[i], res_dgamma_by_dphi[i], dummy, dummy, dummy);
        int numcoeff = dgamma_by_dcoeffs[i].shape(2);
        for (int j = 0; j < dgamma_by_dcoeffs[i].shape(0); ++j) {
            for (int l = 0; l < 3; ++l) {
                auto t1 = res_gamma[i](j, l);
                auto t2 = res_dgamma_by_dphi[i](j, l);
                for (int k = 0; k < numcoeff; ++k)
                    res_B[i](k) += dgamma_by_dcoeffs[i](j, l, k) * t1 + d2gamma_by_dphidcoeffs[i](j, l, k) * t2;

                if(compute_dB) {
                    auto t3 = res_grad_gamma[i](j, l);
                    auto t4 = res_grad_dgamma_by_dphi[i](j, l);
                    for (int k = 0; k < numcoeff; ++k)
                        res_dB[i](k) += dgamma_by_dcoeffs[i](j, l, k) * t3 + d2gamma_by_dphidcoeffs[i](j, l, k) * t4;
                }
            }
        }
        double fak = (currents[i] * 1e-7/gammas[i].shape(0));
        res_B[i] *= fak;
        if(compute_dB)
            res_dB[i] *= fak;
    }
}

void biot_savart_vjp_graph(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<double>& currents, Array& v, vector<Array>& res_gamma, vector<Array>& res_dgamma_by_dphi, Array& vgrad, vector<Array>& res_grad_gamma, vector<Array>& res_grad_dgamma_by_dphi) {
    auto pointsx = AlignedPaddedVec(points.shape(0), 0);
    auto pointsy = AlignedPaddedVec(points.shape(0), 0);
    auto pointsz = AlignedPaddedVec(points.shape(0), 0);
    for (int i = 0; i < points.shape(0); ++i) {
        pointsx[i] = points(i, 0);
        pointsy[i] = points(i, 1);
        pointsz[i] = points(i, 2);
    }

    int num_coils  = gammas.size();
    bool compute_dB = res_grad_gamma.size() > 0;
    Array dummy = Array();

    #pragma omp parallel for
    for(int i=0; i<num_coils; i++) {
        if(compute_dB)
            biot_savart_vjp_kernel<Array, 1>(pointsx, pointsy, pointsz, gammas[i], dgamma_by_dphis[i],
                    v, res_gamma[i], res_dgamma_by_dphi[i],
                    vgrad, res_grad_gamma[i], res_grad_dgamma_by_dphi[i]);
        else
            biot_savart_vjp_kernel<Array, 0>(pointsx, pointsy, pointsz, gammas[i], dgamma_by_dphis[i],
                    v, res_gamma[i], res_dgamma_by_dphi[i],
                    dummy, dummy, dummy);

        double fak = (currents[i] * 1e-7/gammas[i].shape(0));
        res_gamma[i] *= fak;
        res_dgamma_by_dphi[i] *= fak;
        if(compute_dB) {
            res_grad_gamma[i] *= fak;
            res_grad_dgamma_by_dphi[i] *= fak;
        }
    }
}

void biot_savart_vector_potential_vjp_graph(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<double>& currents, Array& v, vector<Array>& res_gamma, vector<Array>& res_dgamma_by_dphi, Array& vgrad, vector<Array>& res_grad_gamma, vector<Array>& res_grad_dgamma_by_dphi) {
    auto pointsx = AlignedPaddedVec(points.shape(0), 0);
    auto pointsy = AlignedPaddedVec(points.shape(0), 0);
    auto pointsz = AlignedPaddedVec(points.shape(0), 0);
    for (int i = 0; i < points.shape(0); ++i) {
        pointsx[i] = points(i, 0);
        pointsy[i] = points(i, 1);
        pointsz[i] = points(i, 2);
    }

    int num_coils  = gammas.size();
    bool compute_dA = res_grad_gamma.size() > 0;
    Array dummy = Array();

    #pragma omp parallel for
    for(int i=0; i<num_coils; i++) {
        if(compute_dA)
            biot_savart_vector_potential_vjp_kernel<Array, 1>(pointsx, pointsy, pointsz, gammas[i], dgamma_by_dphis[i],
                    v, res_gamma[i], res_dgamma_by_dphi[i],
                    vgrad, res_grad_gamma[i], res_grad_dgamma_by_dphi[i]);
        else
            biot_savart_vector_potential_vjp_kernel<Array, 0>(pointsx, pointsy, pointsz, gammas[i], dgamma_by_dphis[i],
                    v, res_gamma[i], res_dgamma_by_dphi[i],
                    dummy, dummy, dummy);

        double fak = (currents[i] * 1e-7/gammas[i].shape(0));
        res_gamma[i] *= fak;
        res_dgamma_by_dphi[i] *= fak;
        if(compute_dA) {
            res_grad_gamma[i] *= fak;
            res_grad_dgamma_by_dphi[i] *= fak;
        }
    }
}
