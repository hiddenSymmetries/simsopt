#include "biot_savart_vjp_impl.h"
#include "biot_savart_vjp_py.h"

// biot_savart_vjp: main VJP used in BiotSavart.vjp()
// ---------------------------------------------------
void biot_savart_vjp(Array& points,
                     std::vector<Array>& gammas,
                     std::vector<Array>& dgamma_by_dphis,
                     std::vector<double>& currents,
                     Array& v,
                     Array& vgrad,
                     std::vector<Array>& dgamma_by_dcoeffs,
                     std::vector<Array>& d2gamma_by_dphidcoeffs,
                     std::vector<Array>& res_B,
                     std::vector<Array>& res_dB)
{
    auto pointsx = AlignedPaddedVec(points.shape(0), 0);
    auto pointsy = AlignedPaddedVec(points.shape(0), 0);
    auto pointsz = AlignedPaddedVec(points.shape(0), 0);
    for (int i = 0; i < static_cast<int>(points.shape(0)); ++i) {
        pointsx[i] = points(i, 0);
        pointsy[i] = points(i, 1);
        pointsz[i] = points(i, 2);
    }

    const int num_coils  = static_cast<int>(gammas.size());
    const bool compute_dB = res_dB.size() > 0;

    // Coil-space adjoints for B and ∇B (existing ones)
    auto res_gamma             = std::vector<Array>(num_coils, Array());
    auto res_dgamma_by_dphi    = std::vector<Array>(num_coils, Array());
    auto res_grad_gamma        = std::vector<Array>(num_coils, Array());
    auto res_grad_dgamma_by_dphi = std::vector<Array>(num_coils, Array());

    // NEW: coil-space adjoints for Hessian (∇∇B); not exposed to Python yet
    auto res_hess_gamma           = std::vector<Array>(num_coils, Array());
    auto res_hess_dgamma_by_dphi  = std::vector<Array>(num_coils, Array());

    // NEW: adjoint of Hessian wrt B; we don't use ∇∇B in Python yet, so set to zero
    Array vhess = xt::zeros<double>(
        std::vector<std::size_t>{
            static_cast<std::size_t>(points.shape(0)), 3u, 3u, 3u
        }
    );

    // Allocate all coil-space adjoints with proper shapes
    for (int i = 0; i < num_coils; ++i) {
        const int num_points = static_cast<int>(gammas[i].shape(0));

        res_gamma[i]          = xt::zeros<double>({num_points, 3});
        res_dgamma_by_dphi[i] = xt::zeros<double>({num_points, 3});

        // We still allocate grad/Hess adjoints even if we don't care about them,
        // to satisfy layout checks in the kernel.
        res_grad_gamma[i]        = xt::zeros<double>({num_points, 3});
        res_grad_dgamma_by_dphi[i] = xt::zeros<double>({num_points, 3});

        res_hess_gamma[i]          = xt::zeros<double>({num_points, 3});
        res_hess_dgamma_by_dphi[i] = xt::zeros<double>({num_points, 3});
    }

    Array dummy = Array();  // used only as vgrad in derivs=0 case

    // Don't understand why, but in parallel this loop segfaults...
    #pragma omp parallel for
    for (int i = 0; i < num_coils; ++i) {
        if (compute_dB) {
            // derivs = 1 → B and ∇B adjoints
            biot_savart_vjp_kernel<Array, 1>(
                pointsx, pointsy, pointsz,
                gammas[i],
                dgamma_by_dphis[i],
                v,
                res_gamma[i],
                res_dgamma_by_dphi[i],
                vgrad,
                res_grad_gamma[i],
                res_grad_dgamma_by_dphi[i],
                vhess,
                res_hess_gamma[i],
                res_hess_dgamma_by_dphi[i]
            );
        } else {
            // derivs = 0 → only B adjoint; pass dummy for vgrad
            biot_savart_vjp_kernel<Array, 0>(
                pointsx, pointsy, pointsz,
                gammas[i],
                dgamma_by_dphis[i],
                v,
                res_gamma[i],
                res_dgamma_by_dphi[i],
                dummy,                      // vgrad not used when derivs=0
                res_grad_gamma[i],          // still need row-major arrays
                res_grad_dgamma_by_dphi[i], // still need row-major arrays
                vhess,
                res_hess_gamma[i],
                res_hess_dgamma_by_dphi[i]
            );
        }

        const int numcoeff = static_cast<int>(dgamma_by_dcoeffs[i].shape(2));
        for (int j = 0; j < static_cast<int>(dgamma_by_dcoeffs[i].shape(0)); ++j) {
            for (int l = 0; l < 3; ++l) {
                const auto t1 = res_gamma[i](j, l);
                const auto t2 = res_dgamma_by_dphi[i](j, l);
                for (int k = 0; k < numcoeff; ++k) {
                    res_B[i](k) += dgamma_by_dcoeffs[i](j, l, k) * t1
                                   + d2gamma_by_dphidcoeffs[i](j, l, k) * t2;
                }

                if (compute_dB) {
                    const auto t3 = res_grad_gamma[i](j, l);
                    const auto t4 = res_grad_dgamma_by_dphi[i](j, l);
                    for (int k = 0; k < numcoeff; ++k) {
                        res_dB[i](k) += dgamma_by_dcoeffs[i](j, l, k) * t3
                                        + d2gamma_by_dphidcoeffs[i](j, l, k) * t4;
                    }
                }
            }
        }

        const double fak = (currents[i] * 1e-7 / gammas[i].shape(0));
        res_B[i] *= fak;
        if (compute_dB) {
            res_dB[i] *= fak;
        }
    }
}


// biot_savart_vjp_graph: returns coil-space adjoints (B and ∇B) directly
// ----------------------------------------------------------------------
void biot_savart_vjp_graph(Array& points,
                           std::vector<Array>& gammas,
                           std::vector<Array>& dgamma_by_dphis,
                           std::vector<double>& currents,
                           Array& v,
                           std::vector<Array>& res_gamma,
                           std::vector<Array>& res_dgamma_by_dphi,
                           Array& vgrad,
                           std::vector<Array>& res_grad_gamma,
                           std::vector<Array>& res_grad_dgamma_by_dphi)
{
    auto pointsx = AlignedPaddedVec(points.shape(0), 0);
    auto pointsy = AlignedPaddedVec(points.shape(0), 0);
    auto pointsz = AlignedPaddedVec(points.shape(0), 0);
    for (int i = 0; i < static_cast<int>(points.shape(0)); ++i) {
        pointsx[i] = points(i, 0);
        pointsy[i] = points(i, 1);
        pointsz[i] = points(i, 2);
    }

    const int num_coils = static_cast<int>(gammas.size());
    const bool compute_dB = (res_grad_gamma.size() > 0);

    // NEW local Hessian adjoints (not exposed to Python)
    auto res_hess_gamma          = std::vector<Array>(num_coils, Array());
    auto res_hess_dgamma_by_dphi = std::vector<Array>(num_coils, Array());
    Array vhess = xt::zeros<double>(
        std::vector<std::size_t>{
            static_cast<std::size_t>(points.shape(0)), 3u, 3u, 3u
        }
    );

    // Ensure Hessian adjoint buffers are allocated with correct shape
    for (int i = 0; i < num_coils; ++i) {
        const int num_points = static_cast<int>(gammas[i].shape(0));
        res_hess_gamma[i]          = xt::zeros<double>({num_points, 3});
        res_hess_dgamma_by_dphi[i] = xt::zeros<double>({num_points, 3});
    }

    Array dummy = Array();

    #pragma omp parallel for
    for (int i = 0; i < num_coils; ++i) {
        if (compute_dB) {
            biot_savart_vjp_kernel<Array, 1>(
                pointsx, pointsy, pointsz,
                gammas[i],
                dgamma_by_dphis[i],
                v,
                res_gamma[i],
                res_dgamma_by_dphi[i],
                vgrad,
                res_grad_gamma[i],
                res_grad_dgamma_by_dphi[i],
                vhess,
                res_hess_gamma[i],
                res_hess_dgamma_by_dphi[i]
            );
        } else {
            biot_savart_vjp_kernel<Array, 0>(
                pointsx, pointsy, pointsz,
                gammas[i],
                dgamma_by_dphis[i],
                v,
                res_gamma[i],
                res_dgamma_by_dphi[i],
                dummy,                      // vgrad unused for derivs=0
                res_grad_gamma[i],          // still real arrays for layout
                res_grad_dgamma_by_dphi[i],
                vhess,
                res_hess_gamma[i],
                res_hess_dgamma_by_dphi[i]
            );
        }

        const double fak = (currents[i] * 1e-7 / gammas[i].shape(0));
        res_gamma[i]          *= fak;
        res_dgamma_by_dphi[i] *= fak;
        if (compute_dB) {
            res_grad_gamma[i]        *= fak;
            res_grad_dgamma_by_dphi[i] *= fak;
        }
    }
}


// biot_savart_vector_potential_vjp_graph
// --------------------------------------
// NOTE: the underlying kernel is currently a stub that throws at runtime,
// but we still need to compile and link this wrapper.
void biot_savart_vector_potential_vjp_graph(Array& points,
                                            std::vector<Array>& gammas,
                                            std::vector<Array>& dgamma_by_dphis,
                                            std::vector<double>& currents,
                                            Array& v,
                                            std::vector<Array>& res_gamma,
                                            std::vector<Array>& res_dgamma_by_dphi,
                                            Array& vgrad,
                                            std::vector<Array>& res_grad_gamma,
                                            std::vector<Array>& res_grad_dgamma_by_dphi)
{
    auto pointsx = AlignedPaddedVec(points.shape(0), 0);
    auto pointsy = AlignedPaddedVec(points.shape(0), 0);
    auto pointsz = AlignedPaddedVec(points.shape(0), 0);
    for (int i = 0; i < static_cast<int>(points.shape(0)); ++i) {
        pointsx[i] = points(i, 0);
        pointsy[i] = points(i, 1);
        pointsz[i] = points(i, 2);
    }

    const int num_coils = static_cast<int>(gammas.size());
    const bool compute_dA = (res_grad_gamma.size() > 0);

    // NEW local Hessian adjoints
    auto res_hess_gamma          = std::vector<Array>(num_coils, Array());
    auto res_hess_dgamma_by_dphi = std::vector<Array>(num_coils, Array());
    Array vhess = xt::zeros<double>(
        std::vector<std::size_t>{
            static_cast<std::size_t>(points.shape(0)), 3u, 3u, 3u
        }
    );

    for (int i = 0; i < num_coils; ++i) {
        const int num_points = static_cast<int>(gammas[i].shape(0));
        res_hess_gamma[i]          = xt::zeros<double>({num_points, 3});
        res_hess_dgamma_by_dphi[i] = xt::zeros<double>({num_points, 3});
    }

    Array dummy = Array();

    #pragma omp parallel for
    for (int i = 0; i < num_coils; ++i) {
        if (compute_dA) {
            biot_savart_vector_potential_vjp_kernel<Array, 1>(
                pointsx, pointsy, pointsz,
                gammas[i],
                dgamma_by_dphis[i],
                v,
                res_gamma[i],
                res_dgamma_by_dphi[i],
                vgrad,
                res_grad_gamma[i],
                res_grad_dgamma_by_dphi[i],
                vhess,
                res_hess_gamma[i],
                res_hess_dgamma_by_dphi[i]
            );
        } else {
            biot_savart_vector_potential_vjp_kernel<Array, 0>(
                pointsx, pointsy, pointsz,
                gammas[i],
                dgamma_by_dphis[i],
                v,
                res_gamma[i],
                res_dgamma_by_dphi[i],
                dummy,                      // vgrad unused
                res_grad_gamma[i],
                res_grad_dgamma_by_dphi[i],
                vhess,
                res_hess_gamma[i],
                res_hess_dgamma_by_dphi[i]
            );
        }

        const double fak = (currents[i] * 1e-7 / gammas[i].shape(0));
        res_gamma[i]          *= fak;
        res_dgamma_by_dphi[i] *= fak;
        if (compute_dA) {
            res_grad_gamma[i]        *= fak;
            res_grad_dgamma_by_dphi[i] *= fak;
        }
    }
}
