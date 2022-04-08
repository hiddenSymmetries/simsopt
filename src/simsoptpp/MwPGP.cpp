#include "MwPGP.h"

vector<double>& _phi_MwPGP(const vector<double>& x, const vector<double>& g, const vector<double>& m_maxima):
        // phi(x_i, g_i) = g_i(x_i) is not on the L2 ball,
        // otherwise set it to zero
        // x and g should be in shape (N, 3)
	double N = len(x) // 3
        vector<double> check_active = np.isclose(
            x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2,
            m_maxima ** 2
        )
        // if triplet is in the active set (on the L2 unit ball)
        // then zero out those three indices
        if np.any(check_active):
            g[check_active, :] = 0.0
        return g  // will need to flatten after

vector<double>& beta_tilde(const vector<double>& x, const vector<double> g, const double alpha, const vector<double>& m_maxima):
    // beta_tilde(x_i, g_i, alpha) = 0_i if the ith triplet
    // is not on the L2 ball, otherwise is equal to different
    // values depending on the orientation of g.
    double N = len(x) // 3;
    check_active = np.isclose(
        x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2,
        m_maxima ** 2
    )
    // if triplet is NOT in the active set (on the L2 unit ball)
    // then zero out those three indices
    Array beta = xt::zeros<double>({N, 3});
    if np.any(~check_active):
        gradient_sphere = 2 * x_shaped
        denom_n = np.linalg.norm(gradient_sphere, axis=1, ord=2)
        n = np.divide(
            gradient_sphere,
            np.array([denom_n, denom_n, denom_n]).T
        )
        g_inds = np.ravel(np.where(~check_active))
        g_shaped = np.copy(g).reshape(N, 3)
        nTg = np.diag(n @ g_shaped.T)
        check_nTg_positive = (nTg > 0)
        beta_inds = np.logical_and(check_active, check_nTg_positive)
        beta_not_inds = np.logical_and(check_active, ~check_nTg_positive)
        N_not_inds = np.sum(np.array(beta_not_inds, dtype=int))
    if np.any(beta_inds):
        beta[beta_inds, :] = g[beta_inds, :]
    if np.any(beta_not_inds):
        beta[beta_not_inds, :] = g_reduced_gradient(
            x[beta_not_inds, :].reshape(3 * N_not_inds),
            alpha,
            g[beta_not_inds, :].reshape(3 * N_not_inds),
            m_maxima[beta_not_inds]
        );
    return beta

Array& g_reduced_gradient(Array& x, double alpha, Array& g, const vector<double> m_maxima) {
        // The reduced gradient of G is simply the
        // gradient step in the L2-projected direction.
        return (x - projection_L2_balls(x - alpha * g, m_maxima)) / alpha;
}
Array& g_reduced_projected_gradient(Array& x, double alpha, Array& g, const vector<double> m_maxima) {
    return phi_MwPGP(x, g, m_maxima) + beta_tilde(x, g, alpha, m_maxima);
}
double find_max_alphaf(Array& x, Array^ p) {
    // Solve a quadratic equation to determine the largest
    // step size alphaf such that the entirety of x - alpha * p
    // lives in the convex space defined by the intersection
    // of the N L2 balls defined in R3, so that
    // (x[0] - alpha * p[0]) ** 2 + (x[1] - alpha * p[1]) ** 2
    // + (x[2] - alpha * p[2]) ** 2 <= 1.
    double N = len(x) // 3
    vector<double> a = p_shaped[:, 0] ** 2 + p_shaped[:, 1] ** 2 + p_shaped[:, 2] ** 2
    vector<double> b = - 2 * (
            x_shaped[:, 0] * p_shaped[:, 0] + x_shaped[:, 1] * p_shaped[:, 1] + x_shaped[:, 2] * p_shaped[:, 2]
        )
    vector<double> c = (x_shaped[:, 0] ** 2 + x_shaped[:, 1] ** 2 + x_shaped[:, 2] ** 2) - self.m_maxima ** 2

    p_nonzero_inds = np.ravel(np.where(a > 0))
    if len(p_nonzero_inds) != 0:
        a = a[p_nonzero_inds]
        b = b[p_nonzero_inds]
        c = c[p_nonzero_inds]
        if np.all((b ** 2 - 4 * a * c) >= 0.0):
            alphaf_plus = np.min((-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a))
        else:
            alphaf_plus = 1e100
    else:
        alphaf_plus = 0.0
    return alphaf_plus;
}

void MwPGP_algorithm(const Array& ATA, const vector<double>& ATb, const vector<double>& m_proxy, const vector<double>& m0, double nu, double delta, double epsilon, double reg_l0, double reg_l1, double reg_l2, double reg_l2_shifted, int max_iter, bool verbose) {
    double alpha = 2.0 / ATA;
    double g = ATA.dot(m0) - ATb;
    double p = self._phi_MwPGP(m0, g);
    double g_alpha_P = g_reduced_projected_gradient(m0, alpha, g);
    double norm_g_alpha_P = np.linalg.norm(g_alpha_P, ord=2);
    // Add contribution from relax-and-split term
    vector<double> ATb_rs = ATb + m_proxy / nu
    
}

void biot_savart(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<Array>& B, vector<Array>& dB_by_dX, vector<Array>& d2B_by_dXdX) {
    auto pointsx = vector_type(points.shape(0), 0);
    auto pointsy = vector_type(points.shape(0), 0);
    auto pointsz = vector_type(points.shape(0), 0);
    int num_points = points.shape(0);
    for (int i = 0; i < num_points; ++i) {
        pointsx[i] = points(i, 0);
        pointsy[i] = points(i, 1);
        pointsz[i] = points(i, 2);
    }
    int num_coils  = gammas.size();

    Array dummyjac = xt::zeros<double>({1, 1, 1});
    Array dummyhess = xt::zeros<double>({1, 1, 1, 1});

    int nderivs = 0;
    if(dB_by_dX.size() == num_coils) {
        nderivs = 1;
        if(d2B_by_dXdX.size() == num_coils) {
            nderivs = 2;
        }
    }

#pragma omp parallel for
    for(int i=0; i<num_coils; i++) {
        if(nderivs == 2)
            biot_savart_kernel<Array, 2>(pointsx, pointsy, pointsz, gammas[i], dgamma_by_dphis[i], B[i], dB_by_dX[i], d2B_by_dXdX[i]);
        else {
            if(nderivs == 1) 
                biot_savart_kernel<Array, 1>(pointsx, pointsy, pointsz, gammas[i], dgamma_by_dphis[i], B[i], dB_by_dX[i], dummyhess);
            else
                biot_savart_kernel<Array, 0>(pointsx, pointsy, pointsz, gammas[i], dgamma_by_dphis[i], B[i], dummyjac, dummyhess);
        }
    }
}

Array biot_savart_B(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<double>& currents){
    auto dB_by_dXs = vector<Array>();
    auto d2B_by_dXdXs = vector<Array>();
    int num_coils = currents.size();
    auto Bs = vector<Array>(num_coils, Array());
    for (int i = 0; i < num_coils; ++i) {
        Bs[i] = xt::zeros<double>({points.shape(0), points.shape(1)});
    }
    biot_savart(points, gammas, dgamma_by_dphis, Bs, dB_by_dXs, d2B_by_dXdXs);
    Array B = xt::zeros<double>({points.shape(0), points.shape(1)});
    for (int i = 0; i < num_coils; ++i) {
        B += currents[i] * Bs[i];
    }
    return B;
}
