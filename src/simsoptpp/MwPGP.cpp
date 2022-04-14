#include "MwPGP.h"

std::tuple<double, double, double> projection_L2_balls(double x1, double x2, double x3, double m_maxima) {
    double dist = sqrt(x1 * x1 + x2 * x2 + x3 * x3) / m_maxima; 
    double denom = std::max(1.0, dist);
    return std::make_tuple(x1 / denom, x2 / denom, x3 / denom);
}

std::tuple<double, double, double> phi_MwPGP(double x1, double x2, double x3, double g1, double g2, double g3, double m_maxima)
{
    // phi(x_i, g_i) = g_i(x_i) if x_i is not on the L2 ball,
    // otherwise set phi to zero
    double atol = 1.0e-8;  // default tolerances from numpy isclose
    double rtol = 1.0e-5;  // default tolerances from numpy isclose
    double dist = x1 * x1 + x2 * x2 + x3 * x3;  // - m_maxima * m_maxima);
    // if triplet is in the active set (on the L2 unit ball)
    // then zero out those three indices
    if (abs(dist - m_maxima * m_maxima) > atol + rtol * m_maxima * m_maxima) {
	return std::make_tuple(g1, g2, g3);
    }
    else {
        return std::make_tuple(0.0, 0.0, 0.0);
    }
}

std::tuple<double, double, double> beta_tilde(double x1, double x2, double x3, double g1, double g2, double g3, double alpha, double m_maxima)
{
    // beta_tilde(x_i, g_i, alpha) = 0_i if the ith triplet
    // is not on the L2 ball, otherwise is equal to different
    // values depending on the orientation of g.
    double ng, dist, denom, normal_vec1, normal_vec2, normal_vec3;
    double atol = 1.0e-8;  // default tolerances from numpy isclose
    double rtol = 1.0e-5;  // default tolerances from numpy isclose
    dist = x1 * x1 + x2 * x2 + x3 * x3; // - m_maxima * m_maxima);
    if (abs(dist - m_maxima * m_maxima) < (atol + rtol * m_maxima * m_maxima)) {
        denom = sqrt(dist);
        normal_vec1 = x1 / denom;
        normal_vec2 = x2 / denom;
        normal_vec3 = x3 / denom;
	ng = normal_vec1 * g1 + normal_vec2 * g2 + normal_vec3 * g3;
        if (ng > 0) { 
	    return std::make_tuple(g1, g2, g3);
	}
        else {
            return g_reduced_gradient(x1, x2, x3, g1, g2, g3, alpha, m_maxima);
	}
    }
    else {
	// if triplet is NOT in the active set (on the L2 unit ball)
        // then zero out those three indices
        return std::make_tuple(0.0, 0.0, 0.0);
    }
}

std::tuple<double, double, double> g_reduced_gradient(double x1, double x2, double x3, double g1, double g2, double g3, double alpha, double m_maxima) 
{
        // The reduced gradient of G is simply the
        // gradient step in the L2-projected direction.
	double proj_L2x, proj_L2y, proj_L2z; 
	std::tie(proj_L2x, proj_L2y, proj_L2z) = projection_L2_balls(x1 - alpha * g1, x2 - alpha * g2, x3 - alpha * g3, m_maxima);
        return std::make_tuple((x1 - proj_L2x) / alpha, (x2 - proj_L2y) / alpha, (x3 - proj_L2z) / alpha);
}

std::tuple<double, double, double> g_reduced_projected_gradient(double x1, double x2, double x3, double g1, double g2, double g3, double alpha, double m_maxima) {
    double psum1, psum2, psum3, bsum1, bsum2, bsum3;
    std::tie(psum1, psum2, psum3) = phi_MwPGP(x1, x2, x3, g1, g2, g3, m_maxima);
    std::tie(bsum1, bsum2, bsum3) = beta_tilde(x1, x2, x3, g1, g2, g3, alpha, m_maxima);
    return std::make_tuple(psum1 + bsum1, psum2 + bsum2, psum3 + bsum3);
}

double find_max_alphaf(double x1, double x2, double x3, double p1, double p2, double p3, double m_maxima) {
    // Solve a quadratic equation to determine the largest
    // step size alphaf such that the entirety of x - alpha * p
    // lives in the convex space defined by the intersection
    // of the N L2 balls defined in R3, so that
    // (x[0] - alpha * p[0]) ** 2 + (x[1] - alpha * p[1]) ** 2
    // + (x[2] - alpha * p[2]) ** 2 <= 1.
    double a, b, c, alphaf_plus;
    double tol = 1e-20;
    a = p1 * p1 + p2 * p2 + p3 * p3;
    b = - 2 * (x1 * p1 + x2 * p2 + x3 * p3);
    c = x1 * x1 + x2 * x2 + x3 * x3 - m_maxima * m_maxima;
    if (a > tol) { 
        if ((b * b - 4 * a * c) >= 0.0) {
            alphaf_plus = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
        }
	else {
            alphaf_plus = 1e100;
	}
    }
    else {
        alphaf_plus = 1e100; // ignore the value
    }
    return alphaf_plus;
}

Array MwPGP_algorithm(Array& ATA, Array& ATb, Array& m_proxy, Array& m0, Array& m_maxima, double alpha, double nu, double delta, double epsilon, double reg_l0, double reg_l1, double reg_l2, double reg_l2_shifted, int max_iter)
{
    // Needs ATA in shape (N, 3, N, 3) and ATb in shape (N, 3)
    int N = ATb.shape(0);
    Array g = xt::zeros<double>({N, 3});
    Array p = xt::zeros<double>({N, 3});
    Array ATAp = xt::zeros<double>({N, 3});
    double norm_g_alpha_p, norm_phi_temp, gamma, gp, pATAp;
    double g_alpha_p1, g_alpha_p2, g_alpha_p3, phi_temp1, phi_temp2, phi_temp3;
    double phig1, phig2, phig3, p_temp1, p_temp2, p_temp3; 
    double alpha_cg, alpha_f;
    vector<double> alpha_fs(N);
    Array x_k1 = m0;
  
    // Set up initial g and p Arrays 
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int ii = 0; ii < 3; ++ii) {
            for (int j = 0; j < N; ++j) {
                for (int kk = 0; kk < 3; ++kk) {
                    g(i, ii) += ATA(i, ii, j, kk) * m0(j, kk);
	        }
	    }
	    g(i, ii) += - ATb(i, ii);
	}
	std::tie(p(i, 0), p(i, 1), p(i, 2)) = phi_MwPGP(m0(i, 0), m0(i, 1), m0(i, 2), g(i, 0), g(i, 1), g(i, 2), m_maxima(i));
    }

    // Main loop over the optimization iterations
    for (int k = 0; k < max_iter; ++k) {
        //if (print_iter) { 
	//m_history(i, k) = x_k(i);
	//}
	
	// compute L2 norm of reduced g and L2 norm of phi(x, g)
	// as well as some dot products needed for the algorithm
	norm_g_alpha_p = 0.0;
	norm_phi_temp = 0.0;
	gp = 0.0;
	pATAp = 0.0;
	ATAp = xt::zeros<double>({N, 3});
        // std::fill(alpha_fs.begin(), alpha_fs.end(), 0.0);     
        #pragma omp parallel for reduction(+: norm_g_alpha_p, norm_phi_temp, gp, pATAp) private(phi_temp1, phi_temp2, phi_temp3, g_alpha_p1, g_alpha_p2, g_alpha_p3)
	for(int i = 0; i < N; ++i) {
	    std::tie(g_alpha_p1, g_alpha_p2, g_alpha_p3) = g_reduced_projected_gradient(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2), g(i, 0), g(i, 1), g(i, 2), alpha, m_maxima(i));
            norm_g_alpha_p += g_alpha_p1 * g_alpha_p1 + g_alpha_p2 * g_alpha_p2 + g_alpha_p3 * g_alpha_p3;
	    std::tie(phi_temp1, phi_temp2, phi_temp3) = phi_MwPGP(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2), g(i, 0), g(i, 1), g(i, 2), m_maxima(i));
            norm_phi_temp += phi_temp1 * phi_temp1 + phi_temp2 * phi_temp2 + phi_temp3 * phi_temp3;
	    gp += g(i, 0) * p(i, 0) + g(i, 1) * p(i, 1) + g(i, 2) * p(i, 2);
            alpha_fs[i] = find_max_alphaf(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2), p(i, 0), p(i, 1), p(i, 2), m_maxima(i));
	    for (int ii = 0; ii < 3; ++ii) {
                for (int j = 0; j < N; ++j) {
                    for (int kk = 0; kk < 3; ++kk) {
                        ATAp(i, ii) += ATA(i, ii, j, kk) * p(j, kk);
	            }
	        }
	    }
	    pATAp += p(i, 0) * ATAp(i, 0) + p(i, 1) * ATAp(i, 1) + p(i, 2) * ATAp(i, 2);
	}
        auto max_i = std::min_element(alpha_fs.begin(), alpha_fs.end()); 
	alpha_f = *max_i;
        alpha_cg = gp / pATAp;
        if (2 * delta * norm_g_alpha_p <= norm_phi_temp) {
	    if (alpha_cg < alpha_f) {
	        // Take a conjugate gradient step
                #pragma omp parallel for
	        for (int i = 0; i < N; ++i) {  
	            for (int ii = 0; ii < 3; ++ii) {  
		    	x_k1(i, ii) += - alpha_cg * p(i, ii);
		    	g(i, ii) += - alpha_cg * ATAp(i, ii);
		    }
		}
		gamma = 0.0;
                #pragma omp parallel for reduction(+: gamma) private(phig1, phig2, phig3)
	        for (int i = 0; i < N; ++i) {     
	            std::tie(phig1, phig2, phig3) = phi_MwPGP(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2), g(i, 0), g(i, 1), g(i, 2), m_maxima(i));
                    gamma += (phig1 * ATAp(i, 0) + phig2 * ATAp(i, 1) + phig3 * ATAp(i, 2)); 
		}    
		gamma = gamma / pATAp;
                #pragma omp parallel for private(p_temp1, p_temp2, p_temp3)
	        for (int i = 0; i < N; ++i) {     
		    std::tie(p_temp1, p_temp2, p_temp3) = phi_MwPGP(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2), g(i, 0), g(i, 1), g(i, 2), m_maxima(i));
		    p(i, 0) = p_temp1 - gamma * p(i, 0);
	            p(i, 1) = p_temp2 - gamma * p(i, 1);
	            p(i, 2) = p_temp3 - gamma * p(i, 2);
		}
	    }
	    else {
		// Take a mixed projected gradient step
                #pragma omp parallel for
	        for (int i = 0; i < N; ++i) {     
                    std::tie(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2)) = projection_L2_balls((x_k1(i, 0) - alpha_f * p(i, 0)) - alpha * (g(i, 0) - alpha_f * ATAp(i, 0)), (x_k1(i, 1) - alpha_f * p(i, 1)) - alpha * (g(i, 1) - alpha_f * ATAp(i, 1)), (x_k1(i, 2) - alpha_f * p(i, 2)) - alpha * (g(i, 2) - alpha_f * ATAp(i, 2)), m_maxima(i));
		}
		g = xt::zeros<double>({N, 3});
                #pragma omp parallel for
	        for (int i = 0; i < N; ++i) {     
                    for (int jj = 0; jj < 3; ++jj) {
                        for (int j = 0; j < N; ++j) {
                            for (int kk = 0; kk < 3; ++kk) {
                                g(i, jj) += ATA(i, jj, j, kk) * x_k1(j, kk);
	                    }
	                }
	                g(i, jj) += - ATb(i, jj);
	            }
	            std::tie(p(i, 0), p(i, 1), p(i, 2)) = phi_MwPGP(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2), g(i, 0), g(i, 1), g(i, 2), m_maxima(i));
                }
	    }
        }
        else {
	    // projected gradient descent method
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) {
	        std::tie(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2)) = projection_L2_balls(x_k1(i, 0) - alpha * g(i, 0), x_k1(i, 1) - alpha * g(i, 1), x_k1(i, 2) - alpha * g(i, 2), m_maxima(i));
	    }
	    g = xt::zeros<double>({N, 3});
            #pragma omp parallel for
	    for (int i = 0; i < N; ++i) {     
                for (int jj = 0; jj < 3; ++jj) {
                    for (int j = 0; j < N; ++j) {
                        for (int kk = 0; kk < 3; ++kk) {
                            g(i, jj) += ATA(i, jj, j, kk) * x_k1(j, kk);
	                }
	            }
	            g(i, jj) += - ATb(i, jj);
	        }
	        std::tie(p(i, 0), p(i, 1), p(i, 2)) = phi_MwPGP(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2), g(i, 0), g(i, 1), g(i, 2), m_maxima(i));
            }
        }
    }
    return x_k1;
}
