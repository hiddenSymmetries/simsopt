#include "MwPGP.h"

Array projection_L2_balls(double x1, double x2, double x3, double m_maxima) {
    Array x_new = xt::zeros<double>({3});
    double dist = sqrt(x1 * x1 + x2 * x2 + x3 * x3) / m_maxima; 
    double denom = std::max(1.0, dist);
    x_new(0) = x1 / denom;
    x_new(1) = x2 / denom;
    x_new(2) = x3 / denom;
    return x_new;
}

Array phi_MwPGP(double x1, double x2, double x3, double g1, double g2, double g3, double m_maxima)
{
    // phi(x_i, g_i) = g_i(x_i) is not on the L2 ball,
    // otherwise set it to zero
    Array g_new = xt::zeros<double>({3});
    double tol = 1.0e-6;
    double dist = abs(x1 * x1 + x2 * x2 + x3 * x3 - m_maxima * m_maxima);
    // if triplet is in the active set (on the L2 unit ball)
    // then zero out those three indices
    if (dist > tol) {
	g_new(0) = g1;
	g_new(1) = g2;
	g_new(2) = g3;
    }
    return g_new;
}

Array beta_tilde(double x1, double x2, double x3, double g1, double g2, double g3, double alpha, double m_maxima)
{
    // beta_tilde(x_i, g_i, alpha) = 0_i if the ith triplet
    // is not on the L2 ball, otherwise is equal to different
    // values depending on the orientation of g.
    double ng, dist, denom, normal_vec1, normal_vec2, normal_vec3;
    double tol = 1.0e-6;
    Array beta = xt::zeros<double>({3});
    dist = abs(x1 * x1 + x2 * x2 + x3 * x3 - m_maxima * m_maxima);
    // if triplet is NOT in the active set (on the L2 unit ball)
    // then zero out those three indices
    denom = sqrt(x1 * x1 + x2 * x2 + x3 * x3);
    normal_vec1 = x1 / denom;
    normal_vec2 = x2 / denom;
    normal_vec3 = x3 / denom;
    if (dist < tol) {
	ng = normal_vec1 * g1 + normal_vec2 * g2 + normal_vec3 * g3;
        if (ng > 0) { 
            beta(0) = g1;
            beta(1) = g2;
            beta(2) = g3;
	}
        else {
            beta = g_reduced_gradient(x1, x2, x3, g1, g2, g3, alpha, m_maxima);
	}
    }
    return beta;
}

Array g_reduced_gradient(double x1, double x2, double x3, double g1, double g2, double g3, double alpha, double m_maxima) {
        // The reduced gradient of G is simply the
        // gradient step in the L2-projected direction.
	Array x = xt::zeros<double>({3});
        x(0) = x1;
        x(1) = x2;
        x(2) = x3;
        return (x - projection_L2_balls(x1 - alpha * g1, x2 - alpha * g2, x3 - alpha * g3, m_maxima)) / alpha;
}

Array g_reduced_projected_gradient(double x1, double x2, double x3, double g1, double g2, double g3, double alpha, double m_maxima) {
    return phi_MwPGP(x1, x2, x3, g1, g2, g3, m_maxima) + beta_tilde(x1, x2, x3, g1, g2, g3, alpha, m_maxima);
}

double find_max_alphaf(double x1, double x2, double x3, double p1, double p2, double p3, double m_maxima) {
    // Solve a quadratic equation to determine the largest
    // step size alphaf such that the entirety of x - alpha * p
    // lives in the convex space defined by the intersection
    // of the N L2 balls defined in R3, so that
    // (x[0] - alpha * p[0]) ** 2 + (x[1] - alpha * p[1]) ** 2
    // + (x[2] - alpha * p[2]) ** 2 <= 1.
    double a, b, c, alphaf_plus;
    a = p1 * p1 + p2 * p2 + p3 * p3;
    b = - 2 * (x1 * p1 + x2 * p2 + x3 * p3);
    c = x1 * x1 + x2 * x2 + x3 * x3 - m_maxima * m_maxima;
    if (a > 0) { 
        if ((b * b - 4 * a * c) >= 0.0) {
            alphaf_plus = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
        }
	else {
            alphaf_plus = 1e100;
	}
    }
    else {
        alphaf_plus = 0.0;
    }
    return alphaf_plus;
}

Array MwPGP_algorithm(Array& ATA, Array& ATb, Array& m_proxy, Array& m0, Array& m_maxima, double alpha, double nu, double delta, double epsilon, double reg_l0, double reg_l1, double reg_l2, double reg_l2_shifted, int max_iter)
{
    // Needs ATA in shape (N, N, 3, 3) and ATb in shape (N, 3)
    int N = ATb.shape(0);
    Array g = xt::zeros<double>({N, 3});
    Array p = xt::zeros<double>({N, 3});
    Array g_alpha_p = xt::zeros<double>({3});
    Array phi_temp = xt::zeros<double>({3});
    Array x_temp = xt::zeros<double>({3});
    double norm_g_alpha_p, norm_phi_temp, x_temp1, x_temp2, x_temp3, gamma;
    double alpha_f, alpha_cg;
    int k = 0;
    int print_iter = 0;
    Array x_k = m0;
    Array x_k1 = xt::zeros<double>({N, 3});
    Array ATAp;
    Array ATAx;
    Array m_history = xt::zeros<double>({N, 3, (int)(max_iter / 10)});
    // Add contribution from relax-and-split term
    Array ATb_rs = ATb + m_proxy / nu;
   
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < 3; ++k) {
            for (int j = 0; j < N; ++j) {
                for (int kk = 0; kk < 3; ++kk) {
                    g(i, k) += ATA(i, j, k, kk) * m0(j, kk);
	        }
	    }
	    g(i, k) += - ATb(i, k);
	}
    }
    for (int i = 0; i < N; ++i) {
	phi_temp = phi_MwPGP(m0(i, 0), m0(i, 1), m0(i, 2), g(i, 0), g(i, 1), g(i, 2), m_maxima(i));
	p(i, 0) = phi_temp(0);
	p(i, 1) = phi_temp(1);
	p(i, 2) = phi_temp(2);
    }
 
    while (k < max_iter) {
        ATAp = xt::zeros<double>({N, 3});
        ATAx = xt::zeros<double>({N, 3});
//#pragma omp parallel for
        for (int i = 0; i < N; ++i) {     
		    //if (print_iter) { 
		    //    m_history(i, k) = x_k(i);
		    // }
		    //g_alpha_p = g_reduced_projected_gradient(x_k(i), x_k(i + 1), x_k(i + 2), g(i), g(i + 1), g(i + 2), alpha, m_maxima((int)(i / 3)));
		    //norm_g_alpha_p = g_alpha_p(0) * g_alpha_p(0)  + g_alpha_p(1) * g_alpha_p(1) + g_alpha_p(2) * g_alpha_p(2);
		    //phi_temp = phi_MwPGP(x_k(i), x_k(i + 1), x_k(i + 2), g(i), g(i + 1), g(i + 2), m_maxima((int)(i / 3)));
		    //norm_phi_temp = phi_temp(0) * phi_temp(0) + phi_temp(1) * phi_temp(1) + phi_temp(2) * phi_temp(2);
            
		    //if (delta < 0) {  //(2 * delta * norm_g_alpha_p <= norm_phi_temp) {
//			alpha_cg = g(i) * p(i) / (p(i) * ATAp(i));
//			alpha_f = find_max_alphaf(x_k(i), x_k(i + 1), x_k(i + 2), p(i), p(i + 1), p(i + 2), m_maxima((int) (i / 3)));
//			if (alpha_cg < alpha_f) {
			    // Take a conjugate gradient step
//			    x_k1(i) = x_k(i) - alpha_cg * p(i);
//			    x_k1(i + 1) = x_k(i + 1) - alpha_cg * p(i + 1);
//			    x_k1(i + 2) = x_k(i + 2) - alpha_cg * p(i + 2);
//			    g(i) = g(i) - alpha_cg * ATAp(i);
//			    g(i + 1) = g(i + 1) - alpha_cg * ATAp(i + 1);
//			    g(i + 2) = g(i + 2) - alpha_cg * ATAp(i + 2);
//			    phi_temp = phi_MwPGP(x_k1(i), x_k1(i + 1), x_k1(i + 2), g(i), g(i + 1), g(i + 2), m_maxima((int)(i / 3)));
//			    gamma = (phi_temp(0) * ATAp(i) + phi_temp(1) * ATAp(i + 1) + phi_temp(2) * ATAp(i + 2)) / (p(i) * ATAp(i) + p(i + 1) * ATAp(i + 1) + p(i + 2) * ATAp(i + 2));
//			    p(i) = phi_temp(0) - gamma * p(i);
//			    p(i + 1) = phi_temp(1) - gamma * p(i + 1);
//			    p(i + 2) = phi_temp(2) - gamma * p(i + 2);
//			}
//			else {
//			    // Take a mixed projected gradient step
//			    x_temp1 = (x_k(i) - alpha_f * p(i)) - alpha * (g(i) - alpha_f * ATAp(i));
//			    x_temp2 = (x_k(i + 1) - alpha_f * p(i + 1)) - alpha * (g(i + 1) - alpha_f * ATAp(i + 1));
//			    x_temp3 = (x_k(i + 2) - alpha_f * p(i + 2)) - alpha * (g(i + 2) - alpha_f * ATAp(i + 2));
//			    x_temp = projection_L2_balls(x_temp1, x_temp2, x_temp3, m_maxima((int)(i / 3)));
			    
//			    x_k1(i) = x_temp(0);
//			    x_k1(i + 1) = x_temp(1);
//			    x_k1(i + 2) = x_temp(2);
//			    g(i) = ATAx(i) - ATb(i);
//			    g(i + 1) = ATAx(i + 1) - ATb(i + 1);
//			    g(i + 2) = ATAx(i + 2) - ATb(i + 2);
//			    phi_temp = phi_MwPGP(x_k1(i), x_k1(i + 1), x_k1(i + 2), g(i), g(i + 1), g(i + 2), m_maxima((int)(i / 3)));
//			    p(i) = phi_temp(0);
//			    p(i + 1) = phi_temp(1);
//			    p(i + 2) = phi_temp(2);
//			}
//		    }
		   // else {
			// Take a projected gradient step
	    x_temp = projection_L2_balls(x_k(i, 0) - alpha * g(i, 0), x_k(i, 1) - alpha * g(i, 1), x_k(i, 2) - alpha * g(i, 2), m_maxima(i));
	    x_k1(i, 0) = x_temp(0); 
	    x_k1(i, 1) = x_temp(1); 
	    x_k1(i, 2) = x_temp(2); 
            for (int ii = 0; ii < 3; ++ii) {    
		for (int j = 0; j < N; ++j) {
	             for (int jj = 0; jj < 3; ++jj) {     
			ATAx(i, ii) += ATA(i, j, ii, jj) * x_k1(j, jj);
			// ATAp(i, ii) += ATA(i, j, ii, jj) * p(j, jj);
		     }
		}
            }
	    g(i, 0) = ATAx(i, 0) - ATb(i, 0);
	    g(i, 1) = ATAx(i, 1) - ATb(i, 1);
	    g(i, 2) = ATAx(i, 2) - ATb(i, 2);
	    phi_temp = phi_MwPGP(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2), g(i, 0), g(i, 1), g(i, 2), m_maxima(i));
	    p(i, 0) = phi_temp(0);
	    p(i, 1) = phi_temp(1);
	    p(i, 2) = phi_temp(2);
		   // }
	//	}
        }
	k = k + 1;
        x_k = x_k1;
    }
    return x_k;
}
