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
    //Array p = xt::zeros<double>({N, 3});
    //Array phi_temp = xt::zeros<double>({3});
    Array x_temp;
    //double norm_g_alpha_p, norm_phi_temp, x_temp1, x_temp2, x_temp3, gamma;
    //Array x_k1 = m0;
    Array x_k1 = xt::zeros<double>({N, 3});
    //Array ATAx;
   
#pragma omp parallel for //firstprivate(m0)
    for (int i = 0; i < N; ++i) {
        for (int ii = 0; ii < 3; ++ii) {
            for (int j = 0; j < N; ++j) {
                for (int kk = 0; kk < 3; ++kk) {
                    g(i, ii) += ATA(i, ii, j, kk) * m0(j, kk);
	        }
	    }
	    g(i, ii) += - ATb(i, ii);
	}
    }
    //for (int i = 0; i < N; ++i) {
//	phi_temp = phi_MwPGP(m0(i, 0), m0(i, 1), m0(i, 2), g(i, 0), g(i, 1), g(i, 2), m_maxima(i));
//	p(i, 0) = phi_temp(0);
//	p(i, 1) = phi_temp(1);
//	p(i, 2) = phi_temp(2);
  //  }
    for (int k = 0; k < max_iter; ++k) {
        //ATAx = xt::zeros<double>({N, 3});
        // x_temp = xt::zeros<double>({3});
//#pragma omp parallel for firstprivate(x_temp)
        printf("k = %d \n", k);
        #pragma omp parallel for private(x_temp)
        for (int i = 0; i < N; ++i) {
	    printf("thread %d, k = %d, iteration %d: x_k1(0) = %f, x_k1(1) = %f, x_k1(2) = %f, g(0) = %e, g(1) = %e, g(2) = %e, alpha = %f, m_maxima = %f \n", omp_get_thread_num(), k, i, x_k1(i, 0), x_k1(i, 1), x_k1(i, 2), g(i, 0), g(i, 1), g(i, 2), alpha, m_maxima(i));
	    x_temp = projection_L2_balls(x_k1(i, 0) - alpha * g(i, 0), x_k1(i, 1) - alpha * g(i, 1), x_k1(i, 2) - alpha * g(i, 2), m_maxima(i));
	    x_k1(i, 0) = x_temp(0); 
	    x_k1(i, 1) = x_temp(1); 
	    x_k1(i, 2) = x_temp(2);
	    //printf("thread %d, iteration %d: x_k1(0) = %f, x_k1(1) = %f, x_k1(2) = %f \n", omp_get_thread_num(), i, x_k1(i, 0), x_k1(i, 1), x_k1(i, 2));
            //x_k1(i, xt::all()) = (projection_L2_balls(x_k1(i, 0) - alpha * g(i, 0), x_k1(i, 1) - alpha * g(i, 1), x_k1(i, 2) - alpha * g(i, 2), m_maxima(i)));
            //for (int ii = 0; ii < 3; ++ii) {    
	    //	for (int j = 0; j < N; ++j) {
	      //       for (int jj = 0; jj < 3; ++jj) {     
		//	ATAx(i, ii) += ATA(i, ii, j, jj) * x_k1(j, jj);
		  //   }
	//	}
          //  }
	    //g(i, 0) = ATAx(i, 0) - ATb(i, 0);
	    //g(i, 1) = ATAx(i, 1) - ATb(i, 1);
	    //g(i, 2) = ATAx(i, 2) - ATb(i, 2);
	}
//#pragma omp barrier
///	g = xt::zeros<double>({N, 3});
//#pragma omp parallel for firstprivate(x_k1)
//	for (int i = 0; i < N; ++i) {     
//            for (int jj = 0; jj < 3; ++jj) {
//                for (int j = 0; j < N; ++j) {
//                    for (int kk = 0; kk < 3; ++kk) {
//                        g(i, jj) += ATA(i, jj, j, kk) * x_k1(j, kk);
//	           }
//	        }
//	        g(i, jj) += - ATb(i, jj);
//	    }

	    // phi_temp = phi_MwPGP(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2), g(i, 0), g(i, 1), g(i, 2), m_maxima(i));
	    // p(i, 0) = phi_temp(0);
	    // p(i, 1) = phi_temp(1);
	    // p(i, 2) = phi_temp(2);
 //       }
        //x_k = x_k1;
    }
    return x_k1;
}
