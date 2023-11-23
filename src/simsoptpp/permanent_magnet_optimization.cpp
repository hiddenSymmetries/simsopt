#include "permanent_magnet_optimization.h"
#include <Eigen/Dense>
#include "simdhelpers.h"
#include "vec3dsimd.h"
#include "xtensor/xsort.hpp"
#include "xtensor/xview.hpp"
#include <functional>
#include <vector>
#include <math.h>

// Project a 3-vector onto the L2 ball with radius m_maxima
std::tuple<double, double, double> projection_L2_balls(double x1, double x2, double x3, double m_maxima) {
    double denom = std::max(1.0, sqrt(x1 * x1 + x2 * x2 + x3 * x3) / m_maxima);
    return std::make_tuple(x1 / denom, x2 / denom, x3 / denom);
}

// Takes a vector and zeros it if it is very close to the L2 ball surface
std::tuple<double, double, double> phi_MwPGP(double x1, double x2, double x3, double g1, double g2, double g3, double m_maxima)
{
    // phi(x_i, g_i) = g_i(x_i) if x_i is not on the L2 ball,
    // otherwise set phi to zero
    double xmag2 = x1 * x1 + x2 * x2 + x3 * x3;
    if (abs(xmag2 - m_maxima * m_maxima) > 1.0e-8 + 1.0e-5 * m_maxima * m_maxima) {
        return std::make_tuple(g1, g2, g3);
    }
    else {
        // if triplet is in the active set (on the L2 unit ball)
        // then zero out those three indices
        return std::make_tuple(0.0, 0.0, 0.0);
    }
}

// Takes a vector and zeros it if it is NOT on the L2 ball surface
std::tuple<double, double, double> beta_tilde(double x1, double x2, double x3, double g1, double g2, double g3, double alpha, double m_maxima)
{
    // beta_tilde(x_i, g_i, alpha) = 0_i if the ith triplet
    // is not on the L2 ball, otherwise is equal to different
    // values depending on the orientation of g.
    double ng, mmax2, dist;
    dist = x1 * x1 + x2 * x2 + x3 * x3;
    mmax2 = m_maxima * m_maxima;
    if (abs(dist - mmax2) < (1.0e-8 + 1.0e-5 * mmax2)) {
        ng = (x1 * g1 + x2 * g2 + x3 * g3) / sqrt(dist);
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

// The reduced gradient of G is simply the
// gradient step in the L2-ball-projected direction.
std::tuple<double, double, double> g_reduced_gradient(double x1, double x2, double x3, double g1, double g2, double g3, double alpha, double m_maxima)
{
    double proj_L2x, proj_L2y, proj_L2z;
    std::tie(proj_L2x, proj_L2y, proj_L2z) = projection_L2_balls(x1 - alpha * g1, x2 - alpha * g2, x3 - alpha * g3, m_maxima);
    return std::make_tuple((x1 - proj_L2x) / alpha, (x2 - proj_L2y) / alpha, (x3 - proj_L2z) / alpha);
}

// This function is just phi + beta_tilde
std::tuple<double, double, double> g_reduced_projected_gradient(double x1, double x2, double x3, double g1, double g2, double g3, double alpha, double m_maxima) {
    double psum1, psum2, psum3, bsum1, bsum2, bsum3;
    std::tie(psum1, psum2, psum3) = phi_MwPGP(x1, x2, x3, g1, g2, g3, m_maxima);
    std::tie(bsum1, bsum2, bsum3) = beta_tilde(x1, x2, x3, g1, g2, g3, alpha, m_maxima);
    return std::make_tuple(psum1 + bsum1, psum2 + bsum2, psum3 + bsum3);
}

// Solve a quadratic equation to determine the largest
// step size alphaf such that the entirety of x - alpha * p
// lives in the L2 ball with radius m_maxima
double find_max_alphaf(double x1, double x2, double x3, double p1, double p2, double p3, double m_maxima) {
    double a, b, c, alphaf_plus;
    double tol = 1e-20;
    a = p1 * p1 + p2 * p2 + p3 * p3;
    c = x1 * x1 + x2 * x2 + x3 * x3 - m_maxima * m_maxima;
    b = - 2 * (x1 * p1 + x2 * p2 + x3 * p3);
    if (a > tol) {
        // c is always negative and a is always positive
        // so alphaf_plus >= 0 always
        alphaf_plus = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
    }
    else {
        alphaf_plus = 1e100; // ignore the value
    }
    return alphaf_plus;
}

// print out all the possible loss terms in the objective function
// and record histories of the dipole moments, objective values, etc.
void print_MwPGP(Array& A_obj, Array& b_obj, Array& x_k1, Array& m_proxy, Array& m_maxima, Array& m_history, Array& objective_history, Array& R2_history, int print_iter, int k, double nu, double reg_l0, double reg_l1, double reg_l2)
{
    int ngrid = A_obj.shape(0);
    int N = m_maxima.shape(0);
    double R2 = 0.0;
    double N2 = 0.0;
    double L2 = 0.0;
    double L1 = 0.0;
    double L0 = 0.0;
    double cost = 0.0;
    double l0_tol = 1e-20;
    Array R2_temp = xt::zeros<double>({ngrid});
#pragma omp parallel for reduction(+: N2, L2, L1, L0)
    for(int i = 0; i < N; ++i) {
	for(int ii = 0; ii < 3; ++ii) {
	    m_history(i, ii, print_iter) = x_k1(i, ii);
	    N2 += (x_k1(i, ii) - m_proxy(i, ii)) * (x_k1(i, ii) - m_proxy(i, ii));
	    L2 += x_k1(i, ii) * x_k1(i, ii);
	    L1 += abs(x_k1(i, ii));
	    L0 += ((abs(m_proxy(i, ii)) < l0_tol) ? 1.0 : 0.0);
	}
    }

    // Computation of R2 takes more work than the other loss terms... need to compute
    // the linear least-squares term.
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_mat(const_cast<double*>(A_obj.data()), ngrid, 3*N);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_v(const_cast<double*>(x_k1.data()), 3*N, 1);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_res(const_cast<double*>(R2_temp.data()), ngrid, 1);
    eigen_res = eigen_mat*eigen_v;
#pragma omp parallel for reduction(+: R2)
    for(int i = 0; i < ngrid; ++i) {
	R2 += (R2_temp(i) - b_obj(i)) * (R2_temp(i) - b_obj(i));
    }

    // rescale loss terms by the hyperparameters
    R2 = 0.5 * R2;
    N2 = 0.5 * N2 / nu;
    L2 = reg_l2 * L2;
    L1 = reg_l1 * L1;
    L0 = reg_l0 * L0;

    // L1, L0, and other nonconvex loss terms are not addressed by this algorithm
    // so they will just be constant and we can omit them from the total cost.
    cost = R2 + N2 + L2;
    objective_history(print_iter) = cost;
    R2_history(print_iter) = R2;
    printf("%d ... %.2e ... %.2e ... %.2e ... %.2e ... %.2e ... %.2e \n", k, R2, N2, L2, L1, L0, cost);
}

// Run the MwPGP algorithm for solving the convex part of
// the permanent magnet optimization problem. This algorithm has
// many optional parameters for additional loss terms.
// See Bouchala, Jiří, et al.On the solution of convex QPQC
// problems with elliptic and other separable constraints with
// strong curvature. Applied Mathematics and Computation 247 (2014): 848-864.
std::tuple<Array, Array, Array, Array> MwPGP_algorithm(Array& A_obj, Array& b_obj, Array& ATb, Array& m_proxy, Array& m0, Array& m_maxima, double alpha, double nu, double epsilon, double reg_l0, double reg_l1, double reg_l2, int max_iter, double min_fb, bool verbose)
{
    // Needs ATb in shape (N, 3)
    int ngrid = A_obj.shape(0);
    int N = ATb.shape(0);
    int print_iter = 0;
    double x_sum;
    Array g = xt::zeros<double>({N, 3});
    Array p = xt::zeros<double>({N, 3});
    Array ATAp = xt::zeros<double>({N, 3});

    // define bunch of doubles, mostly for setting the std::tuples correctly
    double norm_g_alpha_p, norm_phi_temp, gamma, gp, pATAp;
    double g_alpha_p1, g_alpha_p2, g_alpha_p3, phi_temp1, phi_temp2, phi_temp3;
    double phig1, phig2, phig3, p_temp1, p_temp2, p_temp3;
    double alpha_cg, alpha_f;
    vector<double> alpha_fs(N);
    Array x_k1 = m0;
    Array x_k_prev;

    // record the history of the algorithm iterations
    Array m_history = xt::zeros<double>({N, 3, 21});
    Array objective_history = xt::zeros<double>({21});
    Array R2_history = xt::zeros<double>({21});

    // Add contribution from relax-and-split term
    Array ATb_rs = ATb + m_proxy / nu;

    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_mat(const_cast<double*>(A_obj.data()), ngrid, 3*N);

    // Set up initial g and p Arrays
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_v(const_cast<double*>(m0.data()), 1, 3*N);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_res(const_cast<double*>(g.data()), 1, 3*N);

    // A^TA * m + contributions from L2 and relax-and-split terms
    eigen_res = eigen_v*eigen_mat.transpose()*eigen_mat + 2 * eigen_v * (reg_l2 + 1.0 / (2.0 * nu));

    // subtract off A^T * b + m_proxy / nu for fully initialized g
    g -= ATb_rs;

    // initialize p as phi(m0, g)
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        std::tie(p(i, 0), p(i, 1), p(i, 2)) = phi_MwPGP(m0(i, 0), m0(i, 1), m0(i, 2), g(i, 0), g(i, 1), g(i, 2), m_maxima(i));
    }

    // print out the names of the error columns
    if (verbose)
        printf("Iteration ... |Am - b|^2 ... |m-w|^2/v ...   a|m|^2 ...  b|m-1|^2 ...   c|m|_1 ...   d|m|_0 ... Total Error:\n");

    // Main loop over the optimization iterations
    for (int k = 0; k < max_iter; ++k) {

	      x_k_prev = x_k1;

        // compute L2 norm of reduced g and L2 norm of phi(x, g)
        // as well as some dot products needed for the algorithm
        norm_g_alpha_p = 0.0;
        norm_phi_temp = 0.0;
        gp = 0.0;
        pATAp = 0.0;
        ATAp = xt::zeros<double>({N, 3});
        Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_v(const_cast<double*>(p.data()), 1, 3*N);
        Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_res(const_cast<double*>(ATAp.data()), 1, 3*N);
        eigen_res = eigen_v*eigen_mat.transpose()*eigen_mat + 2 * eigen_v * (reg_l2 + 1.0 / (2.0 * nu));
#pragma omp parallel for reduction(+: norm_g_alpha_p, norm_phi_temp, gp, pATAp) private(phi_temp1, phi_temp2, phi_temp3, g_alpha_p1, g_alpha_p2, g_alpha_p3)
        for(int i = 0; i < N; ++i) {
            std::tie(g_alpha_p1, g_alpha_p2, g_alpha_p3) = g_reduced_projected_gradient(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2), g(i, 0), g(i, 1), g(i, 2), alpha, m_maxima(i));
            std::tie(phi_temp1, phi_temp2, phi_temp3) = phi_MwPGP(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2), g(i, 0), g(i, 1), g(i, 2), m_maxima(i));
	    norm_g_alpha_p += g_alpha_p1 * g_alpha_p1 + g_alpha_p2 * g_alpha_p2 + g_alpha_p3 * g_alpha_p3;
            norm_phi_temp += phi_temp1 * phi_temp1 + phi_temp2 * phi_temp2 + phi_temp3 * phi_temp3;
	    gp += g(i, 0) * p(i, 0) + g(i, 1) * p(i, 1) + g(i, 2) * p(i, 2);
            alpha_fs[i] = find_max_alphaf(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2), p(i, 0), p(i, 1), p(i, 2), m_maxima(i));
            pATAp += p(i, 0) * ATAp(i, 0) + p(i, 1) * ATAp(i, 1) + p(i, 2) * ATAp(i, 2);
        }

        // compute step sizes for different descent step types
	auto max_i = std::min_element(alpha_fs.begin(), alpha_fs.end());
        alpha_f = *max_i;
        alpha_cg = gp / pATAp;

        // based on these norms, decide what kind of a descent step to take
        if (norm_g_alpha_p <= norm_phi_temp) {
            if (alpha_cg < alpha_f) {
                // Take a conjugate gradient step
#pragma omp parallel for
                for (int i = 0; i < N; ++i) {
                    for (int ii = 0; ii < 3; ++ii) {
                        x_k1(i, ii) += - alpha_cg * p(i, ii);
                        g(i, ii) += - alpha_cg * ATAp(i, ii);
                    }
                }

                // compute gamma step size
                gamma = 0.0;
#pragma omp parallel for reduction(+: gamma) private(phig1, phig2, phig3)
                for (int i = 0; i < N; ++i) {
                    std::tie(phig1, phig2, phig3) = phi_MwPGP(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2), g(i, 0), g(i, 1), g(i, 2), m_maxima(i));
                    gamma += (phig1 * ATAp(i, 0) + phig2 * ATAp(i, 1) + phig3 * ATAp(i, 2));
                }
                gamma = gamma / pATAp;

                // update p
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

                // update g and p
                Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_v(const_cast<double*>(x_k1.data()), 1, 3*N);
                Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_res(const_cast<double*>(g.data()), 1, 3*N);
                eigen_res = eigen_v*eigen_mat.transpose()*eigen_mat + 2 * eigen_v * (reg_l2 + 1.0 / (2.0 * nu));
#pragma omp parallel for
                for (int i = 0; i < N; ++i) {
                    for (int jj = 0; jj < 3; ++jj) {
                        g(i, jj) += - ATb_rs(i, jj);
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

            // update g and p
            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_v(const_cast<double*>(x_k1.data()), 1, 3*N);
            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_res(const_cast<double*>(g.data()), 1, 3*N);
            eigen_res = eigen_v*eigen_mat.transpose()*eigen_mat + 2 * eigen_v * (reg_l2 + 1.0 / (2.0 * nu));
#pragma omp parallel for
            for (int i = 0; i < N; ++i) {
                for (int jj = 0; jj < 3; ++jj) {
                    g(i, jj) += - ATb_rs(i, jj);
                }
                std::tie(p(i, 0), p(i, 1), p(i, 2)) = phi_MwPGP(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2), g(i, 0), g(i, 1), g(i, 2), m_maxima(i));
            }
        }

	// fairly convoluted way to print every ~ max_iter / 20 iterations
        if (verbose && ((k % (int(max_iter / 5.0)) == 0) || k == 0 || k == max_iter - 1)) {
	    print_MwPGP(A_obj, b_obj, x_k1, m_proxy, m_maxima, m_history, objective_history, R2_history, print_iter, k, nu, reg_l0, reg_l1, reg_l2);
	    if (R2_history(print_iter) < min_fb) break;
            print_iter += 1;
	}

	// check if converged
	x_sum = 0;
#pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            for (int ii = 0; ii < 3; ++ii) {
                x_sum += abs(x_k1(i, ii) - x_k_prev(i, ii));
	    }
	}
	if (x_sum < epsilon) {
            printf("MwPGP algorithm ended early, at iteration %d\n", k);
	    break;
	}
    }
    return std::make_tuple(objective_history, R2_history, m_history, x_k1);
}


// fairly convoluted way to print every ~ K / nhistory iterations
void print_GPMO(int k, int ngrid, int& print_iter, Array& x, double* Aij_mj_ptr, Array& objective_history, Array& Bn_history, Array& m_history, double mmax_sum, double* normal_norms_ptr) 
{	
    int N = x.shape(0);
    double sqrtR2 = 0.0;
    double R2 = 0.0;
    double L2 = mmax_sum;
#pragma omp parallel for schedule(static) reduction(+: R2, sqrtR2)
    for(int i = 0; i < ngrid; ++i) {
	R2 += Aij_mj_ptr[i] * Aij_mj_ptr[i];
	sqrtR2 += abs(Aij_mj_ptr[i]) * sqrt(normal_norms_ptr[i]);
    }
    R2 = 0.5 * R2;
    objective_history(print_iter) = R2;
    Bn_history(print_iter) = sqrtR2 / sqrt(ngrid);
#pragma omp parallel for schedule(static) 
    for (int i = 0; i < N; ++i) {
	for (int ii = 0; ii < 3; ++ii) {
	    m_history(i, ii, print_iter) = x(i, ii);
    	}
    }
    printf("%d ... %.2e ... %.2e \n", k, R2, L2);
    print_iter += 1;
    return;
}

// compute which dipoles are directly adjacent to every dipole
Array connectivity_matrix(Array& dipole_grid_xyz, int Nadjacent)
{
    int Ndipole = dipole_grid_xyz.shape(0);
    Array connectivity_inds = xt::zeros<int>({Ndipole, 2000});
    
    // Compute distances between dipole j and all other dipoles
    // By default computes distance between dipole j and itself
#pragma omp parallel for schedule(static)
    for (int j = 0; j < Ndipole; ++j) {
	vector<double> dist_ij(Ndipole, 1e10);
        for (int i = 0; i < Ndipole; ++i) {
	     dist_ij[i] = sqrt((dipole_grid_xyz(i, 0) - dipole_grid_xyz(j, 0)) * (dipole_grid_xyz(i, 0) - dipole_grid_xyz(j, 0)) + (dipole_grid_xyz(i, 1) - dipole_grid_xyz(j, 1)) * (dipole_grid_xyz(i, 1) - dipole_grid_xyz(j, 1)) + (dipole_grid_xyz(i, 2) - dipole_grid_xyz(j, 2)) * (dipole_grid_xyz(i, 2) - dipole_grid_xyz(j, 2)));
	}
        for (int k = 0; k < 2000; ++k) {
	    auto result = std::min_element(dist_ij.begin(), dist_ij.end());
            int dist_ind = std::distance(dist_ij.begin(), result);
	    connectivity_inds(j, k) = dist_ind;
	    //printf("%d %d %d %f %d\n", j, k, Nadjacent, dist_ij[dist_ind], dist_ind);
            dist_ij[dist_ind] = 1e10; // eliminate the min to get the next min
	}
    }
    return connectivity_inds;
}

// GPMO algorithm with backtracking to fix wyrms -- close cancellations between
// two nearby, oppositely oriented magnets. 
std::tuple<Array, Array, Array, Array, Array> GPMO_backtracking(Array& A_obj, Array& b_obj, Array& mmax, Array& normal_norms, int K, bool verbose, int nhistory, int backtracking, Array& dipole_grid_xyz, int single_direction, int Nadjacent, int max_nMagnets)
{
    int ngrid = A_obj.shape(1);
    int N = int(A_obj.shape(0) / 3);
    int N3 = 3 * N;
    int print_iter = 0;
    int skj_inds;

    Array x = xt::zeros<double>({N, 3});

    // record the history of the algorithm iterations
    Array m_history = xt::zeros<double>({N, 3, nhistory + 1});
    Array objective_history = xt::zeros<double>({nhistory + 1});
    Array Bn_history = xt::zeros<double>({nhistory + 1});

    // print out the names of the error columns
    if (verbose)
        printf("Iteration ... |Am - b|^2 ... lam*|m|^2\n");

    // initialize Gamma_complement with all indices available
    Array Gamma_complement = xt::ones<bool>({N, 3});

    // initialize least-square values to large numbers
    vector<double> R2s(6 * N, 1e50);

    vector<int> skj(K);
    vector<int> skjj(K);
    vector<int> skjj_ind(N);
    vector<double> sign_fac(K);
    vector<double> sk_sign_fac(N);

    double* R2s_ptr = &(R2s[0]);
    double* Aij_ptr = &(A_obj(0, 0));
    double* Gamma_ptr = &(Gamma_complement(0, 0));

    // initialize running matrix-vector product
    Array Aij_mj_sum = -b_obj;
    double* Aij_mj_ptr = &(Aij_mj_sum(0));
    double* normal_norms_ptr = &(normal_norms(0));
    double mmax_sum = 0.0;
    double* mmax_ptr = &(mmax(0));
    
    // get indices for dipoles that are adjacent to dipole j
    Array Connect = connectivity_matrix(dipole_grid_xyz, Nadjacent);

    // if using a single direction, increase j by 3 each iteration
    int j_update = 1;
    if (single_direction >= 0) j_update = 3;
    Array num_nonzeros = xt::zeros<int>({nhistory + 1});
    int num_nonzero = 0;
    int k = 0;

    // Main loop over the optimization iterations
    for (int k = 0; k < K; ++k) {
#pragma omp parallel for schedule(static)
	for (int j = std::max(0, single_direction); j < N3; j += j_update) {

	    // Check all the allowed dipole positions
	    if (Gamma_ptr[j]) {
		double R2 = 0.0;
		double R2minus = 0.0;
		int nj = ngrid * j;

		// Compute contribution of jth dipole component, either with +- orientation
		for(int i = 0; i < ngrid; ++i) {
		    R2 += (Aij_mj_ptr[i] + Aij_ptr[i + nj]) * (Aij_mj_ptr[i] + Aij_ptr[i + nj]); 
		    R2minus += (Aij_mj_ptr[i] - Aij_ptr[i + nj]) * (Aij_mj_ptr[i] - Aij_ptr[i + nj]); 
		}
		R2s_ptr[j] = R2 + (mmax_ptr[j] * mmax_ptr[j]);
		R2s_ptr[j + N3] = R2minus + (mmax_ptr[j] * mmax_ptr[j]);
	    }
	}

	// find the dipole that most minimizes the least-squares term
        skj[k] = int(std::distance(R2s.begin(), std::min_element(R2s.begin(), R2s.end())));
	if (skj[k] >= N3) {
	    skj[k] -= N3;
	    sign_fac[k] = -1.0;
	    skjj[k] = (skj[k] % 3);
	    skj[k] = int(skj[k] / 3.0);
	    sk_sign_fac[skj[k]] = -1.0;
	}
	else {
            sign_fac[k] = 1.0;
	    skjj[k] = (skj[k] % 3);
	    skj[k] = int(skj[k] / 3.0);
	    sk_sign_fac[skj[k]] = 1.0;
	}
	mmax_sum += mmax_ptr[skj[k]] * mmax_ptr[skj[k]];
	skjj_ind[skj[k]] = skjj[k];
        x(skj[k], skjj[k]) = sign_fac[k];

	// Add binary magnet and get rid of the magnet (all three components)
        // from the complement of Gamma
	skj_inds = (3 * skj[k] + skjj[k]) * ngrid;
#pragma omp parallel for schedule(static)
	for(int i = 0; i < ngrid; ++i) {
            Aij_mj_ptr[i] += sign_fac[k] * Aij_ptr[i + skj_inds];
	}
        for (int j = 0; j < 3; ++j) {
            Gamma_complement(skj[k], j) = false;
	    R2s[3 * skj[k] + j] = 1e50;
	    R2s[N3 + 3 * skj[k] + j] = 1e50;
	}

	// backtrack by removing adjacent dipoles that are equal and opposite
	if ((k >= backtracking) and ((k % backtracking) == 0)) {
	    // Loop over all dipoles placed so far
            int wyrm_sum = 0;
	    for (int j = 0; j < k; j++) {
		int jk = skj[j];
		// find adjacent dipoles to dipole at skj[j]
		// Loop over adjacent dipoles and check if have equal and opposite one
	        for(int jj = 0; jj < Nadjacent; ++jj) {
	            int cj = Connect(jk, jj);
		    // Check for nonzero dipole at skj[j] and 
		    // has adjacent dipole that is oppositely oriented
		    if ((sk_sign_fac[jk] != 0.0) && (sk_sign_fac[jk] == (- sk_sign_fac[cj])) && (skjj_ind[jk] == skjj_ind[cj])) {
			 // kill off this pair
			 x(jk, skjj_ind[jk]) = 0.0;
	                 x(cj, skjj_ind[cj]) = 0.0;

			 // Make the pair + components viable options for future optimization
                         for (int jjj = 0; jjj < 3; ++jjj) {
			     Gamma_complement(jk, jjj) = true;
			     Gamma_complement(cj, jjj) = true;
		         }

	                 // Subtract off this pair's contribution to Aij * mj
			 int skj_ind1 = (3 * jk + skjj_ind[jk]) * ngrid;
	                 int skj_ind2 = (3 * cj + skjj_ind[cj]) * ngrid;
#pragma omp parallel for schedule(static)
			 for(int i = 0; i < ngrid; ++i) {
		             Aij_mj_ptr[i] -= sk_sign_fac[jk] * Aij_ptr[i + skj_ind1] + sk_sign_fac[cj] * Aij_ptr[i + skj_ind2];
			 }
	                 mmax_sum -= mmax_ptr[jk] * mmax_ptr[jk];
	                 mmax_sum -= mmax_ptr[cj] * mmax_ptr[cj];
			 // set sign_fac = 0 so that these magnets do not keep getting dewyrmed
			 sk_sign_fac[jk] = 0.0;
			 sk_sign_fac[cj] = 0.0;
			 wyrm_sum += 1;
			 break;
		    }
	        }
	    }
	    printf("%d wyrms removed out of %d possible dipoles\n", wyrm_sum, backtracking);
        }

	// check range here
	num_nonzero = 0;
#pragma omp parallel for schedule(static) reduction(+: num_nonzero)
	for (int j = 0; j < N; ++j) { 
	    for (int jj = 0; jj < 3; ++jj) { 
		if (not Gamma_complement(j, jj)) {
                    num_nonzero += 1; 
		    break; // avoid counting multiple components by breaking inner loop
		} 
	    }            
	}

	if (verbose && (((k % int(K / nhistory)) == 0) || k == 0 || k == K - 1)) {
            print_GPMO(k, ngrid, print_iter, x, Aij_mj_ptr, objective_history, Bn_history, m_history, mmax_sum, normal_norms_ptr);
	    printf("Iteration = %d, Number of nonzero dipoles = %d\n", k, num_nonzero);

            // if stuck at some number of dipoles, break out of the loop
            num_nonzeros(print_iter-1) = num_nonzero;
            if (print_iter > 10 
                && num_nonzeros(print_iter) == num_nonzeros(print_iter - 1) 
                && num_nonzeros(print_iter) == num_nonzeros(print_iter - 2)) {

                printf("Stopping iterations: number of nonzero dipoles "
                       "unchanged over three backtracking cycles");
                break;
            }
	
	}

	// Terminate iterations if a magnet limit is reached
        if ((num_nonzero >= N) || (num_nonzero >= max_nMagnets)) {

            print_GPMO(k, ngrid, print_iter, x, Aij_mj_ptr, objective_history, 
                Bn_history, m_history, mmax_sum, normal_norms_ptr);
	    printf("Iteration = %d, Number of nonzero dipoles = %d\n", k, 
	        num_nonzero);
	    
            if (num_nonzero >= N) {
                printf("Stopping iterations: all dipoles in grid "
                       "are populated");
            }
            else if (num_nonzero >= max_nMagnets) {
                printf("Stopping iterations: maximum number of nonzero "
                       "magnets reached ");
            }

            break;
        }
    }

    return std::make_tuple(objective_history, Bn_history, m_history, num_nonzeros, x);
}

// Run the GPMO algorithm, placing a dipole and all of the closest Nadjacent dipoles down
// all at once each iteration. All of these dipoles are aligned in the same way by assumption 
std::tuple<Array, Array, Array, Array> GPMO_multi(Array& A_obj, Array& b_obj, Array& mmax, Array& normal_norms, int K, bool verbose, int nhistory, Array& dipole_grid_xyz, int single_direction, int Nadjacent)
{
    int ngrid = A_obj.shape(1);
    int N = int(A_obj.shape(0) / 3);
    int N3 = 3 * N;
    int print_iter = 0;

    Array x = xt::zeros<double>({N, 3});

    // record the history of the algorithm iterations
    Array m_history = xt::zeros<double>({N, 3, nhistory + 1});
    Array objective_history = xt::zeros<double>({nhistory + 1});
    Array Bn_history = xt::zeros<double>({nhistory + 1});

    // print out the names of the error columns
    if (verbose)
        printf("Iteration ... |Am - b|^2 ... lam*|m|^2\n");

    // initialize Gamma_complement with all indices available
    Array Gamma_complement = xt::ones<bool>({N, 3});
	
    // initialize least-square values to large numbers    
    vector<double> R2s(6 * N, 1e50);
    vector<int> skj(K);
    vector<int> skjj(K);
    vector<double> sign_fac(K);
    
    double* R2s_ptr = &(R2s[0]);
    double* Aij_ptr = &(A_obj(0, 0));
    double* Gamma_ptr = &(Gamma_complement(0, 0));
    
    // initialize running matrix-vector product
    Array Aij_mj_sum = -b_obj;
    double* Aij_mj_ptr = &(Aij_mj_sum(0));
    double* normal_norms_ptr = &(normal_norms(0));
    double mmax_sum = 0.0;
    double* mmax_ptr = &(mmax(0));
    
    // get indices for dipoles that are adjacent to dipole j
    Array Connect = connectivity_matrix(dipole_grid_xyz, Nadjacent);
    
    // if using a single direction, increase j by 3 each iteration
    int j_update = 1;
    if (single_direction >= 0) j_update = 3;
    
    // Main loop over the optimization iterations
    for (int k = 0; k < K; ++k) {
#pragma omp parallel for schedule(static)
	for (int j = std::max(0, single_direction); j < N3; j += j_update) {
	    // Check all the allowed dipole positions
	    if (Gamma_ptr[j]) {
		int j_ind = int(j / 3);
		double mmax_partial_sum = 0.0;
		double R2 = 0.0;
		double R2minus = 0.0;
		int nj = ngrid * j;

		// Compute contribution of jth dipole component, with +- orientation
		// as well as contributions of all the closest AVAILABLE
		// Nadjacent dipoles, assuming the same orientation
		int cj_counter = 0;
		for (int jj = 0; jj < Nadjacent; ++jj) {
		    int cj = Connect(j_ind, jj); 
		    int cj_ind = 3 * cj + (j % 3);
		    // if this neighbor dipole location is already filled,
		    // look for the next closest neighbor
		    while (not Gamma_ptr[cj_ind]) {
                        cj = Connect(j_ind, Nadjacent + cj_counter);
		        cj_ind = 3 * cj + (j % 3);	
		        cj_counter += 1;
		    }
		    nj = ngrid * cj_ind; // index j and all its neighbors
	    
	    	    // Compute contribution of jth dipole component, either with +- orientation
		    for(int i = 0; i < ngrid; ++i) {
		        R2 += (Aij_mj_ptr[i] + Aij_ptr[i + nj]) * (Aij_mj_ptr[i] + Aij_ptr[i + nj]);
		        R2minus += (Aij_mj_ptr[i] - Aij_ptr[i + nj]) * (Aij_mj_ptr[i] - Aij_ptr[i + nj]); 
		    }
		    mmax_partial_sum += mmax_ptr[cj] * mmax_ptr[cj];
		}
		R2s_ptr[j] = R2 + mmax_partial_sum; 
		R2s_ptr[j + N3] = R2minus + mmax_partial_sum; 
	    }
	}

	// find the dipole (and neighbors) that most minimizes the least-squares term
        skj[k] = int(std::distance(R2s.begin(), std::min_element(R2s.begin(), R2s.end())));
	if (skj[k] >= N3) {
	    skj[k] -= N3;
	    sign_fac[k] = -1.0;
	}
	else {
            sign_fac[k] = 1.0;
	}
	skjj[k] = (skj[k] % 3); 
	skj[k] = int(skj[k] / 3.0);
	// printf("%d\n", skj[k]);

	// Add binary magnets and get rid of the neighboring
	// magnets (all three components) from Gamma_complement
	int cj_counter = 0;
	for (int jj = 0; jj < Nadjacent; ++jj) {
	    int cj = Connect(skj[k], jj); 
	    // printf("%d %d %d %d \n", k, skj[k], jj, cj);
	    int cj_ind = 3 * cj + skjj[k];
	    // if neighbor dipole is already full, look
	    // for the next closest neighbor 
	    while (not Gamma_ptr[cj_ind]) {
		cj = Connect(skj[k], Nadjacent + cj_counter);
		cj_ind = 3 * cj + skjj[k];	
		cj_counter += 1;
	    }
	    x(cj, skjj[k]) = sign_fac[k];	
	    mmax_sum += mmax_ptr[cj] * mmax_ptr[cj];
	    int skj_inds = cj_ind * ngrid;
	    for(int i = 0; i < ngrid; ++i) {
                Aij_mj_ptr[i] += sign_fac[k] * Aij_ptr[i + skj_inds];
	    }
            for (int j = 0; j < 3; ++j) {
                Gamma_complement(cj, j) = false;
	        R2s[3 * cj + j] = 1e50;
	        R2s[N3 + 3 * cj + j] = 1e50;
	    }
	}
	if (verbose && (((k % int(K / nhistory)) == 0) || k == 0 || k == K - 1)) {
            print_GPMO(k, ngrid, print_iter, x, Aij_mj_ptr, objective_history, Bn_history, m_history, mmax_sum, normal_norms_ptr);
	}
    }
    return std::make_tuple(objective_history, Bn_history, m_history, x);
}

// Variant of the GPMO algorithm for solving the permanent magnet optimization 
// problem in which the user has the option to specify arbitrary allowable 
// polarization vectors for each dipole. 
// 
// For the backtracking approach, pairs of nearby magnets will be eliminated
// if the angle between their dipole moments meets or exceeds a certain  
// threshold value specified by the user as thresh_angle (in radians). 
//
// The input K specifies the maximum number of iterations to perform. The 
// algorithm will terminate before K iterations are reached if:
// (1) the number of magnets placed reaches max_nMagnets, or 
// (2) the net increase in magnet count following a backtracking step is zero.
//
// The A matrix should be rescaled by m_maxima since we are assuming all ones 
// in m.
std::tuple<Array, Array, Array, Array, Array> GPMO_ArbVec_backtracking(
    Array& A_obj, Array& b_obj, Array& mmax, Array& normal_norms, 
    Array& pol_vectors, int K, bool verbose, int nhistory, int backtracking, 
    Array& dipole_grid_xyz, int Nadjacent, double thresh_angle, 
    int max_nMagnets, Array& x_init)
{
    int ngrid = A_obj.shape(1);
    int nPolVecs = pol_vectors.shape(1);
    int N = int(A_obj.shape(0) / 3);
    int N3 = 3 * N;
    int NNp = nPolVecs * N;
    int print_iter = 0;
    double cos_thresh_angle = cos(thresh_angle);

    Array x = xt::zeros<double>({N, 3});
    vector<int> x_vec(N);
    vector<int> x_sign(N);

    // record the history of the algorithm iterations
    Array m_history = xt::zeros<double>({N, 3, nhistory + 2});
    Array objective_history = xt::zeros<double>({nhistory + 2});
    Array Bn_history = xt::zeros<double>({nhistory + 2});

    // print out the names of the error columns
    if (verbose)
        printf("Running the GPMO backtracking algorithm with arbitrary "
               "polarization vectors\n");
        printf("Iteration ... |Am - b|^2 ... lam*|m|^2\n");

    // initialize Gamma_complement with all indices available
    Array Gamma_complement = xt::ones<bool>({N});
	
    // initialize least-square values to large numbers    
    vector<double> R2s(2*NNp, 1e50);
    vector<int> skj(K);
    vector<int> skjj(K);
    vector<double> sign_fac(K);
    
    double* R2s_ptr = &(R2s[0]);
    double* Aij_ptr = &(A_obj(0, 0));
    double* Gamma_ptr = &(Gamma_complement(0));
    double* pol_vec_ptr = &(pol_vectors(0,0,0));
    
    // initialize running matrix-vector product
    Array Aij_mj_sum = -b_obj;
    double mmax_sum = 0.0;
    double* Aij_mj_ptr = &(Aij_mj_sum(0));
    double* normal_norms_ptr = &(normal_norms(0));
    double* mmax_ptr = &(mmax(0));

    // Get indices for dipoles that are adjacent to dipole j
    Array Connect = connectivity_matrix(dipole_grid_xyz, Nadjacent);

    int num_nonzero = 0;
    Array num_nonzeros = xt::zeros<int>({nhistory + 2});

    // Initialize the solution according to user input
    initialize_GPMO_ArbVec(x_init, pol_vectors, x, x_vec, x_sign, 
        A_obj, Aij_mj_sum, R2s, Gamma_complement, num_nonzero);
    num_nonzeros(0) = num_nonzero;

    // Save a record of the magnet array as initialized
    print_GPMO(0, ngrid, print_iter, x, Aij_mj_ptr, objective_history, 
        Bn_history, m_history, mmax_sum, normal_norms_ptr);

    // Main loop over the optimization iterations
    for (int k = 0; k < K; ++k) {

#pragma omp parallel for schedule(static)
	for (int j = 0; j < N; j += 1) {

	    // Check all the allowed dipole positions
	    if (Gamma_ptr[j]) {

                for (int m = 0; m < nPolVecs; m++) {

                    int mj = j*nPolVecs + m;
		    double R2 = 0.0;
                    double R2minus = 0.0;

                    // Contribution to the normal field from the mth 
                    // allowable polarization vector
                    for(int i = 0; i < ngrid; ++i) {
                        double bnorm = 0.0;
                        for (int l = 0; l < 3; ++l) {
                            int nj3 = ngrid * (3*j + l);
                            double pol_vec_l = pol_vec_ptr[l+3*(nPolVecs*j+m)];
                            bnorm += pol_vec_l * Aij_ptr[i+nj3];
                        }
                        R2 += (Aij_mj_ptr[i] + bnorm) * (Aij_mj_ptr[i] + bnorm);
                        R2minus += (Aij_mj_ptr[i] - bnorm) * (Aij_mj_ptr[i] - bnorm);
                    }
		    R2s_ptr[mj] = R2 + (mmax_ptr[j] * mmax_ptr[j]);
                    R2s_ptr[mj + NNp] = R2minus + (mmax_ptr[j] * mmax_ptr[j]);
		}
	    }
	}

	// find the dipole that most minimizes the least-squares term
        skj[k] = int(std::distance(R2s.begin(), std::min_element(R2s.begin(), 
                     R2s.end())));
	if (skj[k] >= NNp) {
	    skj[k] -= NNp;
	    sign_fac[k] = -1.0;
	}
	else {
            sign_fac[k] = 1.0;
	}
	skjj[k] = (skj[k] % nPolVecs); 
	skj[k] = int(skj[k] / (double) nPolVecs);
        x_vec[skj[k]] = skjj[k];
        x_sign[skj[k]] = sign_fac[k];

	// Add binary magnet and get rid of the magnet (all three components)
        // from the complement of Gamma 
        for (int l = 0; l < 3; ++l) {
            int pol_ind = l + 3 * (nPolVecs * skj[k] + skjj[k]);
	    int skj_inds = (3 * skj[k] + l) * ngrid;
            x(skj[k], l) = sign_fac[k] * pol_vec_ptr[pol_ind];
#pragma omp parallel for schedule(static)
	    for(int i = 0; i < ngrid; ++i) {
                Aij_mj_ptr[i] += sign_fac[k] * pol_vec_ptr[pol_ind] * Aij_ptr[i + skj_inds];
            }
	}
        Gamma_complement(skj[k]) = false;
        for (int m = 0; m < nPolVecs; ++m) {
	    R2s[skj[k]*nPolVecs + m] = 1e50;
	    R2s[NNp + skj[k]*nPolVecs + m] = 1e50;
        }
        num_nonzero += 1;

        // Backtrack by removing adjacent dipoles that are equal and opposite
        if ((k % backtracking) == 0) {

            int wyrm_sum = 0;

            // Loop over all dipoles
            for (int j = 0; j < N; j++) {

                // Skip if dipole has already been removed
                if (Gamma_complement(j)) continue;

                int m = x_vec[j];

                // Loop over adjacent dipoles and check if a nearby one exceeds
                // the maximum allowable angle difference
                double min_cos_angle = 2.0; // initialize > max possible value
                int cj_min;
                for (int jj = 0; jj < Nadjacent; ++jj) {

                    int cj = Connect(j, jj);

                    // Skip if dipole has not been placed
                    if (Gamma_complement(cj)) continue;

                    // Evaluate angle between moments; save if greatest so far 
                    double cos_angle = 0;
                    for (int l = 0; l < 3; ++l) {
                        cos_angle += x(j, l) * x(cj, l);
                    }
                    if (cos_angle < min_cos_angle) {
                        min_cos_angle = cos_angle;
                        cj_min = cj;
                    }

                }

                // If angle between dipole j and the nearby magnet with the 
                // max angle difference the threshold, eliminate the pair
                if (min_cos_angle <= cos_thresh_angle) {

                    int cm_min = x_vec[cj_min];

                    // Subtract the pair's contribution to Aij * mj
                    #pragma omp parallel for schedule(static)
                    for (int i = 0; i < ngrid; ++i) {
                        for (int l = 0; l < 3; ++l) {
                            int A_ind_k = ngrid * (3*j      + l);
                            int A_ind_c = ngrid * (3*cj_min + l);
                            int pol_ind_k = l + 3*(j*nPolVecs      + m);
                            int pol_ind_c = l + 3*(cj_min*nPolVecs + cm_min);
                            Aij_mj_ptr[i] -= 
                                x_sign[j] * pol_vec_ptr[pol_ind_k] 
                                          * Aij_ptr[i + A_ind_k]
                              + x_sign[cj_min] * pol_vec_ptr[pol_ind_c]
                                               * Aij_ptr[i + A_ind_c];
                        }
                    }

                    // Reset the solution vectors
                    for (int l = 0; l < 3; ++l) {
                        x(j, l) = 0.0;
                        x(cj_min, l) = 0.0;
                    }
                    x_vec[j] = 0;
                    x_vec[cj_min] = 0;
                    x_sign[j] = 0;
                    x_sign[cj_min] = 0;

                    // Indicate that the pair is now available
                    Gamma_complement(j) = true;
                    Gamma_complement(cj_min) = true;

                    // Adjust running totals
                    num_nonzero -= 2;
                    wyrm_sum += 1;

                }
            }
            printf("Backtracking: %d wyrms removed\n", wyrm_sum);
        }

	if (verbose && (((k % int(K / nhistory)) == 0) || k == 0 || k == K - 1)) {
            print_GPMO(k, ngrid, print_iter, x, Aij_mj_ptr, objective_history, 
                       Bn_history, m_history, mmax_sum, normal_norms_ptr);
	    printf("Iteration = %d, Number of nonzero dipoles = %d\n", 
                   k, num_nonzero);

	    // add small amount to the thresh_angle each time we print
            //thresh_angle = thresh_angle + M_PI / 720.0;
            
            // if stuck at some number of dipoles, break out of the loop
            num_nonzeros(print_iter-1) = num_nonzero;
            if (print_iter > 10 
                && num_nonzeros(print_iter) == num_nonzeros(print_iter - 1) 
                && num_nonzeros(print_iter) == num_nonzeros(print_iter - 2)) {

                printf("Stopping iterations: number of nonzero dipoles "
                       "unchanged over three backtracking cycles");
                break;
            }

	}

	// Terminate iterations if a magnet limit is reached
        if ((num_nonzero >= N) || (num_nonzero >= max_nMagnets)) {

            print_GPMO(k, ngrid, print_iter, x, Aij_mj_ptr, objective_history, 
                       Bn_history, m_history, mmax_sum, normal_norms_ptr);
	    printf("Iteration = %d, Number of nonzero dipoles = %d\n", 
                   k, num_nonzero);

            if (num_nonzero >= N) {
                printf("Stopping iterations: all dipoles in grid "
                       "are populated");
            }
	    else if (num_nonzero >= max_nMagnets) {
                printf("Stopping iterations: maximum number of nonzero "
                       "magnets reached ");
            }

            break;
	}

    }

    return std::make_tuple(objective_history, Bn_history, m_history, 
                           num_nonzeros, x);
}

/*  
 *  Initializes the solution vector and related arrays according to a 
 *  user-input initial guess supplied to the GPMO algorithm with arbitrary
 *  vectors.
 */
void initialize_GPMO_ArbVec(Array& x_init, Array& pol_vectors, 
         Array& x, vector<int>& x_vec, vector<int>& x_sign, 
         Array& A_obj, Array& Aij_mj_sum, vector<double>& R2s, 
	 Array& Gamma_complement, int& num_nonzero) {

    // Ensure that size of initialization vector agrees with that of solution
    if (x_init.shape(1) != 3) {
        throw std::runtime_error("Second dimension of initialiation vector "
                  "`x_init` must be 3");
    }
    int N = x_init.shape(0);
    if (x.shape(0) != N) {
        throw std::runtime_error("Number of magnets in initialization array "
 	          " does not match the number of magnets\n in the solution "
		  " vector");
    }
    int nPolVecs = pol_vectors.shape(1);
    int NNp = N*nPolVecs;
    int ngrid = A_obj.shape(1);

    int n_initialized = 0;
    int n_OutOfTol = 0;
    double tol = (double) 4*std::numeric_limits<float>::epsilon();

    double* Aij_ptr = &(A_obj(0, 0));
    double* Aij_mj_ptr = &(Aij_mj_sum(0));
    double* pol_vec_ptr = &(pol_vectors(0,0,0));

    num_nonzero = 0;
    for (int j = 0; j < N; j++) {

        // Do not change solution arrays if dipole is initialized to zero
        if (x_init(j,0) == 0 && x_init(j,1) == 0 && x_init(j,2) == 0) {
	    continue;
        }

        n_initialized++;

	// Otherwise, find the allowable polarization vector and sign that 
	// best match the initial guess for the dipole moment
        double min_sqDiff = std::numeric_limits<double>::infinity();
	int m_min;
	int sign_min;
        for (int m = 0; m < nPolVecs; m++) {
            double sqDiffPos = 0;
            double sqDiffNeg = 0;
            for (int l = 0; l < 3; l++) {
                sqDiffPos += pow(x_init(j,l) - pol_vectors(j,m,l), 2);
                sqDiffNeg += pow(x_init(j,l) + pol_vectors(j,m,l), 2);
	    }
            if (sqDiffPos < min_sqDiff) { 
                min_sqDiff = sqDiffPos; 
		m_min = m;
                sign_min = 1;
	    }
	    if (sqDiffNeg < min_sqDiff) { 
		min_sqDiff = sqDiffNeg; 
		m_min = m;
		sign_min = -1;
	    }
	}

	// Check if the initialized vector is closer to zero than any of the
	// allowable dipole moments
	double sqDiffNull = 0;
	for (int l = 0; l < 3; l++) {
	    sqDiffNull += pow(x_init(j,l), 2);
	}
	if (sqDiffNull < min_sqDiff) {
	    min_sqDiff = sqDiffNull;
	    m_min = 0;
	    sign_min = 0;
	}

	// Update solution vector and associated arrays to the nearest
	// allowable dipole moment vector (if nonzero)
	if (sign_min != 0) {

            num_nonzero++;

            // Solution vector and metadata about type ID and sign
    	    for (int l = 0; l < 3; l++) {
    	        x(j,l) = sign_min * pol_vectors(j,m_min,l);
            }
	    x_vec[j] = m_min;
	    x_sign[j] = sign_min;
            Gamma_complement(j) = false;

            // Running totals for normal field at test points on plasma
            for (int l = 0; l < 3; l++) {
                int pol_ind = l + 3 * (nPolVecs * j + m_min);
                int j_ind = (3 * j + l) * ngrid;
                #pragma omp parallel for schedule(static)
                for(int i = 0; i < ngrid; ++i) {
                    Aij_mj_ptr[i] += 
                        sign_min * pol_vec_ptr[pol_ind] * Aij_ptr[i + j_ind];
                }
            }

            // Assign high values to contributions of the magnet to R2
            for (int m = 0; m < nPolVecs; m++) {
	        R2s[j*nPolVecs + m] = 1e50;
	        R2s[NNp + j*nPolVecs + m] = 1e50;
            }
	}

	// Check if the discrepancy between the initialization vector and the
	// assigned vector exceeds the tolerance for single-precision floats
	for (int l = 0; l < 3; l++) {
	    if (abs(x_init(j,l) - x(j,l)) > tol) {
	        n_OutOfTol++;
		break;
	    }
	}
    }

    if (n_OutOfTol != 0) {
        printf("    WARNING: %d of %d dipoles in the initialization vector "
               "disagree \n    with the allowable dipole moments to single "
	       "floating precision. These \n    dipoles will be initialized "
	       "to the nearest allowable moments or zero.\n", 
	       n_OutOfTol, n_initialized);
    }
}

// Variant of the GPMO algorithm for solving the permanent magnet optimization
// problem in which the user has the option to specify arbitrary allowable 
// polarization vectors for each dipole. The A matrix should be rescaled by 
// m_maxima since we are assuming all ones in m.
std::tuple<Array, Array, Array, Array> GPMO_ArbVec(Array& A_obj, Array& b_obj, Array& mmax, Array& normal_norms, Array& pol_vectors, int K, bool verbose, int nhistory)
{
    int ngrid = A_obj.shape(1);
    int nPolVecs = pol_vectors.shape(1);
    int N = int(A_obj.shape(0) / 3);
    int N3 = 3 * N;
    int NNp = nPolVecs * N;
    int print_iter = 0;

    Array x = xt::zeros<double>({N, 3});

    // record the history of the algorithm iterations
    Array m_history = xt::zeros<double>({N, 3, nhistory + 1});
    Array objective_history = xt::zeros<double>({nhistory + 1});
    Array Bn_history = xt::zeros<double>({nhistory + 1});

    // print out the names of the error columns
    if (verbose)
        printf("Running the GPMO baseline algorithm with arbitrary "
               "polarization vectors\n");
        printf("Iteration ... |Am - b|^2 ... lam*|m|^2\n");

    // initialize Gamma_complement with all indices available
    Array Gamma_complement = xt::ones<bool>({N});
	
    // initialize least-square values to large numbers    
    vector<double> R2s(2*NNp, 1e50);
    vector<int> skj(K);
    vector<int> skjj(K);
    vector<double> sign_fac(K);
    
    double* R2s_ptr = &(R2s[0]);
    double* Aij_ptr = &(A_obj(0, 0));
    double* Gamma_ptr = &(Gamma_complement(0));
    double* pol_vec_ptr = &(pol_vectors(0,0,0));
    
    // initialize running matrix-vector product
    Array Aij_mj_sum = -b_obj;
    double mmax_sum = 0.0;
    double* Aij_mj_ptr = &(Aij_mj_sum(0));
    double* normal_norms_ptr = &(normal_norms(0));
    double* mmax_ptr = &(mmax(0));

    // Main loop over the optimization iterations
    for (int k = 0; k < K; ++k) {
#pragma omp parallel for schedule(static)
	for (int j = 0; j < N; j += 1) {

	    // Check all the allowed dipole positions
	    if (Gamma_ptr[j]) {

                for (int m = 0; m < nPolVecs; m++) {

                    int mj = j*nPolVecs + m;
		    double R2 = 0.0;
                    double R2minus = 0.0;

                    // Contribution to the normal field from the mth 
                    // allowable polarization vector
                    for(int i = 0; i < ngrid; ++i) {
                        double bnorm = 0.0;
                        for (int l = 0; l < 3; ++l) {
                            int nj3 = ngrid * (3*j + l);
                            double pol_vec_l = pol_vec_ptr[l+3*(nPolVecs*j+m)];
                            bnorm += pol_vec_l * Aij_ptr[i+nj3];
                        }
                        R2 += (Aij_mj_ptr[i] + bnorm) * (Aij_mj_ptr[i] + bnorm);
                        R2minus += (Aij_mj_ptr[i] - bnorm) * (Aij_mj_ptr[i] - bnorm);
                    }
		    R2s_ptr[mj] = R2 + (mmax_ptr[j] * mmax_ptr[j]);
                    R2s_ptr[mj + NNp] = R2minus + (mmax_ptr[j] * mmax_ptr[j]);
		}
	    }
	}

	// find the dipole that most minimizes the least-squares term
        skj[k] = int(std::distance(R2s.begin(), std::min_element(R2s.begin(), R2s.end())));
	if (skj[k] >= NNp) {
	    skj[k] -= NNp;
	    sign_fac[k] = -1.0;
	}
	else {
            sign_fac[k] = 1.0;
	}
	skjj[k] = (skj[k] % nPolVecs); 
	skj[k] = int(skj[k] / (double) nPolVecs);

	// Add binary magnet and get rid of the magnet (all three components)
        // from the complement of Gamma 
        for (int l = 0; l < 3; ++l) {
            int pol_ind = l + 3 * (nPolVecs * skj[k] + skjj[k]);
	    int skj_inds = (3 * skj[k] + l) * ngrid;
            x(skj[k], l) = sign_fac[k] * pol_vec_ptr[pol_ind];
#pragma omp parallel for schedule(static)
	    for(int i = 0; i < ngrid; ++i) {
                Aij_mj_ptr[i] += sign_fac[k] * pol_vec_ptr[pol_ind] * Aij_ptr[i + skj_inds];
            }
	}
        Gamma_complement(skj[k]) = false;
        for (int m = 0; m < nPolVecs; ++m) {
	    R2s[skj[k]*nPolVecs + m] = 1e50;
	    R2s[NNp + skj[k]*nPolVecs + m] = 1e50;
        }

	if (verbose && (((k % int(K / nhistory)) == 0) || k == 0 || k == K - 1)) {
            print_GPMO(k, ngrid, print_iter, x, Aij_mj_ptr, objective_history, Bn_history, m_history, mmax_sum, normal_norms_ptr);
	}
    }
    return std::make_tuple(objective_history, Bn_history, m_history, x);
}

// Run the GPMO algorithm for solving 
// the permanent magnet optimization problem.
// The A matrix should be rescaled by m_maxima since we are assuming all ones in m.
std::tuple<Array, Array, Array, Array> GPMO_baseline(Array& A_obj, Array& b_obj, Array& mmax, Array& normal_norms, int K, bool verbose, int nhistory, int single_direction) 
{
    int ngrid = A_obj.shape(1);
    int N = int(A_obj.shape(0) / 3);
    int N3 = 3 * N;
    int print_iter = 0;

    Array x = xt::zeros<double>({N, 3});

    // record the history of the algorithm iterations
    Array m_history = xt::zeros<double>({N, 3, nhistory + 1});
    Array objective_history = xt::zeros<double>({nhistory + 1});
    Array Bn_history = xt::zeros<double>({nhistory + 1});

    // print out the names of the error columns
    if (verbose)
        printf("Iteration ... |Am - b|^2 ... lam*|m|^2\n");

    // initialize Gamma_complement with all indices available
    Array Gamma_complement = xt::ones<bool>({N, 3});
	
    // initialize least-square values to large numbers    
    vector<double> R2s(6 * N, 1e50);
    vector<int> skj(K);
    vector<int> skjj(K);
    vector<double> sign_fac(K);
    
    double* R2s_ptr = &(R2s[0]);
    double* Aij_ptr = &(A_obj(0, 0));
    double* Gamma_ptr = &(Gamma_complement(0, 0));
    
    // initialize running matrix-vector product
    Array Aij_mj_sum = -b_obj;
    double mmax_sum = 0.0;
    double* Aij_mj_ptr = &(Aij_mj_sum(0));
    double* normal_norms_ptr = &(normal_norms(0));
    double* mmax_ptr = &(mmax(0));

    // if using a single direction, increase j by 3 each iteration
    int j_update = 1;
    if (single_direction >= 0) j_update = 3;
    
    // Main loop over the optimization iterations
    for (int k = 0; k < K; ++k) {
#pragma omp parallel for schedule(static)
	for (int j = std::max(0, single_direction); j < N3; j += j_update) {

	    // Check all the allowed dipole positions
	    if (Gamma_ptr[j]) {
		double R2 = 0.0;
		double R2minus = 0.0;
		int nj = ngrid * j;

		// Compute contribution of jth dipole component, either with +- orientation
		for(int i = 0; i < ngrid; ++i) {
		    R2 += (Aij_mj_ptr[i] + Aij_ptr[i + nj]) * (Aij_mj_ptr[i] + Aij_ptr[i + nj]);
		    R2minus += (Aij_mj_ptr[i] - Aij_ptr[i + nj]) * (Aij_mj_ptr[i] - Aij_ptr[i + nj]); 
		}
		R2s_ptr[j] = R2 + (mmax_ptr[j] * mmax_ptr[j]);
		R2s_ptr[j + N3] = R2minus + (mmax_ptr[j] * mmax_ptr[j]);
	    }
	}

	// find the dipole that most minimizes the least-squares term
        skj[k] = int(std::distance(R2s.begin(), std::min_element(R2s.begin(), R2s.end())));
	if (skj[k] >= N3) {
	    skj[k] -= N3;
	    sign_fac[k] = -1.0;
	}
	else {
            sign_fac[k] = 1.0;
	}
	skjj[k] = (skj[k] % 3); 
	skj[k] = int(skj[k] / 3.0);
        x(skj[k], skjj[k]) = sign_fac[k];

	// Add binary magnet and get rid of the magnet (all three components)
        // from the complement of Gamma 
	int skj_inds = (3 * skj[k] + skjj[k]) * ngrid;
#pragma omp parallel for schedule(static)
	for(int i = 0; i < ngrid; ++i) {
            Aij_mj_ptr[i] += sign_fac[k] * Aij_ptr[i + skj_inds];
	}
        for (int j = 0; j < 3; ++j) {
            Gamma_complement(skj[k], j) = false;
	    R2s[3 * skj[k] + j] = 1e50;
	    R2s[N3 + 3 * skj[k] + j] = 1e50;
        }

	if (verbose && (((k % int(K / nhistory)) == 0) || k == 0 || k == K - 1)) {
            print_GPMO(k, ngrid, print_iter, x, Aij_mj_ptr, objective_history, Bn_history, m_history, mmax_sum, normal_norms_ptr);
	}
    }
    return std::make_tuple(objective_history, Bn_history, m_history, x);
}
