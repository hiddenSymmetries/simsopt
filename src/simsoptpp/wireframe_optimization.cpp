#include "wireframe_optimization.h"
#include <limits> // std
#include <vector>
using std::vector;

/*
 *  Greedy Stellarator Coil Optimization (GSCO): optimizes the currents in a
 *  wireframe grid by adding loops of current one by one to minimize an 
 *  objective function.
 */
std::tuple<Array,IntArray,Array,Array,Array,Array> GSCO(
    Array& A_obj, Array& b_obj, double current, double max_current, 
    IntArray& cells, IntArray& free_cells, IntArray& cell_neighbors, 
    double lambda_P, int nIter, Array& x_init, IntArray& cell_count_init, 
    int nHistory){

    int nSegs = A_obj.shape(1);
    int nGrid = A_obj.shape(0);
    int nCells = cells.shape(0);

    // Keeps a running total of the normal field at each test point
    Array Ax_minus_b = -b_obj;

    double* A_ptr           = &(A_obj(0,0));
    double* Ax_minus_b_ptr  = &(Ax_minus_b(0));
    int* cells_ptr          = &(cells(0,0));
    int* cell_neighbors_ptr = &(cell_neighbors(0,0));

    double inf = std::numeric_limits<double>::max();
    vector<double> Chi2_Bs(2*nCells, inf);
    vector<double> Chi2_Ps(2*nCells, inf);
    vector<double> Chi2s(2*nCells, inf);
    double* Chi2_Bs_ptr = &(Chi2_Bs[0]);
    double* Chi2_Ps_ptr = &(Chi2_Ps[0]);
    double* Chi2s_ptr = &(Chi2s[0]);

    // Adjustment to number of history entries to save
    int extra = (nIter % nHistory == 0) ? 1 : 2;

    // Initialize the solution array
    Array x = xt::zeros<double>({nSegs,1});
    Array x_history = xt::zeros<double>({nSegs,nHistory+extra});
    for (int i = 0; i < nSegs; ++i) {
        x(i) = x_init(i);
        x_history(i,0) = x_init(i);
    }
    double* x_ptr = &(x(0,0));

    // Initialize the cell_count array
    IntArray cell_count = xt::zeros<int>({nCells});
    for (int i = 0; i < nCells; ++i) {
        cell_count(i) = cell_count_init(i);
    }
    int* cell_count_ptr = &(cell_count(0));

    Array Chi2_B_history = xt::zeros<double>({nHistory+extra});
    Array Chi2_P_history = xt::zeros<double>({nHistory+extra});
    Array Chi2_history = xt::zeros<double>({nHistory+extra});

    // Initial history values
    int hist_ind = 0;
    double Chi2_B_init = 0.0;
    for (int i = 0; i < nGrid; ++i) {
        Chi2_B_init += Ax_minus_b(i,0) * Ax_minus_b(i,0);
    }
    double tol = 0.001*current;
    double Chi2_P_init = compute_chi2_P(x, tol);
    double Chi2_P_i = Chi2_P_init;

    printf("  Beginning greedy iterations\n");
    printf("%11s %14s %14s %14s\n", "Iteration", "Chi2_B", "Chi2_P", "Chi2");
    printf("  ---------   ------------   ------------   ------------\n");
    record_history(hist_ind, 0, x, 0.5*Chi2_B_init, 0.5*Chi2_P_init, 
                   0.5*Chi2_B_init + 0.5*lambda_P*Chi2_P_init, 
                   x_history, Chi2_B_history, Chi2_P_history, Chi2_history);

    // Indices of cells available for optimization
    vector<int> avail_cells(nCells, 0);
    int nAvail = 0;
    for (int i = 0; i < nCells; ++i) {
        if (free_cells(i) == 1) {
            avail_cells[nAvail] = i;
            nAvail++;
        }
    }
    int* avail_cells_ptr = &(avail_cells[0]);

    // Greedy iterations
    for (int i = 0; i < nIter; ++i) {

        // Asses impact on objective function of a loop at each location
        #pragma omp parallel for schedule(static)
        for (int jj = 0; jj < nAvail; ++jj) {

            int j = avail_cells_ptr[jj];

            // Indices of segments forming the loop
            int ind_tor1 = cells_ptr[j*4 + 0];
            int ind_pol2 = cells_ptr[j*4 + 1];
            int ind_tor3 = cells_ptr[j*4 + 2];
            int ind_pol4 = cells_ptr[j*4 + 3];

            // Check currents in each loop segment to determine eligibility
            bool pos_eligible = true;
            bool neg_eligible = true;
            if (abs(x_ptr[ind_tor1] + current) > max_current ||
                abs(x_ptr[ind_pol2] + current) > max_current ||
                abs(x_ptr[ind_tor3] - current) > max_current ||
                abs(x_ptr[ind_pol4] - current) > max_current) {
                pos_eligible = false;
                Chi2_Bs_ptr[j] = inf;
                Chi2_Ps_ptr[j] = inf;
                Chi2s_ptr[j] = inf;
            }
            if (abs(x_ptr[ind_tor1] - current) > max_current ||
                abs(x_ptr[ind_pol2] - current) > max_current ||
                abs(x_ptr[ind_tor3] + current) > max_current ||
                abs(x_ptr[ind_pol4] + current) > max_current) {
                neg_eligible = false;
                Chi2_Bs_ptr[j+nCells] = inf;
                Chi2_Ps_ptr[j+nCells] = inf;
                Chi2s_ptr[j+nCells] = inf;
            }

            // Calculuate contribution of loop to the squared objective
            if (pos_eligible || neg_eligible) {

                double Chi2_B_pos = 0.0, Chi2_P_pos = 0.0, Chi2_pos = 0.0; 
                double Chi2_B_neg = 0.0, Chi2_P_neg = 0.0, Chi2_neg = 0.0; 

                // Determine impact of the loop on Chi2_B
                for (int m = 0; m < nGrid; ++m) {

                    double bnorm = 0.0;
                    bnorm += A_ptr[nSegs*m + ind_tor1] * current;
                    bnorm += A_ptr[nSegs*m + ind_pol2] * current;
                    bnorm -= A_ptr[nSegs*m + ind_tor3] * current;
                    bnorm -= A_ptr[nSegs*m + ind_pol4] * current;

                    Chi2_B_pos +=  (Ax_minus_b_ptr[m] + bnorm) 
                                 * (Ax_minus_b_ptr[m] + bnorm);
                    Chi2_B_neg +=  (Ax_minus_b_ptr[m] - bnorm) 
                                 * (Ax_minus_b_ptr[m] - bnorm);
                }
                if (pos_eligible) Chi2_Bs_ptr[j]        = Chi2_B_pos;
                if (neg_eligible) Chi2_Bs_ptr[j+nCells] = Chi2_B_neg;

                // Determine the impact of the loop on Chi2_P
                double dChi2_orig = 0.0, dChi2_pos = 0.0, dChi2_neg = 0.0;
                int inds[4] = {ind_tor1, ind_pol2, ind_tor3, ind_pol4};
                double signs[4] = {1.0, 1.0, -1.0, -1.0};
                for (int m = 0; m < 4; ++m) {
                    if (abs(x_ptr[inds[m]]) > tol) {
                        dChi2_orig += 1.0;
                    }
                    if (abs(x_ptr[inds[m]] + signs[m]*current) > tol) {
                        dChi2_pos += 1.0;
                    }
                    if (abs(x_ptr[inds[m]] - signs[m]*current) > tol) {
                        dChi2_neg += 1.0;
                    }
                }
                if (pos_eligible) {
                    Chi2_Ps_ptr[j] = Chi2_P_i + dChi2_pos - dChi2_orig;
                }
                if (neg_eligible) {
                    Chi2_Ps_ptr[j+nCells] = Chi2_P_i + dChi2_neg - dChi2_orig;
                }

                // Total impact
                if (pos_eligible) {
                    Chi2s_ptr[j] = Chi2_Bs_ptr[j] + lambda_P*Chi2_Ps_ptr[j];
                }
                if (neg_eligible) {
                    Chi2s_ptr[j+nCells] = Chi2_Bs_ptr[j+nCells] 
                                              + lambda_P*Chi2_Ps_ptr[j+nCells];
                }
             }

        }

        // Adopt the loop that reduced the squared objective the most
        int opt_ind = int(std::distance(Chi2s.begin(), 
                               std::min_element(Chi2s.begin(), Chi2s.end())));
        double sign = (opt_ind < nCells) ? 1.0 : -1.0;
        int cell_ind = (opt_ind < nCells) ? opt_ind : opt_ind - nCells;
        cell_count(cell_ind) += int(sign);

        int ind_tor1 = cells_ptr[cell_ind*4 + 0];
        int ind_pol2 = cells_ptr[cell_ind*4 + 1];
        int ind_tor3 = cells_ptr[cell_ind*4 + 2];
        int ind_pol4 = cells_ptr[cell_ind*4 + 3];
        x_ptr[ind_tor1] += sign * current;
        x_ptr[ind_pol2] += sign * current;
        x_ptr[ind_tor3] -= sign * current;
        x_ptr[ind_pol4] -= sign * current;
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < nGrid; ++m) {
            Ax_minus_b_ptr[m] += A_ptr[nSegs*m + ind_tor1] * sign * current;
            Ax_minus_b_ptr[m] += A_ptr[nSegs*m + ind_pol2] * sign * current;
            Ax_minus_b_ptr[m] -= A_ptr[nSegs*m + ind_tor3] * sign * current;
            Ax_minus_b_ptr[m] -= A_ptr[nSegs*m + ind_pol4] * sign * current;
        }
        Chi2_P_i = Chi2_Ps[opt_ind];

        if ((i+1) % int(nIter/nHistory) == 0 || i == nIter - 1) {
            hist_ind++;

            // Verify that Chi2_P is as expected
            if (abs(Chi2_Ps[opt_ind] - compute_chi2_P(x, tol)) > 0.00001) {
                printf("ERROR in computing Chi2_P\n");
                printf("From function: %14.4e\n", compute_chi2_P(x, tol));
                printf("From array: %14.4e\n", Chi2_Ps[opt_ind]);
                exit(-1);
            }

            record_history(hist_ind, i+1, x, 0.5*Chi2_Bs[opt_ind], 
                           0.5*Chi2_Ps[opt_ind], 0.5*Chi2s[opt_ind],
                           x_history, Chi2_B_history, Chi2_P_history, 
                           Chi2_history);
        }

    }

    return std::make_tuple(x, cell_count, x_history, 
                           Chi2_B_history, Chi2_P_history, Chi2_history);
}

void record_history(int hist_ind, int iter, Array& x, double Chi2_B, 
                    double Chi2_P, double Chi2, Array& x_history, 
                    Array& Chi2_B_history, Array& Chi2_P_history, 
                    Array& Chi2_history) {

    int nx = x.shape(0);
    for (int i = 0; i < nx; ++i) {
        x_history(i,hist_ind) = x(i);
    }
    Chi2_B_history(hist_ind) = Chi2_B;
    Chi2_P_history(hist_ind) = Chi2_P;
    Chi2_history(hist_ind) = Chi2;

    printf("%11d %14.4e %14.4e %14.4e\n", iter, Chi2_B, Chi2_P, Chi2);

}

double compute_chi2_P(Array& x, double tol) {

    double chi2_P = 0.0;
    int nSegs = x.shape(0);

    for (int i = 0; i < nSegs; ++i) {
        if (abs(x(i,0)) > tol) {
            chi2_P += 1.0;
        }
    }

    return chi2_P;
}



/*
 *  Greedy Stellarator Coil Optimization (GSCO): optimizes the currents in a
 *  wireframe grid by adding loops of current one by one to minimize an 
 *  objective function.
 */
std::tuple<Array,IntArray,Array,Array> GSCO_orig(
    Array& A_obj, Array& b_obj, double current, double max_current, 
    Array& loops, int nIter, Array& x_init, int nHistory){

    int nSegs = A_obj.shape(1);
    int nGrid = A_obj.shape(0);
    int nLoops = loops.shape(0);

    Array Ax_minus_b = -b_obj;

    double* A_ptr          = &(A_obj(0,0));
    double* Ax_minus_b_ptr = &(Ax_minus_b(0));
    double* loops_ptr      = &(loops(0,0));

    double inf = std::numeric_limits<double>::max();
    vector<double> R2s(2*nLoops, inf);
    double* R2s_ptr = &(R2s[0]);

    // Adjustment to number of history entries to save
    int extra = (nIter % nHistory == 0) ? 1 : 2;

    // Initialize the solution array
    Array x = xt::zeros<double>({nSegs,1});
    Array x_history = xt::zeros<double>({nSegs,nHistory+extra});
    for (int i = 0; i < nSegs; ++i) {
        x(i) = x_init(i);
        x_history(i,0) = x_init(i);
    }
    double* x_ptr = &(x(0,0));

    IntArray loop_count = xt::zeros<int>({nLoops});
    Array R2_history = xt::zeros<double>({nHistory+extra});

    // Initial history values
    int hist_ind = 0;
    for (int i = 0; i < nGrid; ++i) {
        R2_history(hist_ind) += Ax_minus_b(i) * Ax_minus_b(i);
    }

    printf("  Beginning greedy iterations\n");

    // Greedy iterations
    for (int i = 0; i < nIter; ++i) {

        // Asses impact on objective function of a loop at each location
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < nLoops; ++j) {

            // Indices of segments forming the loop
            int ind_tor1 = loops_ptr[j*4 + 0];
            int ind_pol2 = loops_ptr[j*4 + 1];
            int ind_tor3 = loops_ptr[j*4 + 2];
            int ind_pol4 = loops_ptr[j*4 + 3];

            // Check currents in each loop segment to determine eligibility
            bool pos_eligible = true;
            bool neg_eligible = true;
            if (abs(x_ptr[ind_tor1] + current) > max_current ||
                abs(x_ptr[ind_pol2] + current) > max_current ||
                abs(x_ptr[ind_tor3] - current) > max_current ||
                abs(x_ptr[ind_pol4] - current) > max_current) {
                pos_eligible = false;
                R2s_ptr[j] = inf;
            }
            if (abs(x_ptr[ind_tor1] - current) > max_current ||
                abs(x_ptr[ind_pol2] - current) > max_current ||
                abs(x_ptr[ind_tor3] + current) > max_current ||
                abs(x_ptr[ind_pol4] + current) > max_current) {
                neg_eligible = false;
                R2s_ptr[j+nLoops] = inf;
            }

            // Calculuate contribution of loop to the squared objective
            if (pos_eligible || neg_eligible) {

                double R2pos = 0.0; 
                double R2neg = 0.0;

                for (int m = 0; m < nGrid; ++m) {

                    double bnorm = 0.0;
                    bnorm += A_ptr[nSegs*m + ind_tor1] * current;
                    bnorm += A_ptr[nSegs*m + ind_pol2] * current;
                    bnorm -= A_ptr[nSegs*m + ind_tor3] * current;
                    bnorm -= A_ptr[nSegs*m + ind_pol4] * current;

                    R2pos +=  (Ax_minus_b_ptr[m] + bnorm) 
                            * (Ax_minus_b_ptr[m] + bnorm);
                    R2neg +=  (Ax_minus_b_ptr[m] - bnorm) 
                            * (Ax_minus_b_ptr[m] - bnorm);
                }
                if (pos_eligible) R2s_ptr[j]        = R2pos;
                if (neg_eligible) R2s_ptr[j+nLoops] = R2neg;
            }

        }

        // Adopt the loop that reduced the squared objective the most
        int opt_ind = int(std::distance(R2s.begin(), 
                                     std::min_element(R2s.begin(), R2s.end())));
        double sign = (opt_ind < nLoops) ? 1.0 : -1.0;
        int loop_ind = (opt_ind < nLoops) ? opt_ind : opt_ind - nLoops;
        loop_count(loop_ind) += int(sign);

        int ind_tor1 = loops_ptr[loop_ind*4 + 0];
        int ind_pol2 = loops_ptr[loop_ind*4 + 1];
        int ind_tor3 = loops_ptr[loop_ind*4 + 2];
        int ind_pol4 = loops_ptr[loop_ind*4 + 3];
        x_ptr[ind_tor1] += sign * current;
        x_ptr[ind_pol2] += sign * current;
        x_ptr[ind_tor3] -= sign * current;
        x_ptr[ind_pol4] -= sign * current;
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < nGrid; ++m) {
            Ax_minus_b_ptr[m] += A_ptr[nSegs*m + ind_tor1] * sign * current;
            Ax_minus_b_ptr[m] += A_ptr[nSegs*m + ind_pol2] * sign * current;
            Ax_minus_b_ptr[m] -= A_ptr[nSegs*m + ind_tor3] * sign * current;
            Ax_minus_b_ptr[m] -= A_ptr[nSegs*m + ind_pol4] * sign * current;
        }

        if ((i+1) % int(nIter/nHistory) == 0 || i == nIter - 1) {
            hist_ind++;
            if (hist_ind >= x_history.shape(1) ||
                hist_ind >= R2_history.shape(0)) {
                printf("ERROR: history arrays are too small\n");
                exit(-1);
            }
            record_history_orig(hist_ind, i+1, x, 0.5*R2s[opt_ind], 
                                x_history, R2_history);
        }

    }

    return std::make_tuple(x, loop_count, x_history, R2_history);
}

void record_history_orig(int hist_ind, int iter, Array& x, double R2, 
                         Array& x_history, Array& R2_history) {

    int nx = x.shape(0);
    for (int i = 0; i < nx; ++i) {
        x_history(i,hist_ind) = x(i);
    }
    R2_history(hist_ind) = R2;

    printf("%5d  %12.4e\n", iter, R2);

}

