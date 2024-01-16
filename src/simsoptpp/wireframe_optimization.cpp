#include "wireframe_optimization.h"
#include <limits> // std
#include <vector>
using std::vector;

/*
 *  Greedy Stellarator Coil Optimization (GSCO): optimizes the currents in a
 *  wireframe grid by adding loops of current one by one to minimize an 
 *  objective function:
 *
 *  f = f_B + lambda_S * f_S
 * 
 *  where:
 *    f_B = 0.5 * (A_obj * x - b_obj)^2
 *    f_S = 0.5 * (number of non-zero elements of x)
 * 
 *    lambda_S = a weighting coefficient
 *    x        = the solution vector
 *    b_obj    = desired normal field on the plasma boundary
 *    A_obj    = inductance matrix relating x to normal field on plasma boundary
 *
 *  Parameters:
 *    Array& A_obj: matrix corresponding to A_obj above
 *    Array& b_obj: column vector corresponding to b_obj above
 *    double current: loop current to add during each iteration
 *    double max_current: maximum magnitude for each element of `x`
 *    IntArray& cells: 4-column matrix giving the indices of the segments 
 *      (elements of x) bordering each cell in the wireframe
 *    IntArray& free_cells: logical array; 1 for each cell that is free
 *      to have current added to the segments around it; otherwise 0
 *    double lambda_P: the weighting factor lambda_P as defined above
 *    int nIter: number of iterations to perform
 *    Array& x_init: initial values of `x`
 *    IntArray& cell_count_init: signed number of loops of current added to
 *      each cell in the wireframe prior to the optimization (optimization will
 *      add to these numbers)
 *    int nHistory: number of intermediate solutions to record, evenly spaced
 *      among the iterations
 *      
 *  Returns (as a tuple):
 *    Array x: solution vector
 *    IntArray cell_count: signed number of current loops added to each cell
 *      in the wireframe
 *    Array x_history: matrix with entries corresponding to the solution at
 *      various iterations, collected with a frequency specified by the 
 *      `nHistory` input parameter. The first column is always the initial
 *      guess `x_init`; the last column is always the solution from the final
 *      iteration.
 *    f_B_history: array with values of the f_B objective function at iterations
 *      corresponding to the columns of `x_history`
 *    f_S_history: array with values of the f_S objective function at iterations
 *      corresponding to the columns of `x_history`
 *    f_history: array with values of the f_S objective function at iterations
 *      corresponding to the columns of `x_history`
 */
std::tuple<Array,IntArray,Array,Array,Array,Array> GSCO(
    Array& A_obj, Array& b_obj, double current, double max_current, 
    IntArray& cells, IntArray& free_cells, double lambda_P, int nIter, 
    Array& x_init, IntArray& cell_count_init, int nHistory){

    int nSegs = A_obj.shape(1);
    int nGrid = A_obj.shape(0);
    int nCells = cells.shape(0);

    // Adjustment to number of history entries to save
    int extra = (nIter % nHistory == 0) ? 1 : 2;

    // Initialize the solution array
    Array x = xt::zeros<double>({nSegs,1});
    Array x_history = xt::zeros<double>({nSegs,nHistory+extra});
    for (int i = 0; i < nSegs; ++i) {
        x(i,0) = x_init(i,0);
        x_history(i,0) = x_init(i,0);
    }
    double* x_ptr = &(x(0,0));

    // Keeps a running total of the normal field at each test point
    Array Ax_minus_b = xt::zeros<double>({nGrid, 1});

    double* Ax_minus_b_ptr  = &(Ax_minus_b(0,0));
    double* A_ptr           = &(A_obj(0,0));
    double* b_obj_ptr       = &(b_obj(0,0));
    int* cells_ptr          = &(cells(0,0));

    // Initialize the vector Ax - b according to initial value
    printf("  Initializing...\n");
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nGrid; ++i) {
        for (int j = 0; j < nSegs; ++j) {
            Ax_minus_b_ptr[i] += A_ptr[i*nSegs + j] * x_ptr[j];
        }
        Ax_minus_b_ptr[i] -= b_obj_ptr[i];
    }

    double inf = std::numeric_limits<double>::max();
    vector<double> f_Bs(2*nCells, inf);
    vector<double> f_Ss(2*nCells, inf);
    vector<double> fs(2*nCells, inf);
    double* f_Bs_ptr = &(f_Bs[0]);
    double* f_Ss_ptr = &(f_Ss[0]);
    double* fs_ptr = &(fs[0]);

    // Initialize the cell_count array
    IntArray cell_count = xt::zeros<int>({nCells});
    for (int i = 0; i < nCells; ++i) {
        cell_count(i) = cell_count_init(i);
    }
    int* cell_count_ptr = &(cell_count(0));

    Array f_B_history = xt::zeros<double>({nHistory+extra});
    Array f_S_history = xt::zeros<double>({nHistory+extra});
    Array f_history = xt::zeros<double>({nHistory+extra});

    // Initial history values
    int hist_ind = 0;
    double f_B_init = 0.0;
    for (int i = 0; i < nGrid; ++i) {
        f_B_init += Ax_minus_b(i,0) * Ax_minus_b(i,0);
    }
    double tol = 0.001*current;
    double f_S_init = compute_chi2_P(x, tol);
    double f_S_i = f_S_init;

    printf("  Beginning GSCO iterations\n");
    printf("%11s %14s %14s %14s\n", "Iteration", "f_B", "f_S", "f");
    printf("  ---------   ------------   ------------   ------------\n");
    record_history(hist_ind, 0, x, 0.5*f_B_init, 0.5*f_S_init, 
                   0.5*f_B_init + 0.5*lambda_P*f_S_init, 
                   x_history, f_B_history, f_S_history, f_history);

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
                f_Bs_ptr[j] = inf;
                f_Ss_ptr[j] = inf;
                fs_ptr[j] = inf;
            }
            if (abs(x_ptr[ind_tor1] - current) > max_current ||
                abs(x_ptr[ind_pol2] - current) > max_current ||
                abs(x_ptr[ind_tor3] + current) > max_current ||
                abs(x_ptr[ind_pol4] + current) > max_current) {
                neg_eligible = false;
                f_Bs_ptr[j+nCells] = inf;
                f_Ss_ptr[j+nCells] = inf;
                fs_ptr[j+nCells] = inf;
            }

            // Calculuate contribution of loop to the squared objective
            if (pos_eligible || neg_eligible) {

                double f_B_pos = 0.0, f_S_pos = 0.0, f_pos = 0.0; 
                double f_B_neg = 0.0, f_S_neg = 0.0, f_neg = 0.0; 

                // Determine impact of the loop on f_B
                for (int m = 0; m < nGrid; ++m) {

                    double bnorm = 0.0;
                    bnorm += A_ptr[nSegs*m + ind_tor1] * current;
                    bnorm += A_ptr[nSegs*m + ind_pol2] * current;
                    bnorm -= A_ptr[nSegs*m + ind_tor3] * current;
                    bnorm -= A_ptr[nSegs*m + ind_pol4] * current;

                    f_B_pos +=  (Ax_minus_b_ptr[m] + bnorm) 
                              * (Ax_minus_b_ptr[m] + bnorm);
                    f_B_neg +=  (Ax_minus_b_ptr[m] - bnorm) 
                              * (Ax_minus_b_ptr[m] - bnorm);
                }
                if (pos_eligible) f_Bs_ptr[j]        = f_B_pos;
                if (neg_eligible) f_Bs_ptr[j+nCells] = f_B_neg;

                // Determine the impact of the loop on f_S
                double df_orig = 0.0, df_pos = 0.0, df_neg = 0.0;
                int inds[4] = {ind_tor1, ind_pol2, ind_tor3, ind_pol4};
                double signs[4] = {1.0, 1.0, -1.0, -1.0};
                for (int m = 0; m < 4; ++m) {
                    if (abs(x_ptr[inds[m]]) > tol) {
                        df_orig += 1.0;
                    }
                    if (abs(x_ptr[inds[m]] + signs[m]*current) > tol) {
                        df_pos += 1.0;
                    }
                    if (abs(x_ptr[inds[m]] - signs[m]*current) > tol) {
                        df_neg += 1.0;
                    }
                }
                if (pos_eligible) f_Ss_ptr[j]        = f_S_i + df_pos - df_orig;
                if (neg_eligible) f_Ss_ptr[j+nCells] = f_S_i + df_neg - df_orig;

                // Total impact
                if (pos_eligible) {
                    fs_ptr[j] = f_Bs_ptr[j] + lambda_P*f_Ss_ptr[j];
                }
                if (neg_eligible) {
                    fs_ptr[j+nCells] = f_Bs_ptr[j+nCells] 
                                              + lambda_P*f_Ss_ptr[j+nCells];
                }
             }

        }

        // Adopt the loop that reduced the squared objective the most
        int opt_ind = int(std::distance(fs.begin(), 
                               std::min_element(fs.begin(), fs.end())));
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
        f_S_i = f_Ss[opt_ind];

        if ((i+1) % int(nIter/nHistory) == 0 || i == nIter - 1) {
            hist_ind++;

            // Verify that f_S is as expected
            if (abs(f_Ss[opt_ind] - compute_chi2_P(x, tol)) > 0.00001) {
                printf("ERROR in computing f_S\n");
                printf("From function: %14.4e\n", compute_chi2_P(x, tol));
                printf("From array: %14.4e\n", f_Ss[opt_ind]);
                exit(-1);
            }

            record_history(hist_ind, i+1, x, 0.5*f_Bs[opt_ind], 
                           0.5*f_Ss[opt_ind], 0.5*fs[opt_ind],
                           x_history, f_B_history, f_S_history, 
                           f_history);
        }

    }

    return std::make_tuple(x, cell_count, x_history, 
                           f_B_history, f_S_history, f_history);
}

void record_history(int hist_ind, int iter, Array& x, double f_B, double f_S, 
                    double f, Array& x_history, Array& f_B_history, 
                    Array& f_S_history, Array& f_history) {

    int nx = x.shape(0);
    for (int i = 0; i < nx; ++i) {
        x_history(i,hist_ind) = x(i);
    }
    f_B_history(hist_ind) = f_B;
    f_S_history(hist_ind) = f_S;
    f_history(hist_ind) = f;

    printf("%11d %14.4e %14.4e %14.4e\n", iter, f_B, f_S, f);

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


