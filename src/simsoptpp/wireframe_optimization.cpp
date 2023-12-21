#include "wireframe_optimization.h"
#include <limits> // std
#include <vector>
using std::vector;

/*
 *  Greedy Stellarator Coil Optimization (GSCO): optimizes the currents in a
 *  wireframe grid by adding loops of current one by one to minimize an 
 *  objective function.
 */
std::tuple<Array,IntArray,Array,Array> GSCO(
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
            record_history(hist_ind, i+1, x, 0.5*R2s[opt_ind], 
                           x_history, R2_history);
        }

    }

    return std::make_tuple(x, loop_count, x_history, R2_history);
}

void record_history(int hist_ind, int iter, Array& x, double R2, 
                    Array& x_history, Array& R2_history) {

    int nx = x.shape(0);
    for (int i = 0; i < nx; ++i) {
        x_history(i,hist_ind) = x(i);
    }
    R2_history(hist_ind) = R2;

    printf("%5d  %12.4e\n", iter, R2);

}

