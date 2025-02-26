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
 *    bool no_crossing: if true, the solution will forbid currents from crossing
 *      within the wireframe 
 *    bool no_new_coils: if true, the solution will forbid the addition of 
 *      currents to loops where no segments already carry current (although it
 *      is still possible for an existing coil to be split into two coils)
 *    bool match_current: if true, added loops of current will match the current
 *      of the loop(s) adjacent to where they are added
 *    Array& A_obj: matrix corresponding to A_obj above
 *    Array& b_obj: column vector corresponding to b_obj above
 *    double default_current: loop current increment to add during each 
 *      iteration to empty loops or to any loop if not matching existing current
 *    double max_current: maximum magnitude for each element of `x`
 *    int max_loop_count: maximum number of current increments to add to a 
 *      given loop
 *    IntArray& loops: 4-column matrix giving the indices of the segments 
 *      (elements of x) bordering each loop in the wireframe
 *    IntArray& free_loops: logical array; 1 for each loop that is free
 *      to have current added to the segments around it; otherwise 0
 *    Intarray& segments: 2-column matrix giving the indices of the nodes
 *      at the beginnings and ends of each segment
 *    IntArray& connections: 4-column matrix giving the indices of the 
 *      segments connected to each node
 *    double lambda_P: the weighting factor lambda_P as defined above
 *    int nIter: maximum number of iterations to perform
 *    Array& x_init: initial values of `x`
 *    IntArray& loop_count_init: signed number of current increments added to
 *      each loop in the wireframe prior to the optimization (optimization will
 *      add to these numbers)
 *    int print_interval: how often to print iteration data to screen (i.e.
 *      the number of iterations between subsequent prints)
 *      
 *  Returns (as a tuple):
 *    Array x: solution vector
 *    IntArray loop_count: signed number of current loops added to each loop
 *      in the wireframe
 *    IntArray iter_history: array with the iteration numbers of the data 
 *      recorded in the history arrays
 *    Array curr_history: array with the signed loop current added at each
 *      iteration, taken to be zero for the initial guess (iteration zero)
 *    Array loop_history: array with the index of the loop to which current
 *      was added at each iteration; taken to be zero for the initial guess
 *      (iteration zero)
 *    Array f_B_history: array with values of the f_B objective function at 
 *      each iteration
 *    Array f_S_history: array with values of the f_S objective function at 
 *      each iteration
 *    Array f_history: array with values of the f_S objective function at 
 *      each iteration
 */
std::tuple<Array,IntArray,IntArray,Array,IntArray,Array,Array,Array> GSCO(
    bool no_crossing, bool no_new_coils, bool match_current,
    Array& A_obj, Array& b_obj, double default_current, double max_current, 
    int max_loop_count, IntArray& loops, IntArray& free_loops, 
    IntArray& segments, IntArray& connections, double lambda_P, int nIter, 
    Array& x_init, IntArray& loop_count_init, int print_interval){

    int nSegs = A_obj.shape(1);
    int nGrid = A_obj.shape(0);
    int nLoops = loops.shape(0);

    // Initialize the solution array
    Array x = xt::zeros<double>({nSegs,1});
    for (int i = 0; i < nSegs; ++i) {
        x(i,0) = x_init(i,0);
    }
    double* x_ptr = &(x(0,0));

    // Make a copy of the loops and free_loops matrices, repeated once
    int twoNLoops = 2*nLoops;
    IntArray loops_rep = xt::zeros<int>({twoNLoops, 4});
    IntArray free_loops_rep = xt::zeros<int>({twoNLoops});
    for (int i = 0; i < nLoops; ++i) {
        for (int j = 0; j < 4; ++j) {
            loops_rep(i,j)        = loops(i,j);
            loops_rep(i+nLoops,j) = loops(i,j);
        }
        free_loops_rep(i)        = free_loops(i);
        free_loops_rep(i+nLoops) = free_loops(i);
    }   

    // Keeps a running total of the normal field at each test point
    Array Ax_minus_b = xt::zeros<double>({nGrid, 1});

    double* Ax_minus_b_ptr  = &(Ax_minus_b(0,0));
    double* A_ptr           = &(A_obj(0,0));
    double* b_obj_ptr       = &(b_obj(0,0));
    int* loops_rep_ptr      = &(loops_rep(0,0));
    int* free_loops_rep_ptr = &(free_loops_rep(0));
    int* segments_ptr       = &(segments(0,0));
    int* connections_ptr    = &(connections(0,0));

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
    vector<double> two_f_Bs(twoNLoops, inf);
    vector<double> two_f_Ss(twoNLoops, inf);
    vector<double> two_fs(twoNLoops, inf);
    double* two_f_Bs_ptr = &(two_f_Bs[0]);
    double* two_f_Ss_ptr = &(two_f_Ss[0]);
    double* two_fs_ptr = &(two_fs[0]);

    // Initialize the loop_count array
    IntArray loop_count = xt::zeros<int>({nLoops});
    for (int i = 0; i < nLoops; ++i) {
        loop_count(i) = loop_count_init(i);
    }
    int* loop_count_ptr = &(loop_count(0));

    IntArray iter_history = xt::zeros<int>({nIter+1});
    Array curr_history = xt::zeros<double>({nIter+1});
    IntArray loop_history = xt::zeros<int>({nIter+1});
    Array f_B_history = xt::zeros<double>({nIter+1});
    Array f_S_history = xt::zeros<double>({nIter+1});
    Array f_history = xt::zeros<double>({nIter+1});
    int* iter_hist_ptr = &(iter_history[0]);
    double* curr_hist_ptr = &(curr_history[0]);
    int* loop_hist_ptr = &(loop_history[0]);
    double* f_B_hist_ptr = &(f_B_history[0]);
    double* f_S_hist_ptr = &(f_S_history[0]);
    double* f_hist_ptr = &(f_history[0]);

    // Initial history values
    int hist_ind = 0;
    int opt_ind_prev;
    double two_f_B_latest = 0.0;
    for (int i = 0; i < nGrid; ++i) {
        two_f_B_latest += Ax_minus_b(i,0) * Ax_minus_b(i,0);
    }
    double tol = 0.001*default_current;
    double two_f_S_latest = 2*compute_f_S(x, tol);
    double two_f_latest = two_f_B_latest + lambda_P*two_f_S_latest;

    printf("  Beginning GSCO iterations\n");
    printf("%11s %14s %14s %14s\n", "Iteration", "f_B", "f_S", "f");
    printf("  ---------   ------------   ------------   ------------\n");
    print_iter(0, 0.5*two_f_B_latest, 0.5*two_f_S_latest, 
               0.5*two_f_B_latest + 0.5*lambda_P*two_f_S_latest);
    record_iter(0, 0.0, 0, 0.5*two_f_B_latest, 0.5*two_f_S_latest, 
                0.5*two_f_B_latest + 0.5*lambda_P*two_f_S_latest, iter_hist_ptr,
                curr_hist_ptr, loop_hist_ptr, f_B_hist_ptr, f_S_hist_ptr, 
                f_hist_ptr);

    vector<double> eligible_curr(twoNLoops, 0);
    double* eligible_curr_ptr = &(eligible_curr[0]);
    vector<int> eligible_inds(twoNLoops, 0);
    int* eligible_inds_ptr = &(eligible_inds[0]);

    // Initialize flags for stopping conditions
    bool accept_current_loop = true;
    bool stop_now = false; 
    bool stop_none_eligible = false; 
    bool stop_undone_loop = false;
    bool stop_last_iter = false;

    // Greedy iterations
    for (int i = 0; i < nIter; ++i) {

        // Determine which loops are eligible for optimization
        check_eligibility(nLoops, default_current, max_current, max_loop_count,
                          no_crossing, no_new_coils, match_current, tol, 
                          loops_rep_ptr, free_loops_rep_ptr, loop_count_ptr, 
                          segments_ptr, connections_ptr, x_ptr, 
                          eligible_curr_ptr);

        // Find the eligible currents; determine f values to exclude
        int nEligible = 0;
        for (int jj = 0; jj < twoNLoops; ++jj) {
            if (eligible_curr_ptr[jj] == 0.0) {
                two_f_Bs_ptr[jj] = inf;
                two_f_Ss_ptr[jj] = inf;
                two_fs_ptr[jj] = inf;
            }
            else{ 
                eligible_inds_ptr[nEligible] = jj;
                nEligible++;
            }
        }

        // Assess impact on objective function of a loop at each location
        #pragma omp parallel for schedule(static)
        for (int jj = 0; jj < nEligible; ++jj) {

            int j = eligible_inds_ptr[jj];

            // Indices of segments forming the loop
            int ind_tor1 = loops_rep_ptr[j*4 + 0];
            int ind_pol2 = loops_rep_ptr[j*4 + 1];
            int ind_tor3 = loops_rep_ptr[j*4 + 2];
            int ind_pol4 = loops_rep_ptr[j*4 + 3];

            // Calculuate contribution of loop to the squared objective
            double two_f_B_pos = 0.0, two_f_S_pos = 0.0, two_f_pos = 0.0; 

            // Determine impact of the loop on f_B
            for (int m = 0; m < nGrid; ++m) {

                double bnorm = 0.0;
                bnorm += A_ptr[nSegs*m + ind_tor1] * eligible_curr_ptr[j];
                bnorm += A_ptr[nSegs*m + ind_pol2] * eligible_curr_ptr[j];
                bnorm -= A_ptr[nSegs*m + ind_tor3] * eligible_curr_ptr[j];
                bnorm -= A_ptr[nSegs*m + ind_pol4] * eligible_curr_ptr[j];

                two_f_B_pos +=  (Ax_minus_b_ptr[m] + bnorm) 
                              * (Ax_minus_b_ptr[m] + bnorm);
            }
            two_f_Bs_ptr[j] = two_f_B_pos;

            // Determine the impact of the loop on f_S
            double two_df_orig = 0.0;
            double two_df = 0.0;
            int inds[4] = {ind_tor1, ind_pol2, ind_tor3, ind_pol4};
            double signs[4] = {1.0, 1.0, -1.0, -1.0};
            for (int m = 0; m < 4; ++m) {
                if (abs(x_ptr[inds[m]]) > tol) {
                    two_df_orig += 1.0;
                }
                if (abs(x_ptr[inds[m]] + signs[m]*eligible_curr_ptr[j]) > tol) {
                    two_df += 1.0;
                }
            }
            two_f_Ss_ptr[j] = two_f_S_latest + two_df - two_df_orig;

            // Total impact
            two_fs_ptr[j] = two_f_Bs_ptr[j] + lambda_P*two_f_Ss_ptr[j];

        }

        // Adopt the loop that reduced the squared objective the most
        int opt_ind = int(std::distance(two_fs.begin(), 
                               std::min_element(two_fs.begin(), two_fs.end())));
        double current = eligible_curr_ptr[opt_ind];
        double sign = (opt_ind < nLoops) ? 1.0 : -1.0;
        int loop_ind = (opt_ind < nLoops) ? opt_ind : opt_ind - nLoops;

        // Check for stopping conditions
        if (nEligible < 1) {
            stop_none_eligible = true;
            accept_current_loop = false;
        }
        else if (i > 0 && (opt_ind + nLoops % (twoNLoops)) == opt_ind_prev) {
            stop_undone_loop = true;
            if (two_fs[opt_ind] > two_f_latest) {
                accept_current_loop = false;
            }
        } 
        else {
            opt_ind_prev = opt_ind;
        }
        stop_last_iter = i + 1 == nIter;
        stop_now = (stop_none_eligible || stop_undone_loop || stop_last_iter);

        // Update the solution if deemed acceptable
        if (accept_current_loop) {
            loop_count(loop_ind) += int(sign);
            int ind_tor1 = loops_rep_ptr[opt_ind*4 + 0];
            int ind_pol2 = loops_rep_ptr[opt_ind*4 + 1];
            int ind_tor3 = loops_rep_ptr[opt_ind*4 + 2];
            int ind_pol4 = loops_rep_ptr[opt_ind*4 + 3];
            x_ptr[ind_tor1] += current;
            x_ptr[ind_pol2] += current;
            x_ptr[ind_tor3] -= current;
            x_ptr[ind_pol4] -= current;
            #pragma omp parallel for schedule(static)
            for (int m = 0; m < nGrid; ++m) {
                Ax_minus_b_ptr[m] += A_ptr[nSegs*m + ind_tor1] * current;
                Ax_minus_b_ptr[m] += A_ptr[nSegs*m + ind_pol2] * current;
                Ax_minus_b_ptr[m] -= A_ptr[nSegs*m + ind_tor3] * current;
                Ax_minus_b_ptr[m] -= A_ptr[nSegs*m + ind_pol4] * current;
            }
            two_f_S_latest = two_f_Ss[opt_ind];
            two_f_B_latest = two_f_Bs[opt_ind];
            two_f_latest = two_fs[opt_ind];

            hist_ind++;

            // Record the loop in to history arrays
            record_iter(hist_ind, current, loop_ind, 
                        0.5*two_f_B_latest, 0.5*two_f_S_latest, 
                        0.5*two_f_latest, 
                        iter_hist_ptr, curr_hist_ptr, loop_hist_ptr, 
                        f_B_hist_ptr, f_S_hist_ptr, f_hist_ptr);
        }

        // Print data if applicable
        bool at_interval = hist_ind % print_interval == 0;
        if ((accept_current_loop && (at_interval || stop_now)) || 
            (stop_now && not at_interval)) {

            print_iter(hist_ind, 0.5*two_f_B_latest, 0.5*two_f_S_latest, 
                       0.5*two_f_latest);

        }

        // Terminate the iterations if a stopping condition has been reached
        if (stop_now) {
            if (stop_none_eligible) {
                printf("  Stopping iterations: no eligible loops\n");
            }
            else if (stop_undone_loop) {
                printf("  Stopping iterations: minimum objective reached\n");
            }
            else if (stop_last_iter) {
                printf("  Stopping iterations: maximum iteration reached\n");
            }
            break;
        }

    }

    auto iter_hist_out = xt::view(iter_history, xt::range(0, hist_ind+1));
    auto curr_hist_out = xt::view(curr_history, xt::range(0, hist_ind+1));
    auto loop_hist_out = xt::view(loop_history, xt::range(0, hist_ind+1));
    auto f_B_hist_out = xt::view(f_B_history, xt::range(0, hist_ind+1));
    auto f_S_hist_out = xt::view(f_S_history, xt::range(0, hist_ind+1));
    auto f_hist_out = xt::view(f_history, xt::range(0, hist_ind+1));
    return std::make_tuple(x, loop_count, iter_hist_out, curr_hist_out,
                           loop_hist_out, f_B_hist_out, f_S_hist_out, 
                           f_hist_out);
}

void record_iter(int iter, double curr, int loop_ind, double f_B, double f_S, 
                 double f, int* iter_hist_ptr, double* curr_hist_ptr, 
                 int* loop_hist_ptr, double* f_B_hist_ptr, double* f_S_hist_ptr,
                 double* f_hist_ptr) {

    iter_hist_ptr[iter] = iter;
    curr_hist_ptr[iter] = curr;
    loop_hist_ptr[iter] = loop_ind;
    f_B_hist_ptr[iter] = f_B;
    f_S_hist_ptr[iter] = f_S;
    f_hist_ptr[iter] = f;

}

void print_iter(int iter, double f_B, double f_S, double f) {

    printf("%11d %14.4e %14.4e %14.4e\n", iter, f_B, f_S, f);

}

double compute_f_S(Array& x, double tol) {

    double f_S = 0.0;
    int nSegs = x.shape(0);

    for (int i = 0; i < nSegs; ++i) {
        if (abs(x(i,0)) > tol) {
            f_S += 0.5;
        }
    }

    return f_S;

}

/*
 * check_eligibility: determines whether each loop is eligible to have an 
 * increment of positive or negative current added to it.
 *
 * 
 *
 */
void check_eligibility(int nLoops, double default_current, double max_current, 
                       int max_loop_count, bool no_crossing, bool no_new_coils, 
                       bool match_current, double tol, int* loops_rep, 
                       int* freeLoops, int* loop_count, int* segments, 
                       int* connections, double* x, double* current) {

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < 2*nLoops; ++i) {

        // Check whether the loop is free to carry current
        if (freeLoops[i] != 1) {
            current[i] = 0.0;
            continue;
        }

        double sign = (i < nLoops) ? 1.0 : -1.0;

        // Check whether adding a loop would exceed the max loop count
        if (max_loop_count > 0) {
            if (abs(loop_count[i % nLoops] + (int) sign) > max_loop_count) {
                current[i] = 0.0;
                continue;
            }
        }

        // Indices of segments forming the loop
        int ind_tor1 = loops_rep[i*4 + 0];
        int ind_pol2 = loops_rep[i*4 + 1];
        int ind_tor3 = loops_rep[i*4 + 2];
        int ind_pol4 = loops_rep[i*4 + 3];
        int loop_inds[4] = {ind_tor1, ind_pol2, ind_tor3, ind_pol4};
        double loop_sgns[4] = {1.0, 1.0, -1.0, -1.0};

        // If no new coils may form, check if loop has existing active segments
        if (no_new_coils) {
            bool no_active_segments = true;
            for (int j = 0; j < 4; ++j) {
                if (abs(x[loop_inds[j]]) > tol) {
                    no_active_segments = false;
                    break;
                }
            }
            if (no_active_segments) {
                current[i] = 0.0;
                continue;
            }
        }

        // If matching existing current, check for existing currents in loop
        double loop_curr = 0.0;
        if (match_current) {

            // If loop contains multiple different current levels, reject
            double abs_curr = 0.0;
            bool mismatch = false;
            for (int j = 0; j < 4; ++j) {
                if (abs(x[loop_inds[j]]) > 0) {
                    if (abs_curr != 0 && abs_curr != abs(x[loop_inds[j]])) {
                        mismatch = true;
                        break;
                    }
                    abs_curr = abs(x[loop_inds[j]]);
                }
            }
            if (mismatch) {
                current[i] = 0.0;
                continue;
            }

            if (abs_curr != 0) {
                loop_curr = sign * abs_curr;
            }
            else {
                loop_curr = sign * default_current;
            }
        }
        else {
            loop_curr = sign * default_current;
        }

        // Check whether adding the loop_curr would exceed the current limit
        bool max_exceeded = false;
        for (int j = 0; j < 4; ++j) {
            double curr_to_add = loop_sgns[j] * loop_curr;
            if (abs(x[loop_inds[j]] + curr_to_add) > max_current) {
                max_exceeded = true;
                break;
            }
        }
        if (max_exceeded) {
            current[i] = 0.0;
            continue;
        }

        // Check for crossings
        if (no_crossing) {
            bool crossing_found = false;

            // Populate an array of the nodes to check, corresponding to the
            // nodes at each end of the two toroidal segments
            int nodes[4] = 
                {segments[loop_inds[0]*2], segments[loop_inds[0]*2 + 1], 
                 segments[loop_inds[2]*2], segments[loop_inds[2]*2 + 1]};

            for (int j = 0; j < 4; ++j) {
    
                int count = 0;
    
                // Count number of current-carrying segments connected to node
                for (int k = 0; k < 4; ++k) {
                    int seg_k = connections[nodes[j]*4 + k];
    
                    // Determine whether connected segment is part of loop;
                    // if so, add the loop current under consideration
                    double curr_to_add = 0.0;
                    for (int l = 0; l < 4; ++l) {
                        if (seg_k == loop_inds[l]) {
                            curr_to_add = loop_sgns[l] * loop_curr;
                            break;
                        }
                    }

                    if (abs(x[seg_k] + curr_to_add) > tol) {
                        count++;
                    }
                }
    
                // Invalidate the loop if it would form a crossing at any node
                if (count > 2) {
                    crossing_found = true;
                    break;
                }
            }
            if (crossing_found) {
                current[i] = 0.0;
                continue;
            }
        }

        // If all tests are passed, adopt the loop current
        current[i] = loop_curr;
    }

}


