#include "NetTorque.h"
#include "simdhelpers.h"
#include "vec3dsimd.h"
#include <cmath>
#include <vector>
#include <omp.h>

// Computes net torque on each dipole by summing pairwise dipole-dipole interactions
// On-the-fly accumulation eliminates the large N×N×3 buffer.
Array net_torque_matrix(Array& magnetMoments, Array& magnetPositions) {
    int N = magnetMoments.shape(0);
    // Allocate output
    Array netTorques = xt::zeros<double>({N, 3});
    double* moments  = magnetMoments.data();     // length N*3
    double* positions= magnetPositions.data();   // length N*3
    double* nettorque = netTorques.data();         // length N*3

    const double mu0 = 4.0 * M_PI * 1e-7;
    int numThreads = omp_get_max_threads();
    // Thread-local buffer: numThreads × N × 3
    // Each thread will accumulate the full net torque for its assigned dipoles here
    std::vector<double> localBuf(numThreads * N * 3, 0.0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double* local = &localBuf[tid * N * 3];

        // Parallelize the outer loop over the TARGET dipole 'k'.
        // Each thread calculates the *full* net torque for the dipoles 'k' it is assigned.
        #pragma omp for schedule(dynamic,1)
        for (int k = 0; k < N; ++k) { // 'k' is the index of the target dipole

            // The net torque for dipole 'k' is the sum of torques on 'k' due to all other dipoles 'p'.
            for (int p = 0; p < N; ++p) { // 'p' is the index of the source dipole
                if (k == p) {
                    continue; // Do not calculate self-interaction
                }

                // Displacement vector from source 'p' to target 'k' (r_prime = x_k - x_p)
                double r_prime_x = positions[3*k + 0] - positions[3*p + 0];
                double r_prime_y = positions[3*k + 1] - positions[3*p + 1];
                double r_prime_z = positions[3*k + 2] - positions[3*p + 2];

                // Distance metrics (r_prime_2, r_prime, r_prime_5)
                double r_prime_2 = r_prime_x*r_prime_x + r_prime_y*r_prime_y + r_prime_z*r_prime_z;
                double r_prime  = std::sqrt(r_prime_2);
                double r_prime_5 = r_prime_2 * r_prime_2 * r_prime; // r'^5

                // Coefficient (mu0 / 4pi) / r'^5
                double coeff = (mu0) / (4.0 * M_PI * r_prime_5);

                // Dot product of m_p . r_prime
                // Uses moments[p] and r_prime
                double mpdotRprime  = moments[3*p + 0]*r_prime_x
                                    + moments[3*p + 1]*r_prime_y
                                    + moments[3*p + 2]*r_prime_z;

                // Compute torque components tau_kp (torque ON k due to p)
                // Formula: coeff * (3 * (m_p . r_prime) * (m_k X r_prime) - r_prime_2 * (m_k X m_p))
                // Uses moments[k], moments[p], and r_prime
                double tx_kp = coeff *
                            (3 * mpdotRprime * (moments[3*k + 1]* r_prime_z - moments[3*k + 2] * r_prime_y) // (m_k X r_prime)_x
                            -r_prime_2 * (moments[3*k + 1] * moments[3*p + 2] - moments[3*k + 2] * moments[3*p + 1]) ); // (m_k X m_p)_x

                double ty_kp = coeff *
                            (3 * mpdotRprime * (moments[3*k + 2]* r_prime_x - moments[3*k + 0] * r_prime_z) // (m_k X r_prime)_y
                            -r_prime_2 * (moments[3*k + 2] * moments[3*p + 0] - moments[3*k + 0] * moments[3*p + 2]) ); // (m_k X m_p)_y

                double tz_kp = coeff *
                            (3 * mpdotRprime * (moments[3*k + 0]* r_prime_y - moments[3*k + 1] * r_prime_x) // (m_k X r_prime)_z
                            -r_prime_2 * (moments[3*k + 0] * moments[3*p + 1] - moments[3*p + 1] * moments[3*p + 0]) ); // (m_k X m_p)_z


                // Accumulate tau_kp (torque ON k) into the local buffer for dipole 'k'.
                // Since 'k' is the parallelized index, each thread writes only to its assigned 'k' locations,
                // avoiding race conditions within this parallel loop.
                local[3*k + 0] += tx_kp;
                local[3*k + 1] += ty_kp;
                local[3*k + 2] += tz_kp;
            }
        }
    }

    // Reduction: Sum the net torques calculated by each thread for each dipole 'k'.
    // This loop correctly sums the values stored at index 'i' (which corresponds to 'k' from the parallel loop)
    // in each thread's local buffer into the final nettorque array at index 'i'.
    for (int i = 0; i < N; ++i) {
        for (int t = 0; t < numThreads; ++t) {
            double* local = &localBuf[t * N * 3];
            nettorque[3*i + 0] += local[3*i + 0];
            nettorque[3*i + 1] += local[3*i + 1];
            nettorque[3*i + 2] += local[3*i + 2];
        }
    }

    return netTorques;
}