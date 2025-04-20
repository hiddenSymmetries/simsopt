#include "NetForce.h"
#include "simdhelpers.h"
#include "vec3dsimd.h"
#include <cmath>
#include <vector>
#include <omp.h>

// Computes net force on each dipole by summing pairwise dipole-dipole interactions
// On-the-fly accumulation eliminates the large N×N×3 buffer.
Array net_force_matrix(Array& magnetMoments, Array& magnetPositions) {
    int N = magnetMoments.shape(0);
    // Allocate output
    Array netForces = xt::zeros<double>({N, 3});
    double* moments  = magnetMoments.data();     // length N*3
    double* positions= magnetPositions.data();   // length N*3
    double* netforce = netForces.data();         // length N*3

    const double mu0 = 4.0 * M_PI * 1e-7;
    int numThreads = omp_get_max_threads();
    // Thread-local buffer: numThreads × N × 3
    std::vector<double> localBuf(numThreads * N * 3, 0.0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double* local = &localBuf[tid * N * 3];

        #pragma omp for schedule(dynamic,1)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < i; ++j) {
                // Displacement
                double r_x = positions[3*j + 0] - positions[3*i + 0];
                double r_y = positions[3*j + 1] - positions[3*i + 1];
                double r_z = positions[3*j + 2] - positions[3*i + 2];
                // Distance metrics
                double r2 = r_x*r_x + r_y*r_y + r_z*r_z;
                double r  = std::sqrt(r2);
                double r5 = r2 * r2 * r;
                // Dot products
                double m1dotR  = moments[3*i + 0]*r_x
                               + moments[3*i + 1]*r_y
                               + moments[3*i + 2]*r_z;
                double m2dotR  = moments[3*j + 0]*r_x
                               + moments[3*j + 1]*r_y
                               + moments[3*j + 2]*r_z;
                double m1dotm2 = moments[3*i + 0]*moments[3*j + 0]
                               + moments[3*i + 1]*moments[3*j + 1]
                               + moments[3*i + 2]*moments[3*j + 2];
                // Coefficient and term4
                double coeff = (3.0 * mu0) / (4.0 * M_PI * r5);
                double term4 = 5.0 * m1dotR * m2dotR / r2;
                // Compute force components
                double fx = coeff * (
                      m1dotR * moments[3*j + 0]
                    + m2dotR * moments[3*i + 0]
                    + m1dotm2 * r_x
                    - term4   * r_x);
                double fy = coeff * (
                      m1dotR * moments[3*j + 1]
                    + m2dotR * moments[3*i + 1]
                    + m1dotm2 * r_y
                    - term4   * r_y);
                double fz = coeff * (
                      m1dotR * moments[3*j + 2]
                    + m2dotR * moments[3*i + 2]
                    + m1dotm2 * r_z
                    - term4   * r_z);
                // Accumulate into thread-local buffers
                local[3*i + 0] += fx;
                local[3*i + 1] += fy;
                local[3*i + 2] += fz;
                local[3*j + 0] += fx;
                local[3*j + 1] += fy;
                local[3*j + 2] += fz;
            }
        }
    }

    // Reduce thread-local buffers into netforce
    for (int t = 0; t < numThreads; ++t) {
        double* local = &localBuf[t * N * 3];
        for (int i = 0; i < N; ++i) {
            netforce[3*i + 0] += local[3*i + 0];
            netforce[3*i + 1] += local[3*i + 1];
            netforce[3*i + 2] += local[3*i + 2];
        }
    }

    return netForces;
}
