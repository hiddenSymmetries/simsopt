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
                // Dot product of r dot m1
                double m1dotR  = moments[3*i + 0]*r_x
                               + moments[3*i + 1]*r_y
                               + moments[3*i + 2]*r_z;

                // Coefficient and term4
                double coeff = (mu0) / (4.0 * M_PI * r5);

                // Compute torque components 
                // Torque = coeff * (3 * m1dotr*[m2 X r] - r2*[m2 X m1])
                double tx = coeff * 
                            (3 * m1dotR * (moments[3*j + 1]* r_z - moments[3*j + 2] * r_y)
                            -r2 * (moments[3*j + 1] * moments[3*i + 2] - moments[3*j + 2] * moments[3*i + 1]) );

                double ty = coeff * 
                            (3 * m1dotR * (moments[3*j + 2]* r_x - moments[3*j + 0] * r_z)
                            -r2 * (moments[3*j + 2] * moments[3*i + 0] - moments[3*j + 0] * moments[3*i + 2]) );

                double tz = coeff * 
                            (3 * m1dotR * (moments[3*j + 0]* r_y - moments[3*j + 1] * r_x)
                            -r2 * (moments[3*j + 0] * moments[3*i + 1] - moments[3*j + 1] * moments[3*i + 0]) );

                // Accumulate into thread-local buffers
                local[3*i + 0] += tx;
                local[3*i + 1] += ty;
                local[3*i + 2] += tz;
                local[3*j + 0] -= tx;
                local[3*j + 1] -= ty;
                local[3*j + 2] -= tz;
            }
        }
    }

    // Reduce thread-local buffers into nettorque
    for (int t = 0; t < numThreads; ++t) {
        double* local = &localBuf[t * N * 3];
        for (int i = 0; i < N; ++i) {
            nettorque[3*i + 0] += local[3*i + 0];
            nettorque[3*i + 1] += local[3*i + 1];
            nettorque[3*i + 2] += local[3*i + 2];
        }
    }

    return netTorques;
}
