#include "NetForce.h"
#include "simdhelpers.h"
#include "vec3dsimd.h"

// Computes the pairwise dipole-dipole force matrix (N x N x 3).
// magnetMoments[i]   = magnetic moment of dipole i
// magnetPositions[i] = position (x,y,z) of dipole i
Array dipole_force_matrix(Array& magnetMoments, Array& magnetPositions)
{
    int N = magnetMoments.shape(0);
    // Initialize ForceMatrix as (N x N x 3), each entry is a 3D array initialized to 0
    Array forceMatrix = xt::zeros<double>({N, N, 3});

    const double eps = 1e-10;
    const double mu  = 4.0 * M_PI * 1e-7; // permeability of free space

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            // Compute R = r_j - r_i
            std::array<double, 3> R;
            for (int k = 0; k < 3; ++k) {
                R[k] = magnetPositions(j, k) - magnetPositions(i, k);
            }

            // Compute magnitude of R (plus eps to avoid singularities)
            double magR = std::sqrt(R[0]*R[0] + R[1]*R[1] + R[2]*R[2] + eps);
            double magR5 = std::pow(magR, 5);

            // Dot products:
            // m1·R, m2·R, m1·m2  (where m1 = magnetMoments[i], m2 = magnetMoments[j])
            double m1dotR = 0.0;
            double m2dotR = 0.0;
            double m1dotm2 = 0.0;
            for (int k = 0; k < 3; ++k) {
                m1dotR   += magnetMoments(i, k) * R[k];
                m2dotR   += magnetMoments(j, k) * R[k];
                m1dotm2  += magnetMoments(i, k) * magnetMoments(j, k);
            }

            // Compute coefficient = 3*mu / (4*pi * magR^5)
            // (Note that 4*pi factor is basically 4 * M_PI, but we'll keep it explicit.)
            double coefficient = (3.0 * mu) / (4.0 * M_PI * magR5);

            // Now compute each term of F_{ij}
            // first_term  = (m1 dot R) * m2
            // second_term = (m2 dot R) * m1
            // third_term  = (m1 dot m2) * R
            // fourth_term = 5 * (m1 dot R) (m2 dot R) / (magR^2) * R
            // Then F = coeff * (first + second + third - fourth)

            double factor_4th = 5.0 * m1dotR * m2dotR / (magR*magR);

            for (int k = 0; k < 3; ++k) {
                double first  = m1dotR       * magnetMoments(j, k);
                double second = m2dotR       * magnetMoments(i, k);
                double third  = m1dotm2      * R[k];
                double fourth = factor_4th   * R[k];
                
                forceMatrix(i, j, k) = coefficient * (first + second + third - fourth);
            }
        }
    }

    return forceMatrix;
}

// net_force_matrix: sums the (N x N x 3) pairwise forces over j to get net force on each i
Array net_force_matrix(Array& magnetMoments, Array& magnetPositions)
{
    // 1) Get the full ForceMatrix for all pairs (i,j).
    Array ForceMatrix = dipole_force_matrix(magnetMoments, magnetPositions);

    // 2) Sum over j to get net force on dipole i.
    int N = magnetMoments.shape(0);
    Array netForce = xt::zeros<double>({N, 3});

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        // Sum over j
       for (int j = 0; j < N; ++j) {
           for (int k = 0; k < 3; ++k) {
               netForce(i, k) += ForceMatrix(i, j, k);
           }
       }
   }
    return netForce; // shape: N x 3
}