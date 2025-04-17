#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cstddef> // for size_t

// Computes the pairwise dipole-dipole force matrix (N x N x 3).
// magnetMoments[i]   = magnetic moment of dipole i
// magnetPositions[i] = position (x,y,z) of dipole i
std::vector<std::vector<std::array<double, 3>>>
dipole_force_matrix(const std::vector<std::array<double, 3>>& magnetMoments,
                    const std::vector<std::array<double, 3>>& magnetPositions)
{
    const size_t N = magnetMoments.size();
    // Initialize ForceMatrix as (N x N), each entry is a 3D array initialized to 0
    std::vector<std::vector<std::array<double, 3>>> forceMatrix(
        N, std::vector<std::array<double, 3>>(N, {0.0, 0.0, 0.0})
    );

    const double eps = 1e-10;
    const double mu  = 4.0 * M_PI * 1e-7; // permeability of free space

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            // Compute R = r_j - r_i
            std::array<double, 3> R;
            for (int k = 0; k < 3; ++k) {
                R[k] = magnetPositions[j][k] - magnetPositions[i][k];
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
                m1dotR   += magnetMoments[i][k] * R[k];
                m2dotR   += magnetMoments[j][k] * R[k];
                m1dotm2  += magnetMoments[i][k] * magnetMoments[j][k];
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

            std::array<double, 3> force_ij = {0.0, 0.0, 0.0};
            double factor_4th = 5.0 * m1dotR * m2dotR / (magR*magR);

            for (int k = 0; k < 3; ++k) {
                double first  = m1dotR       * magnetMoments[j][k];
                double second = m2dotR       * magnetMoments[i][k];
                double third  = m1dotm2      * R[k];
                double fourth = factor_4th   * R[k];
                
                force_ij[k] = coefficient * (first + second + third - fourth);
            }

            // Store in the force matrix
            forceMatrix[i][j] = force_ij;
        }
    }

    return forceMatrix;
}

// net_force_matrix: sums the (N x N x 3) pairwise forces over j to get net force on each i
//std::vector<std::array<double, 3>>
//net_force_matrix(const std::vector<std::array<double, 3>>& magnetMoments,
 //                const std::vector<std::array<double, 3>>& magnetPositions)
//{
    // 1) Get the full ForceMatrix for all pairs (i,j).
//    auto ForceMatrix = dipole_force_matrix(magnetMoments, magnetPositions);

    // 2) Sum over j to get net force on dipole i.
 //   const size_t N = magnetMoments.size();
  //  std::vector<std::array<double, 3>> netForce(N, {0.0, 0.0, 0.0});

    //for (size_t i = 0; i < N; ++i) {
        // Sum over j
      //  for (size_t j = 0; j < N; ++j) {
        //    for (int k = 0; k < 3; ++k) {
          //      netForce[i][k] += ForceMatrix[i][j][k];
           // }
       // }
   // }
    //return netForce; // shape: N x 3
//}

// Example main() to show usage
//int main()