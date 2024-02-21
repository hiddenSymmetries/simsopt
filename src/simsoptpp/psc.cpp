#include "dipole_field.h"
#include "simdhelpers.h"
#include "vec3dsimd.h"
#include <cmath>
#include <Eigen/Dense>

// Calculate the inductance matrix needed for the PSC forward problem
Array L_matrix(Array& points, Array& alphas, Array& deltas, Array& phi, double R, double r)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(alphas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("alphas normal needs to be in row-major storage order");
    if(deltas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("deltas needs to be in row-major storage order");
    
    // points shape should be (3, num_coils)
    int num_coils = alphas.shape(0);  // shape should be (num_coils, num_coils)
    int num_phi = phi.shape(0);  // shape should be (num_phi)
    Array L = xt::zeros<double>({num_coils, num_coils});
    
    // Loop through the evaluation points
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_coils; i++) {
	// Loop through all the dipoles, using all the symmetries
        for (int j = 0; j < num_coils; ++j) {
            if (i == j) L(i, j) = log(8.0 * R / r) - 2.0;
            else {
                int integrand = 0.0;
                for (int k = 0; k < num_phi; ++k) {
                    for (int kk = 0; kk < num_phi; ++kk) {
                        double integrand_numerator = cos(phi(k)) * cos(phi(kk)) * cos(alphas(i, j)) + sin(phi(k)) * sin(phi(kk)) * cos(deltas(i, j)) + cos(phi(k)) * sin(phi(kk)) * sin(deltas(i, j));
                        double x2 = (points(0, j) + cos(phi(kk)) - cos(phi(k)) * cos(deltas(i, j)) + sin(phi(k)) * sin(alphas(i, j)) * sin(deltas(i, j)));
                        double y2 = (points(1, j) + sin(phi(kk)) - sin(phi(k)) * cos(alphas(i, j)));
                        double z2 = (points(2, j) + cos(phi(k)) * sin(deltas(i, j)) + sin(phi(k)) * sin(alphas(i, j)) * cos(deltas(i, j)));
                        x2 = x2 * x2;
                        y2 = y2 * y2;
                        z2 = z2 * z2;
                        double integrand_denominator = sqrt(x2 + y2 + z2);
                        integrand += integrand_numerator / integrand_denominator;
                    }
                }
                L(i, j) = integrand;
            }
        }
    }
    return L;
}