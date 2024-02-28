#include "psc.h"

// Calculate the inductance matrix needed for the PSC forward problem
Array L_matrix(Array& points, Array& alphas, Array& deltas, Array& phi, double R)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(alphas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("alphas normal needs to be in row-major storage order");
    if(deltas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("deltas needs to be in row-major storage order");
    if(phi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("phi needs to be in row-major storage order");
    
    // points shape should be (num_coils, 3)
    int num_coils = alphas.shape(0);  // shape should be (num_coils, num_coils)
    int num_phi = phi.shape(0);  // shape should be (num_phi)
    Array L = xt::zeros<double>({num_coils, num_coils});
    
    // Loop through the evaluation points
    
    // no openmp for now
    // #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_coils; i++) {
        double cai = cos(alphas(i));
        double sai = sin(alphas(i));
        double cdi = cos(deltas(i));
        double sdi = sin(deltas(i));
	// Loop through all the dipoles, using all the symmetries
        for (int j = (i + 1); j < num_coils; ++j) {
            double xj = (points(j, 0) - points(i, 0)) / R;
            double yj = (points(j, 1) - points(i, 1)) / R;
            double zj = (points(j, 2) - points(i, 2)) / R;
            double caj = cos(alphas(j));
            double saj = sin(alphas(j));
            double cdj = cos(deltas(j));
            double sdj = sin(deltas(j));
            int integrand = 0.0;
            for (int k = 0; k < num_phi; ++k) {
                double ck = cos(phi(k));
                double sk = sin(phi(k));
                for (int kk = 0; kk < num_phi; ++kk) {
                    double ckk = cos(phi(kk));
                    double skk = sin(phi(kk));
                    double dl2_x = (-sk * cdi + ck * sai * sdi) * (-skk * cdj + ckk * saj * sdj);
                    double dl2_y = (ck * cai) * (ckk * caj);
                    double dl2_z = (sk * sdi + ck * sai * cdi) * (skk * sdj + ckk * saj * cdj);
                    double x2 = (xj + ckk * cdj + skk * saj * sdj - ck * cdi - sk * sai * sdi);
                    double y2 = (yj + skk * caj - sk * cai);
                    double z2 = (zj + skk * saj * cdj - ckk * sdj - sk * sai * cdi + ck * sdi);
                    double integrand_numerator = (dl2_x + dl2_y + dl2_z);
                    x2 = x2 * x2;
                    y2 = y2 * y2;
                    z2 = z2 * z2;
                    double integrand_denominator = sqrt(x2 + y2 + z2);
                    integrand = integrand + integrand_numerator / integrand_denominator;
                }
            }
            L(i, j) = integrand;
        }
    }
    return L;
}