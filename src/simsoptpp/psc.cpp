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
    int num_coils = alphas.shape(0);  // shape should be (num_coils)
    int num_phi = phi.shape(0);  // shape should be (num_phi)
    Array L = xt::zeros<double>({num_coils, num_coils});
    
    // Loop through the evaluation points
    
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_coils; i++) {
        auto cai = cos(alphas(i));
        auto sai = sin(alphas(i));
        auto cdi = cos(deltas(i));
        auto sdi = sin(deltas(i));
	// Loop through all the dipoles, using all the symmetries
        for (int j = (i + 1); j < num_coils; ++j) {
            auto xj = (points(j, 0) - points(i, 0)) / R;
            auto yj = (points(j, 1) - points(i, 1)) / R;
            auto zj = (points(j, 2) - points(i, 2)) / R;
            auto caj = cos(alphas(j));
            auto saj = sin(alphas(j));
            auto cdj = cos(deltas(j));
            auto sdj = sin(deltas(j));
            auto integrand = 0.0;
            for (int k = 0; k < num_phi; ++k) {
                auto ck = cos(phi(k));
                auto sk = sin(phi(k));
                for (int kk = 0; kk < num_phi; ++kk) {
                    auto ckk = cos(phi(kk));
                    auto skk = sin(phi(kk));
                    auto dl2_x = (-sk * cdi + ck * sai * sdi) * (-skk * cdj + ckk * saj * sdj);
                    auto dl2_y = (ck * cai) * (ckk * caj);
                    auto dl2_z = (sk * sdi + ck * sai * cdi) * (skk * sdj + ckk * saj * cdj);
                    auto integrand_numerator = (dl2_x + dl2_y + dl2_z);
                    auto x2 = (xj + ckk * cdj + skk * saj * sdj - ck * cdi - sk * sai * sdi);
                    auto y2 = (yj + skk * caj - sk * cai);
                    auto z2 = (zj + skk * saj * cdj - ckk * sdj - sk * sai * cdi + ck * sdi);
                    x2 = x2 * x2;
                    y2 = y2 * y2;
                    z2 = z2 * z2;
                    auto integrand_denominator = sqrt(x2 + y2 + z2);
                    integrand += integrand_numerator / integrand_denominator;
                }
            }
            L(i, j) = integrand;
        }
    }
    return L;
}

Array TF_fluxes(Array& points, Array& alphas, Array& deltas, Array& rho, Array& phi, Array& I, Array& normal, double R)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(alphas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("alphas normal needs to be in row-major storage order");
    if(deltas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("deltas needs to be in row-major storage order");
    if(rho.layout() != xt::layout_type::row_major)
          throw std::runtime_error("rho needs to be in row-major storage order");
    if(phi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("phi needs to be in row-major storage order");
    if(I.layout() != xt::layout_type::row_major)
          throw std::runtime_error("I needs to be in row-major storage order");
    if(normal.layout() != xt::layout_type::row_major)
          throw std::runtime_error("normal needs to be in row-major storage order");
          
    // points shape should be (num_coils, 3)
    // I shape should be (num_coils, 3)
    // normal shape should be (num_coils, 3)
    int num_coils = alphas.shape(0);  // shape should be (num_coils)
    int num_phi = phi.shape(0);  // shape should be (num_phi)
    int num_rho = rho.shape(0);  // shape should be (num_rho)
    Array Psi = xt::zeros<double>({num_coils});
    double R2 = R * R;
    double fac = 4e-7 * R2;
    using namespace boost::math;
    
//     #pragma omp parallel for schedule(static)
    for(int j = 0; j < num_coils; j++) {
        auto cdj = cos(deltas(j));
        auto caj = cos(alphas(j));
        auto sdj = sin(deltas(j));
        auto saj = sin(alphas(j));
        auto nx = normal(j, 0);
        auto ny = normal(j, 1);
        auto nz = normal(j, 2);
        auto xj = points(j, 0);
        auto yj = points(j, 1);
        auto zj = points(j, 2);
        double integral = 0.0;
        for (int k = 0; k < num_rho; ++k) {
            auto xx = rho(k);
            for (int kk = 0; k < num_phi; ++kk) {
                auto yy = phi(kk);
                auto x0 = xx * cos(yy);
                auto y0 = xx * sin(yy);
                auto x = x0 * cdj + y0 * saj * sdj + xj;
                auto y = y0 * caj + yj;
                auto z = -x0 * sdj + y0 * saj * cdj + zj;
                auto rho2 = (x - xj) * (x - xj) + (y - yj) * (y - yj);
                auto r2 = rho2 + (z - zj) * (z - zj);
                auto rho = sqrt(rho2);
                auto R2_r2 = R2 + r2;
                auto gamma2 = R2_r2 + 2 * R * rho;
                auto beta2 = R2_r2 - 2 * R * rho;
                auto beta = sqrt(beta2);
                auto k2 = 1 - gamma2 / beta2;
                auto beta_gamma2 = beta * gamma2;
                auto k = sqrt(k2);
                auto ellipe = ellint_2(k);
                auto ellipk = ellint_1(k);
                auto Eplus = R2_r2 * ellipe - gamma2 * ellipk;
                auto Eminus = (R2 - r2) * ellipe + gamma2 * ellipk;
                auto Bx = (x - xj) * (z - zj) * Eplus / (rho2 * beta_gamma2);
                auto By = (y - yj) * (z - zj) * Eplus / (rho2 * beta_gamma2);
                auto Bz = Eminus / beta_gamma2;
                auto Bn = Bx * nx + By * ny + Bz * nz;
                integral += Bn * xx;
            }
        }
        Psi(j) = integral * fac * I(j);
    }
    return Psi;
}