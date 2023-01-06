#include "winding_volume.h"

// Calculate the geometrics factor from the polynomial basis functions 
Array winding_volume_geo_factors(Array& points, Array& coil_points, Array& integration_points, Array& plasma_normal, Array& Phi) {
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(coil_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("coil_points needs to be in row-major storage order");
    if(Phi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("Phi needs to be in row-major storage order");

    int num_points = points.shape(0);
    int num_coil_points = coil_points.shape(0);

    // integration points should be shape (num_coil_points, num_integration_points, 3)
    int num_integration_points = integration_points.shape(1);
    int num_basis_functions = Phi.shape(0);
    Array geo_factor = xt::zeros<double>({num_points, num_coil_points, num_basis_functions});
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_points; i++) {
	double nx = plasma_normal(i, 0);
	double ny = plasma_normal(i, 1);
	double nz = plasma_normal(i, 2);
	double rx = points(i, 0);
	double ry = points(i, 1);
	double rz = points(i, 2);
        for (int jj = 0; jj < num_coil_points; jj++) {
            for (int j = 0; j < num_integration_points; j++) {
	        double rprimex = integration_points(j, 0);
	        double rprimey = integration_points(j, 1);
	        double rprimez = integration_points(j, 2);
	        double rvecx = rx - rprimex;
	        double rvecy = ry - rprimey;
	        double rvecz = rz - rprimez;
	        double rvec_mag = sqrt(rvecx * rvecx + rvecy * rvecy + rvecz * rvecz);
	        double rvec_inv3 = 1.0 / (rvec_mag * (rvec_mag * rvec_mag));
	        double n_cross_rvec_x = ny * rvecz - nz * rvecy;
	        double n_cross_rvec_y = nz * rvecx - nx * rvecz;
	        double n_cross_rvec_z = nx * rvecy - ny * rvecx;
                for (int k = 0; k < num_basis_functions; k++) {
                    geo_factor(i, jj, k) += (n_cross_rvec_x * Phi(k, i, j, 0) + n_cross_rvec_y * Phi(k, i, j, 1) + n_cross_rvec_z * Phi(k, i, j, 2)) * rvec_inv3;
                }
	    }
	}
    }
    return geo_factor;
}   
