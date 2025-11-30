#include "current_voxels.h"

// compute which cells are next to which cells 
Array_INT connections(Array& coil_points, double dx, double dy, double dz)
{
    int Ndipole = coil_points.shape(0);
    double tol = 1e-5;
    
    // Last index indicates +- the x/y/z directions, -1 indicates no neighbor there
    Array_INT connectivity_inds = xt::ones<int>({Ndipole, 6, 6}) * (-1);
    
    // Compute distances between dipole j and all other dipoles
    // By default computes distance between dipole j and itself
// #pragma omp parallel for schedule(static)
    for (int j = 0; j < Ndipole; ++j) {
	vector<double> dist_ij(Ndipole, 1e10);
        for (int i = 0; i < Ndipole; ++i) {
	    if (i != j) {
	        dist_ij[i] = sqrt((coil_points(i, 0) - coil_points(j, 0)) * (coil_points(i, 0) - coil_points(j, 0)) + (coil_points(i, 1) - coil_points(j, 1)) * (coil_points(i, 1) - coil_points(j, 1)) + (coil_points(i, 2) - coil_points(j, 2)) * (coil_points(i, 2) - coil_points(j, 2)));
	    }
        }
	// Need to loop through more than 6 in case some adjacent neighbors are
	// further away than some of the non-adjacent neighbors
	int q = 0;
        for (int k = 0; k < 50; ++k) {
	    auto result = std::min_element(dist_ij.begin(), dist_ij.end());
        int dist_ind = std::distance(dist_ij.begin(), result);
        
	    // Get dx, dy, dz from this ind
	    double dxx = -(coil_points(j, 0) - coil_points(dist_ind, 0));
	    double dyy = -(coil_points(j, 1) - coil_points(dist_ind, 1));
	    double dzz = -(coil_points(j, 2) - coil_points(dist_ind, 2));
	    double dist = dist_ij[dist_ind];
	    int dir_ind = -1;
	    if (dist <= std::max(std::max(dx + tol, dy + tol), dz + tol)) {
    	    if (abs(dxx) <= tol && abs(dyy) <= tol && abs(dzz) <= dz + tol) {
            	    // okay so the cell is adjacent... which direction is it?
            	    if (dzz > 0) dir_ind = 4;
            	    else dir_ind = 5;
                	connectivity_inds(j, q, dir_ind) = dist_ind;
    		q += 1;
                	}
    	    else if (abs(dxx) <= tol && abs(dzz) <= tol && abs(dyy) <= dy + tol) {
            	    // okay so the cell is adjacent... which direction is it?
            	    if (dyy > 0) dir_ind = 2;
            	    else dir_ind = 3;
                	connectivity_inds(j, q, dir_ind) = dist_ind;
    		q += 1;
                	}
    	    else if (abs(dyy) <= tol && abs(dzz) <= tol && abs(dxx) <= dx + tol) {
            	    // okay so the cell is adjacent... which direction is it?
            	    if (dxx > 0) dir_ind = 0;
            	    else dir_ind = 1;
                	connectivity_inds(j, q, dir_ind) = dist_ind;
    		q += 1;
            	}
        }
		dist_ij[dist_ind] = 1e10; // eliminate the min to get the next min
        	}
    }
    return connectivity_inds;
}


// Calculate the geometrics factor from the polynomial basis functions 
Array current_voxels_geo_factors(Array& points, Array& integration_points, Array& plasma_normal, Array& Phi) {
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(Phi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("Phi needs to be in row-major storage order");

    int num_points = points.shape(0);
    int num_coil_points = Phi.shape(1);

    // integration points should be shape (num_coil_points, num_integration_points, 3)
    int num_integration_points = integration_points.shape(1);
    int num_basis_functions = Phi.shape(0);
    Array geo_factor = xt::zeros<double>({num_points, num_coil_points, num_basis_functions});
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_points; i++) {  // loop through quadrature points on plasma surface
	double nx = plasma_normal(i, 0);
	double ny = plasma_normal(i, 1);
	double nz = plasma_normal(i, 2);
	double rx = points(i, 0);
	double ry = points(i, 1);
	double rz = points(i, 2);
        for (int jj = 0; jj < num_coil_points; jj++) {  // loop through grid cells
            for (int j = 0; j < num_integration_points; j++) {  // integrate within a grid cell
	        double rprimex = integration_points(jj, j, 0);
	        double rprimey = integration_points(jj, j, 1);
	        double rprimez = integration_points(jj, j, 2);
	        double rvecx = rx - rprimex;
	        double rvecy = ry - rprimey;
	        double rvecz = rz - rprimez;
	        double rvec_mag = sqrt(rvecx * rvecx + rvecy * rvecy + rvecz * rvecz);
	        double rvec_inv3 = 1.0 / (rvec_mag * (rvec_mag * rvec_mag));
	        double n_cross_rvec_x = ny * rvecz - nz * rvecy;
	        double n_cross_rvec_y = nz * rvecx - nx * rvecz;
	        double n_cross_rvec_z = nx * rvecy - ny * rvecx;
                for (int k = 0; k < num_basis_functions; k++) {  // loop through the 11 linear basis functions
		    // minus sign below because it is really r x nhat but we computed nhat x r
                    geo_factor(i, jj, k) += - (n_cross_rvec_x * Phi(k, jj, j, 0) + n_cross_rvec_y * Phi(k, jj, j, 1) + n_cross_rvec_z * Phi(k, jj, j, 2)) * rvec_inv3;
                }
	    }
	}
    }
    return geo_factor;
} 


// Calculate the geometrics factor from the polynomial basis functions 
std::tuple<Array, Array_INT> current_voxels_flux_jumps(Array& coil_points, Array& Phi, double dx, double dy, double dz) {
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(coil_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("coil_points needs to be in row-major storage order");
    if(Phi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("Phi needs to be in row-major storage order");

    int num_coil_points = coil_points.shape(0);
    // here integration points should be shape (num_coil_points, Nx, Ny, Nz, 3)
    int Nx = Phi.shape(2);
    int Ny = Phi.shape(3);
    int Nz = Phi.shape(4);
    Array_INT Connect = connections(coil_points, dx, dy, dz);

    // here Phi should be shape (n_basis_functions, num_coil_points, Nx, Ny, Nz, 3)
    int num_basis_functions = Phi.shape(0);
    Array flux_factor = xt::zeros<double>({6, num_coil_points, num_basis_functions});
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_coil_points; i++) {
	for (int kk = 0; kk < 6; kk++) {
            int nx = 0;
            int ny = 0;
            int nz = 0;
	    if (kk == 0) nx = 1;
	    if (kk == 1) nx = -1;
	    if (kk == 2) ny = 1;
	    if (kk == 3) ny = -1;
	    if (kk == 4) nz = 1;
	    if (kk == 5) nz = -1;
	    if (kk < 2) {
		int x_ind = 0;
		if (kk == 0) x_ind = Nx - 1;
		for (int j = 0; j < Ny; j++) {
		    for (int p = 0; p < Nz; p++) {
			for (int k = 0; k < num_basis_functions; k++) {
			    flux_factor(kk, i, k) += (nx * Phi(k, i, x_ind, j, p, 0) + ny * Phi(k, i, x_ind, j, p, 1) + nz * Phi(k, i, x_ind, j, p, 2)) * dy * dz;
			}
		    }
		}
	    }
	    else if (kk < 4) {
		int y_ind = 0;
		if (kk == 2) y_ind = Ny - 1;
		for (int j = 0; j < Nx; j++) {
		    for (int p = 0; p < Nz; p++) {
			for (int k = 0; k < num_basis_functions; k++) {
			    flux_factor(kk, i, k) += (nx * Phi(k, i, j, y_ind, p, 0) + ny * Phi(k, i, j, y_ind, p, 1) + nz * Phi(k, i, j, y_ind, p, 2)) * dx * dz;
			}
		    }
		}
	    }
	    else {
		int z_ind = 0;
		if (kk == 4) z_ind = Nz - 1;
		for (int j = 0; j < Nx; j++) {
		    for (int p = 0; p < Ny; p++) {
			for (int k = 0; k < num_basis_functions; k++) {
			    flux_factor(kk, i, k) += (nx * Phi(k, i, j, p, z_ind, 0) + ny * Phi(k, i, j, p, z_ind, 1) + nz * Phi(k, i, j, p, z_ind, 2) * dx * dy);
			}
		    }
		}
            }
	}
    }
    return std::make_tuple(flux_factor, Connect);
}   
