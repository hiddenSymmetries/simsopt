#include "winding_volume.h"

// compute which cells are next to which cells 
Array connections(Array& coil_points, int Nadjacent, int dx, int dy, int dz)
{
    int Ndipole = coil_points.shape(0);
    
    // Last index indicates +- the x/y/z directions
    Array connectivity_inds = xt::zeros<int>({Ndipole, 6, 6});
    
    // Compute distances between dipole j and all other dipoles
    // By default computes distance between dipole j and itself
#pragma omp parallel for schedule(static)
    for (int j = 0; j < Ndipole; ++j) {
	vector<double> dist_ij(Ndipole, 1e10);
        for (int i = 0; i < Ndipole; ++i) {
	    if (i != j) {
	        dist_ij[i] = sqrt((coil_points(i, 0) - coil_points(j, 0)) * (coil_points(i, 0) - coil_points(j, 0)) + (coil_points(i, 1) - coil_points(j, 1)) * (coil_points(i, 1) - coil_points(j, 1)) + (coil_points(i, 2) - coil_points(j, 2)) * (coil_points(i, 2) - coil_points(j, 2)));
	    }
        }
        for (int k = 0; k < 6; ++k) {
	    auto result = std::min_element(dist_ij.begin(), dist_ij.end());
            int dist_ind = std::distance(dist_ij.begin(), result);
	    // Get dx, dy, dz from this ind
	    dxx = coil_points(i, 0) - coil_points(dist_ind, 0);
	    dyy = coil_points(i, 1) - coil_points(dist_ind, 1);
	    dzz = coil_points(i, 2) - coil_points(dist_ind, 2);
	    if (dxx > dx) {
	        connectivity_inds(j, k, 0) = -1; // -1 to indicate no adjacent cell
	    }
	    else if (dxx < -dx) {
	        connectivity_inds(j, k, 1) = -1;
	    }
	    else if (dyy > dy) {
	        connectivity_inds(j, k, 2) = -1;
	    }
	    else if (dyy < -dy) {
	        connectivity_inds(j, k, 3) = -1;
	    }
	    else if (dzz > dz) {
	        connectivity_inds(j, k, 4) = -1;
	    }
	    else if (dzz < -dz) {
	        connectivity_inds(j, k, 5) = -1;
	    }
	    else if ((abs(dxx) > abs(dyy)) && (abs(dxx) > abs(dzz)) && dxx > 0.0) connectivity_inds(j, k, 0) = dist_ind;
	    else if ((abs(dxx) > abs(dyy)) && (abs(dxx) > abs(dzz)) && dxx < 0.0) connectivity_inds(j, k, 1) = dist_ind;
	    else if ((abs(dyy) > abs(dxx)) && (abs(dyy) > abs(dzz)) && dyy > 0.0) connectivity_inds(j, k, 2) = dist_ind;
	    else if ((abs(dyy) > abs(dxx)) && (abs(dyy) > abs(dzz)) && dyy < 0.0) connectivity_inds(j, k, 3) = dist_ind;
	    else if ((abs(dzz) > abs(dxx)) && (abs(dzz) > abs(dyy)) && dzz > 0.0) connectivity_inds(j, k, 4) = dist_ind;
	    else if ((abs(dzz) > abs(dxx)) && (abs(dzz) > abs(dyy)) && dzz < 0.0) connectivity_inds(j, k, 5) = dist_ind;
            dist_ij[dist_ind] = 1e10; // eliminate the min to get the next min
	}
    }
    return connectivity_inds;
}



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


// Calculate the geometrics factor from the polynomial basis functions 
Array winding_volume_flux_jumps(Array& coil_points, Array& integration_points, Array& Phi, double dx, double dy, double dz) {
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(coil_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("coil_points needs to be in row-major storage order");
    if(Phi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("Phi needs to be in row-major storage order");

    int num_coil_points = coil_points.shape(0);
    // here integration points should be shape (num_coil_points, Nx, Ny, Nz, 3)
    int Nx = integration_points.shape(1);
    int Ny = integration_points.shape(2);
    int Nz = integration_points.shape(3);
    Array Connect = connections(coil_points, 6, dx, dy, dz);

    // here Phi should be shape (n_basis_functions, num_coil_points, Nx, Ny, Nz, 3)
    int num_basis_functions = Phi.shape(0);
    Array flux_factor = xt::zeros<double>({6, num_coil_points, num_basis_functions});
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_coil_points; i++) {
        for (int jj = 0; jj < 6; jj++) {
	    for (int kk = 0; kk < 6; kk++) {
	        int cj = Connect(i, jj + 1, kk);
	        if (cj == 0.0) continue
                int nx = 0;
                int ny = 0;
                int nz = 0;
		if (kk == 0) nx = 1;
		if (kk == 1) nx = -1;
		if (kk == 2) ny = 1;
		if (kk == 3) ny = -1;
		if (kk == 4) nz = 1;
		if (kk == 5) nz = -1;
	        // cj < 0 indicates that the neighbor is further than 
	        // directly adjacent to cell i (perhaps because 
	        // the cell is at the end of the mesh), 
	        // so need to zero the flux on that surface
	        if (cj < 0) {
		}
		// cj > 0 indicates an adjacent neighbor
		else {
		}
	    // figure out where the normal vector points
	    double dxx = abs(coil_points[i, 0] - coil_points[cj, 0]); 
	    double dyy = abs(coil_points[i, 1] - coil_points[cj, 1]); 
	    double dzz = abs(coil_points[i, 2] - coil_points[cj, 2]);
	    double nx = 0;
	    double ny = 0;
	    double nz = 0;
	    if (dxx > dyy && dxx > dzz) {
	        nx = 1;
	        int x_ind = (i % (Ny * Nz));
                for (int j = 0; j < Ny; j++) {
                    for (int p = 0; p < Nz; p++) {
                        double rx = integration_points(i, x_ind, j, p, 0);
                        double ry = integration_points(i, x_ind, j, p, 1);
                        double rz = integration_points(i, x_ind, j, p, 2); 
                        for (int k = 0; k < num_basis_functions; k++) {
                            flux_factor(jj, i, k) += (nx * (Phi(k, i, x_ind, j, p, 0) - Phi(k, cj, x_ind, j, p, 0)) + ny * (Phi(k, i, x_ind, j, p, 1) - Phi(k, cj, x_ind, j, p, 1)) + nz * (Phi(k, i, x_ind, j, p, 2) - Phi(k, cj, x_ind, j, p, 2)));
			}
		    }
		}
            }
	    if (dyy > dxx && dyy > dzz) {
	        ny = 1;
	        int y_ind = (i % (Nx * Nz));
                for (int j = 0; j < Nx; j++) {
                    for (int p = 0; p < Nz; p++) {
                        double rx = integration_points(i, j, y_ind, p, 0);
                        double ry = integration_points(i, j, y_ind, p, 1);
                        double rz = integration_points(i, j, y_ind, p, 2); 
                        for (int k = 0; k < num_basis_functions; k++) {
                            flux_factor(jj, i, k) += (nx * (Phi(k, i, j, y_ind, p, 0) - Phi(k, cj, j, y_ind, p, 0)) + ny * (Phi(k, i, j, y_ind, p, 1) - Phi(k, cj, j, y_ind, p, 1)) + nz * (Phi(k, i, j, y_ind, p, 2) - Phi(k, cj, j, y_ind, p, 2)));
			}
		    }
		}
	    }
	    if (dzz > dyy && dzz > dxx) {
	        nz = 1;
	        int z_ind = (i % (Ny * Nz));
                for (int j = 0; j < Nx; j++) {
                    for (int p = 0; p < Ny; p++) {
                        double rx = integration_points(i, j, p, z_ind, 0);
                        double ry = integration_points(i, j, p, z_ind, 1);
                        double rz = integration_points(i, j, p, z_ind, 2); 
                        for (int k = 0; k < num_basis_functions; k++) {
                            flux_factor(jj, i, k) += (nx * (Phi(k, i, j, p, z_ind, 0) - Phi(k, cj, j, p, z_ind, 0)) + ny * (Phi(k, i, j, p, z_ind, 1) - Phi(k, cj, j, p, z_ind, 1)) + nz * (Phi(k, i, j, p, z_ind, 2) - Phi(k, cj, j, p, z_ind, 2)));
			}
		    }
		}
	    }
	}
    }
    return flux_factor;
}   
