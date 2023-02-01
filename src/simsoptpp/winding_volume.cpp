#include "winding_volume.h"

// compute which cells are next to which cells 
Array_INT connections(Array& coil_points, double dx, double dy, double dz)
{
    int Ndipole = coil_points.shape(0);
    double tol = 1e-5;
    
    // Last index indicates +- the x/y/z directions, -1 indicates no neighbor there
    Array_INT connectivity_inds = xt::ones<int>({Ndipole, 6, 6}) * (-1);
    
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
	// Need to loop through more than 6 in case some adjacent neighbors are
	// further away than some of the non-adjacent neighbors
	int q = 0;
        for (int k = 0; k < 30; ++k) {
	    auto result = std::min_element(dist_ij.begin(), dist_ij.end());
        int dist_ind = std::distance(dist_ij.begin(), result);

	    // Get dx, dy, dz from this ind
	    double dxx = -(coil_points(j, 0) - coil_points(dist_ind, 0));
	    double dyy = -(coil_points(j, 1) - coil_points(dist_ind, 1));
	    double dzz = -(coil_points(j, 2) - coil_points(dist_ind, 2));
	    double dist = dist_ij[dist_ind];
	    int dir_ind = -1;
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
	    //if (dir_ind >= 0) {
 	    //    printf("%f %f %f %d %d %d %d %d %d\n", abs(dxx), abs(dyy), abs(dzz), abs(dxx) <= tol, abs(dyy) <= tol, abs(dzz) <= tol, abs(dxx) <= dx + tol, abs(dyy) <= dy + tol, abs(dzz) <= dz + tol);
  	        //printf("%d %d %d %d %f %f %f %f %f %f %f %d\n", j, dist_ind, k, q, dist, dxx, dyy, dzz, dx, dy, dz, dir_ind);
            //}
		dist_ij[dist_ind] = 1e10; // eliminate the min to get the next min
        	}
    }
    return connectivity_inds;
}


// Calculate the geometrics factor from the polynomial basis functions 
Array winding_volume_geo_factors(Array& points, Array& integration_points, Array& plasma_normal, Array& Phi) {
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
std::tuple<Array, Array_INT> winding_volume_flux_jumps(Array& coil_points, Array& Phi, double dx, double dy, double dz) {
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


// Takes a uniform CARTESIAN grid of dipoles, and loops through
// and creates a final set of points which lie between the
// inner and outer toroidal surfaces defined by extending the plasma
// boundary by its normal vectors * some minimum distance. 
Array make_winding_volume_grid(Array& normal_inner, Array& normal_outer, Array& xyz_uniform, Array& xyz_inner, Array& xyz_outer)
{
    // For each toroidal cross-section:
    // For each dipole location:
    //     1. Find nearest point from dipole to the inner surface
    //     2. Find nearest point from dipole to the outer surface
    //     3. Select nearest point that is closest to the dipole
    //     4. Get normal vector of this inner/outer surface point
    //     5. Draw ray from dipole location in the direction of this normal vector
    //     6. If closest point between inner surface and the ray is the 
    //           start of the ray, conclude point is outside the inner surface. 
    //     7. If closest point between outer surface and the ray is the
    //           start of the ray, conclude point is outside the outer surface. 
    //     8. If Step 4 was True but Step 5 was False, add the point to the final grid.
   	
    int num_inner = xyz_inner.shape(0);
    int ngrid = xyz_uniform.shape(0);
    int num_ray = 2000;
    Array final_grid = xt::zeros<double>({ngrid, 3});

    // Loop through every dipole
#pragma omp parallel for schedule(static)
    for (int i = 0; i < ngrid; i++) {
        double X = xyz_uniform(i, 0);
        double Y = xyz_uniform(i, 1);
        double Z = xyz_uniform(i, 2);
           
	// find nearest point on inner/outer toroidal surface
	double min_dist_inner = 1e5;
	double min_dist_outer = 1e5;
	int inner_loc = 0;
	int outer_loc = 0;
        for (int k = 0; k < num_inner; k++) {
	    double x_inner = xyz_inner(k, 0);
	    double y_inner = xyz_inner(k, 1);
	    double z_inner = xyz_inner(k, 2);
	    double x_outer = xyz_outer(k, 0);
	    double y_outer = xyz_outer(k, 1);
	    double z_outer = xyz_outer(k, 2);
	    double dist_inner = (x_inner - X) * (x_inner - X) + (y_inner - Y) * (y_inner - Y) + (z_inner - Z) * (z_inner - Z); 
	    double dist_outer = (x_outer - X) * (x_outer - X) + (y_outer - Y) * (y_outer - Y) + (z_outer - Z) * (z_outer - Z); 
            if (dist_inner < min_dist_inner) {
		min_dist_inner = dist_inner;
	        inner_loc = k;
	    }
            if (dist_outer < min_dist_outer) {
		min_dist_outer = dist_outer;
	        outer_loc = k;
	    }
	}   
	double nx = 0.0;
	double ny = 0.0;
	double nz = 0.0;
	if (min_dist_inner < min_dist_outer) {
            nx = normal_inner(inner_loc, 0);
            ny = normal_inner(inner_loc, 1);
	    nz = normal_inner(inner_loc, 2);
	}
	else {
            nx = normal_outer(outer_loc, 0);
            ny = normal_outer(outer_loc, 1);
	    nz = normal_outer(outer_loc, 2);
	}
	// normalize the normal vectors
	double norm_vec = sqrt(nx * nx + ny * ny + nz * nz);
        double ray_x = nx / norm_vec;
        double ray_y = ny / norm_vec;
        double ray_z = nz / norm_vec;
           
	// Compute all the rays and find the location of minimum ray-surface distance
	double dist_inner_ray = 0.0;
	double dist_outer_ray = 0.0;
	double min_dist_inner_ray = 1e5;
	double min_dist_outer_ray = 1e5;
        int nearest_loc_inner = 0;
        int nearest_loc_outer = 0;
	double ray_equation_x = 0.0;
	double ray_equation_y = 0.0;
        double ray_equation_z = 0.0;
        for (int k = 0; k < num_ray; k++) {
	    ray_equation_x = X + ray_x * (4.0 / ((double) num_ray)) * k;
	    ray_equation_y = Y + ray_y * (4.0 / ((double) num_ray)) * k;
	    ray_equation_z = Z + ray_z * (4.0 / ((double) num_ray)) * k;
	    dist_inner_ray = (xyz_inner(inner_loc, 0) - ray_equation_x) * (xyz_inner(inner_loc, 0) - ray_equation_x) + (xyz_inner(inner_loc, 1) - ray_equation_y) * (xyz_inner(inner_loc, 1) - ray_equation_y) + (xyz_inner(inner_loc, 2) - ray_equation_z) * (xyz_inner(inner_loc, 2) - ray_equation_z);
	    dist_outer_ray = (xyz_outer(outer_loc, 0) - ray_equation_x) * (xyz_outer(outer_loc, 0) - ray_equation_x) + (xyz_outer(outer_loc, 1) - ray_equation_y) * (xyz_outer(outer_loc, 1) - ray_equation_y) + (xyz_outer(outer_loc, 2) - ray_equation_z) * (xyz_outer(outer_loc, 2) - ray_equation_z);
            if (dist_inner_ray < min_dist_inner_ray) {
		min_dist_inner_ray = dist_inner_ray;
		nearest_loc_inner = k;
            }
            if (dist_outer_ray < min_dist_outer_ray) {
		min_dist_outer_ray = dist_outer_ray;
		nearest_loc_outer = k;
	    }
	}
            
	// nearest distance from the inner surface to the ray should be just the original point
        if (nearest_loc_inner > 0) continue;
            
	// nearest distance from the outer surface to the ray should NOT be the original point
        if (nearest_loc_outer > 0) {
            final_grid(i, 0) = X;
	    final_grid(i, 1) = Y;
	    final_grid(i, 2) = Z;
	}
    }
    return final_grid; 
}

Array acc_prox_grad_descent(Eigen::SparseMatrix<double, Eigen::RowMajor> eigen_P, Array& B, Array& I, Array& bB, Array& bI, Array& alpha_initial, double lam, double initial_step, int max_iter) 
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(B.layout() != xt::layout_type::row_major)
          throw std::runtime_error("B needs to be in row-major storage order");	
    if(I.layout() != xt::layout_type::row_major)
          throw std::runtime_error("I needs to be in row-major storage order");
    if(bB.layout() != xt::layout_type::row_major)
          throw std::runtime_error("bB needs to be in row-major storage order");
    if(bI.layout() != xt::layout_type::row_major)
          throw std::runtime_error("bI needs to be in row-major storage order");
    if(alpha_initial.layout() != xt::layout_type::row_major)
          throw std::runtime_error("alpha_initial needs to be in row-major storage order");

    double step_size_i = initial_step;
    int hist_length = 100;
    int N = B.shape(1);
    int N_plasma = B.shape(0);
    Array alpha_opt = alpha_initial;
    Array vi = xt::zeros<double>({N});
    Array BTb = xt::zeros<double>({N});
    Array ITbI = xt::zeros<double>({N});
//     Array vector_constants = xt::zeros<double>({N});
    Array alpha_opt_prev = xt::zeros<double>({N});
    Array fB = xt::zeros<double>({hist_length + 1});
    Array fI = xt::zeros<double>({hist_length + 1});
    Array fK = xt::zeros<double>({hist_length + 1});
    Array f_B = xt::zeros<double>({1});
    Array f_I = xt::zeros<double>({1});
    Array f_K = xt::zeros<double>({1});
    
    // Define Eigen objects for optimization steps
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_B(const_cast<double*>(B.data()), N_plasma, N);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_BT(const_cast<double*>(xt::transpose(B).data()), N, N_plasma);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_I(const_cast<double*>(I.data()), 1, N);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_IT(const_cast<double*>(xt::transpose(I).data()), N, 1);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_b(const_cast<double*>(bB.data()), N_plasma, 1);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_bI(const_cast<double*>(bI.data()), 1, 1);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_BTb(const_cast<double*>(BTb.data()), N, 1);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_ITbI(const_cast<double*>(ITbI.data()), N, 1);
    eigen_BTb = eigen_BT * eigen_b;
    eigen_ITbI = eigen_IT * eigen_bI;
    Array vector_constants = BTb + ITbI;
    
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_BTb_ITbI(const_cast<double*>(vector_constants.data()), N, 1);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_res(const_cast<double*>(alpha_opt.data()), N, 1);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_v(const_cast<double*>(vi.data()), N, 1);
    //     Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_res2(const_cast<double*>(alpha_opt_prev.data()), N, 1);

    // Define Eigen objects for printing
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_fB(const_cast<double*>(f_B.data()), 1, 1);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_fI(const_cast<double*>(f_I.data()), 1, 1);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_fK(const_cast<double*>(f_K.data()), 1, 1);

    int q = 0;
    for (int i = 0; i < max_iter; i++) { 	
        //eigen_v = eigen_res + (i / (i + 3)) * (eigen_res - eigen_res2);
        vi = alpha_opt + (i / (i + 3)) * (alpha_opt - alpha_opt_prev);
        alpha_opt_prev = alpha_opt;
        Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_res(const_cast<double*>(alpha_opt.data()), N, 1);
        Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_v(const_cast<double*>(vi.data()), N, 1);
        eigen_res = eigen_P * (eigen_v + step_size_i * (eigen_BTb_ITbI - eigen_BT * (eigen_B * eigen_v) - eigen_IT * (eigen_I * eigen_v) - lam * eigen_v));
        step_size_i = (1 + sqrt(1 + 4 * step_size_i * step_size_i)) / 2.0;
//         eigen_res = eigen_P * (eigen_v + step_size_i * (eigen_BTb_ITBI - eigen_BT * (eigen_B * eigen_v) - eigen_IT * (eigen_I * eigen_v) - lam * eigen_v));
        if (i % (max_iter / 100) == 0) {
            eigen_fB = (eigen_B * eigen_res - eigen_b).transpose() * (eigen_B * eigen_res - eigen_b);
            eigen_fI = (eigen_I * eigen_res - eigen_bI).transpose() * (eigen_I * eigen_res - eigen_bI);
            eigen_fK = (eigen_res).transpose() * (eigen_res);
            fB(q) = f_B(0);
            fI(q) = f_I(0);
            fK(q) = f_K(0);
            printf("%d %e %e %e %e\n", i, f_B(0), f_I(0), f_K(0), step_size_i);
            q += 1;
        }
    }
    return alpha_opt;
}
