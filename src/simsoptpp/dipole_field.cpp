#include "dipole_field.h"
#include "simdhelpers.h"
#include "vec3dsimd.h"
#include <Eigen/Dense>

// Calculate the B field at a set of evaluation points from N dipoles
// points: where to evaluate the field
// m_points: where the dipoles are located
// m: dipole moments ('orientation')
// everything in xyz coordinates
Array dipole_field_B(Array& points, Array& m_points, Array& m) {
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(m_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("m_points needs to be in row-major storage order");
    if(m.layout() != xt::layout_type::row_major)
          throw std::runtime_error("m needs to be in row-major storage order");

    int num_points = points.shape(0);
    int num_dipoles = m_points.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    Array B = xt::zeros<double>({points.shape(0), points.shape(1)});
   
    // initialize pointers to the beginning of m and the dipole grid
    double* m_points_ptr = &(m_points(0, 0));
    double* m_ptr = &(m(0, 0));
    double fak = 1e-7;  // mu0 divided by 4 * pi factor

    // Loop through the evaluation points by chunks of simd_size
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_points; i += simd_size) {
        auto point_i = Vec3dSimd();
        auto B_i = Vec3dSimd();
        // check that i + k isn't bigger than num_points
        int klimit = std::min(simd_size, num_points - i);
        for(int k = 0; k < klimit; k++){
            for (int d = 0; d < 3; ++d) {
                point_i[d][k] = points(i + k, d);
            }
        }
        for (int j = 0; j < num_dipoles; ++j) {
            Vec3dSimd m_j = Vec3dSimd(m_ptr[3 * j + 0], m_ptr[3 * j + 1], m_ptr[3 * j + 2]);
            Vec3dSimd mp_j = Vec3dSimd(m_points_ptr[3 * j + 0], m_points_ptr[3 * j + 1], m_points_ptr[3 * j + 2]);
            Vec3dSimd r = point_i - mp_j;
            simd_t rmag_2     = normsq(r);
            simd_t rmag_inv   = rsqrt(rmag_2);
            simd_t rmag_inv_3 = rmag_inv * (rmag_inv * rmag_inv);
            simd_t rmag_inv_5 = rmag_inv_3 * (rmag_inv * rmag_inv);
            simd_t rdotm = inner(r, m_j);
            B_i.x += 3.0 * rdotm * r.x * rmag_inv_5 - m_j.x * rmag_inv_3;
            B_i.y += 3.0 * rdotm * r.y * rmag_inv_5 - m_j.y * rmag_inv_3;
            B_i.z += 3.0 * rdotm * r.z * rmag_inv_5 - m_j.z * rmag_inv_3;
        } 
        for(int k = 0; k < klimit; k++){
            B(i + k, 0) = fak * B_i.x[k];
            B(i + k, 1) = fak * B_i.y[k];
            B(i + k, 2) = fak * B_i.z[k];
        }
    }
    return B;
}

Array dipole_field_dB(Array& points, Array& m_points, Array& m) {
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(m_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("m_points needs to be in row-major storage order");
    if(m.layout() != xt::layout_type::row_major)
          throw std::runtime_error("m needs to be in row-major storage order");

    int num_points = points.shape(0);
    int num_dipoles = m_points.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    Array dB = xt::zeros<double>({points.shape(0), points.shape(1), points.shape(1)});
    double* m_points_ptr = &(m_points(0, 0));
    double* m_ptr = &(m(0, 0));
    double fak = 1e-7;
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_points; i += simd_size) {
        auto point_i = Vec3dSimd();
        auto dB_i1   = Vec3dSimd();
        auto dB_i2   = Vec3dSimd();
        int klimit = std::min(simd_size, num_points - i);
        for(int k = 0; k < klimit; k++){
            for (int d = 0; d < 3; ++d) {
                point_i[d][k] = points(i + k, d);
            }
        }
        for (int j = 0; j < num_dipoles; ++j) {
            Vec3dSimd m_j = Vec3dSimd(m_ptr[3 * j], m_ptr[3 * j + 1], m_ptr[3 * j + 2]);
            Vec3dSimd mp_j = Vec3dSimd(m_points_ptr[3 * j], m_points_ptr[3 * j + 1], m_points_ptr[3 * j + 2]);
            Vec3dSimd r = point_i - mp_j;
            simd_t rmag_2     = normsq(r);
            simd_t rmag_inv   = rsqrt(rmag_2);
	    simd_t rmag_inv_2 = rmag_inv * rmag_inv;
            simd_t rmag_inv_3 = rmag_inv * rmag_inv_2;
            simd_t rmag_inv_5 = rmag_inv_3 * rmag_inv_2; 
            simd_t rdotm = inner(r, m_j);
            dB_i1.x += 3.0 * rmag_inv_5 * ((2.0 * m_j.x * r.x + rdotm) - 5.0 * rdotm * r.x * r.x * rmag_inv_2);
            dB_i1.y += 3.0 * rmag_inv_5 * ((m_j.x * r.y + m_j.y * r.x) - 5.0 * rdotm * r.x * r.y * rmag_inv_2);
            dB_i1.z += 3.0 * rmag_inv_5 * ((m_j.x * r.z + m_j.z * r.x) - 5.0 * rdotm * r.x * r.z * rmag_inv_2);
            dB_i2.x += 3.0 * rmag_inv_5 * ((2.0 * m_j.y * r.y + rdotm) - 5.0 * rdotm * r.y * r.y * rmag_inv_2);
            dB_i2.y += 3.0 * rmag_inv_5 * ((m_j.y * r.z + m_j.z * r.y) - 5.0 * rdotm * r.y * r.z * rmag_inv_2);
            dB_i2.z += 3.0 * rmag_inv_5 * ((2.0 * m_j.z * r.z + rdotm) - 5.0 * rdotm * r.z * r.z * rmag_inv_2);
        } 
        for(int k = 0; k < klimit; k++){
            dB(i + k, 0, 0) = fak * dB_i1.x[k];
            dB(i + k, 0, 1) = fak * dB_i1.y[k];
            dB(i + k, 0, 2) = fak * dB_i1.z[k];
            dB(i + k, 1, 1) = fak * dB_i2.x[k];
            dB(i + k, 1, 2) = fak * dB_i2.y[k];
            dB(i + k, 2, 2) = fak * dB_i2.z[k];
	    dB(i + k, 1, 0) = dB(i + k, 0, 1);
	    dB(i + k, 2, 0) = dB(i + k, 0, 2);
	    dB(i + k, 2, 1) = dB(i + k, 1, 2);
	}
    }
    return dB;
}


// Calculate the geometric factor needed for the permanent magnet optimization
std::tuple<Array, Array, Array> dipole_field_Bn(Array& points, Array& m_points, Array& unitnormal, int nfp, int stellsym, Array& phi, Array& b) 
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(m_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("m_points needs to be in row-major storage order");
    if(unitnormal.layout() != xt::layout_type::row_major)
          throw std::runtime_error("unit normal needs to be in row-major storage order");
    if(phi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("phi needs to be in row-major storage order");
    if(b.layout() != xt::layout_type::row_major)
          throw std::runtime_error("b needs to be in row-major storage order");
    
    int nsym = nfp * (stellsym + 1);
    int num_points = points.shape(0);
    int num_dipoles = m_points.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    Array A = xt::zeros<double>({num_points, num_dipoles, 3});
    Array ATb = xt::zeros<double>({num_dipoles, 3});
    Array ATA = xt::zeros<double>({num_dipoles * 3, num_dipoles * 3});
   
    // initialize pointer to the beginning of the dipole grid
    double* m_points_ptr = &(m_points(0, 0));
    double fak = 1e-7;  // mu0 divided by 4 * pi factor

    // Loop through the evaluation points by chunks of simd_size
#pragma omp parallel for schedule(static)
    for(int i = 0; i < num_points; i += simd_size) {
        auto point_i = Vec3dSimd();
        auto n_i = Vec3dSimd();
        // check that i + k isn't bigger than num_points
        int klimit = std::min(simd_size, num_points - i);
        for(int k = 0; k < klimit; k++){
            for (int d = 0; d < 3; ++d) {
                point_i[d][k] = points(i + k, d);
                n_i[d][k] = unitnormal(i + k, d);
            }
        }
	for(int fp = 0; fp < nfp; fp++) {
	    simd_t phi0 = (2 * M_PI / ((simd_t) nfp)) * fp;
	    simd_t sphi0 = xsimd::sin(phi0);
	    simd_t cphi0 = xsimd::cos(phi0);
            for (int j = 0; j < num_dipoles; ++j) {
                auto G_i = Vec3dSimd();
                auto H_i = Vec3dSimd();
	        simd_t phi_sym = phi[j] + phi0;
	        simd_t sphi = xsimd::sin(phi_sym); 
	        simd_t cphi = xsimd::cos(phi_sym); 
                Vec3dSimd mp_j = Vec3dSimd(m_points_ptr[3 * j + 0], m_points_ptr[3 * j + 1], m_points_ptr[3 * j + 2]);
                simd_t mmag = xsimd::sqrt(xsimd::fma(mp_j.x, mp_j.x, mp_j.y * mp_j.y));
		simd_t mp_x_new = mmag * cphi;
		simd_t mp_y_new = mmag * sphi;
		Vec3dSimd mp_j_new = Vec3dSimd(mp_x_new, mp_y_new, mp_j.z);
		Vec3dSimd r = point_i - mp_j_new;
                simd_t rmag_2 = normsq(r);
                simd_t rmag_inv   = rsqrt(rmag_2);
                simd_t rmag_inv_3 = rmag_inv * (rmag_inv * rmag_inv);
                simd_t rmag_inv_5 = rmag_inv_3 * (rmag_inv * rmag_inv);
                simd_t rdotn = inner(r, n_i);
                G_i.x = 3.0 * rdotn * r.x * rmag_inv_5 - n_i.x * rmag_inv_3;
                G_i.y = 3.0 * rdotn * r.y * rmag_inv_5 - n_i.y * rmag_inv_3;
                G_i.z = 3.0 * rdotn * r.z * rmag_inv_5 - n_i.z * rmag_inv_3;
	        
		// stellarator symmetry means dipole grid -> (x, -y, -z)
		if (stellsym > 0) {
		    Vec3dSimd mp_j_stell = Vec3dSimd(mp_x_new, -mp_y_new, -mp_j.z);
		    r = point_i - mp_j_stell;
                    rmag_2 = normsq(r);
                    rmag_inv   = rsqrt(rmag_2);
                    rmag_inv_3 = rmag_inv * (rmag_inv * rmag_inv);
                    rmag_inv_5 = rmag_inv_3 * (rmag_inv * rmag_inv);
                    rdotn = inner(r, n_i);
		    H_i.x = 3.0 * rdotn * r.x * rmag_inv_5 - n_i.x * rmag_inv_3;
                    H_i.y = 3.0 * rdotn * r.y * rmag_inv_5 - n_i.y * rmag_inv_3;
                    H_i.z = 3.0 * rdotn * r.z * rmag_inv_5 - n_i.z * rmag_inv_3;
		}
		for(int k = 0; k < klimit; k++){
		    A(i + k, j, 0) += fak * (G_i.x[k] * cphi0[k] - G_i.y[k] * sphi0[k]);
		    A(i + k, j, 1) += fak * (G_i.x[k] * sphi0[k] + G_i.y[k] * cphi0[k]);
		    A(i + k, j, 2) += fak * G_i.z[k];
		    // if stellsym, flip sign of x component here
		    if (stellsym > 0) {
		        A(i + k, j, 0) += - fak * (H_i.x[k] * cphi0[k] - H_i.y[k] * sphi0[k]);
		        A(i + k, j, 1) += fak * (H_i.x[k] * sphi0[k] + H_i.y[k] * cphi0[k]);
		        A(i + k, j, 2) += fak * H_i.z[k];
		    }
		}
	    }
	}
    }
    // compute ATb
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_mat(const_cast<double*>(A.data()), num_points, 3 * num_dipoles);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_v(const_cast<double*>(b.data()), 1, num_points);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_res(const_cast<double*>(ATb.data()), 1, 3 * num_dipoles);
    eigen_res = eigen_v*eigen_mat;
    // compute ATA
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_res2(const_cast<double*>(ATA.data()), 3 * num_dipoles,  3 * num_dipoles);
    eigen_res2 = eigen_mat.transpose()*eigen_mat;
    return std::make_tuple(A, ATb, ATA);
}

// Takes a uniform grid of dipoles, and loops through
// and creates a final set of points which lie between the
// inner and outer toroidal surfaces corresponding to the permanent
// magnet surface. 
std::tuple<Array, Array> make_final_surface(Array& phi, Array& normal_inner, Array& normal_outer, Array& dipole_grid_rz, Array& r_inner, Array& r_outer, Array& z_inner, Array& z_outer)
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
   	
    int ntheta = normal_inner.shape(1);
    int num_inner = r_inner.shape(1);
    int num_outer = r_outer.shape(1);
    int rz_max = dipole_grid_rz.shape(0);
    int nphi = phi.shape(0);
    int num_ray = 1000;
    int inner_loc = 0;
    int outer_loc = 0;
    int ind_count = 0;
    int nearest_loc = 0;
    int nearest_loc_inner = 0;
    int nearest_loc_outer = 0;
    double min_dist_inner = 0.0;
    double min_dist_outer = 0.0;
    double min_dist_inner_ray = 0.0;
    double min_dist_outer_ray = 0.0;
    //Array dist_inner = xt::zeros<double>({num_inner});
    //Array dist_outer = xt::zeros<double>({num_inner});
    double dist_inner_ray = 0.0;
    double dist_outer_ray = 0.0;
    Array inds = xt::zeros<int>({nphi});

    // initialize gigantic new_grids and chop it later 
    // in the python code
    Array new_grids = xt::zeros<double>({rz_max, 3});
    Array normal_inner_i = xt::zeros<double>({ntheta, 3});
    Array normal_outer_i = xt::zeros<double>({ntheta, 3});
    double ray_equation_r = 0.0;
    double ray_equation_z = 0.0; 
    double ray_dir_r = 0.0;
    double ray_dir_z = 0.0;
    double Rpoint, Zpoint;

    printf("%d %d %d %d %d\n", nphi, ntheta, num_inner, num_outer, rz_max);  

    // #pragma omp parallel for schedule(static)
    for (int i = 0; i < nphi; i++) {
        // ind_count = 0;
	// rotate normal vectors in (r, phi, z) coordinates and set phi component to zero
        // so that we keep everything in the same phi = constant cross-section
        double rot_matrix[3][3] = {{cos(phi(i)), sin(phi(i)), 0}, {-sin(phi(i)), cos(phi(i)), 0}, {0, 0, 1}};
	for (int j = 0; j < ntheta; j++) {
	    printf("%d %d\n", i, j);
            // rotate the normal vectors and ignore the phi component
	    double normal_inner_r = rot_matrix[0][0] * normal_inner(i, j, 0) + rot_matrix[0][1] * normal_inner(i, j, 1) + rot_matrix[0][2] * normal_inner(i, j, 2);
	    double normal_inner_z = normal_inner(i, j, 2);
	    double normal_outer_r = rot_matrix[0][0] * normal_outer(i, j, 0) + rot_matrix[0][1] * normal_outer(i, j, 1) + rot_matrix[0][2] * normal_outer(i, j, 2);
	    double normal_outer_z = normal_outer(i, j, 2);
	    
	    // normalize the rotated unit vectors
            normal_inner_r = normal_inner_r / sqrt(normal_inner_r * normal_inner_r + normal_inner_z * normal_inner_z);
            normal_inner_z = normal_inner_z / sqrt(normal_inner_r * normal_inner_r + normal_inner_z * normal_inner_z);
            normal_outer_r = normal_outer_r / sqrt(normal_outer_r * normal_outer_r + normal_outer_z * normal_outer_z);
            normal_outer_z = normal_outer_z / sqrt(normal_outer_r * normal_outer_r + normal_outer_z * normal_outer_z);
	    printf("%d %d %f %f %f \n", i, j, normal_inner_r, normal_inner_i(j, 1), normal_outer_z);
	    printf("%d %d %f %f %f \n", i, j, normal_outer_r, normal_outer_i(j, 1), normal_outer_z);
        }
	for (int j = 0; j < rz_max; j++) {
	    // Get (R, Z) locations of the points with respect to the magnetic axis
	    Rpoint = dipole_grid_rz(j, 0);
            Zpoint = dipole_grid_rz(j, 1);
           
	    // find nearest point on inner/outer toroidal surface
	    min_dist_inner = 1e5;
	    min_dist_outer = 1e5;
            for (int k = 0; k < num_inner; k++) {
	        double dist_inner = (r_inner(i, k) - Rpoint) * (r_inner(i, k) - Rpoint) + (z_inner(i, k) - Zpoint) * (z_inner(i, k) - Zpoint);
	        double dist_outer = (r_outer(i, k) - Rpoint) * (r_outer(i, k) - Rpoint) + (z_outer(i, k) - Zpoint) * (z_outer(i, k) - Zpoint);
                if (dist_inner < min_dist_inner) {
		    min_dist_inner = dist_inner;
	            inner_loc = k;
		}
                if (dist_outer < min_dist_outer) {
		    min_dist_outer = dist_outer;
	            outer_loc = k;
		}
	    }

	    double dist_inner = (r_inner(i, inner_loc) - Rpoint) * (r_inner(i, inner_loc) - Rpoint) + (z_inner(i, inner_loc) - Zpoint) * (z_inner(i, inner_loc) - Zpoint);
	    double dist_outer = (r_outer(i, outer_loc) - Rpoint) * (r_outer(i, outer_loc) - Rpoint) + (z_outer(i, outer_loc) - Zpoint) * (z_outer(i, outer_loc) - Zpoint);
	    double normal_inner_r = rot_matrix[0][0] * normal_inner(i, nearest_loc, 0) + rot_matrix[0][1] * normal_inner(i, nearest_loc, 1) + rot_matrix[0][2] * normal_inner(i, nearest_loc, 2);
	    double normal_inner_z = normal_inner(i, nearest_loc, 2);
	    double normal_outer_r = rot_matrix[0][0] * normal_outer(i, nearest_loc, 0) + rot_matrix[0][1] * normal_outer(i, nearest_loc, 1) + rot_matrix[0][2] * normal_outer(i, nearest_loc, 2);
	    double normal_outer_z = normal_outer(i, nearest_loc, 2);
	    // normalize the rotated unit vectors
            normal_inner_r = normal_inner_r / sqrt(normal_inner_r * normal_inner_r + normal_inner_z * normal_inner_z);
            normal_inner_z = normal_inner_z / sqrt(normal_inner_r * normal_inner_r + normal_inner_z * normal_inner_z);
            normal_outer_r = normal_outer_r / sqrt(normal_outer_r * normal_outer_r + normal_outer_z * normal_outer_z);
            normal_outer_z = normal_outer_z / sqrt(normal_outer_r * normal_outer_r + normal_outer_z * normal_outer_z);
	    
	    if (dist_inner < dist_outer) {
		nearest_loc = inner_loc;
                ray_dir_r = normal_inner_r;
                // ray_dir(1) = normal_inner(i, nearest_loc, 1)
                ray_dir_z = normal_inner_z;
	    }
	    else {
                nearest_loc = outer_loc;
                ray_dir_r = normal_outer_r;
                // ray_dir(1) = normal_outer(i, nearest_loc, 1)
                ray_dir_z = normal_outer_z;
	    }
            // printf("%d, %f, %f\n", nearest_loc, ray_dir_r, ray_dir_z);
	    min_dist_inner_ray = 1e5;
	    min_dist_outer_ray = 1e5;
            nearest_loc_inner = 0;
            nearest_loc_outer = 0;
            for (int k = 0; k < num_ray; k++) {
	        ray_equation_r = Rpoint + ray_dir_r * (2.0 / ((double) num_ray)) * k;
	        ray_equation_z = Zpoint + ray_dir_z * (2.0 / ((double) num_ray)) * k;
	        dist_inner_ray = (r_inner(i, inner_loc) - ray_equation_r) * (r_inner(i, inner_loc) - ray_equation_r) + (z_inner(i, inner_loc) - ray_equation_z) * (z_inner(i, inner_loc) - ray_equation_z);
	        dist_outer_ray = (r_outer(i, outer_loc) - ray_equation_r) * (r_outer(i, outer_loc) - ray_equation_r) + (z_outer(i, outer_loc) - ray_equation_z) * (z_outer(i, outer_loc) - ray_equation_z);
                if (dist_inner_ray < min_dist_inner_ray) {
		    min_dist_inner_ray = dist_inner_ray;
		    nearest_loc_inner = k;
		}
                if (dist_outer_ray < min_dist_outer_ray) {
		    min_dist_outer_ray = dist_outer_ray;
		    nearest_loc_outer = k;
		}
	    }
	    // printf("%d %d %d %d\n", i, j, nearest_loc_inner, nearest_loc_outer);
            // nearest distance from the inner surface to the ray should be just the original point
	    if (nearest_loc_inner > 0)
                continue;
            // nearest distance from the outer surface to the ray should be NOT be the original point
            if (nearest_loc_outer > 0) {
                new_grids(ind_count, 0) = Rpoint;
                new_grids(ind_count, 1) = phi(i); 
                new_grids(ind_count, 2) = Zpoint;
		ind_count += 1;
	    }
	}
	// count number of elements that were set during ith iteration
	printf("%d\n", ind_count);
	inds(i) = ind_count;
	printf("%d %d\n", i, inds(i));
    }
    // combine inds
    //for (int i = 1; i < nphi; i++) {
//	for (int j = i; j > 0; j = j - 1) {
//	    // printf("%d %d %d\n", i, j, inds(i));
//	    inds(i) += inds(j - 1);
  //      }
    //}
    return std::make_tuple(new_grids, inds);
}
