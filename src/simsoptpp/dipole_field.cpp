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
        // Loops through all the dipoles
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

Array dipole_field_A(Array& points, Array& m_points, Array& m) {
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
    Array A = xt::zeros<double>({points.shape(0), points.shape(1)});
   
    // initialize pointers to the beginning of m and the dipole grid
    double* m_points_ptr = &(m_points(0, 0));
    double* m_ptr = &(m(0, 0));
    double fak = 1e-7;  // mu0 divided by 4 * pi factor

    // Loop through the evaluation points by chunks of simd_size
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_points; i += simd_size) {
        auto point_i = Vec3dSimd();
        auto A_i = Vec3dSimd();

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
            Vec3dSimd mcrossr = cross(m_j, r);
            A_i.x += mcrossr.x * rmag_inv_3;
            A_i.y += mcrossr.y * rmag_inv_3;
            A_i.z += mcrossr.z * rmag_inv_3;
        } 
        for(int k = 0; k < klimit; k++){
            A(i + k, 0) = fak * A_i.x[k];
            A(i + k, 1) = fak * A_i.y[k];
            A(i + k, 2) = fak * A_i.z[k];
        }
    }
    return A;
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

Array dipole_field_dA(Array& points, Array& m_points, Array& m) {
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
    Array dA = xt::zeros<double>({points.shape(0), points.shape(1), points.shape(1)});
    double* m_points_ptr = &(m_points(0, 0));
    double* m_ptr = &(m(0, 0));
    double fak = 1e-7;
    
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_points; i += simd_size) {
        auto point_i = Vec3dSimd();
        auto dA_i1   = Vec3dSimd();
        auto dA_i2   = Vec3dSimd();
        auto dA_i3   = Vec3dSimd();
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
            Vec3dSimd mcrossr = cross(m_j, r);
            dA_i1.x += rmag_inv_3 * (- 3.0 * mcrossr.x * r.x * rmag_inv_2);
            dA_i1.y += rmag_inv_3 * (- m_j.z - 3.0 * mcrossr.x * r.y * rmag_inv_2);
            dA_i1.z += rmag_inv_3 * (m_j.y - 3.0 * mcrossr.x * r.z * rmag_inv_2);
            dA_i2.x += rmag_inv_3 * (m_j.z - 3.0 * mcrossr.y * r.x * rmag_inv_2);
            dA_i2.y += rmag_inv_3 * (- 3.0 * mcrossr.y * r.y * rmag_inv_2);
            dA_i2.z += rmag_inv_3 * (- m_j.x - 3.0 * mcrossr.y * r.z * rmag_inv_2);
            dA_i3.x += rmag_inv_3 * (- m_j.y - 3.0 * mcrossr.z * r.x * rmag_inv_2);
            dA_i3.y += rmag_inv_3 * (m_j.x - 3.0 * mcrossr.z * r.y * rmag_inv_2);
            dA_i3.z += rmag_inv_3 * (- 3.0 * mcrossr.z * r.z * rmag_inv_2);
	} 
        for(int k = 0; k < klimit; k++){
            dA(i + k, 0, 0) = fak * dA_i1.x[k];
            dA(i + k, 0, 1) = fak * dA_i1.y[k];
            dA(i + k, 0, 2) = fak * dA_i1.z[k];
            dA(i + k, 1, 0) = fak * dA_i2.x[k];
            dA(i + k, 1, 1) = fak * dA_i2.y[k];
            dA(i + k, 1, 2) = fak * dA_i2.z[k];
	    dA(i + k, 2, 0) = fak * dA_i3.x[k];
	    dA(i + k, 2, 1) = fak * dA_i3.y[k]; 
            dA(i + k, 2, 2) = fak * dA_i3.z[k];
	}
    }
    return dA;
}

// Calculate the geometric factor needed for the permanent magnet optimization
std::tuple<Array, Array> dipole_field_Bn(Array& points, Array& m_points, Array& unitnormal, int nfp, int stellsym, Array& phi, Array& b, bool cylindrical) 
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
    
    int num_points = points.shape(0);
    int num_dipoles = m_points.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    Array A = xt::zeros<double>({num_points, num_dipoles, 3});
    Array ATb = xt::zeros<double>({num_dipoles, 3});
   
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
	// Loop through all the dipoles, using all the symmetries
        for (int j = 0; j < num_dipoles; ++j) {
            Vec3dSimd mp_j = Vec3dSimd(m_points_ptr[3 * j + 0], m_points_ptr[3 * j + 1], m_points_ptr[3 * j + 2]);
	    for (int stell = 0; stell < (stellsym + 1); ++stell) { 
	        for(int fp = 0; fp < nfp; ++fp) {
	            simd_t phi0 = (2 * M_PI / ((simd_t) nfp)) * fp;
	            simd_t sphi0 = xsimd::sin(phi0);
	            simd_t cphi0 = xsimd::cos(phi0);
		    auto G_i = Vec3dSimd();
                    
                    // Calculate new dipole location after accounting for the symmetries
	            // reflect the y and z-components and then rotate by phi0
		    simd_t mp_x_new = mp_j.x * cphi0 - mp_j.y * sphi0 * pow(-1, stell);
                    simd_t mp_y_new = mp_j.x * sphi0 + mp_j.y * cphi0 * pow(-1, stell);
		    Vec3dSimd mp_j_new = Vec3dSimd(mp_x_new, mp_y_new, mp_j.z * pow(-1, stell));
		    
		    // Calculate new phi location if switching to cylindrical coordinates
		    simd_t mp_phi_new = xsimd::atan2(mp_y_new, mp_x_new);
		    simd_t sphi_new = xsimd::sin(mp_phi_new);
		    simd_t cphi_new = xsimd::cos(mp_phi_new);
		    
		    Vec3dSimd r = point_i - mp_j_new;
                    simd_t rmag_2 = normsq(r);
                    simd_t rmag_inv   = rsqrt(rmag_2);
                    simd_t rmag_inv_3 = rmag_inv * (rmag_inv * rmag_inv);
                    simd_t rmag_inv_5 = rmag_inv_3 * (rmag_inv * rmag_inv);
                    simd_t rdotn = inner(r, n_i);
                    G_i.x = 3.0 * rdotn * r.x * rmag_inv_5 - n_i.x * rmag_inv_3;
                    G_i.y = 3.0 * rdotn * r.y * rmag_inv_5 - n_i.y * rmag_inv_3;
                    G_i.z = 3.0 * rdotn * r.z * rmag_inv_5 - n_i.z * rmag_inv_3;
		    for(int k = 0; k < klimit; k++){
		        A(i + k, j, 2) += fak * G_i.z[k];
		        if (cylindrical) {
			    double Ax_temp = fak * (G_i.x[k] * cphi0[k] + G_i.y[k] * sphi0[k]) * pow(-1, stell);
			    double Ay_temp = fak * (- G_i.x[k] * sphi0[k] + G_i.y[k] * cphi0[k]);
			    A(i + k, j, 0) += fak * (Ax_temp * cphi_new[k] + Ay_temp * sphi_new[k]);
			    A(i + k, j, 1) += fak * ( - Ax_temp * sphi_new[k] + Ay_temp * cphi_new[k]);
		        }
		        else {
			    // rotate by -phi0 and then flip x component
			    // This should be the reverse of what is done to the m vector and the dipole grid
			    // because A * m = A * R^T * R * m and R is an orthogonal matrix both
			    // for a reflection and a rotation. 
			    A(i + k, j, 0) += fak * (G_i.x[k] * cphi0[k] + G_i.y[k] * sphi0[k]) * pow(-1, stell);
			    A(i + k, j, 1) += fak * (- G_i.x[k] * sphi0[k] + G_i.y[k] * cphi0[k]);
		        }
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
    return std::make_tuple(A, ATb);
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
    int num_ray = 2000;
    Array inds = xt::zeros<int>({nphi});

    // initialize new_grids with size of the full uniform grid,
    // and then chop later in the python part of the code
    Array new_grids = xt::zeros<double>({rz_max * nphi, 3});

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nphi; i++) {
        int ind_count = 0;
	double phi_i = phi(i);
        double rot_matrix[3] = {cos(phi_i), sin(phi_i), 0};
	for (int j = 0; j < rz_max; j++) {
	    // Get (R, Z) locations of the points with respect to the magnetic axis
	    double Rpoint = dipole_grid_rz(j, i, 0);
            double Zpoint = dipole_grid_rz(j, i, 2);
           
	    // find nearest point on inner/outer toroidal surface
	    double min_dist_inner = 1e5;
	    double min_dist_outer = 1e5;
	    int inner_loc = 0;
	    int outer_loc = 0;
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
	    
	    // Figure out which surface is closest to the point in question
	    // and then use the normal vector associated with that closest point 
	    
	    // rotate normal vectors in (r, phi, z) coordinates and set phi component to zero
            // so that we keep everything in the same phi = constant cross-section
	    double normal_vec_r = 0.0;
	    double normal_vec_z = 0.0;
	    if (min_dist_inner < min_dist_outer) {
                normal_vec_r = rot_matrix[0] * normal_inner(i, inner_loc, 0) + rot_matrix[1] * normal_inner(i, inner_loc, 1);
	        normal_vec_z = normal_inner(i, inner_loc, 2);
	    }
	    else {
	        normal_vec_r = rot_matrix[0] * normal_outer(i, outer_loc, 0) + rot_matrix[1] * normal_outer(i, outer_loc, 1);
	        normal_vec_z = normal_outer(i, outer_loc, 2);
	    }
	    // normalize the rotated unit vectors
	    double norm_vec = sqrt(normal_vec_r * normal_vec_r + normal_vec_z * normal_vec_z);
            double ray_dir_r = normal_vec_r / norm_vec;
            double ray_dir_z = normal_vec_z / norm_vec;
           
	    // Compute all the rays and find the location of minimum ray-surface distance
	    double dist_inner_ray = 0.0;
	    double dist_outer_ray = 0.0;
	    double min_dist_inner_ray = 1e5;
	    double min_dist_outer_ray = 1e5;
            int nearest_loc_inner = 0;
            int nearest_loc_outer = 0;
	    double ray_equation_r = 0.0;
	    double ray_equation_z = 0.0;
            for (int k = 0; k < num_ray; k++) {
	        ray_equation_r = Rpoint + ray_dir_r * (4.0 / ((double) num_ray)) * k;
	        ray_equation_z = Zpoint + ray_dir_z * (4.0 / ((double) num_ray)) * k;
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
            
	    // nearest distance from the inner surface to the ray should be just the original point
	    if (nearest_loc_inner > 0)
                continue;
            // nearest distance from the outer surface to the ray should be NOT be the original point
            if (nearest_loc_outer > 0) {
                new_grids(ind_count + i * rz_max, 0) = Rpoint;
                new_grids(ind_count + i * rz_max, 1) = phi_i; 
                new_grids(ind_count + i * rz_max, 2) = Zpoint;
		ind_count += 1;
	    }
	}
	inds(i) = ind_count;
    }
    return std::make_tuple(new_grids, inds);
}
