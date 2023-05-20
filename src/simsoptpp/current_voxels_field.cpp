#include "current_voxels_field.h"
#include "simdhelpers.h"
#include "vec3dsimd.h"
#include <cmath>

#if defined(USE_XSIMD)
// Calculate the geometrics factor from the polynomial basis functions 
Array current_voxels_field_B(Array& points, Array& integration_points, Array& J) 
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(integration_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("integration_points needs to be in row-major storage order");
    if(J.layout() != xt::layout_type::row_major)
          throw std::runtime_error("J needs to be in row-major storage order");
    constexpr int simd_size = xsimd::simd_type<double>::size;

    int num_points = points.shape(0);

    // J should be shape (num_coil_points, num_integration_points, 3)
    int num_coil_points = J.shape(0);
    double* J_ptr = &(J(0, 0, 0));
    double* ip_points_ptr = &(integration_points(0, 0, 0));

    // integration points should be shape (num_coil_points, num_integration_points, 3)
    int num_integration_points = integration_points.shape(1);

    double fak = 1e-7;
    Array B = xt::zeros<double>({num_points, 3});
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
        for (int jj = 0; jj < num_coil_points; jj++) {  // loop through grid cells
            for (int j = 0; j < num_integration_points; j++) {  // integrate within a grid cell
                Vec3dSimd J = Vec3dSimd(J_ptr[jj * num_integration_points * 3 + 3 * j + 0], J_ptr[jj * num_integration_points * 3 + 3 * j + 1], J_ptr[jj * num_integration_points * 3 + 3 * j + 2]);
                Vec3dSimd point_j = Vec3dSimd(ip_points_ptr[jj * num_integration_points * 3 + 3 * j + 0], ip_points_ptr[jj * num_integration_points * 3 + 3 * j + 1], ip_points_ptr[jj * num_integration_points * 3 + 3 * j + 2]);
                Vec3dSimd r = point_i - point_j;
                simd_t rmag_2     = normsq(r);
                simd_t rmag_inv   = rsqrt(rmag_2);
                simd_t rmag_inv_3 = rmag_inv * (rmag_inv * rmag_inv);
                Vec3dSimd Jcrossr = cross(J, r);
                B_i.x += Jcrossr.x * rmag_inv_3;
                B_i.y += Jcrossr.y * rmag_inv_3;
                B_i.z += Jcrossr.z * rmag_inv_3;
	    }
	}
        for(int k = 0; k < klimit; k++){
            B(i + k, 0) = fak * B_i.x[k];
            B(i + k, 1) = fak * B_i.y[k];
            B(i + k, 2) = fak * B_i.z[k];
        }
    }
    return B;
} 

Array current_voxels_field_Bext(Array& points, Array& integration_points, Array& Phi, Array& plasma_unitnormal) 
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(integration_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("integration_points needs to be in row-major storage order");
    if(Phi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("Phi needs to be in row-major storage order");

    int num_points = points.shape(0);
    int num_coil_points = integration_points.shape(0);
    int num_basis_functions = Phi.shape(2);  // Phi.shape(0);

    // integration points should be shape (num_coil_points, num_integration_points, 3)
    int num_integration_points = integration_points.shape(1);

    double fak = 1e-7;

    double* Phi_ptr = &(Phi(0, 0, 0, 0));
    double* ip_points_ptr = &(integration_points(0, 0, 0));
    constexpr int simd_size = xsimd::simd_type<double>::size;
    
    Array B = xt::zeros<double>({num_points, num_coil_points, num_basis_functions});
#pragma omp parallel for schedule(static)
    for(int i = 0; i < num_points; i += simd_size) {
        auto point_i = Vec3dSimd();
        auto n_i = Vec3dSimd();
        
	// check that i + k isn't bigger than num_points
        int klimit = std::min(simd_size, num_points - i);
        for(int k = 0; k < klimit; k++){
            for (int d = 0; d < 3; ++d) {
                point_i[d][k] = points(i + k, d);
                n_i[d][k] = plasma_unitnormal(i + k, d);
            }
        }
        for (int jj = 0; jj < num_coil_points; jj++) {  // loop through grid cells
            for (int j = 0; j < num_integration_points; j++) {  // integrate within a grid cell
                Vec3dSimd point_j = Vec3dSimd(ip_points_ptr[jj * num_integration_points * 3 + 3 * j + 0], ip_points_ptr[jj * num_integration_points * 3 + 3 * j + 1], ip_points_ptr[jj * num_integration_points * 3 + 3 * j + 2]);
                Vec3dSimd r = point_i - point_j;
                simd_t rmag_2     = normsq(r);
                simd_t rmag_inv   = rsqrt(rmag_2);
                simd_t rmag_inv_3 = rmag_inv * (rmag_inv * rmag_inv);
    	        for (int kk = 0; kk < num_basis_functions; kk++) {
		    int ind = jj * num_basis_functions * num_integration_points * 3 + j * num_basis_functions * 3 + kk * 3;  
                    Vec3dSimd Phi = Vec3dSimd(Phi_ptr[ind + 0], Phi_ptr[ind + 1], Phi_ptr[ind + 2]);
                    Vec3dSimd Phicrossr = cross(Phi, r);
                    auto B_i = (Phicrossr.x * n_i.x + Phicrossr.y * n_i.y + Phicrossr.z * n_i.z) * rmag_inv_3;
                    //B_i.x += Phicrossr.x * rmag_inv_3;
                    //B_i.y += Phicrossr.y * rmag_inv_3;
                    //B_i.z += Phicrossr.z * rmag_inv_3;

		    for(int k = 0; k < klimit; k++){
		        B(i + k, jj, kk) += B_i[k];
	 	    }
    	        }
    	    }
    	}
    }
    return fak * B;
}

#else
// Calculate the geometrics factor from the polynomial basis functions 
Array current_voxels_field_B(Array& points, Array& integration_points, Array& J) 
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(integration_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("integration_points needs to be in row-major storage order");
    if(J.layout() != xt::layout_type::row_major)
          throw std::runtime_error("J needs to be in row-major storage order");

    int num_points = points.shape(0);

    // J should be shape (num_coil_points, num_integration_points, 3)
    int num_coil_points = J.shape(0);

    // integration points should be shape (num_coil_points, num_integration_points, 3)
    int num_integration_points = integration_points.shape(1);

    double fak = 1e-7;
    Array B = xt::zeros<double>({num_points, 3});
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_points; i++) {  // loop through quadrature points on plasma surface
	double rx = points(i, 0);
	double ry = points(i, 1);
	double rz = points(i, 2);
	double Bx = 0.0;
	double By = 0.0;
	double Bz = 0.0;
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
	        double Jx = J(jj, j, 0);
	        double Jy = J(jj, j, 1);
	        double Jz = J(jj, j, 2);
		double J_cross_rvec_x = Jy * rvecz - Jz * rvecy;
	        double J_cross_rvec_y = Jz * rvecx - Jx * rvecz;
	        double J_cross_rvec_z = Jx * rvecy - Jy * rvecx;
                Bx += J_cross_rvec_x * rvec_inv3;
                By += J_cross_rvec_y * rvec_inv3;
                Bz += J_cross_rvec_z * rvec_inv3;
	    }
	}
	B(i, 0) = fak * Bx;
	B(i, 1) = fak * By;
	B(i, 2) = fak * Bz;
    }
    return B;
} 

// Calculate the geometrics factor from the polynomial basis functions 
Array current_voxels_field_Bext(Array& points, Array& integration_points, Array& Phi, Array& plasma_unitnormal) 
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(integration_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("integration_points needs to be in row-major storage order");
    if(Phi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("Phi needs to be in row-major storage order");

    int num_points = points.shape(0);
    int num_coil_points = integration_points.shape(0);
    int num_basis_functions = Phi.shape(2);  // Phi.shape(0);

    // integration points should be shape (num_coil_points, num_integration_points, 3)
    int num_integration_points = integration_points.shape(1);

    double fak = 1e-7;
    Array B = xt::zeros<double>({num_points, num_coil_points, num_basis_functions});
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_points; i++) {  // loop through quadrature points on plasma surface
    	double rx = points(i, 0);
    	double ry = points(i, 1);
    	double rz = points(i, 2);
    	double nx = plasma_unitnormal(i, 0);
    	double ny = plasma_unitnormal(i, 1);
    	double nz = plasma_unitnormal(i, 2);
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
    	        for (int k = 0; k < num_basis_functions; k++) {
                    double Jx = Phi(jj, j, k, 0);
    	            double Jy = Phi(jj, j, k, 1);
    	            double Jz = Phi(jj, j, k, 2);
                    B(i, jj, k) += ((Jy * rvecz - Jz * rvecy) * nx + (Jz * rvecx - Jx * rvecz) * ny + (Jx * rvecy - Jy * rvecx) * nz) * rvec_inv3;
    	        }
    	    }
    	}
    }
    return fak * B;
} 

#endif
