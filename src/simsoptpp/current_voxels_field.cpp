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

Array current_voxels_field_A(Array& points, Array& integration_points, Array& J) 
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
    Array A = xt::zeros<double>({num_points, 3});
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
        for (int jj = 0; jj < num_coil_points; jj++) {  // loop through grid cells
            for (int j = 0; j < num_integration_points; j++) {  // integrate within a grid cell
                Vec3dSimd J = Vec3dSimd(J_ptr[jj * num_integration_points * 3 + 3 * j + 0], J_ptr[jj * num_integration_points * 3 + 3 * j + 1], J_ptr[jj * num_integration_points * 3 + 3 * j + 2]);
                Vec3dSimd point_j = Vec3dSimd(ip_points_ptr[jj * num_integration_points * 3 + 3 * j + 0], ip_points_ptr[jj * num_integration_points * 3 + 3 * j + 1], ip_points_ptr[jj * num_integration_points * 3 + 3 * j + 2]);
                Vec3dSimd r = point_i - point_j;
                simd_t rmag_2     = normsq(r);
                simd_t rmag_inv   = rsqrt(rmag_2);
                A_i.x += J.x * rmag_inv;
                A_i.y += J.y * rmag_inv;
                A_i.z += J.z * rmag_inv;
	    }
	}
        for(int k = 0; k < klimit; k++){
            A(i + k, 0) = fak * A_i.x[k];
            A(i + k, 1) = fak * A_i.y[k];
            A(i + k, 2) = fak * A_i.z[k];
        }
    }
    return A;
}

// Calculate the geometrics factor from the polynomial basis functions 
Array current_voxels_field_dA(Array& points, Array& integration_points, Array& J) 
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
    Array dA = xt::zeros<double>({points.shape(0), points.shape(1), points.shape(1)});
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_points; i++) {  // loop through quadrature points on plasma surface
        auto point_i = Vec3dSimd();
        auto dA_i1   = Vec3dSimd();
        auto dA_i2   = Vec3dSimd();
        auto dA_i3   = Vec3dSimd();
        
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
                dA_i1.x += J.x * r.x * rmag_inv_3;
                dA_i1.y += J.y * r.x * rmag_inv_3;
                dA_i1.z += J.z * r.x * rmag_inv_3;
                dA_i2.x += J.x * r.y * rmag_inv_3;
                dA_i2.y += J.y * r.y * rmag_inv_3;
                dA_i2.z += J.z * r.y * rmag_inv_3;
                dA_i3.x += J.x * r.z * rmag_inv_3;
                dA_i3.y += J.y * r.z * rmag_inv_3;
                dA_i3.z += J.z * r.z * rmag_inv_3;
	    }
	}
        for(int k = 0; k < klimit; k++){
            dA(i + k, 0, 0) = -fak * dA_i1.x[k];
            dA(i + k, 0, 1) = -fak * dA_i1.y[k];
            dA(i + k, 0, 2) = -fak * dA_i1.z[k];
            dA(i + k, 1, 0) = -fak * dA_i2.x[k];
            dA(i + k, 1, 1) = -fak * dA_i2.y[k];
            dA(i + k, 1, 2) = -fak * dA_i2.z[k];
	    dA(i + k, 2, 0) = -fak * dA_i3.x[k];
	    dA(i + k, 2, 1) = -fak * dA_i3.y[k]; 
            dA(i + k, 2, 2) = -fak * dA_i3.z[k];
	}
    }
    return dA;
}

// Calculate the geometrics factor from the polynomial basis functions 
Array current_voxels_field_dB(Array& points, Array& integration_points, Array& J) 
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
    Array dB = xt::zeros<double>({points.shape(0), points.shape(1), points.shape(1)});
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_points; i++) {  // loop through quadrature points on plasma surface
        auto point_i = Vec3dSimd();
        auto dB_i1   = Vec3dSimd();
        auto dB_i2   = Vec3dSimd();
        auto dB_i3   = Vec3dSimd();
        
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
                simd_t rmag_inv_5 = rmag_inv_3 * (rmag_inv * rmag_inv);
                Vec3dSimd Jcrossr = cross(J, r);
                dB_i1.x += -3 * Jcrossr.x * r.x * rmag_inv_5; 
                dB_i1.y += -J.z * rmag_inv_3 - 3 * Jcrossr.x * r.y * rmag_inv_5;
                dB_i1.z += J.y * rmag_inv_3 - 3 * Jcrossr.x * r.z * rmag_inv_5;
                dB_i2.x += J.z * rmag_inv_3 - 3 * Jcrossr.y * r.x * rmag_inv_5;
                dB_i2.y += -3 * Jcrossr.y * r.y * rmag_inv_5; 
                dB_i2.z += -J.x * rmag_inv_3 - 3 * Jcrossr.y * r.z * rmag_inv_5;
                dB_i3.x += -J.y * rmag_inv_3 - 3 * Jcrossr.z * r.x * rmag_inv_5;
                dB_i3.y += J.x * rmag_inv_3 - 3 * Jcrossr.z * r.y * rmag_inv_5;
                dB_i3.z += -3 * Jcrossr.z * r.z * rmag_inv_5; 
	    }
	}
        for(int k = 0; k < klimit; k++){
            dB(i + k, 0, 0) = fak * dB_i1.x[k];
            dB(i + k, 0, 1) = fak * dB_i1.y[k];
            dB(i + k, 0, 2) = fak * dB_i1.z[k];
            dB(i + k, 1, 0) = fak * dB_i2.x[k];
            dB(i + k, 1, 1) = fak * dB_i2.y[k];
            dB(i + k, 1, 2) = fak * dB_i2.z[k];
	    dB(i + k, 2, 0) = fak * dB_i3.x[k];
	    dB(i + k, 2, 1) = fak * dB_i3.y[k]; 
            dB(i + k, 2, 2) = fak * dB_i3.z[k];
	}
    }
    return dB;
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
                    Vec3dSimd Phi_3d = Vec3dSimd(Phi_ptr[ind + 0], Phi_ptr[ind + 1], Phi_ptr[ind + 2]);
                    Vec3dSimd Phicrossr = cross(Phi_3d, r);
                    auto B_i = (Phicrossr.x * n_i.x + Phicrossr.y * n_i.y + Phicrossr.z * n_i.z) * rmag_inv_3;
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


// Calculate the geometrics factor from the polynomial basis functions 
Array current_voxels_field_A(Array& points, Array& integration_points, Array& J) 
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
    Array A = xt::zeros<double>({num_points, 3});
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_points; i++) {  // loop through quadrature points on plasma surface
	double rx = points(i, 0);
	double ry = points(i, 1);
	double rz = points(i, 2);
	double Ax = 0.0;
	double Ay = 0.0;
	double Az = 0.0;
        for (int jj = 0; jj < num_coil_points; jj++) {  // loop through grid cells
            for (int j = 0; j < num_integration_points; j++) {  // integrate within a grid cell
	        double rprimex = integration_points(jj, j, 0);
	        double rprimey = integration_points(jj, j, 1);
	        double rprimez = integration_points(jj, j, 2);
	        double rvecx = rx - rprimex;
	        double rvecy = ry - rprimey;
	        double rvecz = rz - rprimez;
	        double rvec_mag = sqrt(rvecx * rvecx + rvecy * rvecy + rvecz * rvecz);
	        double rvec_inv = 1.0 / rvec_mag; 
	        double Jx = J(jj, j, 0);
	        double Jy = J(jj, j, 1);
	        double Jz = J(jj, j, 2);
                Ax += Jx * rvec_inv;
                Ay += Jy * rvec_inv;
                Az += Jz * rvec_inv;
	    }
	}
	A(i, 0) = fak * Ax;
	A(i, 1) = fak * Ay;
	A(i, 2) = fak * Az;
    }
    return A;
} 


// Calculate the geometrics factor from the polynomial basis functions 
Array current_voxels_field_dA(Array& points, Array& integration_points, Array& J) 
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
    Array dA = xt::zeros<double>({points.shape(0), points.shape(1), points.shape(1)});
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_points; i++) {  // loop through quadrature points on plasma surface
	double rx = points(i, 0);
	double ry = points(i, 1);
	double rz = points(i, 2);
        auto dA_i1   = Vec3dStd();
        auto dA_i2   = Vec3dStd();
        auto dA_i3   = Vec3dStd();
        for (int jj = 0; jj < num_coil_points; jj++) {  // loop through grid cells
            for (int j = 0; j < num_integration_points; j++) {  // integrate within a grid cell
	        double rprimex = integration_points(jj, j, 0);
	        double rprimey = integration_points(jj, j, 1);
	        double rprimez = integration_points(jj, j, 2);
	        double rvecx = rx - rprimex;
	        double rvecy = ry - rprimey;
	        double rvecz = rz - rprimez;
	        double rvec_mag = sqrt(rvecx * rvecx + rvecy * rvecy + rvecz * rvecz);
	        double rvec_inv = 1.0 / rvec_mag; 
	        double rvec_inv3 = 1.0 / (rvec_mag * (rvec_mag * rvec_mag));
	        double Jx = J(jj, j, 0);
	        double Jy = J(jj, j, 1);
	        double Jz = J(jj, j, 2);
                dA_i1.x += Jx * rvecx * rvec_inv3;
                dA_i1.y += Jy * rvecx * rvec_inv3;
                dA_i1.z += Jz * rvecx * rvec_inv3;
                dA_i2.x += Jx * rvecy * rvec_inv3;
                dA_i2.y += Jy * rvecy * rvec_inv3;
                dA_i2.z += Jz * rvecy * rvec_inv3;
                dA_i3.x += Jx * rvecz * rvec_inv3;
                dA_i3.y += Jy * rvecz * rvec_inv3;
                dA_i3.z += Jz * rvecz * rvec_inv3;
	    }
	}
        dA(i, 0, 0) = -fak * dA_i1.x;
        dA(i, 0, 1) = -fak * dA_i1.y;
        dA(i, 0, 2) = -fak * dA_i1.z;
        dA(i, 1, 0) = -fak * dA_i2.x;
        dA(i, 1, 1) = -fak * dA_i2.y;
        dA(i, 1, 2) = -fak * dA_i2.z;
        dA(i, 2, 0) = -fak * dA_i3.x;
        dA(i, 2, 1) = -fak * dA_i3.y;
        dA(i, 2, 2) = -fak * dA_i3.z;
    }
    return dA;
} 


// Calculate the geometrics factor from the polynomial basis functions 
Array current_voxels_field_dB(Array& points, Array& integration_points, Array& J) 
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
    Array dB = xt::zeros<double>({points.shape(0), points.shape(1), points.shape(1)});
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_points; i++) {  // loop through quadrature points on plasma surface
	double rx = points(i, 0);
	double ry = points(i, 1);
	double rz = points(i, 2);
        auto dB_i1   = Vec3dStd();
        auto dB_i2   = Vec3dStd();
        auto dB_i3   = Vec3dStd();
        for (int jj = 0; jj < num_coil_points; jj++) {  // loop through grid cells
            for (int j = 0; j < num_integration_points; j++) {  // integrate within a grid cell
	        double rprimex = integration_points(jj, j, 0);
	        double rprimey = integration_points(jj, j, 1);
	        double rprimez = integration_points(jj, j, 2);
	        double rvecx = rx - rprimex;
	        double rvecy = ry - rprimey;
	        double rvecz = rz - rprimez;
	        double rvec_mag = sqrt(rvecx * rvecx + rvecy * rvecy + rvecz * rvecz);
	        double rvec_inv = 1.0 / rvec_mag; 
	        double rvec_inv3 = 1.0 / (rvec_mag * (rvec_mag * rvec_mag));
	        double rvec_inv5 = rvec_inv3 * (rvec_inv * rvec_inv);
	        double Jx = J(jj, j, 0);
	        double Jy = J(jj, j, 1);
	        double Jz = J(jj, j, 2);
                double Jcrossr_x = Jy * rvecz - Jz * rvecy;
		double Jcrossr_y = Jz * rvecx - Jx * rvecz;
		double Jcrossr_z = Jx * rvecy - Jy * rvecx;
                dB_i1.x += -3 * Jcrossr_x * rvecx * rvec_inv3;
                dB_i1.y += -Jz * rvec_inv3 - 3 * Jcrossr_x * rvecy * rvec_inv5;
                dB_i1.z += Jy * rvec_inv3 - 3 * Jcrossr_x * rvecz * rvec_inv5;
                dB_i2.x += Jz * rvec_inv3 - 3 * Jcrossr_y * rvecx * rvec_inv5;
                dB_i2.y += -3 * Jcrossr_y * rvecy * rvec_inv5; 
                dB_i2.z += -Jx * rvec_inv3 - 3 * Jcrossr_y * rvecz * rvec_inv5;
                dB_i3.x += -Jy * rvec_inv3 - 3 * Jcrossr_z * rvecx * rvec_inv5;
                dB_i3.y += Jx * rvec_inv3 - 3 * Jcrossr_z * rvecy * rvec_inv5;
                dB_i3.z += -3 * Jcrossr_z * rvecz * rvec_inv5; 
	    }
	}
        dB(i, 0, 0) = fak * dB_i1.x;
        dB(i, 0, 1) = fak * dB_i1.y;
        dB(i, 0, 2) = fak * dB_i1.z;
        dB(i, 1, 0) = fak * dB_i2.x;
        dB(i, 1, 1) = fak * dB_i2.y;
        dB(i, 1, 2) = fak * dB_i2.z;
        dB(i, 2, 0) = fak * dB_i3.x;
        dB(i, 2, 1) = fak * dB_i3.y;
        dB(i, 2, 2) = fak * dB_i3.z;
    }
    return dB;
} 


#endif
