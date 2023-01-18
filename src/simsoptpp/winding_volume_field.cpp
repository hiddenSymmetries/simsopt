#include "winding_volume_field.h"

// Calculate the geometrics factor from the polynomial basis functions 
Array winding_volume_field_B(Array& points, Array& integration_points, Array& J) 
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
Array winding_volume_field_Bext(Array& points, Array& integration_points, Array& Phi) 
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(integration_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("integration_points needs to be in row-major storage order");

    int num_points = points.shape(0);

    int num_coil_points = Phi.shape(1);
    int num_basis_functions = Phi.shape(0);

    // integration points should be shape (num_coil_points, num_integration_points, 3)
    int num_integration_points = integration_points.shape(1);

    double fak = 1e-7;
    Array B = xt::zeros<double>({num_points, 3, num_coil_points, num_basis_functions});
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_points; i++) {  // loop through quadrature points on plasma surface
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
	        for (int k = 0; k < num_basis_functions; k++) {
		    double Jx = Phi(k, jj, j, 0);
	            double Jy = Phi(k, jj, j, 1);
	            double Jz = Phi(k, jj, j, 2);
	   	    double J_cross_rvec_x = Jy * rvecz - Jz * rvecy;
	            double J_cross_rvec_y = Jz * rvecx - Jx * rvecz;
	            double J_cross_rvec_z = Jx * rvecy - Jy * rvecx;
                    B(i, 0, jj, k) += fak * J_cross_rvec_x * rvec_inv3;
                    B(i, 1, jj, k) += fak * J_cross_rvec_y * rvec_inv3;
                    B(i, 2, jj, k) += fak * J_cross_rvec_z * rvec_inv3;
	        }
	    }
	}
    }
    return B;
} 
