#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <omp.h>
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#define _USE_MATH_DEFINES
#include <math.h>

typedef xt::pyarray<double> PyArray;
typedef xt::pytensor<double, 3, xt::layout_type::row_major> Array3D;
using std::vector;

// Generate random dipole moments and positions for N dipoles
void random_dipoles_and_positions(int N, PyArray& moments, PyArray& positions) {
    int size = 3 * N;
    int bounds = N * 10;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-bounds, bounds);

    moments = xt::zeros<double>({size});
    positions = xt::zeros<double>({size});
    for (int i = 0; i < size; ++i) {
        moments(i) = dis(gen);
        positions(i) = dis(gen);
    }
}

// Constructs the rank-3 tensor A_F[j, i, l] as a 3x3x3 array
Array3D build_A_tildeF_tensor(const PyArray& r) {
    double mu0 = 4 * M_PI * 1e-7;
    double r_squared = 0.0;
    for (int i = 0; i < 3; ++i) {
        r_squared += r(i) * r(i);
    }
    double C = (3 * mu0) / (4 * M_PI * pow(r_squared, 2.5));
    
    Array3D A_tildeF = xt::zeros<double>({3, 3, 3});
    
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            for (int l = 0; l < 3; ++l) {
                int delta_ij = (i == j) ? 1 : 0;
                int delta_jl = (j == l) ? 1 : 0;
                int delta_il = (i == l) ? 1 : 0;
                A_tildeF(j, i, l) = delta_ij * r(l)
                                  + r(i) * delta_jl
                                  + delta_il * r(j)
                                  - 5 * r(i) * r(l) * r(j) / r_squared;
                A_tildeF(j, i, l) *= C;
            }
        }
    }
    return A_tildeF;
}

// Builds a (3N, 3N, 3) tensor for N dipoles
Array3D build_A_F_tensor(const PyArray& positions) {
    int N = positions.size() / 3;
    Array3D A_F = xt::zeros<double>({3*N, 3*N, 3});
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int l = 0; l < N; ++l) {
            if (i != l) {
                PyArray r = xt::zeros<double>({3});
                for (int d = 0; d < 3; ++d) {
                    r(d) = positions(3*i + d) - positions(3*l + d);
                }
                Array3D A_tildeF = build_A_tildeF_tensor(r);
                for (int j = 0; j < 3; ++j) {
                    for (int k = 0; k < 3; ++k) {
                        for (int m = 0; m < 3; ++m) {
                            A_F(3*i+j, 3*l+k, m) = A_tildeF(j, k, m);
                        }
                    }
                }
            }
        }
    }
    return A_F;
}

// Computes the force on each individual dipole using the A_F tensor
PyArray dipole_forces_from_A_F(const PyArray& moments, const Array3D& A_F) {
    int N = moments.size() / 3;
    PyArray forces = xt::zeros<double>({3 * N});
    
    // Get raw pointers to data for OpenMP parallelization
    double* moments_ptr = const_cast<double*>(&(moments(0)));
    double* forces_ptr = &(forces(0));
    double* A_F_ptr = const_cast<double*>(&(A_F(0, 0, 0)));
    
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        for (int c = 0; c < 3; ++c) {
                            // Use raw pointer arithmetic for thread-safe access
                            #pragma omp atomic
                            forces_ptr[3*i + c] += moments_ptr[3*i + a] * A_F_ptr[(3*i+a)*(3*N)*3 + (3*j+b)*3 + c] * moments_ptr[3*j + b];
                        }
                    }
                }
            }
        }
    }
    return forces;
}

// Computes the squared 2-norm (sum of squares) of an array
double two_norm_squared(const PyArray& array) {
    double sum = 0.0;
    double* array_ptr = const_cast<double*>(&(array(0)));
    
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < array.size(); ++i) {
        sum += array_ptr[i] * array_ptr[i];
    }
    return sum;
}

// Computes squared force components per dipole
PyArray matrix_force_squared_components_per_dipole(const PyArray& moments, const Array3D& A_F) {
    PyArray forces = dipole_forces_from_A_F(moments, A_F);
    PyArray forces_squared = xt::zeros<double>(forces.shape());
    
    // Get raw pointers to data for OpenMP parallelization
    double* forces_ptr = &(forces(0));
    double* forces_squared_ptr = &(forces_squared(0));
    
    #pragma omp parallel for
    for (int i = 0; i < forces.size(); ++i) {
        forces_squared_ptr[i] = forces_ptr[i] * forces_ptr[i];
    }
    return forces_squared;
}

// Computes the net force vector
PyArray matrix_force(const PyArray& moments, const Array3D& A_F) {
    int N = moments.size() / 3;
    PyArray F = xt::zeros<double>({3});
    
    // Get raw pointers to data for OpenMP parallelization
    double* moments_ptr = const_cast<double*>(&(moments(0)));
    double* F_ptr = &(F(0));
    double* A_F_ptr = const_cast<double*>(&(A_F(0, 0, 0)));
    
    #pragma omp parallel for
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < N; ++i) {
            for (int l = 0; l < N; ++l) {
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        #pragma omp atomic
                        F_ptr[j] += moments_ptr[3*i + a] * A_F_ptr[(3*i+a)*(3*N)*3 + (3*l+b)*3 + j] * moments_ptr[3*l + b];
                    }
                }
            }
        }
    }
    return F;
}

// Diagnostic test
double diagnostic_test(int N, const PyArray& moments = PyArray(), const PyArray& positions = PyArray()) {
    PyArray moments_use, positions_use;
    
    // Check if moments and positions are provided and have correct size
    bool use_provided = (moments.size() == 3*N && positions.size() == 3*N);
    
    if (!use_provided) {
        // Generate random moments and positions
        random_dipoles_and_positions(N, moments_use, positions_use);
        for (int i = 0; i < moments_use.size(); ++i) {
            moments_use(i) *= 1000;
        }
    } else {
        // Use provided moments and positions
        moments_use = moments;
        positions_use = positions;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    Array3D A_F = build_A_F_tensor(positions_use);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> build_time = t2 - t1;
    std::cout << "Time to build A_F: " << build_time.count() << " seconds" << std::endl;

    auto t3 = std::chrono::high_resolution_clock::now();
    PyArray net_forces = dipole_forces_from_A_F(moments_use, A_F);
    auto t4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> force_time = t4 - t3;
    std::cout << "Time to compute net force components: " << force_time.count() << " seconds" << std::endl;

    double force_2norm_squared = two_norm_squared(net_forces);
    if (N < 6) {
        std::cout << "\nDetailed output:" << std::endl;
        std::cout << "\nMoments: [";
        for (int i = 0; i < moments_use.size(); ++i) {
            std::cout << moments_use(i);
            if (i < moments_use.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "\nPositions: [";
        for (int i = 0; i < positions_use.size(); ++i) {
            std::cout << positions_use(i);
            if (i < positions_use.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "\nNet forces: [";
        for (int i = 0; i < net_forces.size(); ++i) {
            std::cout << net_forces(i);
            if (i < net_forces.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    return force_2norm_squared;
} 