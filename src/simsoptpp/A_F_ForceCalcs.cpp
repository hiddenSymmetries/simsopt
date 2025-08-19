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

// Thread-safe version that works with raw arrays
void build_A_tildeF_tensor_raw(const double* r, double* A_tildeF) {
    double mu0 = 4 * M_PI * 1e-7;
    double r_squared = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
    
    // Safety check for very small or zero distance
    if (r_squared < 1e-20) {
        // Return zero tensor for very close or overlapping dipoles
        for (int i = 0; i < 27; ++i) {
            A_tildeF[i] = 0.0;
        }
        return;
    }
    
    double C = (3 * mu0) / (4 * M_PI * pow(r_squared, 2.5));
    
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            for (int l = 0; l < 3; ++l) {
                int delta_ij = (i == j) ? 1 : 0;
                int delta_jl = (j == l) ? 1 : 0;
                int delta_il = (i == l) ? 1 : 0;
                int idx = j * 9 + i * 3 + l;  // Linear index for 3D tensor
                A_tildeF[idx] = (delta_ij * r[l]
                              + r[i] * delta_jl
                              + delta_il * r[j]
                              - 5 * r[i] * r[l] * r[j] / r_squared) * C;
            }
        }
    }
}

// Constructs the rank-3 tensor A_F[j, i, l] as a 3x3x3 array
Array3D build_A_tildeF_tensor(const PyArray& r) {
    double mu0 = 4 * M_PI * 1e-7;
    double r_squared = 0.0;
    for (int i = 0; i < 3; ++i) {
        r_squared += r(i) * r(i);
    }
    
    // Safety check for very small or zero distance
    if (r_squared < 1e-20) {
        // Return zero tensor for very close or overlapping dipoles
        Array3D A_tildeF = xt::zeros<double>({3, 3, 3});
        return A_tildeF;
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
    
    // Get raw pointer to positions for thread-safe access
    double* positions_ptr = const_cast<double*>(&(positions(0)));
    
    #pragma omp parallel for schedule(dynamic,1)
    for (int i = 0; i < N; ++i) {
        for (int l = 0; l < N; ++l) {
            if (i != l) {
                // Use raw arrays instead of PyArray for thread safety
                double r[3];
                for (int d = 0; d < 3; ++d) {
                    r[d] = positions_ptr[3*i + d] - positions_ptr[3*l + d];
                }
                
                // Check for very small distances
                double r_squared = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
                
                if (r_squared > 1e-20) {  // Only process if dipoles are not too close
                    // Use thread-safe raw array version
                    double A_tildeF_raw[27];  // 3x3x3 = 27 elements
                    build_A_tildeF_tensor_raw(r, A_tildeF_raw);
                    
                    // Copy to A_F tensor
                    for (int j = 0; j < 3; ++j) {
                        for (int k = 0; k < 3; ++k) {
                            for (int m = 0; m < 3; ++m) {
                                int idx = j * 9 + k * 3 + m;
                                A_F(3*i+j, 3*l+k, m) = A_tildeF_raw[idx];
                            }
                        }
                    }
                }
                // If r_squared is too small, A_F remains zero for this pair
            }
        }
    }
    
    return A_F;
}

// Helper function to check if a magnet has zero moments
// This optimization skips force calculations for magnets with zero moments
// since they cannot contribute to the net force, saving compute time
bool magnet_has_zero_moments(const double* moments_ptr, int magnet_index) {
    return (moments_ptr[3*magnet_index] == 0.0 && 
            moments_ptr[3*magnet_index + 1] == 0.0 && 
            moments_ptr[3*magnet_index + 2] == 0.0);
}

// Computes the force on each individual dipole using the A_F tensor
PyArray dipole_forces_from_A_F(const PyArray& moments, const Array3D& A_F) {
    int N = moments.size() / 3;
    PyArray forces = xt::zeros<double>({3 * N});
    
    // Safety checks
    if (N <= 0) {
        std::cerr << "Error: Invalid number of dipoles N = " << N << std::endl;
        return forces;
    }
    
    if (A_F.shape(0) != 3*N || A_F.shape(1) != 3*N || A_F.shape(2) != 3) {
        std::cerr << "Error: A_F tensor shape mismatch. Expected (" << 3*N << ", " << 3*N << ", 3), got (" 
                  << A_F.shape(0) << ", " << A_F.shape(1) << ", " << A_F.shape(2) << ")" << std::endl;
        return forces;
    }
    
    // Get raw pointers to data for OpenMP parallelization
    double* moments_ptr = const_cast<double*>(&(moments(0)));
    double* forces_ptr = &(forces(0));
    double* A_F_ptr = const_cast<double*>(&(A_F(0, 0, 0)));
    
    int numThreads = omp_get_max_threads();
    
    // Thread-local buffer: numThreads × 3N
    std::vector<double> localBuf(numThreads * 3 * N, 0.0);
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double* local = &localBuf[tid * 3 * N];
        
        #pragma omp for schedule(dynamic,1)
        for (int i = 0; i < N; ++i) {
            // Skip if magnet i has zero moments
            if (magnet_has_zero_moments(moments_ptr, i)) {
                continue;
            }
            
            for (int j = 0; j < N; ++j) {
                if (i != j) {
                    // Skip if magnet j has zero moments
                    if (magnet_has_zero_moments(moments_ptr, j)) {
                        continue;
                    }
                    
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            for (int c = 0; c < 3; ++c) {
                                // Correct pointer arithmetic for 3D tensor A_F[3*N, 3*N, 3]
                                // A_F(i*3+a, j*3+b, c) = A_F_ptr[(i*3+a)*(3*N)*3 + (j*3+b)*3 + c]
                                size_t idx1 = 3*i + a;  // First index
                                size_t idx2 = 3*j + b;  // Second index
                                size_t idx3 = c;        // Third index
                                size_t linear_idx = idx1 * static_cast<size_t>(3*N) * 3 + idx2 * 3 + idx3;
                                
                                // Bounds checking
                                size_t max_size = static_cast<size_t>(3*N) * static_cast<size_t>(3*N) * 3;
                                if (linear_idx >= max_size) {
                                    std::cerr << "Error: Index out of bounds: " << linear_idx << " >= " << max_size << std::endl;
                                    continue;
                                }
                                
                                // Accumulate into thread-local buffer
                                local[3*i + c] += moments_ptr[3*i + a] * A_F_ptr[linear_idx] * moments_ptr[3*j + b];
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Reduce thread-local buffers into forces
    for (int t = 0; t < numThreads; ++t) {
        double* local = &localBuf[t * 3 * N];
        for (int i = 0; i < 3 * N; ++i) {
            forces_ptr[i] += local[i];
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

// Abbv_Force_Calc(moments, forces, j_index, dipole_grid_xyz, sign = positive)
// moments: 3N list of the dipole moments
// forces: 3N list of the net forces on the dipoles
// j_index: index from moments which will be used to calculate the new forces (i.e. the dipole that is being "activated")
// dipole_grid_xyz: 3N list of the positions of the dipoles 
// sign: orientation of new magnet (which side does north face)

// This function takes the current magnet array with moments and forces, and iterates them to account for changes if 
// a new magnet. First it must identify from moment the indices, m's, of all active magnets. Then from j_index and sign it must identify
// both the index, component, and orientation of the new magnet. Afterwards it will run through force calculations, calculating
// the force vector on m due to magnet j, and adding those to m's sub 3vector in force, and adding the negative to j's sub 3 vector in force
// After this is done for the affect of j on all current m's, calculate the two norm of the modified forces vector. 
// Note: Do this using pointers such that instead of building entirely new copies of the input arrays (only forces is cloned and the clone is modified)
// Note: Code needs to be as fast as possible, minimize overhead and unnnecessary calculations where possible.
// Note: Use PyArrays and Array3D for inputs.

// Step 1: Determine from moments which dipoles (m's) are currently active and from j_index the index, component, and orientation of the new magnet. Additionally once j is known find its position vector from dipole_grid_xyz
// Step 2: For each active magnet m: compute force interactions
     // a) Get position vector for magnet m and calculate displacement vector r pointing from j to m. 
     // b) Use vector r and build_A_tildeF_tensor_raw() to build 3x3x3 interaction matrix between m and j.
     // c) Do matrix operation m A j which results in a resultant vector (length 3) of the force on m due to j.
     // d) In the force array, add the x,y,z components of the resultant vector to the respective x,y,z indices of magnet m.
     // e) In the force array, add the negative of the resultant vector to the indices of magnet j 
//Step 3: Calculate and return the two_norm_squared() of the forces array

std::tuple<PyArray, double> Abbv_Force_Calc(const PyArray& moments, const PyArray& forces, int j_index, const PyArray& dipole_grid_xyz, int sign = 1) {
    int N = moments.size() / 3;
    
    // Extract magnet number and component from j_index
    int j = j_index / 3;           // Magnet number
    int component = j_index % 3;    // Component (0=x, 1=y, 2=z)
    
    // Safety checks
    if (N <= 0 || j < 0 || j >= N) {
        std::cerr << "Error: Invalid magnet number derived from j_index in Abbv_Force_Calc" << std::endl;
        return std::make_tuple(PyArray(), 0.0);
    }
    
    if (component < 0 || component > 2) {
        std::cerr << "Error: Invalid component number derived from j_index in Abbv_Force_Calc" << std::endl;
        return std::make_tuple(PyArray(), 0.0);
    }
    
    if (forces.size() != 3*N || dipole_grid_xyz.size() != 3*N) {
        std::cerr << "Error: Array size mismatch in Abbv_Force_Calc" << std::endl;
        return std::make_tuple(PyArray(), 0.0);
    }
    
    // Create a clone of the forces array to work with
    PyArray forces_clone = forces;
    
    // Get raw pointers for direct array manipulation
    const double* moments_ptr = &(moments(0));
    double* forces_clone_ptr = &(forces_clone(0));
    const double* positions_ptr = &(dipole_grid_xyz(0));
    
    // Step 1: Identify active magnets and get j's position
    std::vector<int> active_magnets;
    for (int m = 0; m < N; ++m) {
        if (!magnet_has_zero_moments(moments_ptr, m)) {
            active_magnets.push_back(m);
        }
    }
    
    // Get position vector for magnet j
    double j_pos[3] = {positions_ptr[3*j], positions_ptr[3*j + 1], positions_ptr[3*j + 2]};
    
    // Step 2: Compute force interactions for each active magnet
    for (int m_idx = 0; m_idx < active_magnets.size(); ++m_idx) {
        int m = active_magnets[m_idx];
        
        if (m == j) continue; // Skip self-interaction
        
        // a) Calculate displacement vector r from j to m
        double r[3];
        for (int d = 0; d < 3; ++d) {
            r[d] = positions_ptr[3*m + d] - j_pos[d];
        }
        
        // b) Build 3x3x3 interaction matrix using build_A_tildeF_tensor_raw
        double A_tildeF[27]; // 3x3x3 = 27 elements
        build_A_tildeF_tensor_raw(r, A_tildeF);
        
        // c) Compute matrix operation m A j (force on m due to j)
        // Build a proper moment vector for magnet j: zeros except for the specific component
        double moment_j[3] = {0.0, 0.0, 0.0};
        moment_j[component] = 1.0; // Set only the specific component to 1
        
        double force_on_m[3] = {0.0, 0.0, 0.0};
        
        // Use the same matrix multiplication approach as dipole_forces_from_A_F
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                for (int c = 0; c < 3; ++c) {
                    // Calculate the linear index for A_tildeF tensor
                    // A_tildeF[a][b][c] = A_tildeF[a*9 + b*3 + c]
                    int idx = a * 9 + b * 3 + c;
                    force_on_m[c] += moments_ptr[3*m + a] * A_tildeF[idx] * moment_j[b];
                }
            }
        }
        
        // Apply sign orientation
        for (int d = 0; d < 3; ++d) {
            force_on_m[d] *= sign;
        }
        
        // d) Add force to magnet m's force vector
        for (int d = 0; d < 3; ++d) {
            forces_clone_ptr[3*m + d] += force_on_m[d];
        }
        
        // e) Add negative force to magnet j's force vector (Newton's 3rd law)
        for (int d = 0; d < 3; ++d) {
            forces_clone_ptr[3*j + d] -= force_on_m[d];
        }
    }
    
    // Step 3: Return both the forces clone and its two_norm_squared
    return std::make_tuple(forces_clone, two_norm_squared(forces_clone));
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