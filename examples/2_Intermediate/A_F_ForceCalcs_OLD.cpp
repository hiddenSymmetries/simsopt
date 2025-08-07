#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <Eigen/Dense>
#include <cmath>
#include <omp.h>

using namespace std;
using namespace Eigen;
// compile with: g++ -std=c++17 -fopenmp -I/usr/include/eigen3 -o A_F_ForceCalcs A_F_ForceCalcs.cpp
// Generate random dipole moments and positions for N dipoles
void random_dipoles_and_positions(int N, VectorXd& moments, VectorXd& positions) {
    int size = 3 * N;
    int bounds = N * 10;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(-bounds, bounds);

    moments.resize(size);
    positions.resize(size);
    for (int i = 0; i < size; ++i) {
        moments(i) = dis(gen);
        positions(i) = dis(gen);
    }
}

// Constructs the rank-3 tensor A_F[j, i, l] as a 3x3x3 array
void build_A_tildeF_tensor(const Vector3d& r, double A_tildeF[3][3][3]) {
    double mu0 = 4 * M_PI * 1e-7;
    double r_squared = r.dot(r);
    double C = (3 * mu0) / (4 * M_PI * pow(r_squared, 2.5));
    // Set all elements to zero
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            for (int l = 0; l < 3; ++l)
                A_tildeF[j][i][l] = 0.0;

    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            for (int l = 0; l < 3; ++l) {
                int delta_ij = (i == j) ? 1 : 0;
                int delta_jl = (j == l) ? 1 : 0;
                int delta_il = (i == l) ? 1 : 0;
                A_tildeF[j][i][l] = delta_ij * r(l)
                                  + r(i) * delta_jl
                                  + delta_il * r(j)
                                  - 5 * r(i) * r(l) * r(j) / r_squared;
                A_tildeF[j][i][l] *= C;
            }
        }
    }
}

// Builds a (3N, 3N, 3) tensor for N dipoles, stored as a flat vector
void build_A_F(const VectorXd& positions, vector<vector<vector<vector<vector<double>>>>> &A_F) {
    int N = positions.size() / 3;
    // A_F[i][l][j][k][m] corresponds to [3*i+j, 3*l+k, m]
    A_F.resize(N, vector<vector<vector<vector<double>>>>(N, vector<vector<vector<double>>>(3, vector<vector<double>>(3, vector<double>(3, 0.0)))));
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int l = 0; l < N; ++l) {
            if (i != l) {
                Vector3d p_i = positions.segment<3>(3 * i);
                Vector3d p_l = positions.segment<3>(3 * l);
                Vector3d r = p_i - p_l;
                double A_tildeF[3][3][3];
                build_A_tildeF_tensor(r, A_tildeF);
                for (int j = 0; j < 3; ++j)
                    for (int k = 0; k < 3; ++k)
                        for (int m = 0; m < 3; ++m)
                            A_F[i][l][j][k][m] = A_tildeF[j][k][m];
            }
        }
    }
}

// Computes the force on each individual dipole using the A_F tensor
void dipole_forces_from_A_F(const VectorXd& moments, const vector<vector<vector<vector<vector<double>>>>>& A_F, VectorXd& forces) {
    int N = moments.size() / 3;
    forces = VectorXd::Zero(3 * N);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        Vector3d force_i = Vector3d::Zero();
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                Vector3d m_i = moments.segment<3>(3 * i);
                Vector3d m_j = moments.segment<3>(3 * j);
                Vector3d force_ij = Vector3d::Zero();
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        for (int c = 0; c < 3; ++c) {
                            force_ij(c) += m_i(a) * A_F[i][j][a][b][c] * m_j(b);
                        }
                    }
                }
                force_i += force_ij;
            }
        }
        forces.segment<3>(3 * i) = force_i;
    }
}

// Computes the squared 2-norm (sum of squares) of an array
double two_norm_squared(const VectorXd& array) {
    return array.squaredNorm();
}

// Diagnostic test
double diagnostic_test(int N, const VectorXd& moments = VectorXd(), const VectorXd& positions = VectorXd()) {
    VectorXd moments_use, positions_use;
    
    if (moments.size() == 0 || positions.size() == 0) {
        // Generate random moments and positions
        random_dipoles_and_positions(N, moments_use, positions_use);
        moments_use = moments_use * 1000;
    } else {
        // Use provided moments and positions
        if (moments.size() != 3*N || positions.size() != 3*N) {
            throw runtime_error("Length of provided moments and positions must be 3*N");
        }
        moments_use = moments;
        positions_use = positions;
    }

    auto t1 = chrono::high_resolution_clock::now();
    vector<vector<vector<vector<vector<double>>>>> A_F;
    build_A_F(positions_use, A_F);
    auto t2 = chrono::high_resolution_clock::now();
    chrono::duration<double> build_time = t2 - t1;
    cout << "Time to build A_F: " << build_time.count() << " seconds" << endl;

    auto t3 = chrono::high_resolution_clock::now();
    VectorXd net_forces;
    dipole_forces_from_A_F(moments_use, A_F, net_forces);
    auto t4 = chrono::high_resolution_clock::now();
    chrono::duration<double> force_time = t4 - t3;
    cout << "Time to compute net force components: " << force_time.count() << " seconds" << endl;

    double force_2norm_squared = two_norm_squared(net_forces);
    if (N < 6) {
        cout << "\nDetailed output:" << endl;
        cout << "\nMoments: [";
        for (int i = 0; i < moments_use.size(); ++i) {
            cout << moments_use(i);
            if (i < moments_use.size() - 1) cout << ", ";
        }
        cout << "]" << endl;
        
        cout << "\nPositions: [";
        for (int i = 0; i < positions_use.size(); ++i) {
            cout << positions_use(i);
            if (i < positions_use.size() - 1) cout << ", ";
        }
        cout << "]" << endl;
        
        cout << "\nNet forces: [";
        for (int i = 0; i < net_forces.size(); ++i) {
            cout << net_forces(i);
            if (i < net_forces.size() - 1) cout << ", ";
        }
        cout << "]" << endl;
    }
    
    return force_2norm_squared;
}

int main() {
    // Test case for two orthogonal dipoles. Should be zero force and gradient.
    VectorXd moments(6), positions(6);
    moments << 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    positions << 1.0, 0.0, 0.0, -1.0, 0.0, 0.0;
    double force_norm_squared = diagnostic_test(2, moments, positions);

    // Test case for 3 collinear dipoles. Should be zero force and gradient.
    moments.resize(9);
    positions.resize(9);
    moments << 0.0, 1.0, 0.0, 0.0, -0.0625, 0.0, 0.0, 1.0, 0.0;
    positions << -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
    force_norm_squared = diagnostic_test(3, moments, positions);

    // Test case for 4 dipoles in a circle. Should be nonzero force and gradient should be zero.
    moments.resize(12);
    positions.resize(12);
    moments << 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0;
    positions << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0;
    force_norm_squared = diagnostic_test(4, moments, positions);

    // Test case for two dipoles.
    // Force should be [ 0.0, -0.1875, 0.0, 0.0, 0.1875, 0.0] * 10^-7
    // Gradient should be [ 0.0, 0.0, -0.09375, 0.0, 0.0, 0.09375] * 10^-7
    moments.resize(6);
    positions.resize(6);
    moments << 0.0, 1.0, 0.0, 0.0, 1.0, 1.0;
    positions << 1.0, 0.0, 0.0, -1.0, 0.0, 0.0;
    force_norm_squared = diagnostic_test(2, moments, positions);
    
    /*vector<int> Ns = {1000,2000,5000,10000, 37500};
    for (int N : Ns) {
        cout << "\nTesting N = " << N << " dipoles:" << endl;
        double force_norm_squared = diagnostic_test(N);
        cout << "Force two norm squared: " << force_norm_squared << endl;
    }*/
    return 0;

    // Benchmark tests for N = 1,000 and N = 10,000 dipoles
    cout << "\n" << string(50, '=') << endl;
    cout << "BENCHMARK TESTS" << endl; 
    cout << string(50, '=') << endl;

    cout << "\nBenchmark 1: N = 1,000 dipoles" << endl;
    double force_norm_squared_1k = diagnostic_test(1000);
    cout << "Force norm squared: " << force_norm_squared_1k << endl;

    cout << "\nBenchmark 2: N = 10,000 dipoles" << endl;
    double force_norm_squared_10k = diagnostic_test(10000);
    cout << "Force norm squared: " << force_norm_squared_10k << endl;
}