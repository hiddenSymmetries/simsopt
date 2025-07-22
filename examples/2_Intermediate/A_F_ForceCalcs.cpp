#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <Eigen/Dense>
#include <cmath>
#include <omp.h>

using namespace std;
using namespace Eigen;

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
double diagnostic_test(int N) {
    VectorXd moments, positions;
    random_dipoles_and_positions(N, moments, positions);
    moments = moments * 1000;

    auto t1 = chrono::high_resolution_clock::now();
    vector<vector<vector<vector<vector<double>>>>> A_F;
    build_A_F(positions, A_F);
    auto t2 = chrono::high_resolution_clock::now();
    chrono::duration<double> build_time = t2 - t1;
    cout << "Time to build A_F: " << build_time.count() << " seconds" << endl;

    auto t3 = chrono::high_resolution_clock::now();
    VectorXd net_forces;
    dipole_forces_from_A_F(moments, A_F, net_forces);
    auto t4 = chrono::high_resolution_clock::now();
    chrono::duration<double> force_time = t4 - t3;
    cout << "Time to compute net force components: " << force_time.count() << " seconds" << endl;

    double force_2norm_squared = two_norm_squared(net_forces);
    return force_2norm_squared;
}

int main() {
    vector<int> Ns = {1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000};
    for (int N : Ns) {
        cout << "\nTesting N = " << N << " dipoles:" << endl;
        double force_norm_squared = diagnostic_test(N);
        cout << "Force two norm squared: " << force_norm_squared << endl;
    }
    return 0;
}