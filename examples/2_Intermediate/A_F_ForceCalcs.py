import time
import numpy as np

def random_dipoles_and_positions(N):
    """
    Generate random dipole moments and positions for N dipoles.
    Returns:
        moments: array of length 3N
        positions: array of length 3N
    """
    bounds = N * 10
    moments = np.random.randint(-bounds, bounds, size=3*N)
    positions = np.random.randint(-bounds, bounds, size=3*N)
    return moments, positions

def build_A_tildeF_tensor(r):
    """
    Constructs the rank-3 tensor A_F[j, i, l] such that:
        F_j = sum_{i,l} m2_i * A_F[j, i, l] * m1_l
    matches the classical dipole-dipole force formula.
    """
    mu0 = 4 * np.pi * 1e-7
    r_squared = np.dot(r, r)
    C = (3 * mu0) / (4 * np.pi * (r_squared ** 2.5))
    A_tildeF = np.zeros((3, 3, 3))
    for j in range(3): # j is the index of the force component
        for i in range(3): # i is the index of the dipole moment of the first dipole
            for l in range(3): # l is the index of the dipole moment of the second dipole
                delta_ij = 1 if i == j else 0  # Kronecker delta for i, j
                delta_jl = 1 if j == l else 0  # Kronecker delta for j, l
                delta_il = 1 if i == l else 0  # Kronecker delta for i, l
                A_tildeF[j, i, l] = (
                    delta_ij * r[l] 
                    + r[i] * delta_jl
                    + delta_il * r[j]
                    - 5 * r[i] * r[l] * r[j] / r_squared
                )
    return C * A_tildeF

def build_A_F(positions):
    """
    Builds a (3N, 3N, 3) tensor for N dipoles, where each (3,3,3) block is filled for i != l with the correct subtensor, and zero for i == l.
    Args:
        moments: array of length 3N
        positions: array of length 3N
    Returns:
        A_F: (3N, 3N, 3) tensor
    """
    N = len(positions) // 3
    A_F = np.zeros((3*N, 3*N, 3))
    for i in range(N):
        for l in range(N):
            if i != l:
                p_i = positions[3*i:3*(i+1)]
                p_l = positions[3*l:3*(l+1)]
                r = p_i - p_l
                A_tildeF = build_A_tildeF_tensor(r)
                for j in range(3):
                    for k in range(3):
                        for m in range(3):
                            A_F[3*i + j, 3*l + k, m] = A_tildeF[j, k, m]
    return A_F

def dipole_forces_from_A_F(moments, A_F):
    """
    Computes the force on each individual dipole using the A_F tensor.
    Args:
        moments: array of length 3N containing the magnetic moments
        A_F: (3N, 3N, 3) tensor containing the interaction matrix
    Returns:
        forces: array of length 3N containing the force on each dipole
    """
    N = len(moments) // 3
    forces = np.zeros(3*N)
    for i in range(N):
        for j in range(N):
            if i != j:  # Skip self-interaction
                interaction_block = A_F[3*i:3*(i+1), 3*j:3*(j+1), :]
                m_i = moments[3*i:3*(i+1)]
                m_j = moments[3*j:3*(j+1)]
                force_ij = np.einsum('a,abc,b->c', m_i, interaction_block, m_j)
                forces[3*i:3*(i+1)] += force_ij
    return forces

def two_norm_squared(array):
    """
    Computes the squared 2-norm (sum of squares) of an array.
    Args:
        array: numpy array of any shape
    Returns:
        float: sum of squares of all elements
    """
    return np.sum(array**2)




def diagnostic_test(N):
    moments, positions = random_dipoles_and_positions(N)
    moments = moments * 1000
    t1 = time.time()
    A_F = build_A_F(positions)
    t2 = time.time()
    print(f"Time to build A_F: {t2-t1:.3f} seconds")
    t3 = time.time()
    net_forces = dipole_forces_from_A_F(moments, A_F)
    t4 = time.time()
    print(f"Time to compute net force components: {t4-t3:.3f} seconds")
    force_2norm_squared = two_norm_squared(net_forces)   
    return force_2norm_squared

if __name__ == "__main__":
    for N in [10, 50, 100, 200, 500, 1000, 2000]:
        print(f"\nTesting N = {N} dipoles:")
        force_norm_squared = diagnostic_test(N)
        print(f"Force two norm squared: {force_norm_squared}")
