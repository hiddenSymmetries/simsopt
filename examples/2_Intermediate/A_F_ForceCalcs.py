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

def gradient_of_force_squared(moments, forces, A_F):
    """
    Computes the gradient of the squared L2-norm of the forces with respect to the moments.

    Args:
        moments (np.ndarray): A 1D array of shape (3*N,) containing the magnetic
                              moments of N dipoles.
        forces (np.ndarray): A 1D array of shape (3*N,) containing the pre-computed
                             force vector.
        A_F (np.ndarray): A 3D array of shape (3*N, 3*N, 3) representing the
                          interaction tensor.

    Returns:
        np.ndarray: A 1D array of shape (3*N,) containing the gradient of the
                    squared force with respect to each component of the moments vector.
    """
    N = len(moments) // 3
    if len(moments) % 3 != 0 or len(forces) % 3 != 0:
        raise ValueError("Length of moments and forces arrays must be a multiple of 3.")

    grad = np.zeros(3*N)
    for L in range(N):  # Index of the moment we are differentiating with respect to (m_L)
        grad_L = np.zeros(3)
        for I in range(N):  # Index of the force component F_I being differentiated
            F_I = forces[3*I:3*(I+1)]
            # Calculate the 3x3 Jacobian block J_IL = dF_I / dm_L
            if I == L:
                # Case 1: Differentiating F_I w.r.t. its own moment m_I.
                J_IL = np.zeros((3, 3))
                for J_prime in range(N):
                    if J_prime == I:
                        continue
                    m_J_prime = moments[3*J_prime:3*(J_prime+1)]
                    A_IJ_prime = A_F[3*I:3*(I+1), 3*J_prime:3*(J_prime+1), :]
                    # Physically correct contraction for the force structure:
                    J_IL += np.einsum('b,dbc->dc', m_J_prime, A_IJ_prime)
            else: # I != L
                # Case 2: Differentiating F_I w.r.t. another moment m_L.
                m_I = moments[3*I:3*(I+1)]
                A_IL = A_F[3*I:3*(I+1), 3*L:3*(L+1), :]
                J_IL = np.einsum('a,adc->dc', m_I, A_IL)
            # Add the contribution to the gradient: (J_IL)^T * F_I
            grad_L += np.dot(J_IL.T, F_I)
        grad[3*L:3*(L+1)] = grad_L
    # Final gradient is 2 * J^T * F
    final_grad = 2 * grad
    return final_grad


def diagnostic_test(N, moments=None, positions=None):
    if moments is None or positions is None:
        moments, positions = random_dipoles_and_positions(N)
        moments = moments * 1000
    else:
        if len(moments) != 3*N or len(positions) != 3*N:
            raise ValueError("Length of provided moments and positions must be 3*N")
    
    t1 = time.time()
    A_F = build_A_F(positions)
    t2 = time.time()
    print(f"\nTime to build A_F: {t2-t1:.3f} seconds")
    t3 = time.time()
    net_forces = dipole_forces_from_A_F(moments, A_F)
    t4 = time.time()
    print(f"Time to compute net force components: {t4-t3:.3f} seconds")
    force_2norm_squared = two_norm_squared(net_forces)  
    t5 = time.time()
    grad = gradient_of_force_squared(moments, net_forces, A_F)
    t6 = time.time()
    print(f"Time to compute gradient: {t6-t5:.3f} seconds")
    if N < 6:
        print("\nDetailed output:")
        print(f"\nMoments:\n{moments}")
        print(f"\nPositions:\n{positions}")
        print(f"\nNet forces:\n{net_forces}")
        print(f"\nGradient:\n{grad}\n\n\n")
    return force_2norm_squared, grad

if __name__ == "__main__":
    # Test case for two orthogonal dipoles. Should be zero force and gradient.
    moments = np.array([ 0.0,  1.0,  0.0,  0.0,  0.0,  1.0])
    positions = np.array([ 1.0,  0.0,  0.0, -1.0,  0.0,  0.0])
    force_norm_squared, grad = diagnostic_test(2, moments, positions)
    
    # Test case for 3 collinear dipoles. Should be zero force and gradient.
    moments = np.array([ 0.0,  1.0,  0.0,   0.0, -0.0625, 0.0,   0.0,  1.0,  0.0])
    positions = np.array([-1.0,  0.0,  0.0,   0.0,  0.0,  0.0,   1.0,  0.0,  0.0])
    force_norm_squared, grad = diagnostic_test(3, moments, positions)
    
     # Test case for 4 dipoles in a circle. Should be nonzero force and gradient should be zero.
    moments = np.array([ 0.0,  1.0,  0.0,  -1.0,  0.0,  0.0,   0.0, -1.0,  0.0,   1.0,  0.0,  0.0])
    positions = np.array([ 1.0,  0.0,  0.0,   0.0,  1.0,  0.0,  -1.0,  0.0,  0.0,   0.0, -1.0,  0.0])
    force_norm_squared, grad = diagnostic_test(4, moments, positions)
    
    # Test case for two  dipoles. S
    # Force should be [ 0.0, -0.1875, 0.0, 0.0, 0.1875, 0.0] * 10^-7
    # Gradient should be [ 0.0, 0.0, -0.09375, 0.0, 0.0, 0.09375] * 10^-7
    moments = np.array([ 0.0,  1.0,  0.0,  0.0,  1.0,  1.0])
    positions = np.array([ 1.0,  0.0,  0.0, -1.0,  0.0,  0.0])
    force_norm_squared, grad = diagnostic_test(2, moments, positions)
    '''
    for N in [2,3,4,5]:
        print(f"\nTesting N = {N} dipoles:")
        force_norm_squared, grad = diagnostic_test(N)
        #print(f"Force two norm squared: {force_norm_squared}")
    '''
