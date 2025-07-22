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

'''
def random_magnet_pair():
    m1 = np.random.randint(-10, 11, size=3)
    m2 = np.random.randint(-10, 11, size=3)
    p1 = np.random.randint(-10, 11, size=3)
    p2 = np.random.randint(-10, 11, size=3)
    return m1, m2, p1, p2
# m1, m2, p1, p2 = random_magnet_pair()
'''
'''
m1 = np.array([2, -1, 3])
m2 = np.array([-1, 4, 2])

p1 = np.array([1, 2, 3])
p2 = np.array([4, 0, -2])
'''
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
                # Each term below corresponds to a term in the classical force formula:
                #   delta_ij * r[l]         --> (m1 . r) * m2_j
                #   r[i] * delta_jl         --> (m2 . r) * m1_j
                #   delta_il * r[j]         --> (m1 . m2) * r_j
                #   -5 * r[i] * r[l] * r[j] / r^2  --> -5 * (m1 . r) * (m2 . r) * r_j / r^2
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

 
'''
def tildematrix_force(m1, m2, p1, p2):
    r = p2 - p1
    A_tildeF = build_A_tildeF_tensor(r)
    # Contract: F_j = sum_{i,l} m2_i * A_{jil} * m1_l
    F = np.einsum('i,jil,l->j', m2, A_tildeF, m1)
    return F

def matrix_force(moments, A_F):
    """
    Computes the force vector F = m A_F m via matrix multiplication
    Args:
        moments: array of length 3N containing the magnetic moments
        A_F: (3N, 3N, 3) tensor containing the interaction matrix
    Returns:
        F: array of length 3 containing the force components
    """
    # Contract: F_j = sum_{i,l} m_i * A_{jil} * m_l
    F = np.einsum('i,ilk,l->k', moments, A_F, moments)
    return F
'''
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
    
    # For each dipole i, compute the force on it
    for i in range(N):
        # Sum over all other dipoles j to get total force on dipole i
        for j in range(N):
            if i != j:  # Skip self-interaction
                # Extract the 3x3 block for interaction between dipoles i and j
                interaction_block = A_F[3*i:3*(i+1), 3*j:3*(j+1), :]  # Shape: (3, 3, 3)
                # Contract with moments of dipoles i and j
                m_i = moments[3*i:3*(i+1)]
                m_j = moments[3*j:3*(j+1)]
                force_ij = np.einsum('a,abc,b->c', m_i, interaction_block, m_j)
                forces[3*i:3*(i+1)] += force_ij
    
    return forces
'''
def matrix_force_squared_components_per_dipole(moments, A_F):
    """
    Computes the squared force components for each dipole using the A_F tensor.
    Args:
        moments: array of length 3N containing the magnetic moments
        A_F: (3N, 3N, 3) tensor containing the interaction matrix
    Returns:
        F2_per_dipole: array of length 3N containing squared force components for each dipole
    """
    forces = dipole_forces_from_A_F(moments, A_F)
    return forces**2
'''
'''
def classic_force(m1, m2, p1, p2):
    mu0 = 4 * np.pi * 1e-7
    r = p2 - p1
    r_x = r[0]
    r_y = r[1]
    r_z = r[2]
    r_squared = r_x**2 + r_y**2 + r_z**2
    r_fifth = r_squared**2.5
    constant = (3*mu0)/(4*np.pi*r_fifth)
    term1 = np.dot(m1, r) * m2
    term2 = np.dot(m2, r) * m1
    term3 = np.dot(m1, m2) * r
    term4 = 5*np.dot(m1, r) * np.dot(m2, r) * r / r_squared
    force = constant * (term1 + term2 + term3 - term4)
    return force

def classic_force_squared_components(m1, m2, p1, p2):
    F = classic_force(m1, m2, p1, p2)
    return F**2
'''
(positions)

def diagnostic_test(N):
    moments, positions = random_dipoles_and_positions(N)
    moments = moments * 1000
    
    t1 = time.time()
    A_F = build_A_F(positions)
    t2 = time.time()
    print(f"Time to build A_F: {t2-t1:.3f} seconds")
    
    t3 = time.time()
    matrix_squared = matrix_force_squared_components_per_dipole(moments, A_F)
    t4 = time.time()
    print(f"Time to compute matrix force squared components: {t4-t3:.3f} seconds")
    
    return matrix_squared

# Example usage:
if __name__ == "__main__":
    # Test performance for different numbers of dipoles
    for N in [10, 50, 100, 200, 500, 1000, 2000]:
        print(f"\nTesting N = {N} dipoles:")
        matrix_squared = diagnostic_test(N)
    
    '''
    # Create random dipole pair
    N = 10000
    print("Number of magnets:", N)
    moments, positions = random_dipoles_and_positions(N)
    moments = moments * 1000
    # Extract individual vectors (each is 3D)
   
    m1 = moments[0:3]
    m2 = moments[3:6]
    m3 = moments[6:9]
    p1 = positions[0:3]
    p2 = positions[3:6]
    p3 = positions[6:9]
    #print("m1:", m1)
    #print("m2:", m2) 
    #print("p1:", p1)
    #print("p2:", p2)
    #print('moments: ', moments)
    #print("positions: ", positions)
    # Calculate forces using classic and matrix methods
    #print("Classic force (m2 on m1):", classic_force(m1, m2, p1, p2))
    #print("     m1 ~A_F m2:         ", tildematrix_force(m1, m2, p1, p2))
    
    # Build full interaction matrices and calculate forces
    t1 = time.time()
    A_F = build_A_F(positions)
    t2 = time.time()
    print(f"Time to build A_F: {t2-t1:.3f} seconds")
    
    print("\nFull interaction matrix A_F shape:", A_F.shape)
    #print("full m^T A_F m:", matrix_force(moments, A_F))
    
    # Test per-dipole forces
    #dipole_forces = dipole_forces_from_A_F(moments, A_F)
    #print("\nPer-dipole forces:",dipole_forces)
    #print("Force on dipole 1:", dipole_forces[0:3])
    #print("Force on dipole 2:", dipole_forces[3:6])
    
    # Test squared components
    t3 = time.time()
    matrix_squared = matrix_force_squared_components_per_dipole(moments, A_F)
    t4 = time.time()
    print(f"\nTime to compute matrix force squared components: {t4-t3:.3f} seconds\n")
    
    #print("\nMatrix force squared components:", matrix_squared)
    #print("Net force (matrix method) squared components for dipole 1:", matrix_squared[0:3])
    #print("Net force (matrix method) squared components for dipole 2:", matrix_squared[3:6])
    #print("Net force squared (matrix method) components for dipole 3:", matrix_squared[6:9])
    #print("\nClassic force squared components:", classic_force_squared_components(m1, m2, p1, p2), "(For just one dipole)")
    #print("\nNet force (classic method) squared on dipole 1:", (classic_force(m1, m2, p1, p2)+classic_force(m1, m3, p1, p3))**2)
    #print("Net force (classic method) squared on dipole 2:", (classic_force(m2, m1, p2, p1)+classic_force(m2, m3, p2, p3))**2)
    #print("Net force (classic method) squared on dipole 3:", (classic_force(m3, m2, p3, p2)+classic_force(m3, m1, p3, p1))**2)
    '''
    
