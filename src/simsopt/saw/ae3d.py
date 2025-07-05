import os
from typing import List
from dataclasses import dataclass, field
import numpy as np
import plotly.graph_objects as go

from .stellgap import ModeContinuum, Harmonic 

__all__ = ['EigModeASCI', 'AE3DEigenvector', 'plot_ae3d_eigenmode', 'continuum_from_ae3d']

@dataclass
class EigModeASCI:
    """
    A class to handle the parsing and storage of eigenmode data from the AE3D output 'eig_mode_asci.dat', which includes
    eigenmode descriptions, Fourier modes, radial points, and eigenvectors.

    Attributes:
        sim_dir (str): Directory where the simulation data files are stored.
        file_path (str): Full path to 'eig_mode_asci.dat'.
        num_eigenmodes (int): Number of eigenmodes.
        num_fourier_modes (int): Number of Fourier modes.
        num_radial_points (int): Number of radial points.
        modes (np.ndarray): Mode numbers (m, n).
        egn_values (np.ndarray): Eigenvalue squares for each mode.
        s_coords (np.ndarray): Radial coordinate values.
        egn_vectors (np.ndarray): Eigenvector data reshaped according to the dimensions.
    """
    sim_dir: str
    file_path: str = field(init=False)
    num_eigenmodes: int = field(init=False)
    num_fourier_modes: int = field(init=False)
    num_radial_points: int = field(init=False)
    modes: np.ndarray = field(init=False)
    egn_values: np.ndarray = field(init=False)
    s_coords: np.ndarray = field(init=False)
    egn_vectors: np.ndarray = field(init=False)

    def __post_init__(self):
        self.file_path = os.path.join(self.sim_dir, 'egn_mode_asci.dat')
        self.load_data()

    def load_data(self):
        r"""
        Loads the eigenmode data from the 'eig_mode_asci.dat' file into the class attributes.
        """
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"Data file {self.file_path} not found.")
        
        data = np.loadtxt(self.file_path)
        it = iter(data)
        self.num_eigenmodes = int(next(it))
        self.num_fourier_modes = int(next(it))
        self.num_radial_points = int(next(it))

        self.modes = np.array([(next(it), next(it)) for _ in range(self.num_fourier_modes)],
                              dtype=[('m', 'int32'), ('n', 'int32')])
        self.egn_values = np.array([next(it) for _ in range(self.num_eigenmodes)])
        self.s_coords = np.array([next(it) for _ in range(self.num_radial_points)])
        self.egn_vectors = np.array([next(it) for _ in range(self.num_eigenmodes * self.num_radial_points * self.num_fourier_modes)]
                                    ).reshape(self.num_eigenmodes, self.num_radial_points, self.num_fourier_modes)

    def get_nearest_eigenvector(self, target_eigenvalue):
        """
        Finds and returns the eigenvector closest to a specified target eigenvalue, along with its
        corresponding eigenvalue and sorted mode numbers.

        Args:
            target_eigenvalue (float): The target eigenvalue to find the closest match to.

        Returns:
            tuple: A tuple containing the closest eigenvalue, the normalized eigenvector, and sorted mode numbers.
        """
        data = [(self.egn_values[I], self.egn_vectors[I]) for I in range(len(self.egn_values))]
        data.sort(key=lambda a: np.abs(a[0]-target_eigenvalue))
        nearest_egn_value, nearest_vector = data[0]
        sort_by_energy = np.argsort(np.sum(-nearest_vector**2, axis=0))
        egn_vector_sorted = nearest_vector[:, sort_by_energy]
        modes_sorted = self.modes[sort_by_energy]
        normalized_egn_vector = egn_vector_sorted / egn_vector_sorted[np.argmax(np.abs(egn_vector_sorted[:, 0])), 0]
        return nearest_egn_value, normalized_egn_vector, modes_sorted
    
    def condition_number(self):
        r"""
        Computes the condition number, defined as the ratio of the largest to the smallest absolute eigenvalue.

        Returns:
            float: The condition number
        """
        eigenvalues = self.egn_values
        abs_eigenvalues = np.abs(eigenvalues)
        max_eigenvalue = np.max(abs_eigenvalues)
        min_eigenvalue = np.min(abs_eigenvalues)
        if min_eigenvalue == 0:
            raise ValueError("The smallest absolute eigenvalue is zero, condition number is undefined.")
        return max_eigenvalue / min_eigenvalue

@dataclass
class AE3DEigenvector:
    """
    Stores the details of a specific eigenvector from AE3D simulations, including the eigenvalue,
    radial coordinates, and a detailed breakdown of harmonics with their amplitudes.

    Attributes:
        eigenvalue (float): The closest eigenvalue found.
        s_coords (np.ndarray): Radial coordinate values.
        harmonics (list[Harmonic]): List of harmonics comprising the eigenvector.
    """
    eigenvalue: float
    s_coords: np.ndarray
    harmonics: List[Harmonic]

    @staticmethod
    def from_eig_mode_asci(eig_mode_asci : EigModeASCI, target_eigenvalue : float):
        """
        Factory method to create an AE3DEigenvector instance from an EigModeASCI data class.

        Args:
            eig_mode_asci (EigModeASCI): The EigModeASCI instance to process.
            target_eigenvalue (float): Target eigenvalue to identify the nearest eigenvector.

        Returns:
            AE3DEigenvector: An initialized AE3DEigenvector object.
        """
        egn_value, egn_vector_sorted, modes_sorted = eig_mode_asci.get_nearest_eigenvector(target_eigenvalue)
        harmonics = [
            Harmonic(m=modes_sorted['m'][i], n=modes_sorted['n'][i], amplitudes=egn_vector_sorted[:, i])
            for i in range(len(modes_sorted))
        ]
        return AE3DEigenvector(eigenvalue=egn_value, s_coords=eig_mode_asci.s_coords, harmonics=harmonics)
    
    def export_to_numpy(self, filename: str, num_harmonics: int = None, resolution_step: int = 1):
        """
        Exports the harmonics to a NumPy file.

        Args:
            filename (str): The name of the file to export to.
            num_harmonics (int, optional): The number of harmonics to export. If None, all harmonics are exported.
        """
        num_harmonics = num_harmonics or len(self.harmonics)
        harmonics_data = {
            'eigenvalue': self.eigenvalue,
            's_coords': self.s_coords[0:-1:resolution_step],
            'harmonics': np.array([
                (h.m, h.n, h.amplitudes[0:-1:resolution_step]) for h in self.harmonics[:num_harmonics]
            ], dtype=object)
        }
        np.save(filename, harmonics_data)
        print(f'Harmonics exported to {filename}')

    @classmethod
    def load_from_numpy(cls, filename: str):
        """
        Loads harmonics from a NumPy file into an AE3DEigenvector instance.

        Args:
            filename (str): The name of the file to load from.

        Returns:
            AE3DEigenvector: An instance of AE3DEigenvector with loaded data.
        """
        harmonics_data = np.load(filename, allow_pickle=True).item()
        harmonics = [
            Harmonic(m=m, n=n, amplitudes=amplitudes) 
            for m, n, amplitudes in harmonics_data['harmonics']
        ]
        return cls(
            eigenvalue=harmonics_data['eigenvalue'],
            s_coords=harmonics_data['s_coords'],
            harmonics=harmonics
        )

def plot_ae3d_eigenmode(mode: AE3DEigenvector, harmonics: int = 5):
    """
    Creates an interactive plot of the top 'harmonics' number of harmonics for a given AE3DEigenvector.

    Args:
        mode (AE3DEigenvector): An instance of AE3DEigenvector containing the eigenmode data to plot.
        harmonics (int): The number of top harmonics to plot (default is 5).

    Returns:
        fig: A Plotly Figure object displaying the harmonics.
    """
    # Create a subplot with 2 rows and 1 column
    fig = go.Figure()

    # Ensure not to exceed the number of available harmonics
    num_harmonics_to_plot = min(harmonics, len(mode.harmonics))

    # Plot each of the top harmonics in the first subplot
    for i in range(num_harmonics_to_plot):
        harmonic = mode.harmonics[i]
        fig.add_trace(
            go.Scatter(x=mode.s_coords, y=harmonic.amplitudes, mode='lines', name=f'(m={harmonic.m}, n={harmonic.n})'))

    # Update axis labels and titles
    fig.update_yaxes(title_text=r"$\text{Electrostatic Potential }\varphi$")
    fig.update_xaxes(title_text=r"$\text{Normalized Flux }s$")
    fig.update_layout(title=f"Eigenvalue: {mode.eigenvalue}")

    return fig

def continuum_from_ae3d(ae3d : EigModeASCI, minevalue = 0.0, maxevalue = 600**2):
    """
    Given an AE3D EigModeASCI object, this function extracts the set of eigenvalues
    between minevalue and maxevalue, and returns a list of ModeContinuum objects. This is 
    used for visualization of the spectral content, which includes the continuum modes. 

    Args:
        ae3d (EigModeASCI): An instance of EigModeASCI containing the eigenmode data.
        minevalue (float): Minimum eigenvalue to consider (default is 0.0).
        maxevalue (float): Maximum eigenvalue to consider (default is 600**2).  

    Returns:
        List[ModeContinuum]: A list of ModeContinuum objects representing the modes within the 
        specified eigenvalue range.
    """
    ModeList = []
    mn_set = set()

    for evalue in ae3d.egn_values:
        if evalue < minevalue:
            continue
        if evalue > maxevalue:
            continue
        evec = AE3DEigenvector.from_eig_mode_asci(ae3d, evalue)
        m, n = evec.harmonics[0].m, evec.harmonics[0].n
        if (m, n) in mn_set:
            for mode in ModeList:
                if (
                    (mode.get_poloidal_mode() == m)
                    and
                    (mode.get_toroidal_mode() == n)
                    ):
                    break
            mode._s.append(evec.s_coords[np.where(evec.harmonics[0].amplitudes==np.max(evec.harmonics[0].amplitudes))[0][0]])
            mode._freq.append(np.sqrt(evalue))
        else:
            mn_set.add((m, n))
            ModeList.append(ModeContinuum(
                n  = evec.harmonics[0].n,
                m  = evec.harmonics[0].m,
                s = [evec.s_coords[np.where(evec.harmonics[0].amplitudes==np.max(evec.harmonics[0].amplitudes))[0][0]]],
                freq = [np.sqrt(evalue)]
            ))
    return ModeList