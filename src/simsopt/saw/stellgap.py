import os
from typing import List
from dataclasses import dataclass
import numpy as np
import plotly.graph_objects as go

__all__ = ['Harmonic','ModeContinuum','AlfvenSpecData']

@dataclass
class Harmonic:
    """
    Represents a harmonic in the Fourier decomposition of an eigenvector.

    Attributes:
        m (int): Poloidal mode number.
        n (int): Toroidal mode number.
        amplitudes (np.ndarray): Array of amplitudes corresponding to radial points.
    """
    m: int
    n: int
    amplitudes: np.ndarray

class ModeContinuum:
    _n: int
    _m: int
    _s: np.array
    _freq: np.array
    r"""
    A class to handle the parsing and storage of continuum modes, which includes
    poloidal and toroidal mode numbers, flux surfaces, and frequencies.
    This class is used to represent the continuum modes extracted from AE3D and STELLGAP
    simulations.
    """
    def __init__(self, m: int, n: int, s = None, freq = None):
        r"""
        Initialize a ModeContinuum instance. s and frequencies can be specified but are not necessary 
        to initialize.

        Args:
            m (int): Poloidal mode number.
            n (int): Toroidal mode number.
            s (np.array, optional): Array of flux surfaces. Defaults to None.
            freq (np.array, optional): Array of frequencies corresponding to the flux surfaces. Defaults to None.
        Raises:
            Exception: If a negative flux label is provided.
        """
        self._m = m
        self._n = n
        self._s = s
        self._freq = freq
        
    def _check_negative_s(self):
        r"""
        Check if any flux label is negative.
        Raises an exception if a negative flux label is found.
        """
        for s in self._s:
            if s < 0:
                self._negative_exception()
            
    def _negative_exception(self):
        r"""
        Raises an exception indicating that a negative flux label was provided.
        The flux label must be positive.
        """
        raise Exception("A negative flux label was provided. The flux label must be positive.")

    def set_poloidal_mode(self, m: int):
        r"""
        Set the poloidal mode number.
        Args:
            m (int): Poloidal mode number.
        """
        self._m = m

    def set_toroidal_mode(self, n: int):
        r"""
        Set the toroidal mode number.
        Args:
            n (int): Toroidal mode number.
        """
        self._n = n

    def set_points(self, s: np.array, freq: np.array):
        r"""
        Set the flux surfaces and frequencies.
        Args:
            s (np.array): Array of flux surfaces.
            freq (np.array): Array of frequencies corresponding to the flux surfaces.
        """
        self._s = s
        self._freq = freq

        self._check_matching_freqs()

    def get_poloidal_mode(self):
        r"""
        Get the poloidal mode number.

        Returns:
            int: Poloidal mode number.
        """
        return self._m

    def get_toroidal_mode(self):
        r"""
        Get the toroidal mode number.
        Returns:
            int: Toroidal mode number.
        """
        return self._n
    
    def get_flux_surfaces(self):
        r"""
        Get the flux surfaces.

        Returns:
            np.array: Array of flux surfaces.
        """
        return self._s
    
    def get_frequencies(self):
        r"""
        Get the frequencies.

        Returns:
            np.array: Array of frequencies.
        """
        return self._freq
    
    def add_point(self, s: float, freq: float):
        r"""
        Add a point to the flux surfaces and frequencies.
        Args:
            s (float): Flux surface value to add.
            freq (float): Frequency value to add.
        Raises:
            Exception: If the flux surface value is negative.
        """
        if s < 0:
            self._negative_exception()

        self._s = np.append(self._s, s)
        self._freq = np.append(self._freq, freq)

class AlfvenSpecData(np.ndarray):
    r"""
    Subclass of numpy.ndarray with dtype specific to STELLGAP output in alfven_spec files.
    """

    def __new__(cls, filenames: List[str]):
        r"""
        Create a new instance of AlfvenSpecData from a list of filenames.

        Args:
            filenames (List[str]): List of filenames containing alfven_spec data.

        Returns:
            AlfvenSpecData: An instance of AlfvenSpecData containing the loaded data.
        """
        if not filenames:
            raise ValueError("No filenames provided")

        data = np.vstack([np.loadtxt(fname,
                                     dtype=[('s', float), ('ar', float),
                                            ('ai', float), ('beta', float),
                                            ('m', int), ('n', int)])
                           for fname in filenames])
        obj = np.asarray(data).view(cls)
        return obj
    
    @classmethod
    def from_dir(cls, directory: str):
        r"""
        Load all alfven_spec data from a specified directory.

        Args:
            directory (str): Path to the directory containing alfven_spec files.

        Returns:
            AlfvenSpecData: An instance of AlfvenSpecData containing the loaded data.
        """
        files = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.startswith('alfven_spec')]
        if not files:
            raise ValueError(f"No alfven_spec files found in the directory {directory}")
        return cls(files)

    def nonzero_beta(self):
        r"""
        Filter entries where beta is not zero.

        Returns:
            AlfvenSpecData: An instance of AlfvenSpecData containing the filtered data.
        """
        return self[self['beta'] != 0]

    def sort_by_s(self):
        r"""
        Sort the array based on the 's' field.
        
        Returns:
            AlfvenSpecData: An instance of AlfvenSpecData sorted by the 's'
            field.
        """
        return self[np.argsort(self['s'])]
    
    def get_modes(self) -> List[ModeContinuum]:
        r"""
        Extract modes from the AlfvenSpecData, creating ModeContinuum instances
        for each unique combination of poloidal (n) and toroidal (m) mode numbers.

        Returns:
            List[ModeContinuum]: A list of ModeContinuum instances, each representing
            a unique mode with its corresponding flux surfaces (s) and frequencies.
        """
        data = self.nonzero_beta()
        modes = [
            ModeContinuum(
                n=n, 
                m=m, 
                s=(filtered_data := np.sort(data[(data['n'] == n) & (data['m'] == m)], order='s'))['s'], 
                freq=np.sqrt(np.abs(filtered_data['ar'] / filtered_data['beta']))
            )
            for n, m in {(a['n'], a['m']) for a in data}
        ]
        return modes
    
    def condition_number(self):
        r"""
        For each s, compute the condition number as the ratio of largest to smallest eigenvalue
        return the array of s and corresponding condition numbers.

        Returns:
            tuple: A tuple containing:
                - s (np.array): Unique flux surface values.
                - condition_numbers (np.array): Condition numbers for each unique flux surface.
        """
        data = self.nonzero_beta().sort_by_s()
        s = np.unique(data['s'])
        condition_numbers = np.array([
            np.max(np.abs(data[data['s'] == s_]['ar'])) / np.min(np.abs(data[data['s'] == s_]['ar']))
            if np.min(np.abs(data[data['s'] == s_]['ar'])) != 0 else np.inf
            for s_ in s
        ])
        return s, condition_numbers


def plot_continuum(overlays: List[List[ModeContinuum]], show_legend: bool = False, normalized_modes=False, yrange=None) -> go.Figure:
    r"""
    Plot the continuum modes using Plotly. Several overlays can be provided. This is useful, for example, in comparing 
    AE3D and STELLGAP results.

    Args:
        overlays: List[List[ModeContinuum]]:
            A list of lists, where each inner list contains ModeContinuum instances representing different overlays.
        show_legend (bool, optional): Whether to show the legend in the plot. Defaults to False.
        normalized_modes (bool, optional): If True, normalize the frequencies by the Alfven frequency. Defaults to False.
        yrange (list, optional): Custom y-axis range for the plot. If None, defaults are used based on normalized_modes.
    """
    fig = go.Figure()
    
    colors = ['blue', 'red', 'green', 'purple']
    markers = ['circle', 'square', 'diamond', 'cross'] 
    
    for idx, modes in enumerate(overlays):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        for md in modes:
            formatted_freq = [f"{f:.10g}" for f in md.get_frequencies()]
            fig.add_trace(go.Scatter(
                x=md.get_flux_surfaces(),
                y=md.get_frequencies(),
                mode='markers',
                name=(f'm={md.get_poloidal_mode()}, '
                      +f'n={md.get_toroidal_mode()} (Overlay {idx+1})'),
                marker=dict(size=3, symbol=marker, color=color),
                line=dict(width=0.5, color=color),
                text=[(f'freq = {f}, '
                       +f'm={md.get_poloidal_mode()}, '
                       +f'n={md.get_toroidal_mode()}') for f in formatted_freq],
                hoverinfo="text+x+y",
                hoverlabel=dict(
                    font_size=16,
                    bgcolor="white",
                    bordercolor=color
                )
            ))

    if normalized_modes:
        yaxis_title = r'$\text{normalized frequency }\omega/\omega_A$'
        yaxis_range = [0, 5]
    else:
        yaxis_title = r'$\text{Frequency }\omega\text{ [kHz]}$'
        yaxis_range = [0, 600]
    
    if yrange is not None:
        yaxis_range = yrange
    
    fig.update_layout(
        autosize=True,
        title=r'$\text{Continuum: }$',
        xaxis_title=r'$\text{Normalized flux }s$',
        yaxis_title=yaxis_title,
        xaxis=dict(range=[np.min([np.min(md.get_flux_surfaces()) for modes in overlays for md in modes]), 
                          np.max([np.max(md.get_flux_surfaces()) for modes in overlays for md in modes])]),
        yaxis=dict(range=yaxis_range),
        legend=dict(
            title=r'$\text{Mode: }$',
            yanchor="top",
            y=1.4,
            xanchor="center",
            x=0.5,
            orientation="h"
        ),
        showlegend=show_legend
    )
    
    return fig