"""
Poincaré plots of magnetic fields.

The :class:`PoincarePlotter` uses an :class:`~simsopt.field.integrator.Integrator`
to compute the crossings of field lines with toroidal planes, caches the
results, and provides methods to plot them. Because the plotter depends on the
integrator (which depends on the magnetic field) in the simsopt dependency
graph, the cached results are discarded when the field changes.
"""

import logging
import os
from math import sqrt
from pathlib import Path

import numpy as np

from .._core import Optimizable
from .integrator import Integrator, SimsoptFieldlineIntegrator, ScipyFieldlineIntegrator

logger = logging.getLogger(__name__)

__all__ = ['PoincarePlotter']


class PoincarePlotter(Optimizable):
    """
    Class to facilitate the calculation of field lines and 
    plotting the results in a Poincare plot. 
    Uses field periodicity to speed up calculation
    """
    def __init__(self, integrator: Integrator, start_points_RZ, phis=None, n_transits=100, add_symmetry_planes=True, store_results=False, phi0=None, nfp=1):
        """
        Initialize the PoincarePlotter. 
        This class uses an Integrator to compute field lines, and takes care of plotting them. 
        If the field is stellarator-symmetric, and symmetry planes are included, then 
        information from identical planes are all plotted together (resulting in better plots with shorter integration). 

        Args:
            integrator: the integrator to be used for the calculation
            start_points_RZ: nx2 array of starting points in cylindrical coordinates (R,Z)
        Kwargs:
            phis (None): angles in [0, 2pi] for which we wish to compute Poincare.
                  *OR* int: number of planes to compute, equally spaced in [0, 2pi/nfp].
            n_transits (100): number of toroidal transits to compute
            add_symmetry_planes (True): if true, we add planes that are identical through field periodicity, increasing the efficiency of the calculation. 
            store_results (False): if true, use a cache on disk to store/retrieve results. 
                The results are stored in a file named 'poincare_data.npz' in the current directory. 
                The results are given a hash based on the MagneticField degrees of freedom and poincare attributes, triggering recomputation only if relevant parameters change.
            phi0 (None): initial angle in the phi plane (default None). if None, the first phi in phis is used.
            nfp (1): number of field periods of the magnetic field. Used to generate planes
                that are identical through field periodicity.
        """
        self.nfp = nfp
        self._start_points_RZ = start_points_RZ
        self.integrator = integrator
        self.n_transits = n_transits
        if isinstance(phis, int):
            self._phis = self.generate_phis(phis, nfp=self.nfp)
        elif phis is None:
            self._phis = np.array([0.0,])
        else:
            self._phis = np.atleast_1d(phis)
        
        if add_symmetry_planes:
            self._phis = self.generate_symmetry_planes(self._phis, nfp=self.nfp)

        self.phi0 = self._phis[0] if phi0 is None else phi0
        self.is_plotter = self.integrator.comm is None or self.integrator.comm.rank == 0  # only rank 0 does plotting
        self.need_to_recompute = True
        self._randomcolors = None
        Optimizable.__init__(self, depends_on=[integrator,])
        self.store_results = store_results
        if store_results:
            # load file form disk if it exists
            self.retrieve_poincare_data()


    @property
    def randomcolors(self):
        """
        Generate a list of random colors for plotting.
        """
        #check if already generated:
        if self._randomcolors is not None:
            return self._randomcolors
        else:
            np.random.seed(0)  # for reproducibility
            self._randomcolors = np.random.rand(len(self.start_points_RZ), 3)
            return self._randomcolors

    @staticmethod
    def generate_phis(nplanes, nfp=1):
        """
        Generate nplanes equally spaced phis in [0, 2pi/nfp].
        Args:
            nplanes: number of planes to generate
            nfp: number of field periods (default: 1)
        Returns:
            phis: list of phis in [0, 2pi/nfp]
        """
        return np.linspace(0, 2*np.pi/nfp, nplanes, endpoint=False)
    
    @staticmethod
    def generate_symmetry_planes(phis, nfp=1):
        """
        Given a list of phis in [0, 2pi/nfp], generate the full list of phis
        in [0, 2pi] by adding the symmetry planes. 
        Args: 
            phis: list of phis in [0, 2pi/nfp]
            nfp: number of field periods (default: 1)
        Returns:
            list_of_phis: list of phis in [0, 2pi] including the symmetry planes
        """
        list_of_phis = [phis + per_idx*2*np.pi/nfp for per_idx in range(nfp)]
        # remove duplicates and sort
        list_of_phis = np.unique(np.concatenate(list_of_phis))
        return list_of_phis
    
    @property
    def phis_for_plotting(self):
        """
        the phis in the first period, useful for plotting
        """
        plot_period = 2*np.pi/self.nfp
        phis_for_plotting = self.phis[np.where(self.phis < plot_period)]
        return phis_for_plotting

    @property
    def start_points_RZ(self):
        """
        start ponts in R,Z for the field line integration
        """
        return self._start_points_RZ

    @start_points_RZ.setter
    def start_points_RZ(self, array):
        if array.shape != self._start_points_RZ.shape:
            self._randomcolors = None  # reset colors 
        self._start_points_RZ = array
        self.recompute_bell()

    @property
    def phis(self):
        """
        all the phi planes used for the calculation
        """
        return self._phis
    
    @phis.setter
    def phis(self, value):
        self._phis = value
        self.recompute_bell()


    @classmethod
    def from_field(cls, field, start_points_RZ, phis=None, n_transits=1, add_symmetry_planes=True,
                   stopping_criteria=None, comm=None, integrator_type='simsopt', nfp=1, **kwargs):
        """
        Helper to create a PoincarePlotter directly from a MagneticField bypassing 
        the manual creation of an Integrator. 

        Parameters mirror PoincarePlotter.__init__, while constructing the appropriate integrator.
        Args:
            field: the magnetic field to be used for the integration
            start_points_RZ: nx2 array of starting points in cylindrical coordinates (R,Z)
            phis: angles in [0, 2pi] for which we wish to compute Poincare.
                  *OR* int: number of planes to compute, equally spaced in [0, 2pi/nfp].
            n_transits: number of toroidal transits to compute
            add_symmetry_planes: if true, we add planes that are identical through field periodicity, increasing the efficiency of the calculation.
            stopping_criteria: list of StoppingCriterion objects that halt integration. Only used if integrator_type is 'simsopt'
            comm: MPI communicator for parallelization
            integrator_type: type of integrator to use ('simsopt' or 'scipy')
            nfp: number of field periods of the magnetic field
            **kwargs: additional arguments to pass to the integrator constructor
        """
        if stopping_criteria is None:
            stopping_criteria = []
        if phis is None:
            phis = 4  # default to 4 planes if not specified
        if integrator_type == 'simsopt':
            integrator = SimsoptFieldlineIntegrator(field, comm=comm, stopping_criteria=stopping_criteria, **kwargs)
        elif integrator_type == 'scipy':
            integrator = ScipyFieldlineIntegrator(field, comm=comm, **kwargs)
        else:
            raise ValueError(f"Integrator type {integrator_type} not supported.")
        return cls(integrator, start_points_RZ, phis=phis, n_transits=n_transits, add_symmetry_planes=add_symmetry_planes, nfp=nfp)


    def recompute_bell(self, parent=None):
        """
        clear the caches when any object on which this depends changes. 
        """
        self._res_phi_hits = None
        self._res_tys = None
        self._lost = None
        self.need_to_recompute = True
    
    def _compute(self):
        """
        Compute the trajectories and plane crossings with the integrator, and
        store them to disk if store_results is True.
        """
        self._res_tys, self._res_phi_hits = self.integrator.compute_poincare_hits(
            self.start_points_RZ, self.n_transits, phis=self.phis, phi0=self.phi0)
        if self.store_results:
            self.save_poincare_data()
        self.need_to_recompute = False

    @property
    def res_tys(self):
        """
        Compute or retrieve the field line trajectories for the Poincare plot.
        If calculation is performed and store_results is True, the results are saved to disk.
        Returns:
            res_tys: list of numpy arrays (one for each particle) containing
                     the trajectory of each field line. Each row of the array contains
                     `[time, x, y, z]`.
        """
        if self.store_results and self._res_tys is None:
            # read from disk if it already exists
            self.retrieve_poincare_data()
        if self._res_tys is None or self.need_to_recompute:
            self._compute()
        return self._res_tys
    
    @property
    def res_phi_hits(self):
        """
        Compute or retrieve the Poincare section hits for the Poincare plot.
        If calculation is performed and store_results is True, the results are saved to disk.
        Returns:
            res_phi_hits: list of numpy arrays (one for each particle) containing
                          the Poincare section hits of each field line. Each row of the array contains
                          `[phi, plane_index, x, y, z]`.
        """
        if self.store_results and self._res_phi_hits is None:
            # read from disk if it already exists
            self.retrieve_poincare_data()
        if self._res_phi_hits is None or self.need_to_recompute:
            self._compute()
        return self._res_phi_hits
    
    @property
    def poincare_hash(self):
        """
        Generate a hash from the MagneticFields dofs, self.phis, self.start_points_RZ, and self.n_transits. 
        Returns: 
            poincare_hash: hash value
        """
        hash_list = self.integrator.field.full_x.tolist() + self.phis.tolist() + self.start_points_RZ.flatten().tolist() + [self.n_transits]
        poincare_hash = hash(tuple(hash_list))
        return poincare_hash
    
    def save_poincare_data(self, filename=None, name=None):
        """
        Save the computed Poincare data (res_phi_hits and res_tys) to disk for later retrieval.
        The data is by default saved in a file named ``poincare_data.npz``, under a key derived
        from a has from the MagneticField DoFs, and the PoincarePlotter settings. This ensures that data calculated in different sessions or even on different
        machines can be loaded. 
        Args:
            filename: optional filename to override the default ``poincare_data.npz``
            name: optional name to override the default hash-derived key prefix.
        """
        if not self.is_plotter:
            return

        if filename is None:
            filename = "poincare_data.npz"
        filename = Path(filename)
        if filename.suffix != ".npz":
            filename = filename.with_suffix(".npz")

        if name is None:
            name = self.poincare_hash
        name = str(name)

        data_to_save = {}
        if filename.exists():
            with np.load(filename, allow_pickle=True) as existing:
                data_to_save = {key: existing[key] for key in existing.files}

        updated = False
        if self._res_phi_hits is not None:
            data_to_save[f"res_phi_{name}"] = np.array(self._res_phi_hits, dtype=object)
            updated = True
        if self._res_tys is not None:
            data_to_save[f"res_tys_{name}"] = np.array(self._res_tys, dtype=object)
            updated = True

        if updated:
            np.savez_compressed(filename, **data_to_save)

    def retrieve_poincare_data(self, name=None, filename=None):
        """
        Check if cached Poincare data is available on disk, and load it
        into the current object. By default, a file named ``poincare_data.npz`` 
        is used, but this can be overridden by passing ``filename``.
        The data is stored in keys derived from a hash of the magnetic field DoFs, 
        and the PoincarePlotter settings. 
        This ensures that data calculated in different sessions or even on different
        machines can be loaded. 
        Args:
            name: optional name to override the hash-derived key prefix.
            filename: optional filename to override the default ``poincare_data.npz``.
        """
        if filename is None:
            filename = "poincare_data.npz"
        filename = Path(filename)
        if filename.suffix != ".npz":
            filename = filename.with_suffix(".npz")
        if name is None:
            name = self.poincare_hash
        name = str(name)

        if not filename.exists():
            logger.debug(f"File {filename} not found. Not loading cached poincare data.")
            return

        res_phi_key = f"res_phi_{name}"
        res_tys_key = f"res_tys_{name}"
        with np.load(filename, allow_pickle=True) as data:
            if res_phi_key in data.files:
                loaded_hits = data[res_phi_key]
                self._res_phi_hits = [np.asarray(arr, dtype=float).copy() for arr in loaded_hits]
            if res_tys_key in data.files:
                loaded_tys = data[res_tys_key]
                self._res_tys = [np.asarray(arr, dtype=float).copy() for arr in loaded_tys]

        if self._res_phi_hits is not None or self._res_tys is not None:
            self.need_to_recompute = False
        return
    
    def particles_to_vtk(self, filename):
        """
        Stores the trajectories in a vtk file
        for visualization in paraview.
        Export particle tracing or field lines to a vtk file.
        """
        from pyevtk.hl import polyLinesToVTK
        x = np.concatenate([xyz[:, 1] for xyz in self.res_tys])
        y = np.concatenate([xyz[:, 2] for xyz in self.res_tys])
        z = np.concatenate([xyz[:, 3] for xyz in self.res_tys])
        ppl = np.asarray([xyz.shape[0] for xyz in self.res_tys])
        data = np.concatenate([i*np.ones((self.res_tys[i].shape[0], )) for i in range(len(self.res_tys))])
        polyLinesToVTK(filename, x, y, z, pointsPerLine=ppl, pointData={'idx': data})

    
    def remove_poincare_data(self, filename=None):
        """
        Clear the saved poincare data file.
        The hash does not take into account the integrator type or tolerances, so if you need higher precision, use this method
        to clear the file and recompute.
        Args:
            filename: filename to remove (default: poincare_data.npz)
        """
        if not self.is_plotter:
            return
        filename = Path("poincare_data.npz" if filename is None else filename)
        if filename.exists():
            os.remove(filename)
            logger.info(f"Removed poincare data file {filename}.")  
        return

        
    @property
    def lost(self): 
        """
        Get the points where the integration stopped due to a stopping criterion.
        This means the last entry of the 'idx' column in the res_phi_hits array is negative.
        Returns:
            lost: list of booleans indicating whether each field line was lost due to a stopping criterion.
        """
        # list comprehension... look at final element, if element 1 negative, then stopping 
        # criterion was encounterd. first stopping criterion is transit number, so ignore.
        if self._lost is None:
            self._lost = [traj[-1, 1] < -1 for traj in self.res_phi_hits]
        return self._lost

    def plane_hits_cyl(self, plane_idx):
        """
        Get the points where the field lines intersect the plane defined by the index.
        Returns a list of arrays containing the R,Z coordinates of the intersection 
        points.  
        Args: 
            plane_idx: index of the plane to get hits for
        Returns:
            hits: list of arrays of shape (n_hits, 2) containing the R,Z points of the hits. 
        """
        hits_cart = self.plane_hits_cart(plane_idx)
        hits = [self.integrator._xyz_to_rphiz(hc)[:, ::2] for hc in hits_cart]
        return hits  # return only R,Z
    
    
    def plane_hits_cart(self, plane_idx):
        """
        Get the points where the field lines hit the plane defined by the index. 
        Returns a list of arrays containing the x,y,z coordinates of the intersection 
        points.
        Args:
            plane_idx: index of the plane to get hits for
        Returns:
            hits: list of arrays of shape (n_hits, 3) containing the x,y,z points of the hits.
        """
        if plane_idx >= len(self.phis):
            raise ValueError(f"Plane index {plane_idx} is larger than the number of planes {len(self.phis)}.")
        
        hits = []
        for traj in self.res_phi_hits:
            hits_xyz = traj[np.where(traj[:, 1] == plane_idx)[0], 2:]  # res_phi_hits col1 = plane idx or stopping criterion
            hits.append(hits_xyz)  # append the xyz points
        return hits  # list of arrays of shape (n_hits, 3)
    
    @staticmethod
    def fix_axes(ax, xlabel='R', ylabel='Z', title=None):
        """
        Label the axes of the plot. 
        Args: 
            ax: matplotlib axis to label
            xlabel: label for the x-axis
            ylabel: label for the y-axis
            title: title for the plot (if None, no title is set)
        """
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')
        if title is not None:
            ax.set_title(title)

    def plot_poincare_plane_idx(self, plane_idx, mark_lost=False, ax=None, **kwargs):
        """
        plot a single cross-section of the field by referencing the index in the PoincarePlotter's phis. 
        *NOTE*: if running parallel, call this function on all ranks.
        Args:
            plane_idx: index of the plane to plot
            mark_lost: if True, mark the field lines that were lost due to stopping criteria in red
            ax: matplotlib axis to plot on (if None, create a new figure and axis)
            **kwargs: additional keyword arguments to pass to the scatter plotter
        Returns:
            fig, ax: the figure and axis objects (only on rank 0, otherwise None, None)
        """
        if plane_idx >= len(self.phis):
            raise ValueError(f"Plane index {plane_idx} is larger than the number of planes {len(self.phis)}.")
        
        # can trigger recompute, so all ranks execute
        hits_thisplane = self.plane_hits_cyl(plane_idx)

        # rank0 only
        if self.is_plotter:
            import matplotlib.pyplot as plt
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure
            
            marker = kwargs.pop('marker', '.')
            s = kwargs.pop('s', 2.5)
            color = kwargs.pop('color', 'random')

            for idx, trajpoints in enumerate(hits_thisplane):
                if color == 'random':
                    this_color = self.randomcolors[idx]
                else: 
                    this_color = color

                this_marker = marker
                this_s = s
                if mark_lost:
                    lost = self.lost[idx]
                    if lost:
                        this_color = 'r'
                        this_marker = 'x'
                        this_s = s*3
                ax.scatter(trajpoints[:, 0], trajpoints[:, 1], marker=this_marker, s=this_s, color=this_color, linewidths=0, **kwargs)
            return fig, ax
        else: 
            return None, None  # other ranks do not plot anything

    def plot_poincare_single(self, phi, prevent_recompute=False, ax=None, mark_lost=False, fix_axes=True, surf=None, include_symmetry_planes=True, **kwargs):
        """
        plot a single cross-section of the field at a given phi value. 
        If this value is not in the list of phis, this phi will be added to the PoincarePlotters planes
        and the sections will be re-computed (unless prevent_recompute is True, in which case an error is raised). 
        *NOTE*: if running parallel, call this function on all ranks. 
        Args: 
            phi: angle in [0, 2pi] at which to plot the Poincare section
            prevent_recompute: if True, do not trigger a recompute if the requested plane is not available
            ax: matplotlib axis to plot on (if None, create a new figure and axis)
            mark_lost: if True, mark the field lines that were lost due to stopping criteria in red
            fix_axes: if True, fix the axes to be equal and labeled (otherwise, deal with the returned axes object)
            surf: if given, a simsopt surface to plot the cross-section of (in black)
            include_symmetry_planes: if True, include all planes that are identical through field periodicity in this plot
            **kwargs: additional keyword arguments to pass to the single plane plotter
        Returns:
            fig, ax: the figure and axis objects (only on rank 0, otherwise None, None)
        """
        if phi not in self.phis:
            if not prevent_recompute:
                self.phis = np.append(self.phis, phi)
                phi_indices = [len(self.phis) - 1]  # index of the newly added plane
            else:
                raise ValueError(f"The requested plane at phi={phi} has not been computed.")
        else:
            if include_symmetry_planes:
                phi_indices = np.where(np.isclose((self.phis - phi) % (2*np.pi/self.nfp), 0))[0]
            else: 
                phi_indices = np.where(np.isclose(self.phis, phi))[0]
        
        # trigger recompute on all ranks if necessary:
        _ = self.res_phi_hits

        if self.is_plotter:
            import matplotlib.pyplot as plt
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure

            for phi_index in phi_indices:
                self.plot_poincare_plane_idx(phi_index, ax=ax, mark_lost=mark_lost, **kwargs)

            if surf is not None:
                # divide by 2pi cause simsopt surf phi is in [0,1]
                cross_section = surf.cross_section(phi=phi/(2*np.pi))
                r_interp = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
                z_interp = cross_section[:, 2]
                ax.plot(r_interp, z_interp, linewidth=1, c='k')
            
            if fix_axes:
                self.fix_axes(ax)
            return fig, ax
        else: 
            return None, None  # other ranks do not plot anything

        


    def plot_poincare_all(self, mark_lost=False, fix_ax=True, **kwargs):
        """
        Plot all the computed poincare planes in a grid. 
        A square grid is generated, so best results are achieved if the plotter uses 4 or 9 planes. 
        *NOTE*: if running parallel, call this function on all ranks. 
        Args:
            mark_lost: if True, mark the field lines that were lost due to stopping criteria in red
            fix_ax: if True, fix the axes to be equal and labeled (otherwise, deal with the returned axes object)
            **kwargs: additional keyword arguments to pass to the single plane plotter
        Returns:
            fig, axs: the figure and axes objects (only on rank 0, otherwise None)
        """
        _ = self.res_phi_hits  #trigger recompute on all ranks if necessary

        if self.is_plotter:
            from math import ceil
            import matplotlib.pyplot as plt
            nrowcol = ceil(sqrt(len(self.phis_for_plotting)))
            fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(8, 5))
            
            axs = np.atleast_1d(axs).ravel()  # make array and flatten

            for section_idx, phi in enumerate(self.phis_for_plotting):  #ony the plane in the first field period
                ax = axs[section_idx]
                self.plot_poincare_single(phi, ax=ax, mark_lost=mark_lost, prevent_recompute=True, fix_axes=fix_ax, **kwargs)
                textstr = f" φ = {phi/np.pi:.2f}π "
                props = dict(boxstyle='round', facecolor='white', edgecolor='black')
                ax.text(0.05, 0.02, textstr, transform=ax.transAxes, fontsize=6,
                        verticalalignment='bottom', bbox=props)

            plt.tight_layout()

            return fig, axs
        else: 
            return None, None  # other ranks do not plot anything

    def plot_fieldline_trajectories_3d(self, engine='mayavi', mark_lost=False, show=True,  **kwargs): 
        """
        Plot the full 3D trajectories of the field lines. 
        Can be very busy if lines are followed for long. 

        Hints: 
            - for mayavi, use tube_radius kwarg to adjust line thickness, opacity for making them transparent. 
        Args:
            engine: 'mayavi' or 'plotly' or 'matplotlib'
            mark_lost: if True, mark the field lines that were lost due to stopping criteria in red
            show: if True, show the plot immediately
            **kwargs: additional keyword arguments to pass to the plotting function
        Returns:
            None

        """
        trajectories = self.res_tys  # trigger recompute if necessary

        if self.is_plotter:
            # unify color kw handling across engines
            if 'color' in kwargs:
                base_color = kwargs.pop('color')
            else:
                base_color = 'random'
            if engine == 'mayavi':
                from mayavi import mlab
                tube_radius = kwargs.pop('tube_radius', 0.005)
                color = base_color
                for idx, traj in enumerate(trajectories): 
                    if color == 'random':
                        this_color = tuple(self.randomcolors[idx])
                    else: 
                        this_color = tuple(color)
                    lost = self.lost[idx] if mark_lost else False
                    this_tube_radius = tube_radius*3 if lost else tube_radius
                    this_color = (1, 0, 0) if lost else this_color
                    mlab.plot3d(traj[:, 1], traj[:, 2], traj[:, 3], tube_radius=this_tube_radius, color=this_color, **kwargs)
                if show:
                    mlab.show()
            if engine == 'plotly':
                import plotly.graph_objects as go
                fig = go.Figure()
                color = base_color
                for idx, traj in enumerate(trajectories): 
                    if color == 'random':
                        this_color = 'rgb({},{},{})'.format(*(self.randomcolors[idx]*255).astype(int))
                    else: 
                        this_color = color
                    lost = self.lost[idx] if mark_lost else False
                    this_width = 6 if lost else 2
                    this_color = 'rgb(255,0,0)' if lost else this_color
                    fig.add_trace(go.Scatter3d(x=traj[:, 1], y=traj[:, 2], z=traj[:, 3], mode='lines', line=dict(color=this_color, width=this_width), **kwargs))
                fig.update_layout(scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'),
                    width=800,
                    margin=dict(r=20, b=10, l=10, t=10))
                if show:
                    fig.show()
            elif engine == 'matplotlib':
                import matplotlib.pyplot as plt
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                color = base_color
                for idx, traj in enumerate(trajectories):
                    if color == 'random':
                        this_color = self.randomcolors[idx]
                    else:
                        this_color = color
                    lost = self.lost[idx] if mark_lost else False
                    this_width = 6 if lost else 2
                    this_color = 'rgb(255,0,0)' if lost else this_color
                    ax.plot(traj[:, 1], traj[:, 2], traj[:, 3], color=this_color, linewidth=this_width, **kwargs)
                if show:
                    plt.show()

    def plot_poincare_in_3d(self, engine='mayavi', mark_lost=False, show=True, **kwargs):
        """
        Plot the Poincare points in 3D. Useful to visualize the poincare planes together with the coils and field lines. 
        Args: 
            engine: 'mayavi' or 'plotly' or 'matplotlib'
            mark_lost: if True, mark the field lines that were lost due to stopping criteria in red
            show: if True, show the plot immediately
            **kwargs: additional keyword arguments to pass to the plotting function
        """
        _ = self.res_phi_hits  # trigger recompute if necessary

        if self.is_plotter:
            # unify color kw handling across engines
            if 'color' in kwargs:
                base_color = kwargs.pop('color')
            else:
                base_color = 'random'
            if engine == 'mayavi':
                from mayavi import mlab
                marker = kwargs.pop('marker', 'sphere')
                scale_factor = kwargs.pop('scale_factor', 0.005)
                color = base_color
                for idx in range(len(self.phis)):
                    plane_hits = self.plane_hits_cart(idx)
                    for traj_idx, hit_group in enumerate(plane_hits):
                        if color == 'random':
                            this_color = tuple(self.randomcolors[traj_idx])
                        else: 
                            this_color = tuple(color)
                        
                        this_scale_factor = scale_factor
                        if mark_lost:
                            lost = self.lost[traj_idx]
                            this_scale_factor = scale_factor*3 if lost else scale_factor
                            this_color = (1, 0, 0) if lost else this_color
                        mlab.points3d(hit_group[:, 0], hit_group[:, 1], hit_group[:, 2], scale_factor=this_scale_factor, color=this_color, mode=marker, **kwargs)
                if show:
                    mlab.show()
            if engine == 'plotly':
                import plotly.graph_objects as go
                fig = go.Figure()
                color = base_color
                for idx in range(len(self.phis)):
                    plane_hits = self.plane_hits_cart(idx)
                    for traj_idx, hit_group in enumerate(plane_hits):
                        if color == 'random':
                            this_color = 'rgb({},{},{})'.format(*(self.randomcolors[traj_idx]*255).astype(int))
                        else: 
                            this_color = color
                        lost = self.lost[traj_idx] if mark_lost else False
                        this_size = 8 if lost else 4
                        this_color = 'rgb(255,0,0)' if lost else this_color
                        fig.add_trace(go.Scatter3d(x=hit_group[:, 0], y=hit_group[:, 1], z=hit_group[:, 2], mode='markers', marker=dict(size=this_size, color=this_color), **kwargs))
                fig.update_layout(scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'),
                    width=800,
                    margin=dict(r=20, b=10, l=10, t=10))
                if show:
                    fig.show()
            elif engine == 'matplotlib':
                import matplotlib.pyplot as plt
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                color = base_color
                for idx in range(len(self.phis)):
                    plane_hits = self.plane_hits_cart(idx)
                    for traj_idx, hit_group in enumerate(plane_hits):
                        if color == 'random':
                            this_color = self.randomcolors[traj_idx]
                        else:
                            this_color = color
                        lost = self.lost[traj_idx] if mark_lost else False
                        this_size = 80 if lost else 40
                        this_color = 'rgb(255,0,0)' if lost else this_color
                        ax.scatter(hit_group[:, 0], hit_group[:, 1], hit_group[:, 2], color=this_color, s=this_size, **kwargs)
                if show:
                    plt.show()  
    # TODO: use res_tys to plot rotational transform
