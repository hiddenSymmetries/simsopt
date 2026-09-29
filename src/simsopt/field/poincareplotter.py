"""
Poincaré plots of magnetic fields.

The :class:`PoincarePlotter` uses an :class:`~simsopt.field.integrator.Integrator`
to compute the crossings of field lines with toroidal planes, caches the
results, and provides methods to plot them. Because the plotter depends on the
integrator (which depends on the magnetic field) in the simsopt dependency
graph, the cached results are discarded when the field changes.
"""

import hashlib
import logging
from math import ceil, sqrt
from pathlib import Path

import numpy as np

from .._core import Optimizable
from .integrator import Integrator, SimsoptFieldlineIntegrator, ScipyFieldlineIntegrator

logger = logging.getLogger(__name__)

__all__ = ['PoincarePlotter']

_ENGINES = ('mayavi', 'plotly', 'matplotlib')


class PoincarePlotter(Optimizable):
    r"""
    Compute and plot Poincaré sections of a magnetic field.

    The plotter traces field lines from ``start_points_RZ`` on the plane
    :math:`\phi_0` with an :class:`~simsopt.field.integrator.Integrator`, and
    records where they cross the toroidal planes ``phis``. The computation is
    only performed when results are requested, and is repeated only when the
    magnetic field (or the plotter settings) change.

    **Symmetry planes.** In a field with ``nfp`` field periods, the planes
    :math:`\phi` and :math:`\phi + 2\pi k/n_\text{fp}` show the same cross
    section. With ``add_symmetry_planes=True``, these equivalent planes are
    added to ``phis``, and the crossings on all of them are drawn together
    when the cross section is plotted. This gives ``nfp`` times as many points
    per cross section for the same integration length.

    **Caching.** If ``cache_file`` is given, results are stored in (and read
    from) that ``.npz`` file under a key computed from the field degrees of
    freedom, the plotter settings, and the integrator settings (see
    :attr:`cache_key`). Results computed earlier, even in another session,
    are then reused, and a change of any of these leads to a new computation.

    **MPI.** If the integrator has an MPI communicator, the field lines are
    distributed over the ranks, and only rank 0 plots. The plotting methods
    must be called on all ranks.

    Args:
        integrator (Integrator): the integrator used to trace the field lines.
            Its tolerances and stopping criteria determine the results.
        start_points_RZ (array): (n,2) array of start points (R, Z) on the plane ``phi0``.
        phis (int or array, optional): toroidal angles of the planes in
            :math:`[0, 2\pi)`, or the number of planes equally spaced in
            :math:`[0, 2\pi/n_\text{fp})`. Defaults to the single plane :math:`\phi=0`.
        n_transits (float): number of toroidal transits to trace.
        add_symmetry_planes (bool): whether to add the planes that are
            equivalent to ``phis`` through the field period symmetry.
        cache_file (str or Path, optional): ``.npz`` file used as a cache. None
            (the default) disables the cache.
        phi0 (float, optional): toroidal angle of the start points. Defaults to
            the first of ``phis``.
        nfp (int): number of field periods of the magnetic field.
    """

    def __init__(self, integrator: Integrator, start_points_RZ, phis=None, n_transits=100, add_symmetry_planes=True,
                 cache_file=None, phi0=None, nfp=1):
        """
        Set up the planes and caches. See the class docstring for the arguments.
        """
        self.nfp = nfp
        self._start_points_RZ = np.atleast_2d(np.asarray(start_points_RZ, dtype=float))
        self.integrator = integrator
        self.n_transits = n_transits
        if isinstance(phis, (int, np.integer)):
            self._phis = self.generate_phis(phis, nfp=self.nfp)
        elif phis is None:
            self._phis = np.array([0.0])
        else:
            self._phis = np.atleast_1d(np.asarray(phis, dtype=float))
        if add_symmetry_planes:
            self._phis = self.generate_symmetry_planes(self._phis, nfp=self.nfp)

        self.phi0 = self._phis[0] if phi0 is None else phi0
        self.is_plotter = self.integrator.comm is None or self.integrator.comm.rank == 0  # only rank 0 plots
        self.cache_file = None if cache_file is None else self._npz_path(cache_file)
        self._res_tys = None
        self._res_phi_hits = None
        self._lost = None
        self._randomcolors = None
        self.need_to_recompute = True
        Optimizable.__init__(self, depends_on=[integrator])
        if self.cache_file is not None:
            self.load_cache()

    @classmethod
    def from_field(cls, field, start_points_RZ, phis=None, n_transits=1, add_symmetry_planes=True,
                   stopping_criteria=None, comm=None, integrator_type='simsopt', nfp=1, **kwargs):
        """
        Create a PoincarePlotter directly from a magnetic field, constructing
        the integrator.

        Args:
            field (MagneticField): the magnetic field.
            start_points_RZ (array): (n,2) array of start points (R, Z).
            phis (int or array, optional): planes, as in the constructor. Defaults to 4 planes per field period.
            n_transits (float): number of toroidal transits to trace.
            add_symmetry_planes (bool): whether to add the equivalent planes.
            stopping_criteria (list, optional): StoppingCriterion objects passed to the integrator.
            comm (MPI.Comm, optional): MPI communicator passed to the integrator.
            integrator_type (str): 'simsopt' or 'scipy'.
            nfp (int): number of field periods of the magnetic field.
            **kwargs: additional arguments for the integrator constructor.

        Returns:
            PoincarePlotter: the plotter.
        """
        if phis is None:
            phis = 4
        integrators = {'simsopt': SimsoptFieldlineIntegrator, 'scipy': ScipyFieldlineIntegrator}
        if integrator_type not in integrators:
            raise ValueError(f"Integrator type {integrator_type} not supported, use one of {list(integrators)}.")
        integrator = integrators[integrator_type](field, comm=comm, stopping_criteria=stopping_criteria, **kwargs)
        return cls(integrator, start_points_RZ, phis=phis, n_transits=n_transits,
                   add_symmetry_planes=add_symmetry_planes, nfp=nfp)

    @classmethod
    def from_poincare_data(cls, integrator, start_points_RZ, res_phi_hits, res_tys=None, **kwargs):
        """
        Create a PoincarePlotter from previously computed results, for example
        the output of :meth:`~simsopt.field.integrator.Integrator.compute_poincare_hits`.
        The results must correspond to the given start points and planes. They
        are discarded, and recomputed, when the magnetic field changes.

        Args:
            integrator (Integrator): the integrator, used if results need to be recomputed.
            start_points_RZ (array): (n,2) array of the start points of the results.
            res_phi_hits (list): plane crossings, one array per field line.
            res_tys (list, optional): trajectories, one array per field line.
            **kwargs: other arguments of the constructor, such as ``phis`` and
                ``n_transits``, which must match the results.

        Returns:
            PoincarePlotter: the plotter.
        """
        plotter = cls(integrator, start_points_RZ, **kwargs)
        if len(res_phi_hits) != len(plotter.start_points_RZ):
            raise ValueError(f"res_phi_hits has {len(res_phi_hits)} field lines, "
                             f"but there are {len(plotter.start_points_RZ)} start points.")
        plotter._res_phi_hits = [np.asarray(hits, dtype=float) for hits in res_phi_hits]
        plotter._res_tys = None if res_tys is None else [np.asarray(ty, dtype=float) for ty in res_tys]
        plotter.need_to_recompute = False
        return plotter

    @property
    def randomcolors(self):
        """
        Random but reproducible colors, one per field line.

        Returns:
            array: (n,3) array of RGB values in [0, 1].
        """
        if self._randomcolors is None:
            self._randomcolors = np.random.default_rng(0).random((len(self.start_points_RZ), 3))
        return self._randomcolors

    @staticmethod
    def generate_phis(nplanes, nfp=1):
        """
        Equally spaced toroidal angles in one field period.

        Args:
            nplanes (int): number of planes.
            nfp (int): number of field periods.

        Returns:
            array: ``nplanes`` angles in :math:`[0, 2\\pi/n_\\text{fp})`.
        """
        return np.linspace(0, 2*np.pi/nfp, nplanes, endpoint=False)

    @staticmethod
    def generate_symmetry_planes(phis, nfp=1):
        """
        Add the planes that are equivalent to ``phis`` through the field
        period symmetry, :math:`\\phi + 2\\pi k/n_\\text{fp}` for :math:`k = 0, \\dots, n_\\text{fp}-1`.

        Args:
            phis (array): toroidal angles in :math:`[0, 2\\pi/n_\\text{fp})`.
            nfp (int): number of field periods.

        Returns:
            array: the sorted, unique toroidal angles in :math:`[0, 2\\pi)`.
        """
        return np.unique(np.concatenate([np.asarray(phis) + k*2*np.pi/nfp for k in range(nfp)]))

    @property
    def phis_for_plotting(self):
        """
        The planes in the first field period, one per distinct cross section.

        Returns:
            array: toroidal angles in :math:`[0, 2\\pi/n_\\text{fp})`.
        """
        return self.phis[self.phis < 2*np.pi/self.nfp]

    @property
    def start_points_RZ(self):
        """
        The start points (R, Z) of the field lines on the plane ``phi0``.
        Setting them discards the results.

        Returns:
            array: (n,2) array of start points.
        """
        return self._start_points_RZ

    @start_points_RZ.setter
    def start_points_RZ(self, array):
        """Set the start points and discard the results."""
        array = np.atleast_2d(np.asarray(array, dtype=float))
        if array.shape != self._start_points_RZ.shape:
            self._randomcolors = None
        self._start_points_RZ = array
        self.recompute_bell()

    @property
    def phis(self):
        """
        The toroidal angles of all planes on which crossings are recorded,
        including the symmetry planes. Setting them discards the results.

        Returns:
            array: toroidal angles in :math:`[0, 2\\pi)`.
        """
        return self._phis

    @phis.setter
    def phis(self, value):
        """Set the planes and discard the results."""
        self._phis = np.atleast_1d(np.asarray(value, dtype=float))
        self.recompute_bell()

    def recompute_bell(self, parent=None):
        """
        Discard the results when the field, the integrator, or the plotter
        settings change. Called by the simsopt dependency graph.

        Args:
            parent (Optimizable, optional): the object that changed.
        """
        self._res_phi_hits = None
        self._res_tys = None
        self._lost = None
        self.need_to_recompute = True

    def _compute(self):
        """
        Compute the trajectories and plane crossings with the integrator, and
        store them in the cache file if there is one.
        """
        self._res_tys, self._res_phi_hits = self.integrator.compute_poincare_hits(
            self.start_points_RZ, self.n_transits, phis=self.phis, phi0=self.phi0)
        self._lost = None
        self.need_to_recompute = False
        if self.cache_file is not None:
            self.save_cache()

    def _ensure_results(self):
        """
        Make the results available: load them from the cache file if possible,
        and compute them otherwise. Must be called on all MPI ranks.
        """
        if not self.need_to_recompute:
            return
        if self.cache_file is not None and self.load_cache():
            return
        self._compute()

    @property
    def res_tys(self):
        """
        The trajectories of the field lines, computed if necessary.

        Returns:
            list: one (m,4) array per field line with rows ``[t, x, y, z]``,
            where ``t`` is the integration variable of the integrator.
        """
        self._ensure_results()
        if self._res_tys is None:  # e.g. loaded or given without trajectories
            self._compute()
        return self._res_tys

    @property
    def res_phi_hits(self):
        """
        The crossings of the field lines with the planes, computed if necessary.

        Returns:
            list: one (k,5) array per field line with rows ``[t, idx, x, y, z]``.
            If ``idx>=0``, the plane ``phis[int(idx)]`` was crossed. The last
            row describes why integration stopped, see
            :meth:`~simsopt.field.integrator.Integrator.compute_poincare_hits`.
        """
        self._ensure_results()
        return self._res_phi_hits

    @property
    def cache_key(self):
        """
        Key under which the results are stored in the cache file: a SHA-256
        hash of the magnetic field degrees of freedom, the planes, start
        points, number of transits and ``phi0``, and the type and settings of
        the integrator. Unlike Python's ``hash``, it is the same in every
        session and on every machine.

        Returns:
            str: hexadecimal hash.
        """
        integrator = self.integrator
        settings = [type(integrator).__name__, len(integrator.stopping_criteria)]
        for attribute in ['tol', 'tmax', '_integrator_type', 'trajectory_points_per_transit']:
            settings.append((attribute, getattr(integrator, attribute, None)))
        settings.append(sorted(getattr(integrator, '_integrator_args', {}).items()))
        digest = hashlib.sha256()
        for array in [integrator.field.full_x, self.phis, self.start_points_RZ, [self.n_transits, self.phi0]]:
            digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
        digest.update(repr(settings).encode())
        return digest.hexdigest()

    @staticmethod
    def _npz_path(filename):
        """
        Path of an ``.npz`` file, adding the suffix if necessary.

        Args:
            filename (str or Path): the file name.

        Returns:
            Path: the path, ending in ``.npz``.
        """
        filename = Path(filename)
        return filename if filename.suffix == '.npz' else filename.with_suffix('.npz')

    def _cache_path(self, filename):
        """
        The cache file to use: ``filename`` if given, otherwise ``cache_file``.

        Args:
            filename (str or Path, optional): the file name.

        Returns:
            Path: the path of the cache file.
        """
        if filename is None:
            if self.cache_file is None:
                raise ValueError("No filename given, and the plotter has no cache_file.")
            return self.cache_file
        return self._npz_path(filename)

    def save_cache(self, filename=None, key=None):
        """
        Store the current results (``res_phi_hits`` and ``res_tys``) in a cache
        file, next to results for other keys that are already in it. Only the
        results are stored, not the plotter itself. Only rank 0 writes.

        Args:
            filename (str or Path, optional): the ``.npz`` file. Defaults to ``cache_file``.
            key (str, optional): the key to store the results under. Defaults to :attr:`cache_key`.
        """
        filename = self._cache_path(filename)
        if not self.is_plotter or (self._res_phi_hits is None and self._res_tys is None):
            return
        key = self.cache_key if key is None else str(key)
        data = {}
        if filename.exists():
            with np.load(filename, allow_pickle=True) as existing:
                data = {name: existing[name] for name in existing.files}
        if self._res_phi_hits is not None:
            data[f"res_phi_{key}"] = np.array(self._res_phi_hits + [None], dtype=object)[:-1]
        if self._res_tys is not None:
            data[f"res_tys_{key}"] = np.array(self._res_tys + [None], dtype=object)[:-1]
        np.savez_compressed(filename, **data)

    def load_cache(self, filename=None, key=None):
        """
        Load results from a cache file, if it contains results for the key.

        Args:
            filename (str or Path, optional): the ``.npz`` file. Defaults to ``cache_file``.
            key (str, optional): the key of the results. Defaults to :attr:`cache_key`.

        Returns:
            bool: whether plane crossings were loaded.
        """
        filename = self._cache_path(filename)
        key = self.cache_key if key is None else str(key)
        if not filename.exists():
            logger.debug(f"Cache file {filename} not found.")
            return False
        with np.load(filename, allow_pickle=True) as data:
            if f"res_phi_{key}" not in data.files:
                return False
            self._res_phi_hits = [np.asarray(hits, dtype=float) for hits in data[f"res_phi_{key}"]]
            if f"res_tys_{key}" in data.files:
                self._res_tys = [np.asarray(ty, dtype=float) for ty in data[f"res_tys_{key}"]]
        self._lost = None
        self.need_to_recompute = False
        return True

    def clear_cache(self, filename=None):
        """
        Delete a cache file. Only rank 0 deletes.

        Args:
            filename (str or Path, optional): the ``.npz`` file. Defaults to ``cache_file``.
        """
        if not self.is_plotter:
            return
        filename = self._cache_path(filename)
        if filename.exists():
            filename.unlink()
            logger.info(f"Removed cache file {filename}.")

    def particles_to_vtk(self, filename):
        """
        Export the field line trajectories to a VTK file, for example for Paraview.

        Args:
            filename (str): the file name, without extension.
        """
        from pyevtk.hl import polyLinesToVTK
        trajectories = self.res_tys
        x = np.concatenate([ty[:, 1] for ty in trajectories])
        y = np.concatenate([ty[:, 2] for ty in trajectories])
        z = np.concatenate([ty[:, 3] for ty in trajectories])
        points_per_line = np.asarray([ty.shape[0] for ty in trajectories])
        line_index = np.concatenate([i*np.ones(ty.shape[0]) for i, ty in enumerate(trajectories)])
        polyLinesToVTK(filename, x, y, z, pointsPerLine=points_per_line, pointData={'idx': line_index})

    @property
    def lost(self):
        """
        Whether each field line was stopped before completing ``n_transits``,
        by a stopping criterion or a failure of the integrator.

        Returns:
            list: one bool per field line.
        """
        if self._lost is None:
            # the terminating row has idx=-1 if the transits were completed
            self._lost = [hits[-1, 1] < -1 for hits in self.res_phi_hits]
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
    def _check_engine(engine):
        """
        Check that the 3D plotting engine is supported.

        Args:
            engine (str): 'mayavi', 'plotly' or 'matplotlib'.

        Raises:
            ValueError: if the engine is not supported.
        """
        if engine not in _ENGINES:
            raise ValueError(f"Unknown engine {engine!r}, use one of {list(_ENGINES)}.")

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
            fig, axs: the figure and axes objects (only on rank 0, otherwise None, None)
        """
        _ = self.res_phi_hits  #trigger recompute on all ranks if necessary

        if self.is_plotter:
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
        Plot the 3D trajectories of the field lines. This can be very busy
        if the field lines are followed for many transits. For mayavi, use
        the ``tube_radius`` keyword to adjust the line thickness and
        ``opacity`` to make the lines transparent.
        *NOTE*: if running in parallel, call this function on all ranks.

        Args:
            engine (str): 'mayavi', 'plotly' or 'matplotlib'.
            mark_lost (bool): if True, draw the lost field lines (see :attr:`lost`) thicker and in red.
            show (bool): if True, show the plot immediately.
            **kwargs: additional keyword arguments for the plotting function of the engine.
        """
        self._check_engine(engine)
        trajectories = self.res_tys  # triggers the computation on all ranks if necessary

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
            elif engine == 'plotly':
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
                    this_color = 'r' if lost else this_color
                    ax.plot(traj[:, 1], traj[:, 2], traj[:, 3], color=this_color, linewidth=this_width, **kwargs)
                if show:
                    plt.show()

    def plot_poincare_in_3d(self, engine='mayavi', mark_lost=False, show=True, **kwargs):
        """
        Plot the plane crossings in 3D, for example together with the coils
        and the field line trajectories.
        *NOTE*: if running in parallel, call this function on all ranks.

        Args:
            engine (str): 'mayavi', 'plotly' or 'matplotlib'.
            mark_lost (bool): if True, draw the crossings of lost field lines (see :attr:`lost`) larger and in red.
            show (bool): if True, show the plot immediately.
            **kwargs: additional keyword arguments for the plotting function of the engine.
        """
        self._check_engine(engine)
        _ = self.res_phi_hits  # triggers the computation on all ranks if necessary

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
            elif engine == 'plotly':
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
                        this_color = 'r' if lost else this_color
                        ax.scatter(hit_group[:, 0], hit_group[:, 1], hit_group[:, 2], color=this_color, s=this_size, **kwargs)
                if show:
                    plt.show()
