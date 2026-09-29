r"""
Object-oriented field line integration.

This module wraps simsopt's field line tracing in classes that hold a
magnetic field together with the settings of the ODE solver to simplify
repeated integrations, and so that the Integrator is an 
:class:`Optimizable`. (i.e. you change a child, like the current in a 
coil, and results can be invalidated)

Two backends are provided:

- :class:`SimsoptFieldlineIntegrator` wraps the C++ routine
  ``simsoptpp.fieldline_tracing`` (the same routine that is used by
  :func:`simsopt.field.tracing.compute_fieldlines`), which solves
  :math:`d\mathbf{x}/dt = \mathbf{B}(\mathbf{x})` in Cartesian coordinates
  and supports simsopt stopping criteria.
- :class:`ScipyFieldlineIntegrator` uses :func:`scipy.integrate.solve_ivp` to
  solve the field line ODE with the toroidal angle :math:`\phi` as the
  independent variable,
  :math:`dR/d\phi = R B_R/B_\phi,\ dZ/d\phi = R B_Z/B_\phi`.
  Because :math:`\phi` is the integration variable, points on a given
  toroidal plane are obtained exactly, which gives crisp Poincaré sections.

Both backends expose the same public methods, which are implemented once in
the :class:`Integrator` base class:

- :meth:`Integrator.compute_poincare_hits`: trace many field lines for a given
  number of toroidal transits and record where they cross a set of toroidal
  planes. This is the data needed for Poincaré plots.
- :meth:`Integrator.integrate_toroidally`: the map that takes a point to where
  its field line ends up after advancing a toroidal angle :math:`\Delta\phi`.
  This is useful for finding the magnetic axis, island O- and X-points, and
  for optimization targets built on these.
- :meth:`Integrator.integrate_fieldlinepoints`: a string of points along a
  single field line, for plotting or for fitting curves and surfaces.

Conventions:

- Field lines are always traced in the direction of increasing :math:`\phi`.
  If :math:`B_\phi<0` at the start point, the field is followed backwards.
  ``delta_phi`` must therefore be non-negative.
- Methods that take a single start point accept either Cartesian coordinates
  ``(x, y, z)`` (``input_coordinates='cartesian'``) or cylindrical coordinates
  ``(R, Z)`` together with the toroidal angle ``phi0``
  (``input_coordinates='cylindrical'``). The output coordinates are chosen
  independently with ``output_coordinates``.
"""

import logging
from math import gcd
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp

import simsoptpp as sopp
from .._core import Optimizable, ObjectiveFailure
from .._core.util import parallel_loop_bounds
from .magneticfield import MagneticField
from .tracing import ToroidalTransitStoppingCriterion

logger = logging.getLogger(__name__)

__all__ = ['Integrator', 'SimsoptFieldlineIntegrator', 'ScipyFieldlineIntegrator']

_COORDINATES = ('cartesian', 'cylindrical')


class Integrator(Optimizable):
    r"""
    Base class for field line integrators.

    The base class implements the public interface (coordinate handling,
    input validation and MPI parallelization) and delegates the actual
    integration to three private methods that each backend implements:

    - ``_integrate_toroidally_cyl(RZ, phi0, delta_phi)``
    - ``_fieldline_rphiz(RZ, phi0, delta_phi, n_points, endpoint)``
    - ``_poincare_single(RZ, phi0, phis, n_transits, return_trajectory)``

    Integrators are Optimizable objects that depend on the magnetic field, so
    that objects built on top of them are notified when the field changes.

    Args:
        field (MagneticField): the magnetic field to integrate.
        comm (MPI.Comm, optional): MPI communicator over which the field lines in
            :meth:`compute_poincare_hits` are distributed.
        stopping_criteria (list, optional): StoppingCriterion objects (from
            :mod:`simsopt.field.tracing`) that stop a field line early. Only used in
            :meth:`compute_poincare_hits`. If ``stopping_criteria[i]`` stops a field
            line, the terminating row of its ``res_phi_hits`` has ``idx=-2-i``.
    """

    def __init__(self, field: MagneticField, comm=None, stopping_criteria=None):
        self.field = field
        self.comm = comm
        self.stopping_criteria = list(stopping_criteria) if stopping_criteria is not None else []
        Optimizable.__init__(self, depends_on=[field])

    @staticmethod
    def _rphiz_to_xyz(array):
        """
        Convert cylindrical coordinates to Cartesian coordinates.

        Args:
            array (array): (3,) or (n,3) array of cylindrical coordinates (R, phi, Z).

        Returns:
            array: (n,3) array of Cartesian coordinates (x, y, z).
        """
        array = np.atleast_2d(array)
        if array.ndim > 2 or array.shape[1] != 3:
            raise ValueError("Input array must be of shape (3,) or (n,3)")
        return np.array([array[:, 0]*np.cos(array[:, 1]), array[:, 0]*np.sin(array[:, 1]), array[:, 2]]).T

    @staticmethod
    def _xyz_to_rphiz(array):
        """
        Convert Cartesian coordinates to cylindrical coordinates.

        Args:
            array (array): (3,) or (n,3) array of Cartesian coordinates (x, y, z).

        Returns:
            array: (n,3) array of cylindrical coordinates (R, phi, Z), with phi in (-pi, pi].
        """
        array = np.atleast_2d(array)
        if array.ndim > 2 or array.shape[1] != 3:
            raise ValueError("Input array must be of shape (3,) or (n,3)")
        return np.array([np.sqrt(array[:, 0]**2 + array[:, 1]**2), np.arctan2(array[:, 1], array[:, 0]), array[:, 2]]).T

    @staticmethod
    def _check_coordinates(input_coordinates, output_coordinates):
        """
        Raise a ValueError if the coordinate specifiers are not 'cartesian' or 'cylindrical'.
        """
        if input_coordinates not in _COORDINATES:
            raise ValueError("input_coordinates must be either 'cartesian' or 'cylindrical'")
        if output_coordinates not in _COORDINATES:
            raise ValueError("output_coordinates must be either 'cartesian' or 'cylindrical'")

    def _parse_start_point(self, start_point, phi0, input_coordinates):
        """
        Convert a start point to cylindrical (R, Z) and the toroidal angle phi0.

        Args:
            start_point (array): (x, y, z) if input_coordinates is 'cartesian', (R, Z) if 'cylindrical'.
            phi0 (float): toroidal angle of the start point. Required for cylindrical
                input, ignored for Cartesian input.
            input_coordinates (str): 'cartesian' or 'cylindrical'.

        Returns:
            tuple: (RZ, phi0), with RZ an array of shape (2,).
        """
        start_point = np.asarray(start_point, dtype=float)
        if input_coordinates == 'cylindrical':
            if phi0 is None:
                raise ValueError("If input_coordinates is 'cylindrical', phi0 must be provided")
            if start_point.shape != (2,):
                raise ValueError("If input_coordinates is 'cylindrical', start_point should be of the form [R, Z]")
            return start_point, float(phi0)
        if start_point.shape != (3,):
            raise ValueError("If input_coordinates is 'cartesian', start_point should be of the form [x, y, z]")
        rphiz = self._xyz_to_rphiz(start_point)[0]
        return rphiz[[0, 2]], rphiz[1]

    @staticmethod
    def _check_delta_phi(delta_phi):
        if delta_phi < 0:
            raise ValueError("delta_phi must be non-negative, field lines are always traced in the direction of increasing phi")

    def integrate_toroidally(self, start_point, delta_phi=2*np.pi, phi0=None,
                             input_coordinates='cartesian', output_coordinates='cartesian'):
        r"""
        Follow the field line through ``start_point`` over a toroidal angle
        ``delta_phi`` and return the end point.

        Integration is always performed in the direction of increasing :math:`\phi`.

        Args:
            start_point (array): (x, y, z) if ``input_coordinates`` is 'cartesian',
                (R, Z) if 'cylindrical'.
            delta_phi (float): toroidal angle to advance, non-negative. May exceed :math:`2\pi`.
            phi0 (float): toroidal angle of the start point. Required if
                ``input_coordinates`` is 'cylindrical', ignored otherwise.
            input_coordinates (str): 'cartesian' or 'cylindrical'.
            output_coordinates (str): 'cartesian' or 'cylindrical'.

        Returns:
            array: the end point, (x, y, z) if ``output_coordinates`` is 'cartesian',
            or (R, Z) on the plane :math:`\phi_0 + \Delta\phi` if 'cylindrical'.
            If integration fails, the entries are NaN.
        """
        self._check_coordinates(input_coordinates, output_coordinates)
        self._check_delta_phi(delta_phi)
        RZ, phi0 = self._parse_start_point(start_point, phi0, input_coordinates)
        if delta_phi == 0:
            RZ_end = RZ
        else:
            RZ_end = self._integrate_toroidally_cyl(RZ, phi0, delta_phi)
        if output_coordinates == 'cylindrical':
            return np.asarray(RZ_end)
        return self._rphiz_to_xyz(np.array([RZ_end[0], phi0 + delta_phi, RZ_end[1]]))[0]

    def integrate_fieldlinepoints(self, start_point, delta_phi=2*np.pi, n_points=None, phi0=None, endpoint=False,
                                  input_coordinates='cartesian', output_coordinates='cartesian'):
        r"""
        Compute points along the single field line through ``start_point``,
        over a toroidal angle ``delta_phi``.

        Unlike :meth:`compute_poincare_hits`, this method traces a single field
        line, accepts the start point in either coordinate system, ignores any
        stopping criteria of the integrator, and returns an array of points
        rather than the raw tracing output.

        Args:
            start_point (array): (x, y, z) if ``input_coordinates`` is 'cartesian',
                (R, Z) if 'cylindrical'.
            delta_phi (float): toroidal angle to trace, non-negative.
            n_points (int, optional): if given, return ``n_points`` points equally
                spaced in :math:`\phi`. If None, return the points at which the
                adaptive solver stepped, whose number depends on the tolerances.
            phi0 (float): toroidal angle of the start point. Required if
                ``input_coordinates`` is 'cylindrical', ignored otherwise.
            endpoint (bool): whether the point at :math:`\phi_0+\Delta\phi` is included.
            input_coordinates (str): 'cartesian' or 'cylindrical'.
            output_coordinates (str): 'cartesian' or 'cylindrical'.

        Returns:
            array: (n,3) array of points, (x, y, z) if ``output_coordinates`` is
            'cartesian', or (R, phi, Z) if 'cylindrical'. The first point is the
            start point, and phi is continuous (not wrapped to :math:`[0, 2\pi)`).

        Raises:
            ObjectiveFailure: if the integration does not reach the requested angle.
        """
        self._check_coordinates(input_coordinates, output_coordinates)
        self._check_delta_phi(delta_phi)
        RZ, phi0 = self._parse_start_point(start_point, phi0, input_coordinates)
        rphiz = self._fieldline_rphiz(RZ, phi0, delta_phi, n_points, endpoint)
        if output_coordinates == 'cylindrical':
            return rphiz
        return self._rphiz_to_xyz(rphiz)

    def compute_poincare_hits(self, start_points_RZ, n_transits, phis=(), phi0=0, return_trajectories=True):
        r"""
        Trace field lines for ``n_transits`` toroidal transits and compute
        where they cross the toroidal planes ``phis``.

        The field lines are distributed over the MPI communicator of the
        integrator, and the results are gathered on all ranks.

        Args:
            start_points_RZ (array): (n,2) array of start points in cylindrical
                coordinates (R, Z), all on the plane :math:`\phi=\phi_0`.
            n_transits (float): number of toroidal transits to trace.
            phis (array): toroidal angles of the planes on which to record
                crossings. If empty, no crossings are recorded.
            phi0 (float): toroidal angle of the start points.
            return_trajectories (bool): whether to return the trajectories.
                Backends for which the trajectories come for free (the Simsopt
                backend) ignore this flag.

        Returns:
            tuple: (res_tys, res_phi_hits)

            - ``res_tys``: list of (m,4) arrays, one per field line, with rows
              ``[t, x, y, z]`` along the trajectory, or None if
              ``return_trajectories`` is False. ``t`` is the integration variable
              of the backend.
            - ``res_phi_hits``: list of (k,5) arrays, one per field line, with rows
              ``[t, idx, x, y, z]``. If ``idx>=0`` the plane ``phis[int(idx)]``
              was crossed. The last row describes how integration terminated:
              ``idx=-1`` if ``n_transits`` were completed, and ``idx=-2-i`` if
              ``stopping_criteria[i]`` stopped the field line. Backend-specific
              reasons to stop are described in the backend classes.
        """
        start_points_RZ = np.atleast_2d(np.asarray(start_points_RZ, dtype=float))
        if start_points_RZ.ndim != 2 or start_points_RZ.shape[1] != 2:
            raise ValueError("start_points_RZ must be of shape (n,2)")
        phis = np.atleast_1d(np.asarray(phis, dtype=float))
        res_tys = []
        res_phi_hits = []
        first, last = parallel_loop_bounds(self.comm, len(start_points_RZ))
        for RZ in start_points_RZ[first:last]:
            logger.info(f'Integrating field line starting at R={RZ[0]}, Z={RZ[1]}, phi={phi0}')
            ty, phi_hits = self._poincare_single(RZ, phi0, phis, n_transits, return_trajectories)
            res_tys.append(ty)
            res_phi_hits.append(phi_hits)
        if self.comm is not None:
            res_tys = [ty for gathered in self.comm.allgather(res_tys) for ty in gathered]
            res_phi_hits = [hits for gathered in self.comm.allgather(res_phi_hits) for hits in gathered]
        if any(ty is None for ty in res_tys):
            res_tys = None
        return res_tys, res_phi_hits

    @staticmethod
    def _fieldline_symmetry(nfp, iota):
        r"""
        Field periods and toroidal transits of a periodic field line on a
        rational surface, for its representation as a
        :class:`~simsopt.geo.CurveXYZFourierSymmetries`.

        For :math:`\iota = n/m`, with :math:`g = \gcd(n, m)`, the field line closes
        after :math:`n_\text{tor} = m/g` toroidal transits, and the island chain
        consists of :math:`g` distinct periodic field lines. If the field period
        symmetry maps these onto each other cyclically, each field line has
        :math:`n_\text{fp}/g` field periods. Using fewer field periods than the
        field line has is always valid, so if :math:`g` does not divide
        :math:`n_\text{fp}`, or if the field periods and transits are not coprime
        (as :class:`~simsopt.geo.CurveXYZFourierSymmetries` requires), the number
        of field periods is reduced until the representation is valid.

        Args:
            nfp (int): number of field periods of the magnetic field.
            iota (tuple, optional): (n, m) with :math:`\iota = n/m`. None for the
                magnetic axis. A negative ``m`` denotes a field line that winds
                counter-clockwise around the axis; this does not change the result.

        Returns:
            tuple: (nfp, ntor) of the periodic field line.
        """
        if iota is None:
            return nfp, 1
        n, m = iota
        if m == 0:
            raise ValueError("iota=(n, m) requires m != 0.")
        g = gcd(n, m)
        ntor = abs(m) // g
        nfp_line = nfp // g if nfp % g == 0 else 1
        while gcd(nfp_line, ntor) != 1:
            nfp_line //= gcd(nfp_line, ntor)
        return nfp_line, ntor

    def periodic_fieldline(self, start_point, order, nfp=1, iota=None, stellsym=True, phi0=None,
                           input_coordinates='cartesian', points_per_dof=10, options=None):
        r"""
        Find the periodic field line near ``start_point``, as a
        :class:`~simsopt.geo.PeriodicFieldLine`. This is the magnetic axis, or,
        if ``iota`` is given, an X- or O-point of an island chain on the
        rational surface :math:`\iota = n/m`.

        The field line closes after :math:`n_\text{tor} = m/\gcd(n, m)` toroidal
        transits (1 for the axis), and has :math:`n_\text{fp}/\gcd(n, m)` field
        periods of its own (see :meth:`_fieldline_symmetry`). For example, each
        of the five field lines of a 5/5 island chain in a five-period field
        closes after one transit and has no field period symmetry, whereas the
        single field line of a 3/4 island chain in a three-period field closes
        after four transits and has three field periods.

        The field line through ``start_point`` is traced over one of its own
        periods, :math:`\Delta\phi = 2\pi n_\text{tor}/n_\text{fp,line}`, and the traced points,
        parametrized by arclength, are fitted with a
        :class:`~simsopt.geo.CurveXYZFourierSymmetries`. This initial guess is
        then refined with :meth:`~simsopt.geo.PeriodicFieldLine.run_code`. The
        start point needs to be close enough to the periodic field line for
        this to converge.

        Args:
            start_point (array): (x, y, z) if ``input_coordinates`` is 'cartesian',
                (R, Z) if 'cylindrical'.
            order (int): number of Fourier modes of the curve. The field line
                closes after :math:`n_\text{tor}` transits, so it needs roughly
                :math:`n_\text{tor}` times the order needed for the magnetic axis.
                Note that PeriodicFieldLine only enforces the field line equation
                at its ``2*order+1`` quadrature points, so ``res['success']`` does
                not guarantee that the order is sufficient.
            nfp (int): number of field periods of the magnetic field.
            iota (tuple, optional): (n, m) with :math:`\iota = n/m` for a field
                line of an island chain. None (the default) for the magnetic axis.
                ``m`` is negative for a field line that winds counter-clockwise
                around the axis.
                If :math:`B_\phi<0`, the curve runs along the field towards
                decreasing :math:`\phi`, and ``curve.ntor`` is negative.
            stellsym (bool): whether the curve is stellarator symmetric. This
                requires the periodic field line to pass through
                :math:`\phi=0, Z=0`.
            phi0 (float): toroidal angle of the start point. Required if
                ``input_coordinates`` is 'cylindrical', ignored otherwise.
            input_coordinates (str): 'cartesian' or 'cylindrical'.
            points_per_dof (int): number of traced points per quadrature point
                of the curve, used for the initial fit.
            options (dict, optional): options passed to
                :class:`~simsopt.geo.PeriodicFieldLine`.

        Returns:
            PeriodicFieldLine: the solved periodic field line. Check
            ``.res['success']`` for convergence.
        """
        from ..geo import CurveXYZFourierSymmetries, PeriodicFieldLine

        nfp, ntor = self._fieldline_symmetry(nfp, iota)
        n_quad = 2*order + 1
        delta_phi = 2*np.pi*ntor/nfp
        xyz = self.integrate_fieldlinepoints(start_point, delta_phi, n_points=points_per_dof*n_quad, phi0=phi0,
                                             endpoint=True, input_coordinates=input_coordinates)
        # PeriodicFieldLine requires the curve to run along B. If B_phi < 0, the
        # traced points are reversed and rotated back by -delta_phi (a symmetry
        # of the field), so that they start at the start point and run towards
        # decreasing phi, and the curve winds with -ntor.
        self.field.set_points(np.ascontiguousarray(xyz[:1]))
        if self.field.B_cyl()[0, 1] < 0:
            rotation = np.array([[np.cos(delta_phi), -np.sin(delta_phi), 0],
                                 [np.sin(delta_phi), np.cos(delta_phi), 0],
                                 [0, 0, 1]])
            xyz = xyz[::-1] @ rotation  # rotates row vectors by -delta_phi
            ntor = -ntor
        # parametrize the traced points by arclength over one period, theta in [0, 1/nfp]
        arclength = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))))
        theta = arclength/arclength[-1]/nfp
        fit = CurveXYZFourierSymmetries(theta, order, nfp, stellsym, ntor=ntor)
        fit.least_squares_fit(np.ascontiguousarray(xyz))

        curve = CurveXYZFourierSymmetries(np.linspace(0, 1/nfp, n_quad, endpoint=False), order, nfp, stellsym,
                                          ntor=ntor, x0=fit.x)
        fieldline = PeriodicFieldLine(self.field, curve, options=options)
        fieldline.run_code(arclength[-1]*nfp)
        if not fieldline.res['success']:
            logger.warning("PeriodicFieldLine did not converge; check .res for details.")
        return fieldline

    def _integrate_toroidally_cyl(self, RZ, phi0, delta_phi):
        """
        Backend hook: return the (R, Z) end point after advancing ``delta_phi``
        from (R, Z) at ``phi0``, or NaNs if integration fails.
        """
        raise NotImplementedError

    def _fieldline_rphiz(self, RZ, phi0, delta_phi, n_points, endpoint):
        """
        Backend hook: return an (n,3) array of (R, phi, Z) points along the
        field line, see :meth:`integrate_fieldlinepoints`.
        """
        raise NotImplementedError

    def _poincare_single(self, RZ, phi0, phis, n_transits, return_trajectory):
        """
        Backend hook: trace a single field line and return ``(ty, phi_hits)``
        in the format of :meth:`compute_poincare_hits`.
        """
        raise NotImplementedError


class SimsoptFieldlineIntegrator(Integrator):
    r"""
    Field line integration using the ``simsoptpp`` routines.

    Integration is performed in three dimensions, solving the ODE

    .. math::
        \frac{d\mathbf{x}(t)}{dt} = \pm\frac{\mathbf{B}(\mathbf{x})}{|\mathbf{B}(\mathbf{x}_0)|}

    where :math:`\mathbf{x}=(x,y,z)` are the Cartesian coordinates and
    :math:`\mathbf{x}_0` is the start point. The sign is chosen such that the
    field line is traced in the direction of increasing :math:`\phi`, and the
    normalization makes :math:`t` approximately the arc length along the field line.

    In :meth:`compute_poincare_hits`, if the integration time ``tmax`` is
    exhausted before ``n_transits`` are completed, ``res_phi_hits`` has no
    terminating row.

    Args:
        field (MagneticField): the magnetic field to integrate
            (BoozerMagneticField is not supported).
        comm (MPI.Comm, optional): MPI communicator to parallelize over.
        stopping_criteria (list, optional): list of StoppingCriterion objects.
            Only used in :meth:`compute_poincare_hits`.
        tol (float): tolerance of the adaptive ODE solver.
        tmax (float): maximum integration time, roughly the maximum field line
            length in meters.
    """

    def __init__(self, field: MagneticField, comm=None, stopping_criteria=None, tol=1e-9, tmax=1e4):
        self.tol = tol
        self.tmax = tmax
        super().__init__(field, comm=comm, stopping_criteria=stopping_criteria)

    def _field_for_tracing(self, start_xyz):
        """
        Return the field scaled such that it points in the direction of
        increasing phi at ``start_xyz`` and has unit strength there.

        Args:
            start_xyz (array): (3,) array with the start point.

        Returns:
            MagneticField: the scaled field.
        """
        self.field.set_points(np.atleast_2d(start_xyz))
        Bstart = self.field.B_cyl()[0]
        sign = -1.0 if Bstart[1] < 0 else 1.0
        return (sign / np.linalg.norm(Bstart)) * self.field

    def _trace(self, RZ, phi0, phis, stopping_criteria):
        """
        Trace a single field line with ``sopp.fieldline_tracing``.

        Args:
            RZ (array): (R, Z) start point.
            phi0 (float): toroidal angle of the start point.
            phis (array): angles of the planes to record crossings on.
            stopping_criteria (list): StoppingCriterion objects.

        Returns:
            tuple: (tys, phi_hits) as numpy arrays.
        """
        start_xyz = self._rphiz_to_xyz(np.array([RZ[0], phi0, RZ[1]]))[0]
        tys, phi_hits = sopp.fieldline_tracing(
            self._field_for_tracing(start_xyz), start_xyz, tmax=self.tmax, tol=self.tol,
            phis=list(phis), stopping_criteria=stopping_criteria)
        return np.array(tys).reshape(-1, 4), np.array(phi_hits).reshape(-1, 5)

    @staticmethod
    def _unwrapped_phi(tys, phi0):
        """
        Continuous toroidal angle along a trajectory starting at ``phi0``.
        The C++ solver takes at most a quarter revolution per step, so
        unwrapping is unambiguous.
        """
        phi = np.unwrap(np.arctan2(tys[:, 2], tys[:, 1]))
        return phi + (phi0 - phi[0])

    def _plane_hits_unwrapped(self, tys, phi_hits, phi0, planes):
        """
        Return the continuous toroidal angle of every plane crossing in
        ``phi_hits``, obtained by lifting the plane angle to the branch closest
        to the trajectory angle at the time of the crossing.
        """
        phi_traj = self._unwrapped_phi(tys, phi0)
        mask = phi_hits[:, 1] >= 0
        hits = phi_hits[mask]
        plane_phi = planes[hits[:, 1].astype(int)]
        phi_approx = np.interp(hits[:, 0], tys[:, 0], phi_traj)
        lifted = plane_phi + 2*np.pi*np.round((phi_approx - plane_phi)/(2*np.pi))
        return hits, lifted

    def _integrate_toroidally_cyl(self, RZ, phi0, delta_phi):
        phi_end = phi0 + delta_phi
        # a small margin ensures the step that crosses phi_end is completed
        criteria = [ToroidalTransitStoppingCriterion(delta_phi/(2*np.pi) + 0.01, False)]
        tys, phi_hits = self._trace(RZ, phi0, [phi_end], criteria)
        hits, lifted = self._plane_hits_unwrapped(tys, phi_hits, phi0, np.array([phi_end]))
        match = np.where(np.abs(lifted - phi_end) < np.pi)[0]
        if len(match) == 0:
            return np.array([np.nan, np.nan])
        rphiz = self._xyz_to_rphiz(hits[match[0], 2:])[0]
        return rphiz[[0, 2]]

    def _fieldline_rphiz(self, RZ, phi0, delta_phi, n_points, endpoint):
        phi_end = phi0 + delta_phi
        criteria = [ToroidalTransitStoppingCriterion(delta_phi/(2*np.pi) + 0.01, False)]
        if n_points is None:
            tys, phi_hits = self._trace(RZ, phi0, [phi_end], criteria)
            phi_traj = self._unwrapped_phi(tys, phi0)
            keep = phi_traj < phi_end
            rphiz = np.column_stack((np.sqrt(tys[keep, 1]**2 + tys[keep, 2]**2), phi_traj[keep], tys[keep, 3]))
            if endpoint:
                end_RZ = self._integrate_toroidally_cyl(RZ, phi0, delta_phi)
                if np.any(np.isnan(end_RZ)):
                    raise ObjectiveFailure("Integration failed")
                rphiz = np.vstack((rphiz, [end_RZ[0], phi_end, end_RZ[1]]))
            return rphiz
        targets = np.linspace(phi0, phi_end, n_points, endpoint=endpoint)
        planes = targets[1:]
        tys, phi_hits = self._trace(RZ, phi0, planes, criteria)
        hits, lifted = self._plane_hits_unwrapped(tys, phi_hits, phi0, planes)
        # each plane is crossed once per transit; keep the crossing at the target angle
        idx = hits[:, 1].astype(int)
        keep = np.abs(lifted - planes[idx]) < np.pi
        hits, idx = hits[keep], idx[keep]
        order = np.argsort(idx)
        hits, idx = hits[order], idx[order]
        if len(np.unique(idx)) != len(planes):
            raise ObjectiveFailure("Integration failed")
        rphiz = self._xyz_to_rphiz(hits[:, 2:])
        rphiz[:, 1] = planes
        return np.vstack(([RZ[0], phi0, RZ[1]], rphiz))

    def _poincare_single(self, RZ, phi0, phis, n_transits, return_trajectory):
        # the transit criterion comes first, so that it has idx=-1
        criteria = [ToroidalTransitStoppingCriterion(n_transits, False)] + self.stopping_criteria
        return self._trace(RZ, phi0, phis, criteria)


class _CriterionEvent:
    """
    Wrap a StoppingCriterion as a terminal event for ``solve_ivp``: the event
    function is -1 where the criterion is satisfied and +1 elsewhere.

    ``solve_ivp`` evaluates events at the start point, after every step, and
    at intermediate points while locating an event. The criterion is not
    evaluated at the start point (as in the C++ tracing routines), and
    ``iter`` counts the steps, which are recognized by a new maximum of phi.
    Points inside the last step are evaluated with the previous ``iter``, so
    that criteria that depend on ``iter`` change sign across the step.
    """
    terminal = True

    def __init__(self, criterion):
        self.criterion = criterion
        self.iter = 0
        self.phi_start = None
        self.phi_max = None

    def __call__(self, phi, rz):
        if self.phi_start is None:
            self.phi_start = self.phi_max = phi
        if phi <= self.phi_start:
            return 1.0
        if phi > self.phi_max:
            self.phi_max = phi
            self.iter += 1
        # points before the end of the last step (evaluated while locating the
        # event) belong to the previous iteration
        it = self.iter if phi >= self.phi_max else self.iter - 1
        R, Z = rz
        return -1.0 if self.criterion(it, phi, R*np.cos(phi), R*np.sin(phi), Z) else 1.0


class ScipyFieldlineIntegrator(Integrator):
    r"""
    Field line integration using :func:`scipy.integrate.solve_ivp`.

    The toroidal angle :math:`\phi` is used as the independent variable, and the ODE

    .. math::
        \frac{dR}{d\phi} = R \frac{B_R}{B_\phi}, \qquad
        \frac{dZ}{d\phi} = R \frac{B_Z}{B_\phi}

    is solved, where :math:`(R,\phi,Z)` are the cylindrical coordinates and
    :math:`(B_R, B_\phi, B_Z)` the cylindrical components of the magnetic field.
    Because :math:`\phi` is the integration variable, points on toroidal
    planes are evaluated exactly.

    This ODE is singular where :math:`B_\phi=0`. Integration is therefore
    stopped when :math:`|B_\phi|/|B|` drops below ``1e-3``, which can happen
    for field lines that approach the coils. In :meth:`compute_poincare_hits`,
    the terminating row of ``res_phi_hits`` then has
    ``idx=-2-len(stopping_criteria)``, as it does when the solver fails.

    Stopping criteria are evaluated as terminal ``solve_ivp`` events, so the
    point where a field line is stopped is located by root finding between
    solver steps. They are called with the step number as ``iter`` and
    :math:`\phi` as ``t``.

    Three dimensional integration, which does not have this limitation, is
    available with :meth:`integrate_3d_fieldlinepoints`.

    Args:
        field (MagneticField): the magnetic field to integrate.
        comm (MPI.Comm, optional): MPI communicator to parallelize over.
        stopping_criteria (list, optional): list of StoppingCriterion objects.
            Only used in :meth:`compute_poincare_hits`.
        integrator_type (str): the ``method`` passed to ``solve_ivp``, for example 'RK45' or 'DOP853'.
        integrator_args (dict, optional): additional keyword arguments for ``solve_ivp``,
            for example ``{'rtol': 1e-9, 'atol': 1e-11}``. The defaults are
            ``rtol=1e-7`` and ``atol=1e-9``.
        trajectory_points_per_transit (int): number of points per toroidal
            transit in the trajectories returned by :meth:`compute_poincare_hits`.
    """

    _bphi_threshold = 1e-3

    def __init__(self, field: MagneticField, comm=None, stopping_criteria=None, integrator_type='RK45',
                 integrator_args=None, trajectory_points_per_transit=100):
        super().__init__(field, comm=comm, stopping_criteria=stopping_criteria)
        self._integrator_type = integrator_type
        self._integrator_args = dict(integrator_args) if integrator_args is not None else {}
        self._integrator_args.setdefault('rtol', 1e-7)
        self._integrator_args.setdefault('atol', 1e-9)
        self.trajectory_points_per_transit = trajectory_points_per_transit

    def _integration_fn_cyl(self, phi, rz):
        r"""
        Right hand side of the field line ODE in cylindrical coordinates,
        :math:`(R B_R/B_\phi, R B_Z/B_\phi)`.

        Args:
            phi (float): toroidal angle, the independent variable.
            rz (array): (R, Z).

        Returns:
            array: (dR/dphi, dZ/dphi).
        """
        if np.any(np.isnan(rz)):
            return np.full_like(rz, np.nan)
        R, Z = rz
        self.field.set_points_cyl(np.array([[R, phi, Z]]))
        B = self.field.B_cyl().flatten()
        return np.array([R*B[0]/B[1], R*B[2]/B[1]])

    def _bphi_event(self, phi, rz):
        r"""
        Event function for ``solve_ivp`` that crosses zero when :math:`|B_\phi|/|B|`
        drops below the threshold, which terminates integration.
        """
        R, Z = rz
        self.field.set_points_cyl(np.array([[R, phi, Z]]))
        B = self.field.B_cyl().flatten()
        return np.abs(B[1]) / np.linalg.norm(B) - self._bphi_threshold
    _bphi_event.terminal = True

    def _solve(self, RZ, phi_span, stopping_criteria=(), **kwargs):
        """
        Call ``solve_ivp`` on the cylindrical field line ODE. The B_phi event is
        the first event, followed by one event per stopping criterion.

        If the field line cannot be started (the right hand side is not finite,
        or :math:`B_\\phi` is already below the threshold), ``solve_ivp`` is not
        called, since its initial step selection does not terminate on
        non-finite input. A result with only the start point is returned instead,
        with status -1 or 1 respectively.
        """
        RZ = np.asarray(RZ, dtype=float)
        status = None
        if not np.all(np.isfinite(self._integration_fn_cyl(phi_span[0], RZ))):
            status, message = -1, "Right hand side is not finite at the start point."
        elif self._bphi_event(phi_span[0], RZ) < 0:
            status, message = 1, "B_phi is below the threshold at the start point."
        if status is not None:
            t_events = [np.array([phi_span[0]]) if status == 1 else np.array([])]
            t_events += [np.array([]) for _ in stopping_criteria]
            return SimpleNamespace(t=np.array([phi_span[0]]), y=RZ[:, None], sol=None, t_events=t_events,
                                   status=status, message=message, success=False)
        events = [self._bphi_event] + [_CriterionEvent(c) for c in stopping_criteria]
        return solve_ivp(self._integration_fn_cyl, phi_span, RZ, events=events,
                         method=self._integrator_type, **self._integrator_args, **kwargs)

    def _stop_idx(self, sol):
        """
        The ``idx`` of the terminating row of ``res_phi_hits`` for a solution
        of :meth:`_solve` with the integrator's stopping criteria.
        """
        if sol.status == 0:
            return -1
        if sol.status == 1:
            triggered = [len(t) > 0 for t in sol.t_events[1:]]
            if any(triggered):
                return -2 - triggered.index(True)
        return -2 - len(self.stopping_criteria)  # B_phi event or solver failure

    def _integrate_toroidally_cyl(self, RZ, phi0, delta_phi):
        sol = self._solve(RZ, [phi0, phi0 + delta_phi])
        if sol.status != 0:
            return np.array([np.nan, np.nan])
        return sol.y[:, -1]

    def _fieldline_rphiz(self, RZ, phi0, delta_phi, n_points, endpoint):
        phi_end = phi0 + delta_phi
        if n_points is None:
            sol = self._solve(RZ, [phi0, phi_end])
            phis, rz = sol.t, sol.y
            if not endpoint:
                phis, rz = phis[:-1], rz[:, :-1]
        else:
            phis = np.linspace(phi0, phi_end, n_points, endpoint=endpoint)
            sol = self._solve(RZ, [phi0, phi_end], t_eval=phis)
            rz = sol.y
        if sol.status != 0:
            raise ObjectiveFailure("Integration failed")
        return np.column_stack((rz[0], phis, rz[1]))

    def _poincare_single(self, RZ, phi0, phis, n_transits, return_trajectory):
        phi_end = phi0 + 2*np.pi*n_transits
        # angles at which the planes are crossed, excluding the start point
        offsets = np.mod(phis - phi0, 2*np.pi)
        offsets[np.isclose(offsets, 0)] = 2*np.pi
        n_laps = int(np.ceil(n_transits))
        hit_phis = (phi0 + offsets[None, :] + 2*np.pi*np.arange(n_laps)[:, None]).ravel()
        hit_idx = np.tile(np.arange(len(phis)), n_laps)
        order = np.argsort(hit_phis, kind='stable')
        hit_phis, hit_idx = hit_phis[order], hit_idx[order]
        in_range = hit_phis <= phi_end
        hit_phis, hit_idx = hit_phis[in_range], hit_idx[in_range]

        sol = self._solve(RZ, [phi0, phi_end], stopping_criteria=self.stopping_criteria, dense_output=True)
        phi_stop = sol.t[-1]
        rz_stop = sol.y[:, -1]
        interpolate = sol.sol if (sol.sol is not None and len(sol.t) > 1) else None

        reached = hit_phis <= phi_stop if interpolate is not None else np.zeros(len(hit_phis), dtype=bool)
        hit_phis, hit_idx = hit_phis[reached], hit_idx[reached]
        rz_hits = interpolate(hit_phis) if len(hit_phis) > 0 else np.zeros((2, 0))
        xyz_hits = self._rphiz_to_xyz(np.column_stack((rz_hits[0], hit_phis, rz_hits[1])))
        stop_idx = self._stop_idx(sol)
        stop_xyz = self._rphiz_to_xyz(np.array([rz_stop[0], phi_stop, rz_stop[1]]))
        phi_hits = np.vstack((np.column_stack((hit_phis, hit_idx, xyz_hits)),
                              np.column_stack(([phi_stop], [stop_idx], stop_xyz))))

        if not return_trajectory:
            return None, phi_hits
        n_traj = int(np.ceil(n_transits*self.trajectory_points_per_transit)) + 1
        traj_phis = np.linspace(phi0, phi_end, n_traj)
        traj_phis = traj_phis[traj_phis <= phi_stop] if interpolate is not None else traj_phis[:1]
        rz_traj = interpolate(traj_phis) if interpolate is not None else RZ[:, None]
        xyz_traj = self._rphiz_to_xyz(np.column_stack((rz_traj[0], traj_phis, rz_traj[1])))
        return np.column_stack((traj_phis, xyz_traj)), phi_hits

    def _integration_fn_3d(self, t, xyz):
        """
        Right hand side of the arc length parametrized field line ODE,
        :math:`\\mathbf{B}/|\\mathbf{B}|`.

        Args:
            t (float): arc length, the independent variable (unused).
            xyz (array): Cartesian coordinates (x, y, z).

        Returns:
            array: the unit vector along the field.
        """
        self.field.set_points(xyz[None, :])
        B = self.field.B().flatten()
        return B/np.linalg.norm(B)

    def integrate_3d_fieldlinepoints(self, start_point, l_total, n_points, phi0=None,
                                     input_coordinates='cartesian', output_coordinates='cartesian'):
        r"""
        Integrate a field line in three dimensions over a given length.
        This method is specific to the Scipy backend. It solves

        .. math::
            \frac{d\boldsymbol{\gamma}(\tau)}{d\tau} = \frac{\mathbf{B}(\boldsymbol{\gamma}(\tau))}{|\mathbf{B}(\boldsymbol{\gamma}(\tau))|}

        where :math:`\tau` is the arc length along the field line and
        :math:`\boldsymbol{\gamma}(\tau)` its position. Unlike the other
        methods, this follows the field direction (not increasing :math:`\phi`),
        and works for field lines where :math:`B_\phi` changes sign, such as
        field lines that are caught by the coils.

        Args:
            start_point (array): (x, y, z) if ``input_coordinates`` is 'cartesian',
                (R, Z) if 'cylindrical'.
            l_total (float): length of the field line to integrate. A full
                transit is roughly :math:`2\pi R_0`, with :math:`R_0` the major radius.
            n_points (int): number of points to return, equally spaced in arc length.
            phi0 (float): toroidal angle of the start point. Required if
                ``input_coordinates`` is 'cylindrical', ignored otherwise.
            input_coordinates (str): 'cartesian' or 'cylindrical'.
            output_coordinates (str): 'cartesian' or 'cylindrical'.

        Returns:
            array: (n_points,3) array of points, (x, y, z) or (R, phi, Z).
        """
        self._check_coordinates(input_coordinates, output_coordinates)
        RZ, phi0 = self._parse_start_point(start_point, phi0, input_coordinates)
        start_xyz = self._rphiz_to_xyz(np.array([RZ[0], phi0, RZ[1]]))[0]
        sol = solve_ivp(self._integration_fn_3d, [0, l_total], start_xyz, t_eval=np.linspace(0, l_total, n_points),
                        method=self._integrator_type, **self._integrator_args)
        if output_coordinates == 'cartesian':
            return sol.y.T
        return self._xyz_to_rphiz(sol.y.T)
