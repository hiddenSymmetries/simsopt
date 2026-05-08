# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
This module provides a class that handles the DESC equilibrium code.
"""

import logging
import os
import re
import shutil
import tempfile

import numpy as np

logger = logging.getLogger(__name__)

from simsopt.mhd import ProfileSpline, ProfilePolynomial, Vmec
from simsopt._core import Optimizable

# TODO: remove this import after merging the surface conversion.
from constellaration.geometry.surface_utils_desc import (
    to_desc_fourier_rz_toroidal_surface,
    from_desc_fourier_rz_toroidal_surface,
)
from constellaration.geometry.surface_rz_fourier import to_simsopt, from_simsopt

from typing import Protocol, runtime_checkable, Any, Optional

try:
    from desc.equilibrium import Equilibrium as DescEquilibrium
    from desc.profiles import (
        SplineProfile as DescSplineProfile,
        PowerSeriesProfile as DescPowerSeriesProfile,
    )
    from desc.vmec import VMECIO
    from desc.compat import rescale as desc_rescale
    desc_available = True
except ImportError as e:
    desc_available = False
    logger.debug(str(e))

__all__ = ["Desc"]


@runtime_checkable
class DescEquilibriumProtocol(Protocol):
    """A Protocol for desc.equilibrium.Equilibrium objects. This Protocol determines
    the basic set of attributes and methods that a DESC Equilibrium must have in order
    to be used within Simsopt.

    Running,
        ```
        from desc.equilibrium import Equilibrium
        eq = Equilibrium(...)
        isinstance(eq, DescEquilibriumProtocol)
        ```
    will check the DESC Equilibrium object has the attributes and methods defined by the DescEquilibriumProtocol.
    If False, then `eq` does not have the necessary structure to be used within Simsopt.
    This check should be implemented by all methods that rely directly (though not indirectly)
    on the DESC Equilibrium object.
    """

    surface: Any
    pressure: Any
    current: Optional[Any]
    iota: Optional[Any]
    Psi: float

    def solve(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def compute(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def save(self, *args: Any, **kwargs: Any) -> Any:
        pass


class Desc(Optimizable):
    """An Optimizable object for interfacing with the DESC code to solve Ideal MHD equilibria.

    In addition to passing in a Surface object representing the plasma boundary, the edge toroidal
    flux, the pressure profile, and one of the iota or current profile should be prescribed. Both
    rotational transform and current profiles can be passed in. When solve() is called, one of the
    profiles must be chosen to be used.

    The profiles should represent p(rho), I(rho) (or iota(rho)) where rho is Desc's normalized radial
    coordinate; the profiles should not represent derivatives such as p'(rho), I'(rho). rho
    relates to toroidal flux psi as rho = (psi / psi_edge)^2, so it is standard for profiles
    to be even functions of rho, such as p(rho) = p0 + p2 * rho^2 + ...

    The degrees of freedom are psi and any dofs associated to the boundary and profiles.

    Example:
        boundary = SurfaceRZFourier().make_rotating_ellipse(major_radius=1,
            minor_radius=0.1, elongation=1.2)
        # p(rho) = p0 - p2 * rho^2
        pressure_profile = ProfilePolynomial([1e4, 0.0, -1e4])
        # I(rho) = 0
        current_profile = ProfilePolynomial([0.0])
        psi = 5.0
        eq = Desc(boundary,
                            psi=psi,
                            pressure_profile=pressure_profile,
                            current_profile=current_profile)
        eq.solve()


    Args:
        boundary (Surface): Surface object representing the plasma boundary.
        psi (float, optional): Edge toroidal flux [Webers]. Defaults to 1.0.
        pressure_profile (Profile, optional): Pressure profile. Defaults to None.
        iota_profile (Profile, optional):  Rotational transform profile. Defaults to None.
        current_profile (Profile, optional):  Current profile. Defaults to None.
        which_profile (str): One of ["current", "iota"].
            Specify which profile to use as the second profile (current or iota) during
            equilibrium solves. Using "auto" will choose whichever profile is not None,
            and will default to using the current profile if both profiles are not None.
            This attribute can be updated at any time using e.g. `eq.which_profile = "iota"`.
        n_knots (int, optional): Number of knots to use in spline profile interpolation. Defaults to 20.
        degree (int, optional): Degree of spline profile interpolants. Defaults to 3.
        eq (Equilibrium, optional): Optionally pass in a Desc Equilibrium object which will be used
            for evaluation etc. Default to None.
        **kwargs: Keyword arguments passed to DESC Equilibrium() constructor. See the documentation of
            Equilibium object for an up to date list of the keyword arguments.
    """

    def __init__(
        self,
        boundary,
        psi=1.0,
        pressure_profile=None,
        current_profile=None,
        iota_profile=None,
        which_profile="auto",
        n_knots=20,
        degree=3,
        eq=None,
        **kwargs,
    ) -> None:
        
        if not desc_available:
            raise RuntimeError("simsopt requires the desc python package to use Desc.")

        if pressure_profile is None:
            raise ValueError("pressure_profile is a required input.")
        if current_profile is None and iota_profile is None:
            raise ValueError("One of current_profile or iota_profile is required.")

        self._boundary = boundary
        self._pressure_profile = pressure_profile
        self._current_profile = current_profile
        self._iota_profile = iota_profile
        self._which_profile = which_profile
        self.need_to_run_code = True

        self.n_knots = n_knots
        self.degree = degree

        x0 = np.array([psi])
        names = ["psi"]
        depends_on = [boundary, pressure_profile]
        if current_profile is not None:
            depends_on.append(current_profile)
        if iota_profile is not None:
            depends_on.append(iota_profile)
        Optimizable.__init__(
            self,
            x0=x0,
            names=names,
            depends_on=depends_on,
        )

        self._init_desc_equilibrium(eq=eq, **kwargs)

    @property
    def which_profile(self):
        return self._which_profile

    @which_profile.setter
    def which_profile(self, which_profile):
        which_profile_options = ["auto", "current", "iota"]
        if which_profile not in which_profile_options:
            raise ValueError(f"which_profile must be one of {which_profile_options}")
        self._which_profile = which_profile

    @property
    def boundary(self):
        return self._boundary

    @boundary.setter
    def boundary(self, boundary):
        if boundary is not self._boundary:
            logger.debug("Replacing surface in boundary setter")
            self.remove_parent(self._boundary)
            self._boundary = boundary
            self.append_parent(boundary)
            self.need_to_run_code = True

    @property
    def pressure_profile(self):
        return self._pressure_profile

    @pressure_profile.setter
    def pressure_profile(self, pressure_profile):
        if pressure_profile is not self._pressure_profile:
            logger.debug("Replacing pressure_profile in setter")
            if self._pressure_profile is not None:
                self.remove_parent(self._pressure_profile)
            self._pressure_profile = pressure_profile
            if pressure_profile is not None:
                self.append_parent(pressure_profile)
                self.need_to_run_code = True

    @property
    def current_profile(self):
        return self._current_profile

    @current_profile.setter
    def current_profile(self, current_profile):
        if current_profile is not self._current_profile:
            logger.debug("Replacing current_profile in setter")
            if self._current_profile is not None:
                self.remove_parent(self._current_profile)
            self._current_profile = current_profile
            if current_profile is not None:
                self.append_parent(current_profile)
                self.need_to_run_code = True

    @property
    def iota_profile(self):
        return self._iota_profile

    @iota_profile.setter
    def iota_profile(self, iota_profile):
        if iota_profile is not self._iota_profile:
            logger.debug("Replacing iota_profile in setter")
            if self._iota_profile is not None:
                self.remove_parent(self._iota_profile)
            self._iota_profile = iota_profile
            if iota_profile is not None:
                self.append_parent(iota_profile)
                self.need_to_run_code = True

    def recompute_bell(self, parent=None):
        """Flag that the equilibrium needs to be recomputed.

        Args:
            parent: Parent object (optional).
        """
        self.need_to_run_code = True

    @staticmethod
    def surface_to_desc(surface):
        """Convert any Simsopt Surface to a DESC FourierRZToroidalSurface.

        Args:
            surface (Surface): A Simsopt Surface.

        Returns:
            FourierRZToroidalSurface: boundary shape for DESC.
        """
        boundary_rz = surface.to_RZFourier()
        boundary_constellaration = from_simsopt(boundary_rz)
        return to_desc_fourier_rz_toroidal_surface(boundary_constellaration)

    @staticmethod
    def surface_from_desc(eq):
        """Get a SurfaceRZFourier object from a DESC equilibrium.

        Args:
            eq (Equilibrium): a DESC equilibrium object.

        Returns:
            SurfaceRZFourier: boundary shape as a SurfaceRZFourier.
        """
        # TODO: this will create a new optimizable object. Instead, overwrite the
        # TODO: existing dofs
        boundary_constellaration = from_desc_fourier_rz_toroidal_surface(eq.surface)
        boundary = to_simsopt(boundary_constellaration)
        return boundary

    def profiles_to_desc(self):
        """Convert simsopt profiles to DESC Profile objects.

        Returns:
            tuple: (pressure, current, iota) of Desc Profile objects (or None).
        """
        pressure = self._profile_to_desc(self.pressure_profile)

        if self.current_profile is not None:
            current = self._profile_to_desc(self.current_profile)
        else:
            current = None

        if self.iota_profile is not None:
            iota = self._profile_to_desc(self.iota_profile)
        else:
            iota = None
        return pressure, current, iota

    @classmethod
    def profiles_from_desc(cls, eq, n_knots=20, degree=3):
        """
        Convert the profiles of the Equilibrium object to Simsopt profiles.

        Args:
            n_knots (int, optional): number of knots to use in splines. defaults to 20.
            degree (int, optional): degree of splines. defaults to 3.

        Returns:
            tuple: (pressure_profile, current_profile, iota_profile) of Simsopt Profile objects (or None).
        """
        # TODO: this will create a new optimizable object. Instead, overwrite the
        # TODO: existing dofs
        pressure_profile = cls._profile_from_desc(
            eq.pressure, n_knots=n_knots, degree=degree
        )

        if eq.current is not None:
            current_profile = cls._profile_from_desc(
                eq.current, n_knots=n_knots, degree=degree
            )
        else:
            current_profile = None

        if eq.iota is not None:
            iota_profile = cls._profile_from_desc(
                eq.iota, n_knots=n_knots, degree=degree
            )
        else:
            iota_profile = None

        return pressure_profile, current_profile, iota_profile

    @staticmethod
    def _profile_to_desc(prof):
        """Convert a Simsopt ProfileSpline or ProfilePolynomial to a DESC Profile.

        Args:
            prof (Profile): Simsopt Profile object (ProfileSpline or ProfilePolynomial).

        Returns:
            DESC Profile object (DescSplineProfile or DescPowerSeriesProfile).

        Raises:
            ValueError: If profile type is not ProfileSpline or ProfilePolynomial.
        """
        if isinstance(prof, ProfilePolynomial):
            prof_desc = DescPowerSeriesProfile(params=prof.local_full_x)
        elif isinstance(prof, ProfileSpline):
            if prof.degree == 1:
                method = "linear"
            else:
                method = "cubic2"
            # TODO: prof.s is not a supported attribute of ProfileSpline
            rho = prof.s
            prof_desc = DescSplineProfile(
                values=prof.local_full_x, knots=rho, method=method
            )
        else:
            raise ValueError(
                "Profile must be one of ProfileSpline or ProfilePolynomial."
            )
        return prof_desc

    @staticmethod
    def _profile_from_desc(prof_desc, n_knots=20, degree=3):
        """Convert a DESC Profile to a Simsopt Profile.

        If prof_desc is a PowerSeriesProfile, returns a ProfilePolynomial.
        Otherwise, returns a ProfileSpline in rho-space.

        Args:
            prof_desc (DESC Profile): A DESC Profile object.
            n_knots (int, optional): Number of knots in the spline. Defaults to 20.
                Only used if prof_desc is not a PowerSeriesProfile.
            degree (int, optional): Degree of the spline. Defaults to 3.
                Only used if prof_desc is not a PowerSeriesProfile.

        Returns:
            Profile: A ProfilePolynomial or ProfileSpline in rho-space.
        """
        if isinstance(prof_desc, DescPowerSeriesProfile):
            # Extract polynomial coefficients
            params = np.copy(prof_desc.params)
            if prof_desc.sym:
                coeffs = np.zeros(2 * len(params) - 1)
                coeffs[::2] = params
            else:
                coeffs = np.copy(params)
            prof = ProfilePolynomial(coeffs)
        else:
            # Convert to spline in rho-space
            rho_knots = np.linspace(0, 1, n_knots)
            prof_spl = prof_desc.to_spline(knots=rho_knots, method="cubic2")
            prof = ProfileSpline(rho_knots, np.array(prof_spl.params), degree=degree)
        return prof

    def _init_desc_equilibrium(self, eq=None, **kwargs):
        """Create a DESC equilibrium from given information.

        Raises:
            TypeError: Desc Equilibrium must be an instance of DescEquilibriumProtocol.

        Args:
            eq (Equilibrium, optional): Optionally pass in a Desc Equilibrium object which will be used
                for evaluation etc. If None, a new Equilibrium object will be created. Defaults to None.
            **kwargs: Keyword arguments passed to DESC Equilibrium() constructor. See the documentation of
                Equilibium object for an up to date list of the keyword arguments.

        Return:
            None
        """

        boundary_desc = self.surface_to_desc(self.boundary)
        pressure, current, iota = self.profiles_to_desc()

        if eq is None:
            # Determine which profile to use at initialization
            use_current = self._determine_which_profile()
            if use_current:
                iota = None
            else:
                current = None

            eq = DescEquilibrium(
                surface=boundary_desc,
                pressure=pressure,
                current=current,
                iota=iota,
                Psi=self.get("psi"),
                **kwargs,
            )

        # check protocol
        if not isinstance(eq, DescEquilibriumProtocol):
            raise TypeError("DESC Equilibrium does not match DescEquilibriumProtocol.")

        self.eq = eq

        return

    def _determine_which_profile(self):
        """Determine which profile to use: current or iota.

        Returns:
            bool: If True, use current profile. Otherwise use iota profile.
        """

        if self.which_profile == "auto":
            if self.current_profile is not None:
                use_current = True
            else:
                use_current = False
        else:
            use_current = self.which_profile == "current"
        return use_current

    def update_desc_equilibrium(self):
        """
        Update a DESC equilibrium with current (boundary, profiles, psi).

        Return:
            None
        """
        if not self.need_to_run_code:
            # dont update unless (boundary, profiles, psi) have changed
            return

        boundary_desc = self.surface_to_desc(self.boundary)
        pressure, current, iota = self.profiles_to_desc()

        self.eq.surface = boundary_desc
        self.eq.pressure = pressure
        self.eq.Psi = self.get("psi")

        use_current = self._determine_which_profile()
        if use_current:
            self.eq.current = current
            self.eq.iota = None
        else:
            self.eq.current = None
            self.eq.iota = iota

        return

    def rescale(self, **kwargs):
        """Rescale the Desc Equilibium using desc.compat.rescale.
        The Optimizable boundary, profiles, and psi will be scaled accordingly.

        Example:
            desc_eq = Desc(...)
            desc_eq.rescale(L=("a", 1.0), B=("<B>", 1.0), scale_pressure=True)

        Args:
            **kwargs: Keyword arguments passed to desc.compat.rescale. See the documentation of
                desc.compat.rescale for an up to date list of the keyword arguments.

        Return:
            None
        """
        eq = desc_rescale(self.eq, **kwargs)

        # get scaled variables
        psi = eq.Psi
        boundary = self.surface_from_desc(eq)
        pressure_profile, current_profile, iota_profile = self.profiles_from_desc(
            eq, n_knots=self.n_knots, degree=self.degree
        )

        # update dofs
        self.boundary = boundary
        self.pressure_profile = pressure_profile
        self.current_profile = current_profile
        self.iota_profile = iota_profile
        self.set("psi", psi)
        self.eq = eq

        scale_pressure = kwargs.get("scale_pressure", False)
        self.need_to_run_code = not scale_pressure

        return

    def solve(self, *args, **kwargs):
        """Call the Equilibium.solve() method of the Desc Equilibrium
        Any args, kwargs will be passed to the solve() method.

        Args:
            **args: Arguments passed to solve(). See the documentation of
                Equilibium.solve for an up to date list of the keyword arguments.
            **kwargs: Keyword arguments passed to Equilibium.solve(). See the documentation of
                Equilibium.solve() for an up to date list of the keyword arguments.

        Return:
            None
        """
        if not self.need_to_run_code:
            return
        self.update_desc_equilibrium()
        self.eq.solve(*args, **kwargs)
        self.need_to_run_code = False

    def to_vmec(self, vmec_options={}, **kwargs):
        """Convert equilibrium to a Vmec object.
        There are no gaurantees that Vmec will be able to evaluate the equilibrium.

        Args:
            vmec_options (dict, optional): Dictionary of vmec.indata fields to override
                the defaults. Supported keys and their defaults:
                - ``mpol`` (int): Poloidal mode number. Defaults to ``max(boundary.mpol, 10)``.
                - ``ntor`` (int): Toroidal mode number. Defaults to ``max(boundary.ntor, 10)``.
                - ``ntheta`` (int): Poloidal grid points. Defaults to ``2 * mpol + 6``.
                - ``nzeta`` (int): Toroidal grid points. Defaults to ``2 * ntor + 4``.
                - ``delt`` (float): Time step. Defaults to ``0.5``.
                - ``nstep`` (int): Number of steps per call. Defaults to ``200``.
                - ``ns_array`` (list): Radial grid sizes. Defaults to ``[16, 32, 64, 101, 0]``.
                - ``niter_array`` (list): Iteration limits per grid. Defaults to ``[1000, 2000, 3000, 10000, 0]``.
                - ``ftol_array`` (list): Force tolerances per grid. Defaults to ``[1e-16, 1e-16, 1e-16, 1e-13, 0.0]``.
                - ``lfreeb`` (bool): Free-boundary flag. Defaults to ``False``.
                - ``pres_scale`` (float): Pressure scaling factor. Defaults to ``1.0``.
                Any remaining keys are applied directly to ``vmec.indata``.
            kwargs: Keyword arguments passed to the Vmec constructor, such as
                ``mpi`` or ``verbose``. See the Vmec documentation for available options.

        Returns:
            Vmec: Vmec object representing the equilibrium.

        """
        vmec = Vmec(**kwargs)

        vmec.boundary = self.boundary
        boundary_rz = self.boundary.to_RZFourier()

        vmec.indata.phiedge = self.get("psi")
        vmec.indata.lasym = not self.boundary.stellsym
        vmec.indata.nfp = self.boundary.nfp

        vmec.indata.mpol = vmec_options.pop("mpol", max(boundary_rz.mpol, 10))
        vmec.indata.ntor = vmec_options.pop("ntor", max(boundary_rz.ntor, 10))

        # Vmec default
        vmec.indata.ntheta = vmec_options.pop("ntheta", 2 * vmec.indata.mpol + 6)
        vmec.indata.nzeta = vmec_options.pop("nzeta", 2 * vmec.indata.ntor + 4)

        vmec.indata.delt = vmec_options.pop('delt', 0.5)
        vmec.indata.nstep = vmec_options.pop('nstep', 200)
        ns_array = vmec_options.pop("ns_array", [16, 32, 64, 101, 0])
        vmec.indata.ns_array[:len(ns_array)] = ns_array 
        niter_array = vmec_options.pop("niter_array", [1000, 2000, 3000, 10000, 0])
        vmec.indata.niter_array[:len(niter_array)] = niter_array 
        ftol_array = vmec_options.pop("ftol_array", [1.0e-16, 1.0e-16, 1.0e-16, 1.0e-13, 0.0])
        vmec.indata.ftol_array[:len(ftol_array)] = ftol_array

        vmec.indata.lfreeb = vmec_options.pop('lfreeb', False)

        vmec.pressure_profile = self.pressure_profile
        if isinstance(self.pressure_profile, ProfileSpline):
            vmec.indata.pmass_type = "cubic_spline"
        vmec.indata.pres_scale = vmec_options.pop('pres_scale', 1.0)

        use_current = self._determine_which_profile()
        if use_current:
            if isinstance(self.current_profile, ProfileSpline):
                vmec.indata.pcurr_type = "cubic_spline_i"
            vmec.current_profile = self.current_profile
            vmec.indata.ncurr = 1.0
        else:
            vmec.iota_profile = self.iota_profile
            if isinstance(self.iota_profile, ProfileSpline):
                vmec.indata.piota_type = "cubic_spline"
            vmec.indata.ncurr = 0
        
        for k, v in vmec_options.items():
            eval(f"vmec.indata.{k} = {v}")

        return vmec

    @classmethod
    def from_eq(cls, eq, n_knots=20, degree=3):
        """Build a Desc object from a Desc Equilibrium object.
        The profiles are interpolated to spline profiles. The equilibrium object
        will used for evaluation etc.

        Args:
            eq (Equilibrium): a Desc Equilibrium
            n_knots (str): Number of knots to use in interpolating the profile.
            degree (str): Degree of the spline to use for the profiles.

        Returns:
            Desc: Optimizable object representing the equilibrium.
        """
        # check protocol
        if not isinstance(eq, DescEquilibriumProtocol):
            raise TypeError("DESC Equilibrium does not match DescEquilibriumProtocol.")

        boundary = cls.surface_from_desc(eq)

        pressure_profile, current_profile, iota_profile = cls.profiles_from_desc(
            eq, n_knots=n_knots, degree=degree
        )

        psi = eq.Psi

        return cls(
            boundary=boundary,
            psi=psi,
            pressure_profile=pressure_profile,
            current_profile=current_profile,
            iota_profile=iota_profile,
            n_knots=n_knots,
            degree=degree,
            eq=eq,
        )

    @classmethod
    def from_input_file(cls, filename, n_knots=20, degree=3, **kwargs):
        """Build a Desc object from a Desc or Vmec *input* file.
        The profiles are interpolated to spline profiles.

        Example:
            eq = Desc.from_input_file("input.precise_QA")

        Args:
            filename (str): Name of the input file to load.
            n_knots (str): Number of knots to use in interpolating the profile.
            degree (str): Degree of the spline to use for the profiles.
            kwargs: Key-word arguments to pass to the Equilibrium object, such as L or M.

        Return:
            Desc: Optimizable object representing the equilibrium.
        """
        # DESC writes a `<filename>_desc` sidecar file when loading a VMEC input.
        # Copy to a temp dir so we don't need write access to the source directory.
        with open(filename, "r") as f:
            header = f.read(8192)
        is_vmec = bool(re.search(r"&INDATA", header, re.IGNORECASE))

        if is_vmec:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = os.path.join(tmpdir, os.path.basename(filename))
                shutil.copy2(filename, tmp_path)
                eq = DescEquilibrium.from_input_file(tmp_path, **kwargs)
        else:
            eq = DescEquilibrium.from_input_file(filename, **kwargs)

        return cls.from_eq(eq, n_knots=n_knots, degree=degree)

    @classmethod
    def from_file(cls, filename, n_knots=20, degree=3):
        """Build a Desc object from a Desc hdf5 or pickle file,
        using the eq.load(filename) method. The profiles are interpolated
        to spline profiles.

        Example:
            eq = Desc.from_file("saved_equilibrium.hdf5")

        Args:
            filename (str): Name of the file to load.
            n_knots (str): Number of knots to use in interpolating the profile.
            degree (str): Degree of the spline to use for the profiles.

        Return:
            Desc: Optimizable object representing the equilibrium.
        """
        eq = DescEquilibrium.load(filename)
        return cls.from_eq(eq, n_knots=n_knots, degree=degree)

    def to_wout(self, filename):
        """Write a Vmec style wout (Net-CDF) file from the DESC equilibrium.

        Args:
            filename (str): File path of output data.

        Returns:
            None
        """
        self.update_desc_equilibrium()
        VMECIO.save(self.eq, filename)

    def to_vmec_input(self, filename, *args, **kwargs):
        """Write a Vmec input file from the DESC equilibrium.

        Args:
            filename (str): File path.
            *args: Arguments to VMECIO.write_vmec_input.
            **kwargs: Keyword arguments VMECIO.to write_vmec_input.

        Returns:
            None
        """
        self.update_desc_equilibrium()
        VMECIO.write_vmec_input(self.eq, filename, *args, **kwargs)

    def save_equilibrium(self, filename, *args, **kwargs):
        """Save the Desc equilibrum using the Equilibrium.save() method.

        Args:
            filename (str): File path.
            *args: Positional arguments passed to Equilibrium.save().
            **kwargs: Keyword arguments passed to Equilibrium.save().

        Returns:
            None
        """
        self.update_desc_equilibrium()
        self.eq.save(filename, *args, **kwargs)

    def compute(self, *args, **kwargs):
        """Compute requested quantities from the equilibrium using the eq.compute() method.
            Equilibrium will be solved prior to calling compute(), if necessary.

        Args:
            *args: Positional arguments passed to DESC Equilibrium.compute().
            **kwargs: Keyword arguments passed to DESC Equilibrium.compute(). See the
                documentation of Equilibrium.compute() for available options.

        Returns:
            dict: Result of eq.compute(*args, **kwargs).
        """

        self.solve()
        return self.eq.compute(*args, **kwargs)
