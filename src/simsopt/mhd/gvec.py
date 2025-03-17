# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
This module provides a class that provides an optimizable for the GVEC equilibrium code.

The Gvec optimizable class is designed to be similar to the Vmec optimizable, but the interfaces are not identical.
"""

import copy
import logging
from pathlib import Path
from typing import Optional, Mapping
import shutil

import numpy as np

logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI
    from ..util.mpi import MpiPartition
except ImportError as e:
    MPI = None
    MpiPartition = None
    logger.debug(str(e))

try:
    import gvec
    from gvec import State
    from gvec.scripts.run import run_stages
except ImportError as e:
    gvec = None
    State = None
    logger.debug(str(e))

from .._core.optimizable import Optimizable
from .._core.util import ObjectiveFailure
from ..geo import Surface, SurfaceRZFourier
from simsopt.mhd.profiles import Profile, ProfilePolynomial, ProfileScaled

__all__ = ["Gvec"]


class Gvec(Optimizable):
    """
    This class represents the optimizable for the GVEC equilibrium code.

    Args:
        base_parameter_file: The base parameter file to use for the GVEC run. This file contains the common/default parameters and is adapted for each run.
        restart_file: (optional) A GVEC Statefile to restart each run from.
        delete_intermediates: Whether to delete intermediate files after the run.
        mpi: The MPI partition to use for running GVEC. Note that GVEC is parallelized using OpenMP and MPI parallelization is not available with SIMSOPT.

    ToDo: doc
    """

    def __init__(
        self,
        base_parameters: Mapping,
        restart_file: str | Path | None = None,
        delete_intermediates: bool = False,
        mpi: Optional[MpiPartition] = None,
    ):
        self.base_parameters = base_parameters # nested parameters
        self.restart_file = Path(restart_file) if restart_file else None
        self.delete_intermediates = delete_intermediates
        self.mpi = mpi

        if self.restart_file and not self.restart_file.exists():
            raise FileNotFoundError(f"Restart file {self.restart_file} not found")

        if MPI:
            if self.mpi is None:
                self.mpi = MpiPartition()
            self._mpi_id = f"_{self.mpi.rank_world:03d}"
        else:
            if self.mpi:
                logger.warning("MPI not available, ignoring MpiPartition")
            self._mpi_id = ""

        self.run_count: int = 0  # number of times GVEC has been run
        self.run_required: bool = True  # flag for caching e.g. set after changing parameters
        self.run_successful: bool = False  # flag for whether the last run was successful
        self._state: State = None  # state object representing the GVEC equilibrium

        # set boundary object from the parameters
        if "nfp" not in self.base_parameters:
            raise ValueError("nfp not found in base parameter file")
        if self.base_parameters["X1_sin_cos"] != "_cos_" or self.base_parameters["X2_sin_cos"] != "_sin_":
            raise NotImplementedError("Not stellarator symmetric configurations not supported.")
        if self.base_parameters["X1_mn_max"] != self.base_parameters["X2_mn_max"]:
            raise ValueError("X1_mn_max and X2_mn_max must be equal")
        
        M, N = self.base_parameters["X1_mn_max"]
        self._boundary = SurfaceRZFourier(
            nfp=self.base_parameters["nfp"],
            stellsym=True,
            mpol=M,
            ntor=N,
        )

        for (m, n), value in self.base_parameters.get("X1_b_cos", {}).items():
            self._boundary.set_rc(m, n, value)
        for (m, n), value in self.base_parameters.get("X2_b_sin", {}).items():
            if m == 0:
                self._boundary.set_zs(m, n, -value)
            else:
                self._boundary.set_zs(m, -n, value)

        self._boundary.local_full_x = self._boundary.get_dofs()

        # set profiles
        if self.base_parameters["pres"].get("type", "polynomial") == "polynomial":
            self._pressure = ProfileScaled(
                ProfilePolynomial(self.base_parameters["pres"]["coefs"]), self.base_parameters["pres"]["scale"]
            )
        else:
            raise NotImplementedError(f"Pressure profile {self.base_parameters['pres']['type']} not supported.")
        if self.base_parameters["iota"].get("type", "polynomial") == "polynomial":
            self._iota = ProfilePolynomial(self.base_parameters["iota"]["coefs"])
        else:
            raise NotImplementedError(f"Rotational transform profile {self.base_parameters['iota']['type']} not supported.")
        self._current = None  # ToDo: implement current profile

        self._phiedge = self.base_parameters["phiedge"]

        # init Optimizable
        x0 = self.get_dofs()
        fixed = np.full(len(x0), True)
        names = ['phiedge']
        depends = [self._boundary]
        super().__init__(
            x0=x0, 
            fixed=fixed, 
            names=names,
            depends_on=depends,
            external_dof_setter=self.__class__.set_dofs,
        )
    
    @classmethod
    def from_parameter_file(
        cls, 
        base_parameter_file: str | Path,
        restart_file: str | Path | None = None,
        delete_intermediates: bool = False,
        mpi: Optional[MpiPartition] = None,
    ):
        parameters = gvec.util.read_parameters(base_parameter_file)
        return cls(parameters, restart_file, delete_intermediates, mpi)

    def recompute_bell(self, parent=None):
        """Set the recomputation flag"""
        logger.debug(f"recompute_bell called by {parent}: run_required = {self.run_required}")
        self.run_required = True

    def get_dofs(self):
        """Return the DOFs owned by the GVEC optimizable.
        
        This does not include any DoFs owned by the boundary or profile objects.
        """
        return np.array([self._phiedge])

    def set_dofs(self, x):
        """Set the DoFs owned by the GVEC optimizable.
        
        This does not include any DoFs owned by the boundary or profile objects.
        """
        if len(x) != 1:
            raise ValueError(f"Expected 1 DOF, got {len(x)}")
        self.phiedge = x[0]
    
    def run(self, force: bool = False) -> None:
        """
        Run GVEC (via subprocess) if `run_required` or `force` is True.
        """
        if not self.run_required and not force:
            if not force:
                logger.debug("run() called but no need to re-run GVEC")
                return
            else:
                logger.debug("run() forced")

        if self._state is not None:
            self._state.finalize()

        if self.delete_intermediates and self.run_successful:
            logger.debug("Deleting output from previous run")
            shutil.rmtree(self.rundir)
        
        self.run_count += 1
        logger.debug(f"Preparing to run GVEC run number {self.run_count}")

        # create run directory
        self.rundir = Path(f"gvec{self._mpi_id}-{self.run_count:03d}")
        if self.rundir.exists():
            logger.warning(f"Run directory {self.rundir} already exists, replacing")
            shutil.rmtree(self.rundir)
        self.rundir.mkdir()

        # prepare parameter file
        params = self.prepare_parameters()

        # configure restart
        if self.restart_file:
            # ToDo: needs test
            logger.info(f"Running GVEC with restart from {self.restart_file}")
            if self.restart_file.is_absolute():
                restart_file = self.restart_file
            else:
                restart_file = ".." / self.restart_file
        else:
            logger.info("Running GVEC")
            restart_file = None
        
        # run GVEC
        self.run_successful = False
        try:
            with gvec.util.chdir(self.rundir):
                stagedir, statefile, diagnostics = run_stages(params, restart_file)
        except RuntimeError as e:
            logger.error(f"GVEC failed with: {e}")
            raise ObjectiveFailure("Run GVEC failed.") from e
        self.run_successful = True
        
        # read output/statefile
        logger.debug(f"Reading state: {self.rundir / stagedir / 'parameter.ini'}, {self.rundir / statefile}")
        self._state = State(self.rundir / stagedir / "parameter.ini", self.rundir / statefile)

        logger.debug("GVEC finished")
        self.run_required = False

    def prepare_parameters(self) -> Mapping:
        """
        Prepare DOFs as parameters for the GVEC input file as keyword arguments which can be passed to `run_stages`.
        """
        params = copy.deepcopy(self.base_parameters)

        if not isinstance(self._boundary, SurfaceRZFourier):
            raise NotImplementedError(f"Boundary object {self._boundary} not supported.")
        # boundary object -> parameters
        params["X1_b_cos"] = {}
        params["X2_b_sin"] = {}
        params["X1_a_cos"] = {}
        params["X2_a_sin"] = {}
        for m in range(self._boundary.mpol + 1):
            for n in range(-self._boundary.ntor, self._boundary.ntor + 1):
                if X1c := self._boundary.get_rc(m, n):
                    params["X1_b_cos"][m, n] = X1c
                    if m == 0:
                        params["X1_a_cos"][0, n] = X1c
                if X2s := self._boundary.get_zs(m, n):
                    params["X2_b_sin"][m, n] = X2s
                    if m == 0:
                        params["X2_a_sin"][0, n] = X2s
        # flip signs (GVEC default torus coordinates have clockwise zeta)
        params = gvec.util.flip_parameters_zeta(params)
        # perturb boundary when restarting
        if self.restart_file:  # ToDo: check how perturbation interacts with run_stages
            perturbation = {"X1pert_b_cos": {}, "X2pert_b_sin": {}, "X1_b_cos": {}, "X2_b_sin": {}, "boundary_perturb": True}
            for (m, n), v in params.items():
                perturbation["X1pert_b_cos"][m, n] = v - self.base_parameters["X1_b_cos"].get((m, n), 0)
                perturbation["X2pert_b_sin"][m, n] = v - self.base_parameters["X2_b_sin"].get((m, n), 0)
                perturbation["X1_b_cos"][m, n] = self.base_parameters["X1_b_cos"].get((m, n), 0)
                perturbation["X2_b_sin"][m, n] = self.base_parameters["X2_b_sin"].get((m, n), 0)
            params |= perturbation
        # pressure profile
        params["pres"] = self.profile_to_params(self._pressure)
        # iota profile
        if self._iota is None:  # current constraint - set initial iota = 0
            params["iota"] = dict(
                type="polynomial",
                coefs=[0.0]
            )
        else:
            params["iota"] = self.profile_to_params(self._iota)
        # toroidal current profile
        if self._current is None:  # iota constraint
            if "I_tor" in params:
                del params["I_tor"]
        else:
            params["I_tor"] = self.profile_to_params(self._current)

        # local DOFs
        params["phiedge"] = self._phiedge
        # default values
        params["init_average_axis"] = True
        params["outputIter"] = self.base_parameters["maxIter"]
        return params

    @staticmethod
    def profile_to_params(profile: Profile) -> dict:
        """Convert a simsopt.Profile object to a dictionary of parameters for GVEC."""
        if isinstance(profile, ProfilePolynomial):
            return dict(
                type="polynomial",
                coefs=profile.local_full_x
            )
        elif isinstance(profile, ProfileScaled) and isinstance(profile.base, ProfilePolynomial):
            return dict(
                type="polynomial",
                coefs=profile.base.local_full_x,
                scale=profile.get("scalefac"),
            )
        else:
            s = np.linspace(0, 1, 101)
            return dict(
                type="interpolation",
                vals=profile(s),
                rho2=s
            )


    # === INPUT VARIABLES === #

    @property
    def phiedge(self) -> float:
        """The toroidal magnetic flux at the last closed flux surface."""
        return self._phiedge
    
    @phiedge.setter
    def phiedge(self, phiedge: float):
        """Set the toroidal magnetic flux at the last closed flux surface."""
        if phiedge != self._phiedge:
            self._phiedge = phiedge
            self.run_required = True

    @property
    def boundary(self) -> Surface:
        """The plasma boundary."""
        return self._boundary

    @boundary.setter
    def boundary(self, boundary):
        """Set the plasma boundary."""
        if not isinstance(boundary, SurfaceRZFourier):
            raise TypeError(f"Boundary object {boundary} not supported.")
        if not boundary.stellsym:
            raise ValueError("Non-stellarator symmetric configurations not supported.")

        if boundary is not self._boundary:
            logging.debug("Replacing surface in boundary setter")
            self.remove_parent(self._boundary)
            self._boundary = boundary
            self.append_parent(boundary)
            self.run_required = True

    @property
    def pressure_profile(self) -> Profile:
        """
        The pressure profile in terms of the normalized toroida flux $s$.

        Represented as a scaled polynomial profile.
        """
        return self._pressure

    @property
    def iota_profile(self) -> Profile | None:
        """
        The rotational transform profile in terms of the normalized toroidal flux $s$.

        To evaluate the rotational transform from the equilibrium, use `iota`.
        """
        return self._iota
    
    @property
    def current_profile(self) -> Profile | None:
        """
        The toroidal current profile in terms of the normalized toridal flux $s$.

        To evaluate the toroidal current from the equilibrium, use `I_tor`.
        """
        return self._current

    # === RETURN FUNCTIONS === #

    def state(self) -> State:
        """Compute the GVEC state object, representing the equilibrium."""
        self.run()
        return self._state

    def iota(self) -> np.ndarray:
        """Compute the rotational transform profile, linearly spaced in $s=\rho^2$."""
        self.run()
        rho = np.sqrt(np.linspace(0, 1, 101))
        ds = gvec.Evaluations(rho=rho, theta=None, zeta=None, state=self._state)
        self._state.compute(ds, "iota")
        return ds.iota.data
        
    def iota_axis(self) -> float:
        """Compute the rotational transform at the magnetic axis."""
        self.run()
        ds = gvec.Evaluations(rho=[0], theta=None, zeta=None, state=self._state)
        self._state.compute(ds, "iota")
        return ds.iota.item()

    def iota_edge(self) -> float:
        """Compute the rotational transform at the edge."""
        self.run()
        ds = gvec.Evaluations(rho=[1], theta=None, zeta=None, state=self._state)
        self._state.compute(ds, "iota")
        return ds.iota.item()

    def I_tor(self) -> np.ndarray:
        """Compute the toroidal current profile, linearly spaced in $s=\rho^2$."""
        self.run()
        rho = np.sqrt(np.linspace(0, 1, 101))
        ds = gvec.Evaluations(rho=rho, theta="int", zeta="int", state=self._state)
        self._state.compute(ds, "I_tor")
        return ds.I_tor.data

    def I_tor_edge(self) -> float:
        """Compute the toroidal current at the edge."""
        self.run()
        ds = gvec.Evaluations(rho=[1], theta="int", zeta="int", state=self._state)
        self._state.compute(ds, "I_tor")
        return ds.I_tor.data

    def volume(self) -> float:
        """Compute the volume enclosed by the flux surfaces."""
        self.run()
        ds = gvec.Evaluations(rho="int", theta="int", zeta="int", state=self._state)
        self._state.compute(ds, "V")
        return ds.V.item()

    def aspect(self) -> float:
        """Compute the aspect ratio of the plasma."""
        self.run()
        ds = gvec.Evaluations(rho="int", theta="int", zeta="int", state=self._state)
        self._state.compute(ds, "major_radius", "minor_radius")
        return (ds.major_radius / ds.minor_radius).item()

    def vacuum_well(self) -> np.ndarray:
        """Compute the vacuum magnetic well W, given by the formula

        W = V'' / V'

        where V' is the derivative of the flux surface volume with
        respect to the normalized toroidal flux. Negative values of W are
        favorable for stability to interchange modes.
        """
        self.run()
        rho = np.linspace(0, 1, 101)
        ds = gvec.Evaluations(rho=rho, theta="int", zeta="int", state=self._state)
        self._state.compute(ds, "dV_dPhi_n", "dV_dPhi_n2")
        return (ds.dV_dPhi_n2 / ds.dV_dPhi_n).data

    def vacuum_well_edge(self) -> float:
        """Compute the vacuum magnetic well W, given by the formula

        W = V'' / V'

        where V' is the derivative of the flux surface volume with
        respect to the normalized toroidal flux. Negative values of W are
        favorable for stability to interchange modes.
        """
        self.run()
        ds = gvec.Evaluations(rho=[1], theta="int", zeta="int", state=self._state)
        self._state.compute(ds, "dV_dPhi_n", "dV_dPhi_n2")
        return (ds.dV_dPhi_n2 / ds.dV_dPhi_n).item()
