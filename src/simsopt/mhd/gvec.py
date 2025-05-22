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
from typing import Optional, Mapping, Literal, Union, Optional
import shutil

import numpy as np

logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI
    from simsopt.util.mpi import MpiPartition
except ImportError as e:
    MPI = None
    MpiPartition = None
    logger.debug(str(e))

try:
    import gvec
    from gvec import State
except ImportError as e:
    gvec = None
    State = None
    logger.debug(str(e))

from simsopt._core.optimizable import Optimizable
from simsopt._core.util import ObjectiveFailure
from simsopt.geo import Surface, SurfaceRZFourier
from simsopt.mhd.profiles import Profile, ProfilePolynomial, ProfileScaled

__all__ = ["Gvec"]

default_parameters = dict(
    ProjectName = "SIMSOPT-GVEC",
    whichInitEquilibrium = 0,

    X1X2_deg = 5,
    LA_deg = 5,

    MaxIter = 100,
    MinimizerType = 10,
    minimize_tol = 1e-6,

    sgrid = dict(
        grid_type = 4,
        nElems = 3,
    )
)

class Gvec(Optimizable):
    """
    This class represents the optimizable for the GVEC equilibrium code.

    Args:
        phiedge: the prescribed total toroidal magnetic flux.
        boundary: the prescribed plasma boundary.
        pressure: the prescribed pressure profile.
        iota: the prescribed (or initial) rotational transform profile.
        current: the prescribed toroidal current profile. if None, GVEC will run in iota-constraint mode, otherwise it will run in current-constraint mode with the given iota as initial guess.
        parameters: a dictionary of GVEC parameters.
        restart_file: path to a GVEC Statefile to restart from.
        delete_intermediates: whether to delete intermediate files after the run.
        mpi: the MPI partition to use. For now, GVEC is best run with ngroups = nprocs and multiple OpenMP threads assigned to each MPI process.
    """

    def __init__(
        self,
        phiedge: Optional[float] = None,
        boundary: Optional[Surface] = None,
        pressure: Optional[Profile] = None,
        iota: Union[Profile, float, None] = None,
        current: Union[Profile, float, None] = None,
        parameters: Mapping = {},
        restart_file: Union[str, Path, None] = None,
        delete_intermediates: bool = False,
        mpi: Optional[MpiPartition] = None,
    ):
        # init arguments which are not DoFs
        self.parameters = gvec.util.CaseInsensitiveDict(parameters) # nested parameters
        self.restart_file = Path(restart_file) if restart_file else None
        self.delete_intermediates = delete_intermediates
        self.mpi = mpi

        if self.restart_file and not self.restart_file.exists():
            raise FileNotFoundError(f"Restart file {self.restart_file} not found")

        # auxiliary attributes
        self.run_count: int = 0  # number of times GVEC has been run (this MPI-process only)
        self.run_required: bool = True  # flag for caching e.g. set after changing parameters
        self.run_successful: bool = False  # flag for whether the last run was successful
        self._state: State = None  # state object representing the GVEC equilibrium

        # init MPI
        if MPI:
            if self.mpi is None:
                self.mpi = MpiPartition()  # default to ngroups = nprocs
            self._mpi_id = f"_{self.mpi.rank_world:03d}"
        else:
            if self.mpi:
                logger.warning("MPI not available, ignoring MpiPartition")
            self._mpi_id = ""

        # fill default parameters
        for key, value in default_parameters.items():
            if key not in self.parameters:
                self.parameters[key] = value
            elif isinstance(value, Mapping):
                for subkey, subvalue in value.items():
                    if subkey not in self.parameters[key]:
                        self.parameters[key][subkey] = subvalue

        # init DoFs
        if phiedge is None:
            if "phiedge" in self.parameters:
                phiedge = self.parameters["phiedge"]
            else:
                phiedge = 1.0
        self._phiedge = phiedge

        if boundary is None:
            if "X1_mn_max" in self.parameters:
                boundary = self.boundary_from_params(self.parameters)
            else:
                boundary = SurfaceRZFourier()
        self._boundary = boundary

        if pressure is None:
            if "pres" in self.parameters:
                pressure = self.profile_from_params(self.parameters["pres"])
            else:
                pressure = ProfilePolynomial([0.0])
        elif isinstance(pressure, float):
            pressure = ProfilePolynomial([pressure])
        self._pressure = pressure

        if iota is None:
            if "iota" in self.parameters:
                iota = self.profile_from_params(self.parameters["iota"])
            else:
                iota = ProfilePolynomial([0.0])
        elif isinstance(iota, float):
            iota = ProfilePolynomial([iota])
        self._iota = iota

        if current is None and "Itor" in self.parameters:
            current = self.profile_from_params(self.parameters["Itor"])
        elif isinstance(current, float):
            current = ProfilePolynomial([current])
        self._current = current

        # init Optimizable
        x0 = self.get_dofs()
        fixed = np.full(len(x0), True)
        names = ['phiedge']
        depends = [self._boundary, self._pressure, self._iota]
        if current is not None:
            depends += [self._current]
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
        parameter_file: Union[str, Path],
        **kwargs,
    ):
        parameters = gvec.util.read_parameters(parameter_file)
        return cls(parameters=parameters, **kwargs)
    
    def recompute_bell(self, parent=None):
        """Set the recomputation flag"""
        logger.debug(f"recompute_bell called by {parent}: run_required = {self.run_required}")
        self.run_required = True

    def get_dofs(self):
        """Return the DOFs owned by the GVEC optimizable.
       o
        This does not include any DoFs owned by dependent objects (boundary or profiles).
        """
        return np.array([self._phiedge])

    def set_dofs(self, x):
        """Set the DoFs owned by the GVEC optimizable.
        
        This does not include any DoFs owned by dependent objects (boundary or profiles).
        """
        if len(x) != 1:
            raise ValueError(f"Expected 1 DOF, got {len(x)}")
        self.phiedge = x[0]
    
    @property
    def logger(self):
        return logging.getLogger("gvec")
    
    def run(self, force: bool = False) -> None:
        """
        Run GVEC (via subprocess) if `run_required` or `force` is True.
        """
        if not self.run_required:
            if force:
                logger.debug("re-run forced")
            else:
                logger.debug("no run required")
                return

        if self._state is not None:
            self._state.finalize()

        if self.delete_intermediates and self.run_successful:
            logger.debug("deleting output from previous run")
            shutil.rmtree(self.rundir)
        
        self.run_count += 1
        logger.debug(f"preparing to run GVEC run number {self.run_count}")

        # create run directory
        self.rundir = Path(f"gvec{self._mpi_id}-{self.run_count:03d}")
        if self.rundir.exists():
            logger.warning(f"run directory {self.rundir} already exists, replacing")
            shutil.rmtree(self.rundir)
        self.rundir.mkdir()

        # prepare parameter file
        params = self.prepare_parameters()

        # configure restart
        if self.restart_file:
            # ToDo: needs test
            logger.info(f"junning GVEC with restart from {self.restart_file}")
            if self.restart_file.is_absolute():
                restart_file = self.restart_file
            else:
                restart_file = ".." / self.restart_file
        else:
            logger.info("running GVEC")
            restart_file = None
        
        # run GVEC
        self.run_successful = False
        try:
            with gvec.util.chdir(self.rundir):
                gvec.util.write_parameters(self.serialize_parameters(params), "parameter.toml")
                stagedir, statefile, diagnostics = gvec.scripts.run.run_stages(params, restart_file)
        except RuntimeError as e:
            logger.error(f"GVEC failed with: {e}")
            raise ObjectiveFailure("Run GVEC failed.") from e
        self.run_successful = True
        
        # read output/statefile
        logger.debug(f"Reading state: {self.rundir / stagedir / 'parameter.ini'}, {self.rundir / statefile}")
        self._state = State(self.rundir / stagedir / "parameter.ini", self.rundir / statefile)

        logger.debug("GVEC finished")
        self.run_required = False
    
    @staticmethod
    def serialize_parameters(params: dict) -> dict:  # ToDo: deprecate with GVEC v1.1
        for key, value in params.items():
            if isinstance(value, Mapping):
                params[key] = Gvec.serialize_parameters(value)
            elif isinstance(value, np.ndarray):
                params[key] = value.tolist()
            else:
                params[key] = value
        return params

    def prepare_parameters(self) -> Mapping:
        """
        Prepare DOFs as parameters for the GVEC input file as keyword arguments which can be passed to `run_stages`.
        """
        if isinstance(self.boundary, SurfaceRZFourier):
            boundary = self.boundary
        else:
            logger.debug(f"Converting boundary from {self.boundary.__class__} to SurfaceRZFourier")
            boundary = self.boundary.to_RZFourier()

        if not boundary.stellsym:
            raise NotImplementedError("Non-stellarator symmetric configurations not supported with simsopt yet.")

        params = copy.deepcopy(self.parameters)
        
        # set paths to other files absolute
        for key in [
            "vmecwoutfile",
            "boundary_filename",
            "hmap_ncfile",
            ]:
            if key in params:
                params[key] = str(Path(params[key]).absolute())

        # boundary object -> parameters
        params["nfp"] = boundary.nfp
        for key in ["X1", "X2", "LA"]:
            if f"{key}_mn_max" not in params:
                params[f"{key}_mn_max"] = (boundary.mpol, boundary.ntor)
        params["init_average_axis"] = True
        params["X1_b_cos"] = {}
        params["X2_b_sin"] = {}
        params["X1_a_cos"] = {}
        params["X2_a_sin"] = {}
        for m in range(boundary.mpol + 1):
            for n in range(-boundary.ntor, boundary.ntor + 1):
                if X1c := boundary.get_rc(m, n):
                    params["X1_b_cos"][m, n] = X1c
                if X2s := boundary.get_zs(m, n):
                    params["X2_b_sin"][m, n] = X2s
        # flip signs (GVEC default torus coordinates have clockwise zeta)
        params = gvec.util.flip_boundary_zeta(params)

        # perturb boundary when restarting
        if self.restart_file:  # ToDo: check how perturbation interacts with run_stages (wait for gvec update of current_constraint fix)
            perturbation = {"X1pert_b_cos": {}, "X2pert_b_sin": {}, "X1_b_cos": {}, "X2_b_sin": {}, "boundary_perturb": True}
            for (m, n), v in params.items():
                perturbation["X1pert_b_cos"][m, n] = v - self.parameters["X1_b_cos"].get((m, n), 0)
                perturbation["X2pert_b_sin"][m, n] = v - self.parameters["X2_b_sin"].get((m, n), 0)
                perturbation["X1_b_cos"][m, n] = self.parameters["X1_b_cos"].get((m, n), 0)
                perturbation["X2_b_sin"][m, n] = self.parameters["X2_b_sin"].get((m, n), 0)
            params |= perturbation

        # profiles
        params["pres"] = self.profile_to_params(self._pressure)
        params["iota"] = self.profile_to_params(self._iota)
        if self._current is None:  # iota constraint
            if "Itor" in params:
                del params["Itor"]
        else:  # current constraint
            params["Itor"] = self.profile_to_params(self._current)

        # local DOFs
        params["phiedge"] = self._phiedge
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
    
    @staticmethod
    def profile_from_params(params: Mapping) -> Profile:
        """Convert a dictionary of parameters to a simsopt.Profile object."""
        if params["type"] == "polynomial":
            profile = ProfilePolynomial(params["coefs"])
        elif params["type"] in ["bspline", "interpolation"]:
            raise NotImplementedError(f"Profile of type {params['type']} is not supported for converting a GVEC parameter file to a simsopt.Profile object. Use 'polynomial' instead.")
        else:
            raise ValueError(f"Unknown profile type {params['type']}")
        
        if "scale" in params:
            profile = ProfileScaled(profile, params["scale"])
        return profile
    
    @staticmethod
    def boundary_from_params(params: Mapping) -> Surface:
        """Convert a dictionary of parameters to a simsopt.Surface object."""
        if "nfp" not in params:
            raise ValueError("nfp is a required parameter")
        if params["X1_sin_cos"] != "_cos_" or params["X2_sin_cos"] != "_sin_":
            raise NotImplementedError("Not stellarator symmetric configurations not supported by the GVEC-SIMSOPT interface for now.")
        if params["X1_mn_max"] != params["X2_mn_max"]:
            raise ValueError("X1_mn_max and X2_mn_max must be equal")
        
        M, N = params["X1_mn_max"]
        boundary = SurfaceRZFourier(
            nfp=params["nfp"],
            stellsym=True,
            mpol=M,
            ntor=N,
        )

        params = gvec.util.flip_parameters_zeta(params)
        for (m, n), value in params.get("X1_b_cos", {}).items():
            boundary.set_rc(m, n, value)
        for (m, n), value in params.get("X2_b_sin", {}).items():
            boundary.set_zs(m, n, value)

        boundary.local_full_x = boundary.get_dofs()
        return boundary

    # === INPUT VARIABLES === #

    @property
    def phiedge(self) -> float:
        """The toroidal magnetic flux at the last closed flux surface."""
        return self._phiedge
    
    @phiedge.setter
    def phiedge(self, phiedge: float):
        """Set the toroidal magnetic flux at the last closed flux surface."""
        if phiedge == self._phiedge:
            return
        logging.debug(f"setting phiedge to {phiedge}")
        self._phiedge = phiedge
        self.run_required = True

    @property
    def boundary(self) -> Surface:
        """The plasma boundary."""
        return self._boundary

    @boundary.setter
    def boundary(self, boundary: Surface):
        """Set the plasma boundary."""
        if boundary is self._boundary:
            return
        logging.debug("setting plasma boundary")
        self.remove_parent(self._boundary)
        self._boundary = boundary
        self.append_parent(self._boundary)
        self.run_required = True

    @property
    def pressure_profile(self) -> Profile:
        """
        The pressure profile in terms of the normalized toroida flux $s$.

        Represented as a scaled polynomial profile.
        """
        return self._pressure
    
    @pressure_profile.setter
    def pressure_profile(self, pressure_profile: Union[Profile, float]):
        if pressure_profile is self._pressure:
            return
        logging.debug("setting pressure profile")
        if isinstance(pressure_profile, float):
            pressure_profile = ProfilePolynomial([pressure_profile])
        self.remove_parent(self._pressure)
        self._pressure = pressure_profile
        self.append_parent(self._pressure)
        self.run_required = True

    @property
    def iota_profile(self) -> Profile:
        """
        The rotational transform profile in terms of the normalized toroidal flux $s$.

        To evaluate the rotational transform from the equilibrium, use `iota`.
        If `current_profile` is set, this profile is only used to set the initial iota profile.
        """
        return self._iota
    
    @iota_profile.setter
    def iota_profile(self, iota_profile: Union[Profile, float]):
        if iota_profile is self._iota:
            return
        if self._current is None:
            logging.debug("setting iota profile (GVEC configured with iota-constraint)")
        else:
            logging.debug("setting initial iota profile (GVEC configured with current-constraint)")
        if isinstance(iota_profile, float):
            iota_profile = ProfilePolynomial([iota_profile])
        self.remove_parent(self._iota)
        self._iota = iota_profile
        self.append_parent(self._iota)
        self.run_required = True
    
    @property
    def current_profile(self) -> Union[Profile, None]:
        """
        The toroidal current profile in terms of the normalized toridal flux $s$.

        To evaluate the toroidal current from the equilibrium, use `I_tor`.
        If this profile is set, GVEC will run in "current-constraint" mode, otherwise it will run in "iota-constraint" mode.
        """
        return self._current
    
    @current_profile.setter
    def current_profile(self, current_profile: Union[Profile, float, None]):
        if current_profile is self._current:
            return
        logging.debug("setting toroidal current profile")
        if isinstance(current_profile, float):
            current_profile = ProfilePolynomial([current_profile])
        if self._current is not None:
            self.remove_parent(self._current)
        self._current = current_profile
        if current_profile is not None:
            self.append_parent(self._current)
        self.run_required = True

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

    def current(self) -> np.ndarray:
        """Compute the toroidal current profile, linearly spaced in $s=\rho^2$."""
        self.run()
        rho = np.sqrt(np.linspace(0, 1, 101))
        ds = gvec.Evaluations(rho=rho, theta="int", zeta="int", state=self._state)
        self._state.compute(ds, "I_tor")
        return ds.I_tor.data

    def current_edge(self) -> float:
        """Compute the toroidal current at the edge."""
        self.run()
        ds = gvec.Evaluations(rho=[1], theta="int", zeta="int", state=self._state)
        self._state.compute(ds, "I_tor")
        return ds.I_tor.data

    def volume(self) -> float:
        """Compute the volume enclosed by the flux surfaces.
        
        This method integrates the Jacobian determiant over the whole domain.
        The result should be the same as the volume computed from a Surface object.
        """
        self.run()
        ds = gvec.Evaluations(rho="int", theta="int", zeta="int", state=self._state)
        self._state.compute(ds, "V")
        return ds.V.item()

    def aspect(self) -> float:
        """Compute the aspect ratio of the configuration.
        
        This method computes the effective major and minor radii by integration of the
        Jacobian determinant. The result should be the same as the aspect ratio
        computed from a Surface object.
        """
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
