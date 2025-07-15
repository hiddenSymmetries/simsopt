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
from typing import Optional, Literal, Union
from collections.abc import Mapping, Sequence
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

    X1X2_deg = 5,
    LA_deg = 5,

    totalIter = 10000,
    minimize_tol = 1e-5,

    sgrid = dict(
        grid_type = 4,
        nElems = 5,
    )
)

_return_functions = set()

def return_function(method):
    """
    Decorator to mark a method as a return function.
    
    This is used to indicate that the method returns a value that is computed from the GVEC state.
    """
    _return_functions.add(method)
    return method

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
        keep_gvec_intermediates: Optional[Literal['all', 'stages']] = None,
        mpi: Optional[MpiPartition] = None,
    ):
        # init arguments which are not DoFs
        self.parameters = gvec.util.CaseInsensitiveDict(parameters) # nested parameters
        self.restart_file = Path(restart_file) if restart_file else None
        self.delete_intermediates = delete_intermediates
        self.keep_gvec_intermediates = keep_gvec_intermediates
        self.mpi = mpi

        if self.restart_file and not self.restart_file.exists():
            raise FileNotFoundError(f"Restart file {self.restart_file} not found")

        # auxiliary attributes
        self.run_count: int = 0  # number of times GVEC has been run (this MPI-process only)
        self.run_required: bool = True  # flag for caching e.g. set after changing parameters
        self.run_successful: bool = False  # flag for whether the last run was successful
        self._state: State = None  # state object representing the GVEC equilibrium
        self._runobj: gvec.Run = None  # the gvec run object, used to access the state & diagnostics

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

        if iota is None and "iota" in self.parameters:
            iota = self.profile_from_params(self.parameters["iota"])
        elif isinstance(iota, float):
            iota = ProfilePolynomial([iota])
        self._iota = iota

        if current is None and "I_tor" in self.parameters:
            current = self.profile_from_params(self.parameters["I_tor"])
        elif isinstance(current, float):
            current = ProfilePolynomial([current])
        self._current = current

        # init Optimizable
        x0 = self.get_dofs()
        fixed = np.full(len(x0), True)
        names = ['phiedge']
        depends = [self._boundary, self._pressure]
        if iota is not None:
            depends += [self._iota]
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
            logger.info(f"running GVEC with restart from {self.restart_file}")
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
            self._runobj = gvec.run(params, restart_file, runpath=self.rundir, quiet=True, keep_intermediates=self.keep_gvec_intermediates)
        except RuntimeError as e:
            logger.error(f"GVEC failed with: {e}")
            raise ObjectiveFailure("Run GVEC failed.") from e
        self.run_successful = True
        
        # read output/statefile
        self._state = self._runobj.state

        logger.debug("GVEC finished")
        self.run_required = False

    def prepare_parameters(self) -> Mapping:
        """
        Prepare DOFs as parameters for the GVEC input file as keyword arguments which can be passed to `run_stages`.
        """
        if isinstance(self.boundary, SurfaceRZFourier):
            boundary = self.boundary
        else:
            logger.debug(f"Converting boundary from {self.boundary.__class__} to SurfaceRZFourier")
            boundary = self.boundary.to_RZFourier()

        params = copy.deepcopy(self.parameters)
        
        # set paths to other files absolute
        for key in [
            "vmecwoutfile",
            "boundary_filename",
            "hmap_ncfile",
            ]:
            if key in params:
                params[key] = str(Path(params[key]).absolute())

        params = self.boundary_to_params(boundary, params)

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
        if self._iota is None and self._current is None:
            raise RuntimeError("GVEC requires either an iota or a current profile to be set.")

        params["pres"] = self.profile_to_params(self._pressure)
        if self._iota is None:
            if "iota" in params:
                del params["iota"]
        else:
            params["iota"] = self.profile_to_params(self._iota)
        if self._current is None:  # iota constraint
            if "I_tor" in params:
                del params["I_tor"]
            if "picard_current" in params:
                del params["picard_current"]
        else:  # current optimization
            params["I_tor"] = self.profile_to_params(self._current)
            params["picard_current"] = "auto"

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
    def boundary_to_params(boundary: SurfaceRZFourier, append: Optional[Mapping] = None) -> dict:
        """Convert a simsopt.SurfaceRZFourier object into GVEC boundary parameters.
        
        The output parameters will include the (non-boundary) contents of the `append` dictionary, if provided.
        """
        if append is None:
            params = {}
        else:
            params = copy.deepcopy(append)

        params["nfp"] = boundary.nfp
        params["init_average_axis"] = True

        # keep higher mn_max if set previously (e.g. for interior modes)
        for key in ["X1", "X2", "LA"]:
            m, n = params.get(f"{key}_mn_max", (0, 0))
            params[f"{key}_mn_max"] = (max(boundary.mpol, m), max(boundary.ntor, n))
        
        params["X1_sin_cos"] = "_cos_" if boundary.stellsym else "_sin_cos_"
        params["X2_sin_cos"] = "_sin_" if boundary.stellsym else "_sin_cos_"
                
        for Xi in ["X1", "X2"]:
            for ab in ["a", "b"]:
                for sincos in ["sin", "cos"]:
                    params[f"{Xi}_{ab}_{sincos}"] = {}
        for m in range(boundary.mpol + 1):
            for n in range(-boundary.ntor, boundary.ntor + 1):
                if X1c := boundary.get_rc(m, n):
                    params["X1_b_cos"][m, n] = X1c
                if not boundary.stellsym and (X1s := boundary.get_rs(m, n)):
                    params["X1_b_sin"][m, n] = X1s
                if not boundary.stellsym and (X2c := boundary.get_zc(m, n)):
                    params["X2_b_cos"][m, n] = X2c
                if X2s := boundary.get_zs(m, n):
                    params["X2_b_sin"][m, n] = X2s
        for Xi in ["X1", "X2"]:
            for ab in ["a", "b"]:
                for sincos in ["sin", "cos"]:
                    if len(params[f"{Xi}_{ab}_{sincos}"]) == 0:
                        del params[f"{Xi}_{ab}_{sincos}"]
        
        # change (R,phi,Z) -> (R,Z,phi)
        params = gvec.util.flip_boundary_zeta(params)
        # ensure right-handed (R,Z,phi)
        if not gvec.util.check_boundary_direction(params):
            params = gvec.util.flip_boundary_theta(params)
        return params
    
    @staticmethod
    def boundary_from_params(params: Mapping) -> SurfaceRZFourier:
        """Convert a dictionary of parameters to a simsopt.SurfaceRZFourier object.
        
        Note that simsopt assumes a (right-handed) (R,phi,Z) coordinate system,
        while GVEC uses a (R,Z,phi) coordinate system. The toroidal angle therefore increases
        in the clockwise, rather than counter-clockwise direction, when viewed from above.
        """
        hmap = params.get("which_hmap", 1)
        if hmap != 1:
            logger.warning(f"interpreting GVEC boundary with {hmap:=} as SurfaceRZFourier")
        if params["X1_mn_max"] != params["X2_mn_max"]:
            raise NotImplementedError("X1_mn_max != X2_mn_max is not supported.")
    
        nfp = params["nfp"]
        M, N = params["X1_mn_max"]
        stellsym = params["X1_sin_cos"] == "cos" and params["X2_sin_cos"] == "sin"

        boundary = SurfaceRZFourier(
            nfp=nfp,
            stellsym=stellsym,
            mpol=M,
            ntor=N,
        )

        # set default values to 0
        boundary.set_rc(0, 0, 0.0)
        boundary.set_rc(1, 0, 0.0)
        boundary.set_zs(1, 0, 0.0)

        params = gvec.util.flip_boundary_zeta(params)
        if "cos" in params["X1_sin_cos"]:
            for (m, n), value in params.get("X1_b_cos", {}).items():
                boundary.set_rc(m, n, value)
        if "sin" in params["X1_sin_cos"]:
            for (m, n), value in params.get("X1_b_sin", {}).items():
                boundary.set_rs(m, n, value)
        if "cos" in params["X2_sin_cos"]:
            for (m, n), value in params.get("X2_b_cos", {}).items():
                boundary.set_zc(m, n, value)
        if "sin" in params["X2_sin_cos"]:
            for (m, n), value in params.get("X2_b_sin", {}).items():
                boundary.set_zs(m, n, value)

        boundary.fix_all()
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
        """The plasma boundary as a simsopt.Surface object.
        
        Note that the default coordinate system for VMEC uses a toroidal angle in the opposite
        direction of GVEC's toroidal angle (counter-clockwise vs clockwise when viewed from above).
        """
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
        """The pressure profile in terms of the normalized toroidal flux $s$."""
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
    def iota_profile(self) -> Union[Profile, None]:
        """
        The rotational transform profile in terms of the normalized toroidal flux $s$.

        To evaluate the rotational transform from the equilibrium, use `iota`.
        If `current_profile` is set, this profile is only used to set the initial iota profile.

        Note that the default coordinate system for VMEC uses a toroidal angle in the opposite
        direction of GVEC's toroidal angle (counter-clockwise vs clockwise when viewed from above).
        Therefore the iota profile typically has the opposite sign compared to VMEC.
        """
        return self._iota
    
    @iota_profile.setter
    def iota_profile(self, iota_profile: Union[Profile, float, None]):
        if iota_profile is self._iota:
            return
        logger.debug("setting iota profile")
        if isinstance(iota_profile, float):
            iota_profile = ProfilePolynomial([iota_profile])
        if self._iota is not None:
            self.remove_parent(self._iota)
        self._iota = iota_profile
        if iota_profile is not None:
            self.append_parent(self._iota)
        self.run_required = True
    
    @property
    def current_profile(self) -> Union[Profile, None]:
        """
        The toroidal current profile in terms of the normalized toridal flux $s$.

        To evaluate the toroidal current from the equilibrium, use `I_tor`.
        If this profile is set, GVEC will run using "current-optimization",
        otherwise it will run with "iota-constraint".

        Note that the default coordinate system for VMEC uses a toroidal angle in the opposite
        direction of GVEC's toroidal angle (counter-clockwise vs clockwise when viewed from above).
        Therefore the current profile typically has the opposite sign compared to VMEC.
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
        """Compute the gvec.State object, representing the equilibrium.
        
        The State object allows evaluating the configuration at any position
        using the same accuracy and discretization as used during the minimization.
        """
        self.run()
        return self._state

    @return_function
    def iota_axis(self) -> float:
        """Compute the rotational transform at the magnetic axis."""
        ev = self.state().evaluate("iota", rho=0.0, theta=None, zeta=None)
        return ev.iota.item()

    @return_function
    def iota_edge(self) -> float:
        """Compute the rotational transform at the edge."""
        ev = self.state().evaluate("iota", rho=1.0, theta=None, zeta=None)
        return ev.iota.item()

    @return_function
    def current_edge(self) -> float:
        """Compute the toroidal current at the edge."""
        ev = self.state().evaluate("I_tor", rho=1.0, theta="int", zeta="int")
        return ev.I_tor.item()

    @return_function
    def volume(self) -> float:
        """Compute the volume enclosed by the flux surfaces.
        
        This method integrates the Jacobian determiant over the whole domain.
        The result should be the same as the volume computed from a Surface object.
        """
        ev = self.state().evaluate("V", rho="int", theta="int", zeta="int")
        return ev.V.item()

    @return_function
    def aspect(self) -> float:
        """Compute the aspect ratio of the configuration.
        
        This method computes the effective major and minor radii by integration of the
        Jacobian determinant. The result should be the same as the aspect ratio
        computed from a Surface object.
        """
        ev = self.state().evaluate("major_radius", "minor_radius", rho="int", theta="int", zeta="int")
        return (ev.major_radius / ev.minor_radius).item()

    @return_function
    def vacuum_well(self) -> float:
        """Compute the depth of the vacuum magnetic well.

        Compute a single number W that summarizes the vacuum magnetic well,
        given by the formula

        W = (dV/ds(s=0) - dV/ds(s=1)) / (dV/ds(s=0)

        where dVds is the derivative of the flux surface volume with
        respect to the normalized toroidal flux s=Phi_n. Positive values
        of W are favorable for stability to interchange modes. This formula
        for W is motivated by the fact that

        d^2 V / d s^2 < 0

        is favorable for stability. Integrating over s from 0 to 1
        and normalizing gives the above formula for W. Notice that W
        is dimensionless, and it scales as the square of the minor
        radius.
        """
        ev = self.state().evaluate(
            "dV_dPhi_n", rho=[1e-4, 1.0], theta="int", zeta="int"
        )
        Vp1 = ev.sel(rho=1.0).dV_dPhi_n.item()
        Vp0 = ev.sel(rho=1e-4).dV_dPhi_n.item()
        return (Vp0 - Vp1) / Vp0

    return_fn_map = {func.__name__: func for func in _return_functions}
