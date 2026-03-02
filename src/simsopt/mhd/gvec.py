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
from collections.abc import Mapping
import shutil

import numpy as np
import matplotlib.pyplot as plt

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
from simsopt._core.types import RealArray
from simsopt.geo import Surface, SurfaceScaled, SurfaceRZFourier
from simsopt.mhd.profiles import Profile, ProfilePolynomial, ProfileScaled
from simsopt.mhd.gvec_dofs import GVECSurfaceDoFs

__all__ = ["Gvec", "GVECQuantity", "GVECSurfaceDoFs"]

default_parameters = dict(
    ProjectName="SIMSOPT-GVEC",
    X1X2_deg=5,
    LA_deg=5,
    totalIter=10000,
    minimize_tol=1e-5,
    sgrid=dict(
        grid_type=4,
        nElems=5,
    ),
)


class Gvec(Optimizable):
    """
    This class represents the optimizable for the GVEC equilibrium code.

    Args:
        phiedge: the prescribed total toroidal magnetic flux.
        boundary: the prescribed plasma boundary.
        pressure: the prescribed pressure profile.
        iota: the prescribed (or initial) rotational transform profile.
        current: the prescribed toroidal current profile. if None, GVEC will run in prescribed-iota mode, otherwise it will run in prescribed-current mode with the given iota as initial guess.
        parameters: a dictionary of GVEC parameters.
        restart: ...
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
        restart: Union[Literal["first", "last"], Path, str, None] = None,
        delete_intermediates: bool = True,
        keep_failures: bool = False,
        keep_gvec_intermediates: Optional[Literal["all", "stages"]] = None,
        mpi: Optional[MpiPartition] = None,
    ):
        # init arguments which are not DoFs
        self.parameters = gvec.util.CaseInsensitiveDict(parameters)  # nested parameters
        self.delete_intermediates = delete_intermediates
        self.keep_failures = keep_failures
        self.keep_gvec_intermediates = keep_gvec_intermediates
        self.mpi = mpi

        if restart in [None, "first", "last"]:
            self.restart = restart
        elif isinstance(restart, (str, Path)):
            if not Path(restart).is_dir():
                raise ValueError(
                    f"restart path {restart} does not exist or is not a directory."
                )
            self.restart = gvec.find_state(restart)
        else:
            raise ValueError(
                "restart must be 'first', 'last', a path to a GVEC run directory or None."
            )

        # auxiliary attributes
        self.run_count: int = (
            -1
        )  # number of times GVEC has been run (0-indexed, this MPI-process only)
        self.run_required: bool = (
            True  # flag for caching e.g. set after changing parameters
        )
        self.run_successful: bool = (
            False  # flag for whether the last run was successful
        )
        self._state: State = None  # state object representing the GVEC equilibrium
        self._runobj: gvec.Run = (
            None  # the gvec run object, used to access the state & diagnostics
        )

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
        names = ["phiedge"]
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

    @classmethod
    def from_rundir(
        cls,
        rundir: Union[str, Path],
        **kwargs,
    ):
        state = gvec.find_state(rundir)
        parameter_toml = list(Path(rundir).glob("parameter*.toml"))
        if len(parameter_toml) == 1:
            self = cls.from_parameter_file(parameter_toml[0], **kwargs)
        else:
            self = cls.from_parameter_file(state.parameterfile, **kwargs)
        self._state = state
        self.run_required = False
        return self

    def recompute_bell(self, parent=None):
        """Set the recomputation flag"""
        logger.debug(
            f"recompute_bell called by {parent}: run_required = {self.run_required}"
        )
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
            elif not self.run_successful:
                logger.debug("no run required, cached run not successful")
                raise ObjectiveFailure("cached GVEC run was not successful")
            else:
                logger.debug("no run required")
                return

        self.run_count += 1
        logger.debug(f"preparing to run GVEC run number {self.run_count}")

        # create run directory
        if self.run_successful:
            previous_rundir = self.rundir
        else:
            previous_rundir = None
        self.rundir = Path(f"gvec{self._mpi_id}-{self.run_count:03d}")
        if self.rundir.exists():
            logger.warning(f"run directory {self.rundir} already exists, replacing")
            shutil.rmtree(self.rundir)
        self.rundir.mkdir()

        # configure restart
        if self.restart is None:
            restart = None
            logger.info("running GVEC")
        elif isinstance(self.restart, State):
            restart = self.restart
            logger.info(f"running GVEC from {self.restart}")
        elif self.restart == "first":
            if self.run_count == 0 and (not MPI or self.mpi.proc0_world):
                restart = None
                logger.info("running GVEC (first run)")
            else:
                if MPI:
                    restart = gvec.find_state("gvec_000-000")
                else:
                    restart = gvec.find_state("gvec-000")
                logger.info("running GVEC from first run")
        elif self.restart == "last":
            if self.run_successful:
                restart = self._state
                logger.info("running GVEC from previous run")
            else:
                restart = None
                logger.info("running GVEC (previous run not successful)")
        else:
            raise RuntimeError(f"invalid value for restart: {self.restart}")

        # prepare parameter file
        params = self.prepare_parameters(restart)
        gvec.util.write_parameters(params, self.rundir / "parameters.toml")

        # run GVEC
        self.run_required = False
        self.run_successful = False
        self._state = None

        try:
            self._runobj = gvec.run(
                params,
                restart,
                runpath=self.rundir,
                quiet=True,
                keep_intermediates=self.keep_gvec_intermediates,
            )
            fig = self._runobj.plot_diagnostics_minimization()
            fig.savefig(self.rundir / "iterations.png")
            plt.close(fig)
        except RuntimeError as e:
            logger.error(f"GVEC failed with: {e}")
            if not self.keep_failures:
                shutil.rmtree(self.rundir)
            raise ObjectiveFailure("Run GVEC failed.") from e

        self.run_successful = True
        self._state = self._runobj.state
        logger.debug(f"GVEC finished in {self._runobj.GVEC_iter_used} iterations")

        # remove previous rundir
        if (
            self.delete_intermediates
            and previous_rundir
            and (
                self.restart != "first"
                or previous_rundir not in [Path("gvec-000"), Path("gvec_000-000")]
            )
        ):
            logger.debug("deleting output from previous run")
            shutil.rmtree(previous_rundir)

    def prepare_parameters(self, restart: Union[State, None] = None) -> Mapping:
        """
        Prepare DOFs as parameters for the GVEC input file as keyword arguments which can be passed to `run_stages`.
        """
        params = copy.deepcopy(self.parameters)

        # set paths to other files absolute
        for key in [
            "vmecwoutfile",
            "boundary_filename",
            "hmap_ncfile",
        ]:
            if key in params:
                params[key] = str(Path(params[key]).resolve())

        params = self.boundary_to_params(self.boundary, params)

        # non-RZphi coordinate frame
        if params.get("which_hmap", 1) != 1 and not (
            isinstance(self.boundary, GVECSurfaceDoFs)
            or (isinstance(self.boundary, SurfaceScaled) and isinstance(self.boundary.surf, GVECSurfaceDoFs))
        ):
            raise TypeError(f"Non-RZphi coordinate frame (hmap={params.get('which_hmap')}) selected, but boundary is of type {type(self.boundary)}. "
                            "Use GVECSurfaceDoFs to represent generic boundary DoFs.")

        # perturb boundary when restarting
        if restart:
            perturbation = gvec.util.CaseInsensitiveDict(boundary_perturb=True)
            for Xi in ["X1", "X2"]:
                for scs in ["sin", "cos"]:
                    perturbation[f"{Xi}pert_b_{scs}"] = {}
                    perturbation[f"{Xi}_b_{scs}"] = {}
                    # set boundary modes to values from restart
                    for (m, n), v in restart.parameters.get(
                        f"{Xi}_b_{scs}", {}
                    ).items():
                        perturbation[f"{Xi}_b_{scs}"][m, n] = v
                    # set boundary perturbation to difference between current and restart
                    for (m, n), v in params.get(f"{Xi}_b_{scs}", {}).items():
                        v = v - restart.parameters[f"{Xi}_b_{scs}"].get((m, n), 0)
                        if v != 0.0:
                            perturbation[f"{Xi}pert_b_{scs}"][m, n] = v
                    if len(perturbation[f"{Xi}pert_b_{scs}"]) == 0:
                        del perturbation[f"{Xi}pert_b_{scs}"]
                    if len(perturbation[f"{Xi}_b_{scs}"]) == 0:
                        del perturbation[f"{Xi}_b_{scs}"]
            params.update(perturbation)

        # profiles
        if self._iota is None and self._current is None:
            raise RuntimeError(
                "GVEC requires either an iota or a current profile to be set."
            )

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
            return dict(type="polynomial", coefs=profile.local_full_x)
        elif isinstance(profile, ProfileScaled) and isinstance(
            profile.base, ProfilePolynomial
        ):
            return dict(
                type="polynomial",
                coefs=profile.base.local_full_x,
                scale=profile.get("scalefac"),
            )
        else:
            s = np.linspace(0, 1, 101)
            return dict(type="interpolation", vals=profile(s), rho2=s)

    @staticmethod
    def profile_from_params(params: Mapping) -> Profile:
        """Convert a dictionary of parameters to a simsopt.Profile object."""
        if params["type"] == "polynomial":
            profile = ProfilePolynomial(params["coefs"])
        elif params["type"] in ["bspline", "interpolation"]:
            raise NotImplementedError(
                f"Profile of type {params['type']} is not supported for converting a GVEC parameter file to a simsopt.Profile object. Use 'polynomial' instead."
            )
        else:
            raise ValueError(f"Unknown profile type {params['type']}")

        if "scale" in params:
            profile = ProfileScaled(profile, params["scale"])
        return profile

    @staticmethod
    def boundary_to_params(
        boundary: Union[Surface, SurfaceScaled, GVECSurfaceDoFs], append: Optional[Mapping] = None
    ) -> dict:
        """Convert a simsopt.SurfaceRZFourier object into GVEC boundary parameters.

        The output parameters will include the (non-boundary) contents of the `append` dictionary, if provided.
        """
        if isinstance(boundary, SurfaceScaled):
            boundary = boundary.surf
        if not isinstance(boundary, (SurfaceRZFourier, GVECSurfaceDoFs)):
            boundary = boundary.to_RZFourier()
        
        params = boundary.to_gvec_parameters()

        if append is not None:
            params = copy.deepcopy(append) | params

        return params

    @staticmethod
    def boundary_from_params(params: Mapping) -> Union[SurfaceRZFourier, GVECSurfaceDoFs]:
        """Convert a dictionary of parameters to a simsopt.SurfaceRZFourier object.

        Note that simsopt assumes a (right-handed) (R,phi,Z) coordinate system,
        while GVEC uses a (R,Z,phi) coordinate system. The toroidal angle therefore increases
        in the clockwise, rather than counter-clockwise direction, when viewed from above.
        """

        # RZphi with phi clockwise when viewed from above
        if params.get("which_hmap", 1) == 1:
            boundary = SurfaceRZFourier.from_gvec_parameters(params)
        
        # generic boundary type: only contains boundary modes, cannot be evaluated
        else:
            boundary = GVECSurfaceDoFs.from_gvec_parameters(params)

        boundary.fix_all()
        return boundary

    @property
    def state(self) -> State:
        """Return the gvec.State object, representing the equilibrium, rerunning GVEC if necessary.

        The State object allows evaluating the configuration at any position
        using the same accuracy and discretization as used during the minimization.

        As this is not a number, this is not a 'return function' in the usual sense.
        """
        self.run()
        return self._state

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

    # === RETURN FUNCTIONS - TO BE SIMILAR TO VMEC === #

    def aspect(self):
        """Return the effective aspect ratio."""
        ev = self.state.evaluate("aspect_ratio")
        return ev.aspect_ratio.item()

    def volume(self):
        """Return the volume inside the last closed flux surface."""
        ev = self.state.evaluate("V")
        return ev.V.item()

    def iota_axis(self):
        """Return the rotational transform on axis."""
        ev = self.state.evaluate("iota", rho=1e-4)
        return ev.iota.item()

    def iota_edge(self):
        """Return the rotational transform at the boundary."""
        ev = self.state.evaluate("iota", rho=1.0)
        return ev.iota.item()

    def mean_iota(self):
        """Return the mean rotational transform."""
        ev = self.state.evaluate("iota_avg2")
        return ev.iota_avg2.item()

    def mean_shear(self):
        """Return the average magnetic shear."""
        ev = self.state.evaluate("shear_avg2")
        return ev.shear_avg2.item()

    def vacuum_well(self) -> float:
        """Return the depth of the vacuum magnetic well."""
        ev = self.state.evaluate("vacuum_magnetic_well_depth")
        return ev.vacuum_magnetic_well_depth.item()

    return_fn_map = {
        "aspect": aspect,
        "volume": volume,
        "iota_axis": iota_axis,
        "iota_edge": iota_edge,
        "mean_iota": mean_iota,
        "mean_shear": mean_shear,
        "vacuum_well": vacuum_well,
    }


class GVECQuantity(Optimizable):
    """
    This optimizable computes a specified quantity as a dependent optimizable from the GVEC equilibrium.
    """

    def __init__(self, eq: Gvec, quantity: str, **kwargs):
        self.eq = eq
        self.quantity = quantity
        self.kwargs = kwargs

        super().__init__(depends_on=[eq])

    @property
    def ev(self):
        """
        Evaluate the specified quantity from the GVEC state.

        Returns:
            The evaluated quantity as an xarray.Dataset.
        """
        return self.eq.state.evaluate(self.quantity, **self.kwargs)

    def J(self) -> np.ndarray:
        return self.ev[self.quantity].data

    return_fn_map = {"J": J}


class Elongation(Optimizable):
    """
    This optimizable computes the elongation of the plasma cross-sections using PCA.

    Applying a PCA on the boundary points gives us the principal axes for the cross-section.
    We can then take the minimum and maximum values along the principal axes to get an estimate for the "length" and "width" of the cross-section.
    """

    def __init__(self, eq: Gvec, zeta: Union[float, RealArray, Literal["int"]] = "int"):
        self.eq = eq
        self.zeta = zeta

        super().__init__(depends_on=[eq])

    def J(self):
        from scipy.linalg import svd

        ev = self.eq.state.evaluate(
            "pos", rho=[1.0], theta="int", zeta=self.zeta
        ).squeeze()

        elongation = np.zeros(len(ev.zeta))
        for z, zeta in enumerate(ev.zeta):
            pos = ev.pos.sel(zeta=zeta).squeeze().transpose("xyz", "pol").values
            # weighting with arclength
            tan = np.sqrt(
                np.sum(
                    (np.roll(pos, 1, axis=1) - np.roll(pos, -1, axis=1)) ** 2, axis=0
                )
            )
            tan /= np.sum(tan)
            cen0 = np.mean(pos, axis=1)
            cen = np.mean((pos - cen0[:, None]) * tan[None, :], axis=1) + cen0
            delta = pos - cen[:, None]

            # PCA
            sigma = delta * np.sqrt(tan[None, :])
            Sigma = sigma @ sigma.T
            U, S, V = svd(Sigma)

            u1 = U[:, 0]
            u2 = U[:, 1]
            # S[2] should be negligible (planar cross-sections)

            lim1 = np.array([np.min(delta.T @ u1), np.max(delta.T @ u1)])
            lim2 = np.array([np.min(delta.T @ u2), np.max(delta.T @ u2)])
            len1 = lim1[1] - lim1[0]
            len2 = lim2[1] - lim2[0]

            elongation[z] = len1 / len2
        return elongation

    def max(self):
        return np.max(self.J())

    return_fn_map = {"J": J, "max": max}
