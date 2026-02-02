import logging
import re

import numpy as np

from simsopt.geo.surface import SurfaceScaled
from .._core.optimizable import Optimizable
from .._core.dev import SimsoptRequires

try:
    import gvec
except ImportError:
    gvec = None

logger = logging.getLogger(__name__)

__all__ = ["GVECSurfaceDoFs"]


class GVECSurfaceDoFs(Optimizable):
    """
    This class is used to represent the Fourier-space dofs of
    a boundary surface for the GVEC optimizable. This class only keeps
    track of the Fourier coefficients and does not translate them to
    real-space coordinates, leaving that task entirely to GVEC.
    This is useful when using GVEC with an underlying coordinate frame
    which is not supported by simsopt (yet), e.g. the G-Frame.

    This class does not presently implement the
    :obj:`simsopt.geo.surface.Surface` interface, e.g. there is not a
    ``gamma()`` or ``to_RZFourier()`` function.

    Args:
        mpol: Maximum poloidal Fourier mode number represented.
        ntor: The maximum toroidal Fourier mode number represented, divided by ``nfp``.
        nfp: Number of field periods.
        stellsym: Whether to use stellarator symmetry.
    """

    def __init__(self, mpol: int, ntor: int, nfp: int = 1, stellsym: bool = True, **kwargs):
        self.mpol = mpol
        self.ntor = ntor
        self.nfp = nfp
        self.stellsym = stellsym
        
        names = self._make_names()
        x0 = np.zeros(len(names))
        super().__init__(names=names, x0=x0, **kwargs)

    def _make_names(self):
        """
        Create the list of names for the dofs.
        """
        names = []
        if self.stellsym:
            for var, sincos in [("X1", "c"), ("X2", "s")]:
                names += self._make_names_helper(var, sincos)
        else:
            for var in ["X1", "X2"]:
                for sincos in ["c", "s"]:
                    names += self._make_names_helper(var, sincos)
        return names
    
    def _make_names_helper(self, var, sincos):
        r"""Generate names with format f'{var}{sincos}({m},{n})'."""
        names = []
        if sincos == "c":
            names.append(f"{var}c(0,0)")
        for n in range(1, self.ntor + 1):
            names.append(f"{var}{sincos}(0,{n})")
        for m in range(1, self.mpol + 1):
            for n in range(-self.ntor, self.ntor + 1):
                names.append(f"{var}{sincos}({m},{n})")
        return names
    
    @staticmethod
    def split_dof_name(name):
        """Split a dof name into its components."""
        var, sc, m, n = re.match(r"(X\d)([cs])\((\-?\d+),(\-?\d+)\)", name).groups()
        sc = "cos" if sc == "c" else "sin"
        return var, sc, int(m), int(n)
    
    def change_resolution(self, mpol, ntor, stellsym=None):
        """
        Increase or decrease the number of degrees of freedom. This
        function is useful for increasing the size of the parameter
        space during stage-1 optimization. If ``mpol`` and ``ntor``
        are increased or unchanged, there is no loss of information.
        If ``mpol`` or ``ntor`` are decreased, information is lost.

        Args:
            mpol: The new maximum poloidal mode number.
            ntor: The new maximum toroidal mode number, divided by ``nfp``.
            stellsym: Whether to use stellarator symmetry. If ``None``,
                the current value of ``stellsym`` is used.
        Returns:
            A new :obj:`GVECSurfaceDoFs` instance with the desired resolution.
        """
        if stellsym is None:
            stellsym = self.stellsym
        surf = self.__class__(mpol, ntor, nfp=self.nfp, stellsym=stellsym)

        # copy over the dofs that exist in both
        # new dofs are initialized to zero, extra dofs are discarded
        for dof in self.local_full_dof_names:
            if dof in surf.local_full_dof_names:
                surf.set(dof, self.get(dof))

        return surf
    
    def autoscale(self) -> SurfaceScaled:
        """
        Create a ``SurfaceScaled`` object wrapping this surface,
        with the same degrees of freedom but scaled with a factor
        max(m^2 + (n * nfp)^2, 1) to equalize the length scales of
        the Fourier coefficients.

        Returns:
            a ``SurfaceScaled`` instance.
        """
        factor = np.zeros(self.local_full_x.size)
        for i, name in enumerate(self.local_full_dof_names):
            var, sc, m, n = self.split_dof_name(name)
            factor[i] = 1 / max(m**2 + (n * self.nfp)**2, 1)
        return SurfaceScaled(self, factor)

    def fixed_range(self, mmin, mmax, nmin, nmax, fixed=True):
        """
        Set the 'fixed' property for a range of `m` and `n` values.

        All modes with `m` in the interval [`mmin`, `mmax`] and `n` in the
        interval [`nmin`, `nmax`] will have their fixed property set to
        the value of the `fixed` parameter. Note that `mmax` and `nmax`
        are included (unlike the upper bound in python's range(min,
        max).)
        """
        if mmin < 0:
            raise ValueError("mmin must be non-negative.")
        if mmax > self.mpol:
            raise ValueError("mmax exceeds maximum poloidal mode number. Change resolution first.")
        if nmax > self.ntor:
            raise ValueError("nmax exceeds maximum toroidal mode number. Change resolution first.")

        fn = self.fix if fixed else self.unfix
        vars = ["X1c", "X2s"] if self.stellsym else ["X1c", "X1s", "X2c", "X2s"]
        for m in range(mmin, mmax + 1):
            if m == 0:
                if nmin <= 0 < nmax:
                    fn("X1c(0,0)")
                    if not self.stellsym:
                        fn("X2c(0,0)")
                for n in range(max(nmin, 1), nmax + 1):
                    for var in vars:
                        fn(f"{var}({m},{n})")
            else:
                for n in range(nmin, nmax + 1):
                    for var in vars:
                        fn(f"{var}({m},{n})")

    @classmethod
    @SimsoptRequires(gvec is not None, "from_gvec_parameters method requires the gvec package")
    def from_gvec_parameters(cls, parameters: dict, **kwargs) -> 'GVECSurfaceDoFs':
        """
        Create a surface from a GVEC parameter dictionary, which does not necessarily use cylindrical
        coordinates. For cylindrical coordinates (``which_hmap=1``), use ``SurfaceRZFourier`` instead.

        Args:
            parameters: GVEC parameter dictionary.
            **kwargs: additional keyword arguments to pass to the ``GVECSurfaceDoFs`` constructor.
        Returns:
            GVECSurfaceDoFs object.
        """
        nfp = parameters["nfp"]
        M = max(parameters["X1_mn_max"][0], parameters["X2_mn_max"][0])
        N = max(parameters["X1_mn_max"][1], parameters["X2_mn_max"][1])
        stellsym = (
            parameters.get("X1_sin_cos", "_cos_") == "_cos_"
            and parameters.get("X2_sin_cos", "_sin_") == "_sin_"
        )

        self = cls(
            nfp=nfp,
            stellsym=stellsym,
            mpol=M,
            ntor=N,
        )

        for xi in ["X1", "X2"]:
            for sincos in ["sin", "cos"]:
                sc = sincos[0]
                for (m, n), value in parameters.get(f"{xi}_b_{sincos}", {}).items():
                    self.set(f"{xi}{sc}({m},{n})", value)

        return self
    
    @SimsoptRequires(gvec is not None, "to_gvec_parameters method requires the gvec package")
    def to_gvec_parameters(self) -> dict:
        """
        Generate a GVEC parameter dictionary representing this surface.

        Returns:
            GVEC parameter dictionary.
        """
        parameters = {}
        parameters["nfp"] = self.nfp
        parameters["init_average_axis"] = True

        for key in ["X1", "X2", "LA"]:
            parameters[f"{key}_mn_max"] = [self.mpol, self.ntor]

        parameters["X1_sin_cos"] = "_cos_" if self.stellsym else "_sin_cos_"
        parameters["X2_sin_cos"] = "_sin_" if self.stellsym else "_sin_cos_"
        parameters["LA_sin_cos"] = "_sin_" if self.stellsym else "_sin_cos_"

        if self.stellsym:
            for key in ["X1_b_cos", "X2_b_sin"]:
                parameters[key] = {}
        else:
            for Xi in ["X1", "X2"]:
                for sincos in ["sin", "cos"]:
                    parameters[f"{Xi}_b_{sincos}"] = {}

        for name in self.local_full_dof_names:
            var, sincos, m, n = self.split_dof_name(name)
            # only set non-zero modes
            if value := self.get(name):
                parameters[f"{var}_b_{sincos}"][m, n] = value
        
        # ensure right-handed (X1,X2,zeta) / counter-clockwise theta
        if not gvec.util.check_boundary_direction(parameters):
            parameters = gvec.util.flip_boundary_theta(parameters)

        return parameters