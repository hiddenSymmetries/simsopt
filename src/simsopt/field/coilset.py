import numpy as np
import re

from simsopt._core.optimizable import DOFs, Optimizable
from .biotsavart import BiotSavart
from simsopt.geo import (SurfaceRZFourier, Surface, CurveXYZFourier, CurveCurveDistance, CurveSurfaceDistance, LpCurveCurvature,
                         MeanSquaredCurvature, ArclengthVariation,
                         CurveLength)
from simsopt.geo import plot, curves_to_vtk, create_equally_spaced_curves
from simsopt._core.finite_difference import FiniteDifference
from simsopt._core import make_optimizable
from simsopt.objectives import SquaredFlux, QuadraticPenalty
from .coil import Coil, coils_via_symmetries, Current, load_coils_from_makegrid_file, coils_to_makegrid

__all__ = ['CoilSet', 'ReducedCoilSet']


class CoilSet(Optimizable):
    """
    A set of coils as a single optimizable object, and a surface on which 
    it evaluates the field.

    The surfaces will be cast as a SurfaceRZFourier, as it will evaluate a field
    on the surface using the RZFourier/VMEC convention. 
    The surfaces range will be adapted to 'half period' or 'field period'
    respectively if it is stellarator symmetric or not.

    The Coilset contains convenience methods to acces coil optimization penalties: 
    flux_penalty, length_penalty, cc_distance_penalty, cs_distance_penalty,
    lp_curvature_penalty, meansquared_curvature_penalty, arc_length_variation_penalty
    and total_length.

    These functions can then be used to define a FOCUS-like optimization problem
    (note that the FOCUS algorithm is not a least-squares algorithm) that can be passed
    to scipy.minimize, or the CoilSet can be used as a parent for a SPEC NormalField
    object. 
    """

    def __init__(self, base_coils=None, coils=None, surface=None):

        #set the surface, change its's range if necessary
        if surface is None:
            self._surface = SurfaceRZFourier().copy(range=SurfaceRZFourier.RANGE_HALF_PERIOD)
        #stellarator symmetric surfaces are more efficiently evaluated on a half-period
        else:
            if surface.stellsym:
                if surface.deduced_range is surface.RANGE_HALF_PERIOD:
                    self._surface = surface
                else:
                    newsurf = surface.to_RZFourier().copy(range=surface.RANGE_HALF_PERIOD)
                    self._surface = newsurf
            else:
                if surface.deduced_range is surface.RANGE_FIELD_PERIOD:
                    self._surface = surface
                else:
                    newsurf = surface.to_RZFourier().copy(range=surface.RANGE_FIELD_PERIOD)
                    self._surface = newsurf

        # set the coils
        if base_coils is not None:
            self._base_coils = base_coils
            if coils is None:
                self.coils = coils_via_symmetries([coil.curve for coil in base_coils], [coil.current for coil in base_coils], nfp=self._surface.nfp, stellsym=self._surface.stellsym)
            else:
                self.coils = coils
        else:
            if coils is not None:
                raise ValueError("If base_coils is None, coils must be None as well")
            base_curves = self._circlecurves_around_surface(self._surface, coils_per_period=10)
            base_currents = [Current(1e5) for _ in base_curves]
            # fix the currents on the default coilset
            for current in base_currents:
                current.fix_all()
            base_coils = [Coil(curv, curr) for (curv, curr) in zip(base_curves, base_currents)]
            self._base_coils = base_coils
            self.coils = coils_via_symmetries([coil.curve for coil in base_coils], [coil.current for coil in base_coils], nfp=self._surface.nfp, stellsym=self._surface.stellsym)

        self.bs = BiotSavart(self.coils)
        self.bs.set_points(self._surface.gamma().reshape((-1, 3)))
        super().__init__(depends_on=base_coils)

    @classmethod
    def for_surface(cls, surf, coil_current=1e5, coils_per_period=5, nfp=None, current_constraint="fix_all", **kwargs):
        """
        Create a CoilSet for a given surface. The coils are created using
        :py:func:`simsopt.geo.create_equally_spaced_curves` with the given parameters.

        The keyword arguments ``coils_per_period``, ``order``, ``R0``, ``R1``,
        ``factor``, and ``use_stellsym`` are passed to the
        :py:func:`~simsopt.geo.create_equally_spaced_curves` function.

        Args:
            surf: The surface for which to create the coils
            total_current: the total current in the CoilSet
            coils_per_period: the number of coils per field period
            nfp: The number of field periods.
            current_constraint: ``"fix_one"`` or ``"fix_all"`` or ``"free_all"``
            coils_per_period: The number of coils per field period
            order: The order of the Fourier expansion
            R0: major radius of a torus on which the initial coils are placed
            R1: The radius of the coils
            factor: if R0 or R1 are None, they are factor times 
                    the first sine/cosine coefficient (factor > 1)
            use_stellsym: Whether to use stellarator symmetry
        """
        if nfp is None:
            nfp = surf.nfp
        base_curves = CoilSet._circlecurves_around_surface(surf, coils_per_period=coils_per_period, **kwargs)
        base_currents = [Current(coil_current) for _ in base_curves]
        if current_constraint == "fix_one":
            base_currents[0].fix_all()
        elif current_constraint == "fix_all":
            [base_current.fix_all() for base_current in base_currents]
        elif current_constraint == "free_all":
            pass
        else:
            raise ValueError("current_constraint must be 'fix_one', 'fix_all' or 'free_all'")
        base_coils = [Coil(curv, curr) for (curv, curr) in zip(base_curves, base_currents)]
        coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym=True)
        return cls(base_coils, coils, surf)

    @classmethod
    def from_makegrid_file(cls, makegrid_file, surface, order=10, ppp=10):
        """
        Create a CoilSet from a MAKEGRID file and a surface.
        """
        coils = load_coils_from_makegrid_file(makegrid_file, order=order, ppp=ppp)
        return cls(base_coils=coils, coils=coils, surface=surface)

    def to_makegrid_file(self, filename):
        """
        Write the CoilSet to a MAKEGRID file
        """
        coils_to_makegrid(filename,
                          [coil.curve for coil in self.coils],
                          [coil.current for coil in self.coils],
                          nfp=1)

    def reduce(self, target_function, nsv='nonzero'):
        """
        Return a ReducedCoilSet based on this CoilSet using the SVD of the mapping
        given by target_function.

        Args:
            target_function: a function that takes a CoilSet and maps to a different space that is relevant for the optimization problem. 
                The SVD will be calculated on the Jacobian of f: coilDOFs -> target.
                For example a function that calculates the fourier coefficients of the
                normal field on the surface. 
            nsv: The number of singular vectors to keep, integer or ``"nonzero"`` (for all nonzero singular values)
        """
        return ReducedCoilSet.from_function(self, target_function, nsv)

    @property
    def surface(self):
        return self._surface

    @surface.setter
    def surface(self, surface: 'Surface'):
        """
        set a new surface for the CoilSet. Change its range if necessary.
        """
        # Changing surface requires changing the BiotSavart object. We also modify its range
        if surface.nfp != self.surface.nfp:
            raise ValueError("nfp must be equal to the current surface nfp")
        if surface.stellsym:
            if surface.deduced_range is not surface.RANGE_HALF_PERIOD:
                newsurf = surface.to_RZFourier().copy(range=surface.RANGE_HALF_PERIOD)
                surface = newsurf
        else:  # not stellsym
            if surface.deduced_range is not surface.RANGE_FIELD_PERIOD:
                newsurf = surface.to_RZFourier().copy(range=surface.RANGE_FIELD_PERIOD)
                surface = newsurf
        self.bs.set_points(surface.gamma().reshape((-1, 3)))
        self._surface = surface

    @property
    def base_coils(self):
        return self._base_coils

    @base_coils.setter
    def base_coils(self, base_coils):
        for coilparent in self._base_coils:
            self.remove_parent(coilparent)
        self._base_coils = base_coils
        self._coils = coils_via_symmetries([coil.curve for coil in base_coils], [coil.current for coil in base_coils], nfp=self._surface.nfp, stellsym=self._surface.stellsym)
        self.bs = BiotSavart(self._coils)
        self.bs.set_points(self._surface.gamma().reshape((-1, 3)))
        for coilparent in self._base_coils:
            self.append_parent(coilparent)

    @staticmethod
    def _circlecurves_around_surface(surf, coils_per_period=4, order=6, R0=None, R1=None, use_stellsym=None, factor=2.):
        """
        return a set of base curves for a surface using the surfaces properties where possible
        """
        nfp = surf.nfp
        if use_stellsym is None:
            use_stellsym = surf.stellsym
        if R0 is None:
            R0 = surf.to_RZFourier().get_rc(0, 0)
        if R1 is None:
            # take the magnitude of the first-order poloidal Fourier coefficient
            R1 = np.sqrt(surf.to_RZFourier().get_rc(1, 0)**2 + surf.to_RZFourier().get_zs(1, 0)**2) * factor
        return create_equally_spaced_curves(coils_per_period, nfp, stellsym=use_stellsym, R0=R0, R1=R1, order=order)

    def get_dof_orders(self):
        """
        get the Fourier order of the degrees of freedom  corresponding to the Fourier
        coefficients describing the coils. 
        Useful for reducing the trust region for the higher order coefficients.
        in optimization. 

        Returns:
            An array with the fourier order of each degree of freedom.
        """
        dof_names = self.dof_names
        orders = np.zeros(self.dof_size)  # dofs which are not Fourier coefficients are treated as zeroth' order.
        # test if coils are CurveXYZFourier:
        if type(self.coils[0].curve) is not CurveXYZFourier:
            raise ValueError("Coils must be of type CurveXYZFourier")
        # run through names, extract order using regular expression
        for name in dof_names:
            if name.startswith("Curve"):
                match = re.search(r'\((\d+)\)', name)
                if match:
                    order = int(match.group(1))
                    orders[dof_names.index(name)] = order
        return orders

    def flux_penalty(self, target=None):
        """
        Return the penalty function for the quadratic flux penalty on 
        the surface
        """
        target = SquaredFlux(self.surface, self.bs, target=target)
        # make self parent as self's dofs will propagate to coils
        for parent in target.parents:
            target.remove_parent(parent)
        target.append_parent(self)
        return target

    def length_penalty(self, TOTAL_LENGTH, f):
        """
        Return a QuadraticPenalty on the total length of the coils 
        if it is larger than TARGET_LENGTH (do not penalize if shorter)
        Args:
            TOTAL_LENGTH: The threshold length above which the penalty is applied
            f: type of penalty, "min", "max" or "identity"
        """
        # only calculate length of base_coils, others are equal
        coil_multiplication_factor = len(self.coils) / len(self.base_coils)
        # summing optimizables makes new optimizables
        lenth_optimizable = sum(CurveLength(coil.curve) for coil in self.base_coils)*coil_multiplication_factor
        target = QuadraticPenalty(lenth_optimizable, TOTAL_LENGTH, f)
        # make self parent as self's dofs will propagate to coils
        for parent in target.parents:
            target.remove_parent(parent)
        target.append_parent(self)
        # return the penalty function
        return target

    def cc_distance_penalty(self, DISTANCE_THRESHOLD):
        """
        Return a penalty function for the distance between coils
        Args: 
            DISTANCE_THRESHOLD: The threshold distance below which the penalty is applied
        """
        curves = [coil.curve for coil in self.coils]
        target = CurveCurveDistance(curves, DISTANCE_THRESHOLD, num_basecurves=len(self.base_coils))
        # make self parent as self's dofs will propagate to coils
        for parent in target.parents:
            target.remove_parent(parent)
        target.append_parent(self)
        # return the penalty function
        return target

    def cs_distance_penalty(self, DISTANCE_THRESHOLD):
        """
        Return a penalty function for the distance between coils and the surface
        Args:
            DISTANCE_THRESHOLD: The threshold distance below which the penalty is applied
        """
        curves = [coil.curve for coil in self.coils]
        return CurveSurfaceDistance(curves, self.surface, DISTANCE_THRESHOLD)

    def lp_curvature_penalty(self, CURVATURE_THRESHOLD, p=2):
        """
        Return a penalty function on the curvature of the coils that is 
        the Lp norm of the curvature of the coils. Defaults to L2. 
        Args:
            CURVATURE_THRESHOLD: The threshold curvature above which the penalty is applied
            p: The p in the Lp norm for calculating the penalty
        """
        base_curves = [coil.curve for coil in self.base_coils]
        target = sum(LpCurveCurvature(curve, p, CURVATURE_THRESHOLD) for curve in base_curves)
        # make self parent as self's dofs will propagate to coils
        for parent in target.parents:
            target.remove_parent(parent)
        target.append_parent(self)
        # return the penalty function
        return target

    def meansquared_curvature_penalty(self):
        """
        Return a penalty function on the mean squared curvature of the coils
        """
        target = sum(MeanSquaredCurvature(coil.curve) for coil in self.base_coils)
        for parent in target.parents:
            target.remove_parent(parent)
        target.append_parent(self)
        # return the penalty function
        return target

    def meansquared_curvature_threshold(self, CURVATURE_THRESHOLD):
        """
        Return a penalty function on the mean squared curvature of the coils
        """
        meansquaredcurvatures = [MeanSquaredCurvature(coil.curve) for coil in self.base_coils]
        target = sum(QuadraticPenalty(msc, CURVATURE_THRESHOLD, "max") for msc in meansquaredcurvatures)
        for parent in target.parents:
            target.remove_parent(parent)
        target.append_parent(self)
        # return the penalty function
        return target

    def arc_length_variation_penalty(self):
        """
        Return a penalty function on the arc length variation of the coils
        """
        target = sum(ArclengthVariation(coil.curve) for coil in self.base_coils)
        for parent in target.parents:
            target.remove_parent(parent)
        target.append_parent(self)
        # return the penalty function
        return target

    def total_length_penalty(self):
        """
        Return the total length of the coils
        """
        multiplicity = len(self.coils) / len(self.base_coils)
        target = sum(CurveLength(coil.curve) for coil in self.base_coils)*multiplicity
        for parent in target.parents:
            target.remove_parent(parent)
        target.append_parent(self)
        # return the penalty function
        return target

    @property
    def total_length(self):
        """
        Return the total length of the coils
        """
        multiplicity = len(self.coils) / len(self.base_coils)
        lengthoptimizable = sum(CurveLength(coil.curve) for coil in self.base_coils)*multiplicity
        return lengthoptimizable.J()

    def to_vtk(self, filename, add_biotsavart=True, close=False):
        """
        Write the CoilSet and its surface to a vtk file
        """
        curves_to_vtk([coil.curve for coil in self.coils], filename+'_coils', close=close)
        if self.surface is not None:
            if add_biotsavart:
                pointData = {"B_N": np.sum(self.bs.B().reshape((len(self.surface.quadpoints_phi), len(self.surface.quadpoints_theta), 3)) * self.surface.unitnormal(), axis=2)[:, :, None]}
            else:
                pointData = None
            self.surface.to_vtk(filename+'_surface', extra_data=pointData)

    def plot(self, **kwargs):
        """
        Plot the coil's curve. This method is just shorthand for calling
        the :obj:`~simsopt.geo.Curve/Surface.plot()` function on the
        underlying Curve. All arguments are passed to
        :obj:`simsopt.geo.Curve/Surface.plot()`
        """
        #plot the coils, which are already a list:
        plot(self.coils, **kwargs, show=False)
        #generate a full torus for plotting, plot it:
        self.surface.to_RZFourier().copy(range=SurfaceRZFourier.RANGE_FULL_TORUS).plot(**kwargs, show=True)


class ReducedCoilSet(CoilSet):
    """
    An Optimizable that replaces a CoilSet and has as degrees of 
    freedom linear combinations of the CoilSet degrees of freedom 

    Single-stage optimization is often accompanied with an explosion
    in the numbers of degrees of freedom needed to accurately represent
    the stellarator, as each coilneeds 6 DOFs per coil per Fourier order. 
    This makes solving the optimization problem harder in two ways: 
    1: the computational cost of finite-difference gradient evaluation is
    linear in the number of DOFs, and 2: the optimization problem can become 
    rank-deficient: changes to your DOFs cancel each other, and the minmum 
    is no longer unique. 

    This class solves these problems by allowing physics-based dimensionality
    reduction of the optimization space using singluar value-decomposition
    of a fast-evalable map. 

    A ReducedCoilSet consists of a Coilset, and the three elements of an SVD of
    the Jacobian defined by a mapping from the coils' DOFs to a physically relevant, 
    fast-evaluating target.  
    """

    def __init__(self, coilset=None, nsv=None, s_diag=None, u_matrix=None, vh_matrix=None, function=None, **kwargs):
        """
        create a ReducedCoilSet from a CoilSet and the three elements of an SVD of the Jacobian of a mapping. 
        """
        # create standards if initialized with None
        if coilset is None:  # need 'is' (idenity comparison), overloaded __eq__ fails here.
            coilset = CoilSet()
        if nsv is None:
            nsv = coilset.dof_size

        # raise error if any of [s_diag, u_matrix, vh_matrix] are None but not all of them are:
        if any([s_diag is None, u_matrix is None, vh_matrix is None]) \
                and not all([s_diag is None, u_matrix is None, vh_matrix is None]):
            raise ValueError("If any of [s_diag, u_matrix, vh_matrix] are None, all must be None")

        if s_diag is None:
            if nsv > coilset.dof_size:
                raise ValueError("nsv must be equal to or smaller than the coilset's number of DOFs if initializing with None")
            s_diag = np.ones(nsv)
            u_matrix = np.eye(nsv)
            vh_matrix = np.eye(nsv)

        self.function = function
        # Reproduce CoilSet's attributes
        self._coilset = coilset
        self.bs = coilset.bs
        self.bs.set_points(self.surface.gamma().reshape((-1, 3)))
        # Add the SVD matrices
        self._u_matrix = u_matrix
        self._vh_matrix = vh_matrix
        self._coil_x0 = np.copy(self.coilset.x)  # DID NOT COPY BEFORE, CHECK CONVERGENCE
        self._s_diag = s_diag
        if nsv > len(s_diag):
            raise ValueError("nsv must be equal to or smaller than the number of singular values")
        self._nsv = nsv
        Optimizable.__init__(self, x0=np.zeros_like(self._s_diag[:nsv]), names=self._make_names(), external_dof_setter=ReducedCoilSet.set_dofs, **kwargs)

    @classmethod
    def from_function(cls, coilset, target_function, nsv='nonzero'):
        """
        Reduce an existing CoilSet to a ReducedCoilSet using the SVD of the mapping
        given by target_function.

        Args:
            coilset: The CoilSet to reduce
            target_function: a function that takes a CoilSet and maps to a different space that is relevant for the optimization problem. 
                The SVD will be calculated on the Jacobian of f: coilDOFs -> target
                For example a function that calculates the fourier coefficients of the
                normal field on the surface. 
            nsv: The number of singular values to keep, either an integer or ``"nonzero"``. defaults to ``"nonzero"``.
                If an integer is given, this number of largest singular values will be kept.
                If ``"nonzero"``, all nonzero singular values are kept.
        """
        optfunction = make_optimizable(target_function, coilset)
        fd = FiniteDifference(optfunction.J)
        jaccers = fd.jac()
        u_matrix, s_diag, vh_matrix = np.linalg.svd(jaccers)
        if nsv == 'nonzero':
            nsv = len(s_diag)
        return cls(coilset, nsv, s_diag, u_matrix, vh_matrix, target_function)

    def recalculate_reduced_basis(self, target_function=None):
        """
        Recalculate the reduced basis using a new target function. 
        """
        thisfunction = target_function or self.function
        if thisfunction is None:
            raise ValueError("No target function provided")
        self.function = thisfunction
        optfunction = make_optimizable(thisfunction, self.coilset)
        fd = FiniteDifference(optfunction.J)
        jaccers = fd.jac()
        u_matrix, s_diag, vh_matrix = np.linalg.svd(jaccers)
        self.nsv = min(self.nsv, len(s_diag))
        self._u_matrix = u_matrix
        self._s_diag = s_diag
        self._vh_matrix = vh_matrix
        self._coil_x0 = np.copy(self.coilset.x)
        x = np.zeros(self.nsv)
        self.replace_dofs(DOFs(x, self._make_names()))

    @property
    def coilset(self):
        return self._coilset

    @coilset.setter
    def coilset(self, coilset):
        raise ValueError("You cannot change the CoilSet of a ReducedCoilSet.")

    @property
    def base_coils(self):
        return self.coilset.base_coils

    @base_coils.setter
    def base_coils(self, base_coils):
        self.coilset.base_coils = base_coils
        self.recalculate_reduced_basis(self.function)

    @property
    def coils(self):
        return self.coilset.coils

    @coils.setter
    def coils(self, *args, **kwargs):
        raise ValueError("You cannot change the coils of a ReducedCoilSet. Change the base_coils instead.")

    @property
    def surface(self):
        return self._coilset.surface

    @surface.setter
    def surface(self, surface: 'Surface'):
        """
        setting a new surface requires re-calculating
        the SVD matrices. 
        """
        self._coilset.surface = surface
        self.recalculate_reduced_basis()

    @property
    def nsv(self):
        """
        number of singular values used by the ReducedCoilSet
        """
        return self._nsv

    @nsv.setter
    def nsv(self, nsv: int):
        """
        Set the number of singular values to use. 
        args: 
            nsv: the number of singular values to use, int or string 'nonzero'
        """
        if nsv == 'nonzero':
            nsv = len(self._s_diag)
        if nsv > len(self._s_diag):
            raise ValueError("nsv must be equal to or smaller than the number of singular values")
        self._nsv = nsv
        x = np.zeros(nsv)
        x[:len(self.x)] = self.x[:len(x)]
        self.replace_dofs(DOFs(x, self._make_names()))

    @property
    def rsv(self):
        """
        right-singular vectors of the svd
        """
        return [np.array(self._vh_matrix[i, :]) for i in range(self.nsv)]

    @property
    def lsv(self):
        """
        left-singular vectors of the svd
        """
        return [np.array(self._u_matrix.T[i, :]) for i in range(self.nsv)]

    @property
    def singular_values(self):
        """
        singular values of the svd
        """
        return self._s_diag[:self.nsv]

    def get_dof_orders(self):
        """
        Not available for a ReducedCoilSet
        """
        raise ValueError("Not available for a ReducedCoilSet")

    def _make_names(self):
        names = [f"sv{n}" for n in range(len(self._s_diag[:self.nsv]))]
        return names

    def set_dofs(self, x):
        #check if correct!
        if len(x) != self.nsv:
            raise ValueError("Wrong number of DOFs")
        padx = np.pad(x, (0, len(self._coil_x0)-self.nsv), mode='constant', constant_values=0)  # create zero-padded vector of length of coil DOFs.
        self.coilset.x = self._coil_x0 + (padx) @ self._vh_matrix  # translate reduced vectors amplitude to movement of coils.

    def plot_singular_vector(self, n, eps=1e-4, show_delta_B=True, engine='mayavi', show=False, **kwargs):
        """
        Plot the n-th singular vector. 
        args: 
            n: the index of the singular vector
            eps: the magnitude of the displacement. 
            show_delta_B: whether to show the change in the magnetic field
            engine: the plotting engine to use. Defaults to 'mayavi'
        """
        if engine == 'mayavi':
            from mayavi import mlab
            if n > self.nsv:
                raise ValueError("n must be smaller than the number of singular values")
            singular_vector = np.array(self.rsv[n])

            plotsurf = self.surface.copy(range=SurfaceRZFourier.RANGE_FULL_TORUS)
            if show_delta_B:
                bs = self.bs
                initial_points = self.bs.get_points_cart_ref()
                bs.set_points(plotsurf.gamma().reshape((-1, 3)))

            current_x = np.copy(self.x)
            startpositions = [np.copy(coil.curve.gamma()) for coil in self.coilset.coils]
            plot(self.coilset.coils, close=True, engine='mayavi', tube_radius=0.02, color=(0, 0, 0), show=False, **kwargs)
            if show_delta_B:
                startB = np.copy(np.sum(bs.B().reshape((plotsurf.quadpoints_phi.size, plotsurf.quadpoints_theta.size, 3)) * plotsurf.unitnormal()*-1, axis=2))
                startB = np.concatenate((startB, startB[:1, :]), axis=0)
                startB = np.concatenate((startB, startB[:, :1]), axis=1)

            # Perturb the coils by the singular vector
            self.coilset.x = self.coilset.x + singular_vector*eps
            newpositions = [np.copy(coil.curve.gamma()) for coil in self.coilset.coils]
            if show_delta_B:
                changedB = np.copy(np.sum(bs.B().reshape((plotsurf.quadpoints_phi.size, plotsurf.quadpoints_theta.size, 3)) * plotsurf.unitnormal()*-1, axis=2))
                # close the plot
                changedB = np.concatenate((changedB, changedB[:1, :]), axis=0)
                changedB = np.concatenate((changedB, changedB[:, :1]), axis=1)
            # plot the displacement vectors
            for newcoilpos, startcoilpos in zip(newpositions, startpositions):
                diffs = (0.05/eps) * (startcoilpos - newcoilpos)
                x = startcoilpos[:, 0]
                y = startcoilpos[:, 1]
                z = startcoilpos[:, 2]
                # enlarge the difference vectors for better visibility
                dx = diffs[:, 0]
                dy = diffs[:, 1]
                dz = diffs[:, 2]
                mlab.quiver3d(x, y, z, dx, dy, dz, line_width=4, **kwargs)

            if show_delta_B:
                plot([plotsurf,], engine='mayavi', wireframe=False, close=True, scalars=changedB-startB, colormap='Reds', show=False, **kwargs)
            else:
                plot([plotsurf,], engine='mayavi', wireframe=False, close=True, colormap='Reds', show=False, **kwargs)
            # plot the original coils again
            self.x = current_x
            plot(self.coilset.coils, close=True, tube_radius=0.02, engine='mayavi', color=(1, 1, 1), show=show, **kwargs)
            # set bs set points back
            if show_delta_B:
                bs.set_points(initial_points)
            return changedB, startB
        else:
            raise ValueError("plotting a ReducedCoilSet is only implemented in Mayavi. you can access and plot the surface and coils attributes directly.")
