from math import pi
import numpy as np

from simsopt._core.optimizable import Optimizable
from simsopt._core.derivative import Derivative
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curve import RotatedCurve
import simsoptpp as sopp


__all__ = ['Coil', 'PassiveSuperconductingCoil', 
           'Current', 'coils_via_symmetries', 'psc_coils_via_symmetries',
           'load_coils_from_makegrid_file',
           'apply_symmetries_to_currents', 'apply_symmetries_to_curves',
           'coils_to_makegrid', 'coils_to_focus']


class Coil(sopp.Coil, Optimizable):
    """
    A :obj:`Coil` combines a :obj:`~simsopt.geo.curve.Curve` and a
    :obj:`Current` and is used as input for a
    :obj:`~simsopt.field.biotsavart.BiotSavart` field.
    """

    def __init__(self, curve, current):
        self._curve = curve
        self._current = current
        sopp.Coil.__init__(self, curve, current)
        Optimizable.__init__(self, depends_on=[curve, current])

    def vjp(self, v_gamma, v_gammadash, v_current):
        return self.curve.dgamma_by_dcoeff_vjp(v_gamma) \
            + self.curve.dgammadash_by_dcoeff_vjp(v_gammadash) \
            + self.current.vjp(v_current)

    def plot(self, **kwargs):
        """
        Plot the coil's curve. This method is just shorthand for calling
        the :obj:`~simsopt.geo.curve.Curve.plot()` function on the
        underlying Curve. All arguments are passed to
        :obj:`simsopt.geo.curve.Curve.plot()`
        """
        return self.curve.plot(**kwargs)
    
class PassiveSuperconductingCoil(sopp.Coil, Optimizable):
    """
    A :obj:`Coil` combines a :obj:`~simsopt.geo.curve.Curve` and a
    :obj:`Current` and is used as input for a
    :obj:`~simsopt.field.biotsavart.BiotSavart` field.
    """

    def __init__(self, curve, current, psc_array, index=0):
        self._curve = curve
        self._index = index
        names = curve.local_dof_names
        try: 
            # self._grid_xyz = self._curve.get_dofs()[-3:]
            # self.setup_full_grid()
            # Fix the curve shape and center point coordinate
            # print(curve.local_dof_names)
            for i in range(2 * self._curve.order + 1):
                self._curve.fix(names[i])
            self._curve.fix(names[2 * self._curve.order + 5])
            self._curve.fix(names[2 * self._curve.order + 6])
            self._curve.fix(names[2 * self._curve.order + 7])
            # print(curve.local_dof_names)
        except:
            'do nothing'
        self._current = current
        self._current.fix_all()  # currents cannot be freely optimized
        self._psc_array = psc_array
        # self._bs = bs
        # self._TF_coils = bs._coils
        # self.I_TF = np.array([coil.current.get_value() for coil in self._TF_coils])
        # self.dl_TF = np.array([coil.curve.gammadash() for coil in self._TF_coils])
        # self.gamma_TF = np.array([coil.curve.gamma() for coil in self._TF_coils])
        self.npsc = self._psc_array.num_psc
        sopp.Coil.__init__(self, curve, current)
        Optimizable.__init__(self, depends_on=[curve, current])
        
    # def setup_full_grid(self):
    #     """
    #     Initialize the field-period and stellarator symmetrized grid locations
    #     and normal vectors for all the PSC coils. Note that when alpha_i
    #     and delta_i angles change, coil_i location does not change, only
    #     its normal vector changes. 
    #     """
    #     nn = 1
    #     contig = np.ascontiguousarray
    #     self._grid_xyz_all = np.zeros((self._psc_array.symmetry, 3))
    #     q = 0
    #     for fp in range(self._psc_array.nfp):
    #         for stell in self._psc_array.stell_list:
    #             phi0 = (2 * np.pi / self._psc_array.nfp) * fp
    #             # get new locations by flipping the y and z components, then rotating by phi0
    #             self._grid_xyz_all[nn * q: nn * (q + 1), 0] = self._grid_xyz[0] * np.cos(phi0) - self._grid_xyz[1] * np.sin(phi0) * stell
    #             self._grid_xyz_all[nn * q: nn * (q + 1), 1] = self._grid_xyz[0] * np.sin(phi0) + self._grid_xyz[1] * np.cos(phi0) * stell
    #             self._grid_xyz_all[nn * q: nn * (q + 1), 2] = self._grid_xyz[2] * stell
    #             q += 1
    #     self._grid_xyz_all = contig(self._grid_xyz_all)

    def vjp(self, v_gamma, v_gammadash, v_current):
        # print(v_current, v_current[0])
        # print(self.curve.local_dof_names)
        # print(self.curve.dgammadash_by_dcoeff_vjp(v_gammadash), 
              # self.psc_current_contribution_vjp(self.dkappa_dcoef_vjp(v_current)))
        # print(self.curve.dgamma_by_dcoeff_vjp(v_gamma),
        #       self.curve.dgammadash_by_dcoeff_vjp(v_gammadash),
        #       self.psc_current_contribution_vjp(self.dkappa_dcoef_vjp(v_current)))
        # print('here ', self.dkappa_dcoef_vjp(v_current), self.psc_current_contribution_vjp(self.dkappa_dcoef_vjp(v_current)))
        return self.psc_current_contribution_vjp(self.dkappa_dcoef_vjp(v_current))
            
    def psc_current_contribution_vjp(self, v_current):
        # self.update_variables()
        indices = np.hstack((self._index, self._index + self.npsc))
        Linv_partial = self._psc_array.L_inv[self._index, self._index]
        Linv = np.vstack((Linv_partial, Linv_partial)).T
        psi_deriv = self._psc_array.dpsi[indices]
        # print('dpsi = ', -Linv, psi_deriv, v_current, np.ravel((-Linv * psi_deriv) @ v_current))
        # print(Linv.shape, indices, psi_deriv.shape, v_current.shape, ((-Linv * psi_deriv) @ v_current).shape)
        return Derivative({self: np.ravel((-Linv * psi_deriv) @ v_current).tolist() })  #  np.ravel((-Linv * psi_deriv) @ v_current).tolist()  # 
    
    def dkappa_dcoef_vjp(self, v_current):
        dofs = self._curve.get_dofs()  # should already be only the orientation dofs
        dofs = dofs[2 * self._curve.order + 1:2 * self._curve.order + 5]  # don't need the coordinate variables
        # print('dofs = ', dofs)
        dofs = dofs / np.sqrt(np.sum(dofs ** 2))  # normalize the quaternion
        w = dofs[0]
        x = dofs[1]
        y = dofs[2]
        z = dofs[3]
        # alphas = 2.0 * np.arcsin(np.sqrt(dofs[:, 1] ** 2 + dofs[:, 3] ** 2))
        # deltas = 2.0 * np.arccos(np.sqrt(dofs[:, 0] ** 2 + dofs[:, 1] ** 2))
        # dofs[3] = cos(alpha / 2.0) * cos(delta / 2.0)
        # dofs[4] = sin(alpha / 2.0) * cos(delta / 2.0)
        # dofs[5] = cos(alpha / 2.0) * sin(delta / 2.0)
        # dofs[6] = -sin(alpha / 2.0) * sin(delta / 2.0)
        dalpha_dw = (2 * x * (2 * (x ** 2 + y ** 2) - 1)) / \
            (4 * (w * x + y * z) ** 2 + (1 - 2 * ( x ** 2 + y ** 2)) ** 2)
        dalpha_dx = (w * (-0.5 - x ** 2 + y ** 2) - 2 * x * y * z) / \
            (x ** 2 * (w ** 2 + 2 * y ** 2 - 1) + 2 * w * x * y * z + x ** 4 + y ** 4 + y ** 2 * (z ** 2 - 1) + 0.25)
        dalpha_dy = (z * (-0.5 + x ** 2 - y ** 2) - 2 * x * y * w) / \
            (x ** 2 * (w ** 2 + 2 * y ** 2 - 1) + 2 * w * x * y * z + x ** 4 + y ** 4 + y ** 2 * (z ** 2 - 1) + 0.25)
        dalpha_dz = (2 * y * (2 * (x ** 2 + y ** 2) - 1)) / \
            (4 * (w * x + y * z) ** 2 + (1 - 2 * ( x ** 2 + y ** 2)) ** 2)
        ddelta_dw = -y / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
        ddelta_dx = z / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
        ddelta_dy = -w / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
        ddelta_dz = x / np.sqrt(-(w * y - x * z - 0.5) * (w * y - x * z + 0.5))
        
        dalpha = np.array([dalpha_dw, dalpha_dx, dalpha_dy, dalpha_dz])
        ddelta = np.array([ddelta_dw, ddelta_dx, ddelta_dy, ddelta_dz])
        # dalpha = np.hstack((np.zeros(2 * self._curve.order + 1), np.array([dalpha_dw, dalpha_dx, dalpha_dy, dalpha_dz])))
        # dalpha = np.hstack((dalpha, np.zeros(3)))
        # ddelta = np.hstack((np.zeros(2 * self._curve.order + 1), np.array([ddelta_dw, ddelta_dx, ddelta_dy, ddelta_dz])))
        # ddelta = np.hstack((ddelta, np.zeros(3)))
        deriv = np.vstack((dalpha, ddelta))
        return deriv * v_current[0]  # Derivative({self: deriv})
    
    # def update_variables(self):
    #     dofs = self._curve.get_dofs()
    #     dofs = dofs[2 * self._curve.order + 1:]  # don't need the coordinate variables
    #     self.alphas = 2.0 * np.arcsin(np.sqrt(dofs[1] ** 2 + dofs[3] ** 2))
    #     self.deltas = 2.0 * np.arccos(np.sqrt(dofs[0] ** 2 + dofs[1] ** 2))
        # kappas = np.ravel(np.array([self.alphas, self.deltas]))
        
        # self._psc_array.self._index
        
        # self.coil_normals = np.array(
        #     [np.cos(self.alphas) * np.sin(self.deltas),
        #       -np.sin(self.alphas),
        #       np.cos(self.alphas) * np.cos(self.deltas)]
        # ).T
        # # deal with -0 terms in the normals, which screw up the arctan2 calculations
        # self.coil_normals[
        #     np.logical_and(np.isclose(self.coil_normals, 0.0), 
        #                    np.copysign(1.0, self.coil_normals) < 0)
        #     ] *= -1.0
                
        # Apply discrete symmetries to the alphas and deltas and coordinates
        # self.update_alphas_deltas()
        
        ##### NOOOOOOOO
        # Recompute the inductance matrices with the newly rotated coils
        # L_total = sopp.L_matrix(
        #     self._psc_array._grid_xyz_all / self._psc_array.R, 
        #     self._psc_array.alphas_total, 
        #     self._psc_array.deltas_total,
        #     self._psc_array.quad_points_phi,
        #     self._psc_array.quad_weights
        # )
        # L_total = (L_total + L_total.T)
        # # Add in self-inductances
        # np.fill_diagonal(L_total, (np.log(8.0 * self._psc_array.R / self._psc_array.a) - 2.0) * 4 * np.pi)
        # self.L = L_total * self._psc_array.R
        # Linv = self._psc_array.L_inv[:self.npsc, :self.npsc]
        # self.update_psi()
        # self.I = -self.L_inv[:self.npsc, :] @ self.psi_total * 1e7
        
    # def update_alphas_deltas(self):
    #     """
    #     Initialize the field-period and stellarator symmetrized normal vectors
    #     for all the PSC coils. This is required whenever the alpha_i
    #     and delta_i angles change.
    #     """
    #     # Made this unnecessarily fast xsimd calculation for setting the 
    #     # variables from the discrete symmetries
    #     contig = np.ascontiguousarray
    #     (self.coil_normals_all, self.alphas_total, self.deltas_total, 
    #       self.aaprime_aa, self.aaprime_dd, self.ddprime_dd, self.ddprime_aa
    #       ) = sopp.update_alphas_deltas_xsimd(
    #           contig(self.coil_normals), 
    #           self._psc_array.nfp, 
    #           int(self._psc_array.stellsym)
    #     )
    #     self.alphas_total = contig(self.alphas_total)
    #     self.deltas_total = contig(self.deltas_total)
    #     self.coil_normals_all = contig(self.coil_normals_all)
        
    # def psi_deriv(self):
    #     """
    #     Should return gradient of the inductance matrix L that satisfies 
    #     L^(-1) * Psi = I for the PSC arrays.
    #     Returns
    #     -------
    #         grad_psi: 1D numpy array, shape (2 * num_psc) 
    #             The gradient of the psi vector with respect to the PSC angles
    #             alpha_i and delta_i. 
    #     """
    #     nn = self.npsc
    #     contig = np.ascontiguousarray
    #     psi_deriv = np.zeros(2 * nn)
    #     q = 0
    #     for fp in range(self._psc_array.nfp):
    #         for stell in self._psc_array.stell_list:
    #             dpsi = sopp.dpsi_dkappa(
    #                 contig(self.I_TF),
    #                 contig(self.dl_TF),
    #                 contig(self.gamma_TF),
    #                 contig(self._grid_xyz_all[q * nn: (q + 1) * nn, :]),
    #                 contig(self.alphas_total[q * nn: (q + 1) * nn]),
    #                 contig(self.deltas_total[q * nn: (q + 1) * nn]),
    #                 contig(self.coil_normals_all[q * nn: (q + 1) * nn, :]),
    #                 contig(self._psc_array.quad_points_rho),
    #                 contig(self._psc_array.quad_points_phi),
    #                 self._psc_array.quad_weights,
    #                 self._psc_array.R,
    #             ) 
    #             psi_deriv[:nn] += dpsi[:nn] * self.aaprime_aa[q * nn:(q + 1) * nn] + dpsi[nn:] * self.ddprime_aa[q * nn:(q + 1) * nn]
    #             psi_deriv[nn:] += dpsi[:nn] * self.aaprime_dd[q * nn:(q + 1) * nn] + dpsi[nn:] * self.ddprime_dd[q * nn:(q + 1) * nn]
    #             q += 1
    #     return psi_deriv * (1.0 / self.gamma_TF.shape[1]) / self._psc_array.nfp / (self._psc_array.stellsym + 1.0)  # Factors because TF fields get overcounted
    
    # def update_psi(self):
    #     """
    #     Update the flux grid with the new normal vectors.
    #     """
    #     contig = np.ascontiguousarray
    #     flux_grid = sopp.flux_xyz(
    #         contig(self._grid_xyz_all), 
    #         contig(self.alphas_total),
    #         contig(self.deltas_total), 
    #         contig(self._psc_array.quad_points_rho), 
    #         contig(self._psc_array.quad_points_phi), 
    #     )
    #     flux_grid = np.array(flux_grid).reshape(-1, 3)
    #     self._bs.set_points(contig(flux_grid))
    #     N = len(self._psc_array.quad_points_rho)
    #     # # Update the flux values through the newly rotated coils
    #     self.psi_total = sopp.flux_integration(
    #         contig(self._bs.B().reshape(len(self.alphas_total), N, N, 3)),
    #         contig(self._psc_array.quad_points_rho),
    #         contig(self.coil_normals_all),
    #         self._psc_array.quad_weights
    #     )
    #     self.psi = self.psi_total[:self.npsc]
    #     self._bs.set_points(self._psc_array.plasma_points)

    def plot(self, **kwargs):
        """
        Plot the coil's curve. This method is just shorthand for calling
        the :obj:`~simsopt.geo.curve.Curve.plot()` function on the
        underlying Curve. All arguments are passed to
        :obj:`simsopt.geo.curve.Curve.plot()`
        """
        return self.curve.plot(**kwargs)


class CurrentBase(Optimizable):

    def __init__(self, **kwargs):
        Optimizable.__init__(self, **kwargs)

    def __mul__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        return ScaledCurrent(self, other)

    def __rmul__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        return ScaledCurrent(self, other)

    def __truediv__(self, other):
        assert isinstance(other, float) or isinstance(other, int)
        return ScaledCurrent(self, 1.0/other)

    def __neg__(self):
        return ScaledCurrent(self, -1.)

    def __add__(self, other):
        return CurrentSum(self, other)

    def __sub__(self, other):
        return CurrentSum(self, -other)

    # https://stackoverflow.com/questions/11624955/avoiding-python-sum-default-start-arg-behavior
    def __radd__(self, other):
        # This allows sum() to work (the default start value is zero)
        if other == 0:
            return self
        return self.__add__(other)


class Current(sopp.Current, CurrentBase):
    """
    An optimizable object that wraps around a single scalar degree of
    freedom. It represents the electric current in a coil, or in a set
    of coils that are constrained to use the same current.
    """

    def __init__(self, current, dofs=None, **kwargs):
        sopp.Current.__init__(self, current)
        if dofs is None:
            CurrentBase.__init__(self, external_dof_setter=sopp.Current.set_dofs,
                                 x0=self.get_dofs(), **kwargs)
        else:
            CurrentBase.__init__(self, external_dof_setter=sopp.Current.set_dofs,
                                 dofs=dofs, **kwargs)

    def vjp(self, v_current):
        return Derivative({self: v_current})

    @property
    def current(self):
        return self.get_value()


class ScaledCurrent(sopp.CurrentBase, CurrentBase):
    """
    Scales :mod:`Current` by a factor. To be used for example to flip currents
    for stellarator symmetric coils.
    """

    def __init__(self, current_to_scale, scale, **kwargs):
        self.current_to_scale = current_to_scale
        self.scale = scale
        sopp.CurrentBase.__init__(self)
        CurrentBase.__init__(self, depends_on=[current_to_scale], **kwargs)

    def vjp(self, v_current):
        return self.scale * self.current_to_scale.vjp(v_current)

    def get_value(self):
        return self.scale * self.current_to_scale.get_value()


class CurrentSum(sopp.CurrentBase, CurrentBase):
    """
    Take the sum of two :mod:`Current` objects.
    """

    def __init__(self, current_a, current_b):
        self.current_a = current_a
        self.current_b = current_b
        sopp.CurrentBase.__init__(self)
        CurrentBase.__init__(self, depends_on=[current_a, current_b])

    def vjp(self, v_current):
        return self.current_a.vjp(v_current) + self.current_b.vjp(v_current)

    def get_value(self):
        return self.current_a.get_value() + self.current_b.get_value()


def apply_symmetries_to_curves(base_curves, nfp, stellsym):
    """
    Take a list of ``n`` :mod:`simsopt.geo.curve.Curve`s and return ``n * nfp *
    (1+int(stellsym))`` :mod:`simsopt.geo.curve.Curve` objects obtained by
    applying rotations and flipping corresponding to ``nfp`` fold rotational
    symmetry and optionally stellarator symmetry.
    """
    flip_list = [False, True] if stellsym else [False]
    curves = []
    for k in range(0, nfp):
        for flip in flip_list:
            for i in range(len(base_curves)):
                if k == 0 and not flip:
                    curves.append(base_curves[i])
                else:
                    rotcurve = RotatedCurve(base_curves[i], 2*pi*k/nfp, flip)
                    curves.append(rotcurve)
    return curves


def apply_symmetries_to_currents(base_currents, nfp, stellsym):
    """
    Take a list of ``n`` :mod:`Current`s and return ``n * nfp * (1+int(stellsym))``
    :mod:`Current` objects obtained by copying (for ``nfp`` rotations) and
    sign-flipping (optionally for stellarator symmetry).
    """
    flip_list = [False, True] if stellsym else [False]
    currents = []
    for k in range(0, nfp):
        for flip in flip_list:
            for i in range(len(base_currents)):
                current = ScaledCurrent(base_currents[i], -1.) if flip else base_currents[i]
                currents.append(current)
    return currents


def coils_via_symmetries(curves, currents, nfp, stellsym):
    """
    Take a list of ``n`` curves and return ``n * nfp * (1+int(stellsym))``
    ``Coil`` objects obtained by applying rotations and flipping corresponding
    to ``nfp`` fold rotational symmetry and optionally stellarator symmetry.
    """

    assert len(curves) == len(currents)
    curves = apply_symmetries_to_curves(curves, nfp, stellsym)
    currents = apply_symmetries_to_currents(currents, nfp, stellsym)
    coils = [Coil(curv, curr) for (curv, curr) in zip(curves, currents)]
    return coils


def psc_coils_via_symmetries(curves, currents, nfp, stellsym, psc_array):
    """
    Take a list of ``n`` curves and return ``n * nfp * (1+int(stellsym))``
    ``Coil`` objects obtained by applying rotations and flipping corresponding
    to ``nfp`` fold rotational symmetry and optionally stellarator symmetry.
    """

    assert len(curves) == len(currents)
    curves = apply_symmetries_to_curves(curves, nfp, stellsym)
    currents = apply_symmetries_to_currents(currents, nfp, stellsym)
    inds = np.arange(psc_array.num_psc)
    coils = [PassiveSuperconductingCoil(curv, curr, psc_array, ind) for (curv, curr, ind) in zip(curves, currents, inds)]
    return coils


def load_coils_from_makegrid_file(filename, order, ppp=20):
    """
    This function loads a file in MAKEGRID input format containing the Cartesian coordinates 
    and the currents for several coils and returns an array with the corresponding coils. 
    The format is described at
    https://princetonuniversity.github.io/STELLOPT/MAKEGRID

    Args:
        filename: file to load.
        order: maximum mode number in the Fourier expansion.
        ppp: points-per-period: number of quadrature points per period.

    Returns:
        A list of ``Coil`` objects with the Fourier coefficients and currents given by the file.
    """
    with open(filename, 'r') as f:
        all_coils_values = f.read().splitlines()[3:] 

    currents = []
    flag = True
    for j in range(len(all_coils_values)-1):
        vals = all_coils_values[j].split()
        if flag:
            currents.append(float(vals[3]))
            flag = False
        if len(vals) > 4:
            flag = True

    curves = CurveXYZFourier.load_curves_from_makegrid_file(filename, order=order, ppp=ppp)
    coils = [Coil(curves[i], Current(currents[i])) for i in range(len(curves))]

    return coils


def coils_to_makegrid(filename, curves, currents, groups=None, nfp=1, stellsym=False):
    """
    Export a list of Curve objects together with currents in MAKEGRID input format, so they can 
    be used by MAKEGRID and FOCUS. The format is introduced at
    https://princetonuniversity.github.io/STELLOPT/MAKEGRID
    Note that this function does not generate files with MAKEGRID's *output* format.

    Args:
        filename: Name of the file to write.
        curves: A python list of Curve objects.
        currents: Coil current of each curve.
        groups: Coil current group. Coils in the same group will be assembled together. Defaults to None.
        nfp: The number of field periodicity. Defaults to 1.
        stellsym: Whether or not following stellarator symmetry. Defaults to False.
    """

    assert len(curves) == len(currents)
    coils = coils_via_symmetries(curves, currents, nfp, stellsym)
    ncoils = len(coils)
    if groups is None:
        groups = np.arange(ncoils) + 1
    else:
        assert len(groups) == ncoils
        # should be careful. SIMSOPT flips the current, but actually should change coil order
    with open(filename, "w") as wfile:
        wfile.write("periods {:3d} \n".format(nfp)) 
        wfile.write("begin filament \n")
        wfile.write("mirror NIL \n")
        for icoil in range(ncoils):
            x = coils[icoil].curve.gamma()[:, 0]
            y = coils[icoil].curve.gamma()[:, 1]
            z = coils[icoil].curve.gamma()[:, 2]
            for iseg in range(len(x)):  # the last point matches the first one;
                wfile.write(
                    "{:23.15E} {:23.15E} {:23.15E} {:23.15E}\n".format(
                        x[iseg], y[iseg], z[iseg], coils[icoil].current.get_value()
                    )
                )
            wfile.write(
                "{:23.15E} {:23.15E} {:23.15E} {:23.15E} {:} {:10} \n".format(
                    x[0], y[0], z[0], 0.0, groups[icoil], coils[icoil].curve.name
                )
            )
        wfile.write("end \n")
    return


def coils_to_focus(filename, curves, currents, nfp=1, stellsym=False, Ifree=False, Lfree=False):
    """
    Export a list of Curve objects together with currents in FOCUS format, so they can 
    be used by FOCUS. The format is introduced at
    https://princetonuniversity.github.io/FOCUS/rdcoils.pdf
    This routine only works with curves of type CurveXYZFourier,
    not other curve types.

    Args:
        filename: Name of the file to write.
        curves: A python list of CurveXYZFourier objects.
        currents: Coil current of each curve.
        nfp: The number of field periodicity. Defaults to 1.      
        stellsym: Whether or not following stellarator symmetry. Defaults to False.
        Ifree: Flag specifying whether the coil current is free. Defaults to False.
        Lfree: Flag specifying whether the coil geometry is free. Defaults to False.
    """
    from simsopt.geo import CurveLength

    assert len(curves) == len(currents)
    ncoils = len(curves)
    if stellsym:
        symm = 2  # both periodic and symmetric
    elif nfp > 1 and not stellsym:
        symm = 1  # only periodicity
    else:
        symm = 0  # no periodicity or symmetry
    if nfp > 1:
        print('Please note: FOCUS sets Nfp in the plasma file.')
    with open(filename, 'w') as f:
        f.write('# Total number of coils \n')
        f.write('  {:d} \n'.format(ncoils))
        for i in range(ncoils):
            assert isinstance(curves[i], CurveXYZFourier)
            nf = curves[i].order
            xyz = curves[i].full_x.reshape((3, -1))
            xc = xyz[0, ::2]
            xs = np.concatenate(([0.], xyz[0, 1::2]))
            yc = xyz[1, ::2]
            ys = np.concatenate(([0.], xyz[1, 1::2]))
            zc = xyz[2, ::2]
            zs = np.concatenate(([0.], xyz[2, 1::2]))
            length = CurveLength(curves[i]).J()
            nseg = len(curves[i].quadpoints)
            f.write('#------------{:d}----------- \n'.format(i+1))
            f.write('# coil_type  symm  coil_name \n')
            f.write('  {:d}   {:d}  {:} \n'.format(1, symm, curves[i].name))
            f.write('# Nseg current Ifree Length Lfree target_length \n')
            f.write('  {:d} {:23.15E} {:d} {:23.15E} {:d} {:23.15E} \n'.format(nseg, currents[i].get_value(), Ifree, length, Lfree, length))
            f.write('# NFcoil \n')
            f.write('  {:d} \n'.format(nf))
            f.write('# Fourier harmonics for coils ( xc; xs; yc; ys; zc; zs) \n')
            for r in [xc, xs, yc, ys, zc, zs]:  # 6 lines
                for k in range(nf+1):
                    f.write('{:23.15E} '.format(r[k]))
                f.write('\n')
        f.write('\n')
    return
