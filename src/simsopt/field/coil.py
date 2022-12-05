from math import pi
import numpy as np

from simsopt._core.optimizable import Optimizable
from simsopt._core.derivative import Derivative
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curve import RotatedCurve, Curve
import simsoptpp as sopp


__all__ = ['Coil', 'Current', 'coils_via_symmetries',
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
