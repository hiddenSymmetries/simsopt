from math import pi
import numpy as np
from simsopt.geo.curve import RotatedCurve


class CoilCollection():
    """
    Given some input coils and currents, this performs the reflection and
    rotation to generate a full set of stellarator coils.
    """

    def __init__(self, coils, currents, nfp, stellarator_symmetry):
        self._base_coils = coils
        self._base_currents = currents
        self.coils = []
        self.currents = []
        flip_list = [False, True] if stellarator_symmetry else [False]
        self.map = []
        self.current_sign = []
        for k in range(0, nfp):
            for flip in flip_list:
                for i in range(len(coils)):
                    if k == 0 and not flip:
                        self.coils.append(self._base_coils[i])
                        self.currents.append(self._base_currents[i])
                    else:
                        rotcoil = RotatedCurve(coils[i], 2*pi*k/nfp, flip)
                        self.coils.append(rotcoil)
                        self.currents.append(-self._base_currents[i] if flip else currents[i])
                    self.map.append(i)
                    self.current_sign.append(-1 if flip else +1)
        dof_ranges = [(0, len(self._base_coils[0].get_dofs()))]
        for i in range(1, len(self._base_coils)):
            dof_ranges.append((dof_ranges[-1][1], dof_ranges[-1][1] + len(self._base_coils[i].get_dofs())))
        self.dof_ranges = dof_ranges

    def set_dofs(self, dofs):
        assert len(dofs) == self.dof_ranges[-1][1]
        for i in range(len(self._base_coils)):
            self._base_coils[i].set_dofs(dofs[self.dof_ranges[i][0]:self.dof_ranges[i][1]])

    def get_dofs(self):
        return np.concatenate([coil.get_dofs() for coil in self._base_coils])

    def reduce_coefficient_derivatives(self, derivatives, axis=0):
        """
        Add derivatives for all those coils that were obtained by rotation and
        reflection of the initial coils.
        """
        assert len(derivatives) == len(self.coils) or len(derivatives) == len(self._base_coils)
        res = len(self._base_coils) * [None]
        for i in range(len(derivatives)):
            if res[self.map[i]] is None:
                res[self.map[i]]  = derivatives[i]
            else:
                res[self.map[i]] += derivatives[i]
        return np.concatenate(res, axis=axis)
