
class ParameterFreeCurve(sgpp.Curve, Curve):

    """ This class can for example be used to describe a magnetic axis. """

    def __init__(self, xyz):
        numquadpoints = xyz.shape[0]
        quadpoints = np.linspace(0, 1, numquadpoints, endpoint=False)
        sgpp.Curve.__init__(self, quadpoints)
        Curve.__init__(self)
        self.xyz

    def num_dofs(self):
        return 2*self.order+1

    def get_dofs(self):
        return np.concatenate(self.coefficients)

    def set_dofs_impl(self, dofs):
        counter = 0
        for i in range(self.order+1):
            self.coefficients[0][i] = dofs[i]
        for i in range(self.order):
            self.coefficients[1][i] = dofs[self.order + 1 + i]
        for d in self.dependencies:
            d.invalidate_cache()
