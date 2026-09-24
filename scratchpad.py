from simsopt.geo import CurveXYZFourier
from simsopt._core import Optimizable

class BunchOfCurves(Optimizable):
    """
    Create an optimizable set of curves, given a list of the orders
    of each curve, or a list of curves. 
    """
    def __init__(self, order_list, curve_list=None):
        if curve_list is None:
            curve_list=[]
            for order in order_list:
                curve = CurveXYZFourier(quadpoints=30, order=order)
                curve_list.append(curve)
        Optimizable.__init__(self, depends_on=curve_list)

orderlist = [1, 2, 2, 2, 1]
curveset0 = BunchOfCurves(order_list=orderlist)
curveset1 = BunchOfCurves(order_list=orderlist)
curveset1.x = curveset0.x # does not 
print(curveset1.dof_names)