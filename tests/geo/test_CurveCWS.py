from simsopt.geo import CurveCWS
from simsopt.geo import SurfaceRZFourier

filename = "/home/joaobiu/PIC/vmec_equilibria/W7-X/Standard/wout.nc"

s = SurfaceRZFourier.from_wout(filename, range="full torus", ntheta=32, nphi=32)  #range = 'full torus', 'field period', 'half period'
sdofs = s.get_dofs()
print(sdofs)

cws = CurveCWS(s.mpol, s.ntor, sdofs, 20, 0, s.nfp, s.stellsym)

cws.set_dofs([1, 0, 0, 0])

cws.plot()
