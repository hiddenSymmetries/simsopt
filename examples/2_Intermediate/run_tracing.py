from simsopt.util.permanent_magnet_helper_functions import *
from simsopt.geo import SurfaceRZFourier
from simsopt._core.optimizable import Optimizable
from mpi4py import MPI
comm = MPI.COMM_WORLD
IN_DIR = 'wv_QA_poincare/'
input_name = 'input.LandremanPaul2021_QA'
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name

nphi = 64  # nphi = ntheta >= 64 needed for accurate full-resolution runs
ntheta = 64
qphi = nphi * 4
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_vmec_input(
    surface_filename, range="full torus",
    quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta
)
bs = Optimizable.from_file(IN_DIR + 'BiotSavart.json')
print(bs.B())
trace_fieldlines(bs, 'poincare_qa', 'qa', s_plot, comm, IN_DIR)
