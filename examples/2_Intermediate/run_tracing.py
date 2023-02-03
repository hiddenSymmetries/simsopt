from simsopt.util.permanent_magnet_helper_functions import *
from simsopt.geo import SurfaceRZFourier
from mpi4py import MPI
comm = MPI.COMM_WORLD
IN_DIR = 'wv_QA/'
input_name = 'input.LandremanPaul2021_QA'
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name

nphi = 32  # nphi = ntheta >= 64 needed for accurate full-resolution runs
ntheta = 32
qphi = s.nfp * nphi * 2
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_vmec_input(
    surface_filename, range="full torus",
    quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta
)
bs = Optimizable.from_file(IN_DIR + 'BiotSavart.json')
trace_fieldlines(bs, 'poincare_qa', 'qa', s_plot, comm, IN_DIR)
