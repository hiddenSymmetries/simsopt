from simsopt.mhd.vmec import Vmec
from simsopt.util.mpi import MpiPartition
from simsopt.util import comm_world
from simsopt._core import Optimizable
import time
import sys

mpi = MpiPartition(ngroups=8)
comm = comm_world

# # Make the QFM surfaces
qfm_surf = Optimizable.from_file(str(sys.argv[1]))

# Run VMEC with new QFM surface
t1 = time.time()
print("VMEC beginning: ")
### Always use the QA VMEC file and just change the boundary
vmec_input = "../../tests/test_files/input.LandremanPaul2021_QA"
equil = Vmec(vmec_input, mpi)
equil.boundary = qfm_surf
equil.run()
t2 = time.time()
print("Running VMEC took ", t2 - t1, " s")

# from simsopt.field.magneticfieldclasses import InterpolatedField

# out_dir = Path(out_dir)
# n = 20
# rs = np.linalg.norm(s_plot.gamma()[:, :, 0:2], axis=2)
# zs = s_plot.gamma()[:, :, 2]
# rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
# rrange = (np.min(rs), np.max(rs), n)
# phirange = (0, 2 * np.pi / s_plot.nfp, n * 2)
# zrange = (0, np.max(zs), n // 2)
# degree = 2  # 2 is sufficient sometimes
# bs.set_points(s_plot.gamma().reshape((-1, 3)))
# B_PSC.set_points(s_plot.gamma().reshape((-1, 3)))
# bsh = InterpolatedField(
#     bs + B_PSC, degree, rrange, phirange, zrange, True, nfp=s_plot.nfp, stellsym=s_plot.stellsym
# )
# bsh.set_points(s_plot.gamma().reshape((-1, 3)))
# from simsopt.field.tracing import compute_fieldlines, \
#     plot_poincare_data, \
#     IterationStoppingCriterion, SurfaceClassifier, \
#     LevelsetStoppingCriterion
# from simsopt.util import proc0_print


# # set fieldline tracer parameters
# nfieldlines = 8
# tmax_fl = 10000

# R0 = np.linspace(16.5, 17.5, nfieldlines)  # np.linspace(s_plot.get_rc(0, 0) - s_plot.get_rc(1, 0) / 2.0, s_plot.get_rc(0, 0) + s_plot.get_rc(1, 0) / 2.0, nfieldlines)
# Z0 = np.zeros(nfieldlines)
# phis = [(i / 4) * (2 * np.pi / s_plot.nfp) for i in range(4)]
# print(rrange, zrange, phirange)
# print(R0, Z0)

# t1 = time.time()
# # compute the fieldlines from the initial locations specified above
# sc_fieldline = SurfaceClassifier(s_plot, h=0.03, p=2)
# sc_fieldline.to_vtk(str(out_dir) + 'levelset', h=0.02)

# fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
#     bsh, R0, Z0, tmax=tmax_fl, tol=1e-20, comm=comm,
#     phis=phis,
#     # phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
#     stopping_criteria=[IterationStoppingCriterion(50000)])
# t2 = time.time()
# proc0_print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
# # make the poincare plots
# if comm is None or comm.rank == 0:
#     plot_poincare_data(fieldlines_phi_hits, phis, out_dir / 'poincare_fieldline.png', dpi=100, surf=s_plot)
