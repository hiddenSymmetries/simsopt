import numpy as np
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.coilcollection import CoilCollection
from simsopt.util.zoo import get_ncsx_data
from simsopt.util.constants import PROTON_MASS, ONE_EV

MASS = PROTON_MASS
EKIN = 5000 * ONE_EV
v2 = 2 * EKIN/MASS

nphi = 3

# phi_hits = []
# for i in range(nparticles):
#     phi_hits.append(np.load(f'/tmp/phihits/24_gc_phi_hits_{i}.npy'))

# phi_hits = np.load('24_bs.npy',allow_pickle=True)
# phi_hits = np.load('phi_24_n_1000_bsh.npy',allow_pickle=True)
phi_hits = np.load(f'phi_{nphi}_n_4000_many_u_bsh.npy',allow_pickle=True)

nfp = 3
phis = [(i/nphi)*(2*np.pi) for i in range(nphi)]
functions_next = [ [] for k in range(nphi) ]
functions_same = [ [] for k in range(nphi) ]
functions_prev = [ [] for k in range(nphi) ]
functions_exit = [ [] for k in range(nphi) ]
for i in range(phi_hits.shape[0]):
    for j in range(phi_hits[i].shape[0]-1):
        this_idx = int(phi_hits[i][j, 1])
        next_idx = int(phi_hits[i][j+1, 1])
        dt = phi_hits[i][j+1, 0] - phi_hits[i][j, 0]
        rin = np.linalg.norm(phi_hits[i][j, 2:4])
        rout = np.linalg.norm(phi_hits[i][j+1, 2:4])
        zin = phi_hits[i][j, 4]
        zout = phi_hits[i][j+1, 4]
        vin = phi_hits[i][j, 5]/np.sqrt(v2)
        vout = phi_hits[i][j+1, 5]/np.sqrt(v2)
        data = np.asarray([rin, zin, vin, rout, zout, vout, dt])
        if next_idx == this_idx:
            functions_same[this_idx].append(data)
        elif next_idx == (this_idx+1) % nphi:
            functions_next[this_idx].append(data)
        elif next_idx == (this_idx-1) % nphi:
            functions_prev[this_idx].append(data)
        elif next_idx < 0:
            functions_exit[this_idx].append(data)

for phiidx in range(nphi):
    functions_same[phiidx] = np.asarray(functions_same[phiidx])
    functions_prev[phiidx] = np.asarray(functions_prev[phiidx])
    functions_next[phiidx] = np.asarray(functions_next[phiidx])
    functions_exit[phiidx] = np.asarray(functions_exit[phiidx])


coils, currents, ma = get_ncsx_data(Nt_coils=25, Nt_ma=20)

# stellarator = CoilCollection(coils, currents, 3, True)
# coils = stellarator.coils
# currents = stellarator.currents
# bs = BiotSavart(coils, currents)
# data = functions_next[0]
# AbsB = bs.set_points(np.ascontiguousarray(data[:, 0:3])).AbsB()
# mus = (v2-data[:, -1])[:, None]/(2*AbsB)
# print("mus", np.min(mus), np.max(mus), mus)
# data = functions_same[0]
# AbsB = bs.set_points(np.ascontiguousarray(data[:, 0:3])).AbsB()
# mus = (v2-data[:, -1])[:, None]/(2*AbsB)
# print("mus", np.min(mus), np.max(mus))


import matplotlib.pyplot as plt
plt.figure()
phiidx = 0


def write_to_evtk(datas, shift, vtk_file):
    rs = np.concatenate([data[:, 0]  for data in datas if data.size>0])
    zs = np.concatenate([data[:, 1] for data in datas if data.size>0])
    vs = np.concatenate([data[:, 2] for data in datas if data.size>0])

    colors = np.concatenate([(i+shift)*np.ones((datas[i].shape[0], )) for i in range(len(datas)) if datas[i].size>0])

    rout = np.concatenate([data[:, 3] for data in datas if data.size>0])
    zout = np.concatenate([data[:, 4] for data in datas if data.size>0])
    vout = np.concatenate([data[:, 5] for data in datas if data.size>0])
    dt = np.concatenate([data[:, 6] for data in datas if data.size>0])
    from pyevtk.hl import pointsToVTK
    pointsToVTK(f"./vtk/{vtk_file}", rs, zs, vs, data={
        "color": colors, "dt": dt, "r": rout, "z": zout, "v": vout})

fig, axs = plt.subplots(2, nphi//nfp, figsize=(20, 7))

signdict = {1: "forward", -1: "backward"}
for phiidx in range(nphi//nfp):
    f_prev = np.vstack([functions_prev[phiidx + i*(nphi//nfp)] for i in range(nfp) if functions_prev[phiidx + i*(nphi//nfp)].size > 0])
    f_next = np.vstack([functions_next[phiidx + i*(nphi//nfp)] for i in range(nfp) if functions_next[phiidx + i*(nphi//nfp)].size > 0])
    f_same = np.vstack([functions_same[phiidx + i*(nphi//nfp)] for i in range(nfp) if functions_same[phiidx + i*(nphi//nfp)].size > 0])
    f_exit = np.vstack([functions_exit[phiidx + i*(nphi//nfp)] for i in range(nfp) if functions_exit[phiidx + i*(nphi//nfp)].size > 0])
    write_to_evtk([f_prev, f_next, f_same, f_exit], 0, f"{phiidx}")
    write_to_evtk([f_prev], 0, f"{phiidx}_prev")
    write_to_evtk([f_next], 1, f"{phiidx}_next")
    write_to_evtk([f_same], 2, f"{phiidx}_same")
    write_to_evtk([f_exit], 3, f"{phiidx}_exit")


from scipy.spatial import KDTree
from scipy.interpolate import LinearNDInterpolator
prev_trees = []
next_trees = []
same_trees = []
exit_trees = []

for phiidx in range(nphi//nfp):
    f_prev = np.vstack([functions_prev[phiidx + i*(nphi//nfp)] for i in range(nfp) if functions_prev[phiidx + i*(nphi//nfp)].size > 0])
    f_next = np.vstack([functions_next[phiidx + i*(nphi//nfp)] for i in range(nfp) if functions_next[phiidx + i*(nphi//nfp)].size > 0])
    f_same = np.vstack([functions_same[phiidx + i*(nphi//nfp)] for i in range(nfp) if functions_same[phiidx + i*(nphi//nfp)].size > 0])
    f_exit = np.vstack([functions_exit[phiidx + i*(nphi//nfp)] for i in range(nfp) if functions_exit[phiidx + i*(nphi//nfp)].size > 0])


    prev_trees.append((KDTree(f_prev[:, :3]), f_prev[:, 3:], LinearNDInterpolator(f_prev[:, :3], f_prev[:, 3:])))
    next_trees.append((KDTree(f_next[:, :3]), f_next[:, 3:], LinearNDInterpolator(f_next[:, :3], f_next[:, 3:])))
    same_trees.append((KDTree(f_same[:, :3]), f_same[:, 3:], LinearNDInterpolator(f_same[:, :3], f_same[:, 3:])))
    exit_trees.append((KDTree(f_exit[:, :3]), f_exit[:, 3:], LinearNDInterpolator(f_exit[:, :3], f_exit[:, 3:])))

mod = nphi//nfp
def mapeval(phiidx, rzv):
    dist_prev, idx_prev = prev_trees[phiidx][0].query(rzv, k=1)
    dist_next, idx_next = next_trees[phiidx][0].query(rzv, k=1)
    dist_same, idx_same = same_trees[phiidx][0].query(rzv, k=1)
    dist_exit, idx_exit = exit_trees[phiidx][0].query(rzv, k=1)
    which = np.argmin([dist_prev, dist_next, dist_same, dist_exit])
    if which == 0:
        # return prev_trees[phiidx][1][idx_prev, :], (phiidx-1)%mod, "prev"
        return prev_trees[phiidx][2](rzv)[0, :], (phiidx-1)%mod, "prev"
    elif which == 1:
        # return next_trees[phiidx][1][idx_next, :], (phiidx+1)%mod, "next"
        return next_trees[phiidx][2](rzv)[0, :], (phiidx+1)%mod, "next"
    elif which == 2:
        # return same_trees[phiidx][1][idx_same, :], phiidx, "same"
        return same_trees[phiidx][2](rzv)[0, :], phiidx, "same"
    elif which == 3:
        # return exit_trees[phiidx][1][idx_exit, :], -1, "exit"
        return exit_trees[phiidx][2](rzv)[0, :], -1, "exit"
    else:
        raise RuntimeError("wtf")

import IPython; IPython.embed()
import sys; sys.exit()
for i in range(3):#phi_hits.shape[0]:
    print("###################################")
    print(f"################# i = {i} ##########")
    print("###################################")
    r = np.linalg.norm(phi_hits[i][0, 2:4])
    z = phi_hits[i][0, 4]
    v = phi_hits[i][0, 5]/np.sqrt(v2)
    rzv = np.asarray([r, z, v])
    phiidx = int(phi_hits[i][0, 1]) % mod
    t = phi_hits[i][0, 0]
    for j in range(1, phi_hits[i].shape[0]):
        rzvt, phiidx, _ = mapeval(phiidx, rzv)
        rzv = rzvt[:-1]
        t += rzvt[-1]
        r = np.linalg.norm(phi_hits[i][j, 2:4])
        z = phi_hits[i][j, 4]
        v = phi_hits[i][j, 5]/np.sqrt(v2)
        # print(f"Map: r={rzv[0]:.3f}, z={rzv[1]:.3f}, v={rzv[2]:.3f}, Actual: r={r:.3f}, z={z:.3f}, v={v:.3f}")
        print(f"t={t:.3e}, err={np.abs(np.concatenate([rzv, [t]])-np.asarray([r, z, v, phi_hits[i][j, 0]]))}")

print(mapeval(0, f_prev[10, :3]))
print(mapeval(0, f_next[10, :3]))
print(mapeval(0, f_same[10, :3]))
print(mapeval(0, f_exit[10, :3]))

