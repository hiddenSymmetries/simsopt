from math import sqrt
import numpy as np
import simsgeopp as sgpp
import logging


logger = logging.getLogger(__name__)


def compute_gc_radius(m, vperp, q, absb):
    return m*vperp/(abs(q)*absb)

def gc_to_fullorbit_initial_guesses(field, xyz_inits, vtangs, vtotal, m, q):
    nparticles = xyz_inits.shape[0]
    xyz_inits_full = np.zeros_like(xyz_inits)
    v_inits = np.zeros((nparticles, 3))
    rgs = np.zeros((nparticles, ))
    field.set_points(xyz_inits)
    Bs = field.B()
    AbsBs = field.AbsB()
    eB = Bs/AbsBs
    ez = np.asarray(nparticles*[[0., 0., -1.]])
    ez -= eB * np.sum(eB*ez, axis=1)[:, None]
    ez *= 1./np.linalg.norm(ez, axis=1)[:, None]
    Bperp = np.cross(eB, ez, axis=1)
    Bperp *= 1./np.linalg.norm(Bperp, axis=1)[:, None]
    vperp2s = vtotal**2 - vtangs**2
    for i in range(nparticles):
        rgs[i] = compute_gc_radius(m, sqrt(vperp2s[i]), q, AbsBs[i, 0])
        xyz_inits_full[i, :] = xyz_inits[i, :] + rgs[i] * ez[i, :]
        v_inits[i, :] = -sqrt(vperp2s[i]) * Bperp[i, :] + vtangs[i] * eB[i, :]
    print("rgs", rgs)
    return xyz_inits_full, v_inits, rgs


def trace_particles_starting_on_axis(axis, field, nparticles, tmax=1e-4, seed=1,
                                     mass=1.67e-27, charge=1, Ekinev=9000,
                                     umin=-1, umax=+1, phis=[], stopping_criteria=[], mode='gc'):
    assert mode in ['gc', 'full']
    e = 1.6e-19
    Ekin = Ekinev*e
    m = mass
    q = charge*e
    vtotal = sqrt(2*Ekin/m)  # Ekin = 0.5 * m * v^2 <=> v = sqrt(2*Ekin/m)

    tol = 1e-9

    np.random.seed(seed)
    us = np.random.uniform(low=umin, high=umax, size=(nparticles, ))
    vtangs = us*vtotal
    xyz_inits = axis[np.random.randint(0, axis.shape[0], size=(nparticles, )), :]
    if mode == 'full':
        xyz_inits, v_inits, _ = gc_to_fullorbit_initial_guesses(field, xyz_inits, vtangs, vtotal, m, q)
    res_tys = []
    res_phi_hits = []

    loss_ctr = 0
    for i in range(nparticles):
        if mode == 'gc':
            res_ty, res_phi_hit = sgpp.particle_guiding_center_tracing(
                field, xyz_inits[i, :],
                m, q, vtotal, vtangs[i], tmax, tol, phis=phis, stopping_criteria=stopping_criteria)
        else:
            res_ty, res_phi_hit = sgpp.particle_fullorbit_tracing(
                field, xyz_inits[i, :], v_inits[i, :],
                m, q, tmax, tol, phis=phis, stopping_criteria=stopping_criteria)
        res_tys.append(np.asarray(res_ty))
        res_phi_hits.append(res_phi_hit)
        logger.debug(f"{i+1:3d}/{nparticles}, t_final={res_ty[-1][0]}")
        if res_ty[-1][0] < tmax - 1e-15:
            loss_ctr += 1
    logger.debug(f'Particles lost {loss_ctr}/{nparticles}={loss_ctr/nparticles}')
    return res_tys, res_phi_hits


def compute_fieldlines(field, r0, nlines, linestep=0.01, tmax=200, phis=[], stopping_criteria=[]):
    xyz_inits = np.zeros((nlines, 3))
    xyz_inits[:, 0] = np.asarray([r0 + i*linestep for i in range(nlines)])
    tol = 1e-7
    res_tys = []
    res_phi_hits = []
    for i in range(nlines):
        res_ty, res_phi_hit = sgpp.fieldline_tracing(
            field, xyz_inits[i, :],
            tmax, tol, phis=phis, stopping_criteria=stopping_criteria)
        res_tys.append(np.asarray(res_ty))
        res_phi_hits.append(res_phi_hit)
        logger.debug(f"{i+1}/{nlines}, t_final={res_ty[-1][0]}")
    return res_tys, res_phi_hits


def particles_to_vtk(res_tys, filename):
    from pyevtk.hl import polyLinesToVTK
    x = np.concatenate([xyz[:, 1] for xyz in res_tys])
    y = np.concatenate([xyz[:, 2] for xyz in res_tys])
    z = np.concatenate([xyz[:, 3] for xyz in res_tys])
    ppl = np.asarray([xyz.shape[0] for xyz in res_tys])
    data = np.concatenate([i*np.ones((res_tys[i].shape[0], )) for i in range(len(res_tys))])
    polyLinesToVTK(filename, x, y, z, pointsPerLine=ppl, pointData={'idx': data})


def signed_distance_from_surface(xyz, surface):
    gammas = surface.gamma().reshape((-1, 3))
    from scipy.spatial.distance import cdist
    dists = cdist(xyz, gammas)
    mins = np.argmin(dists, axis=1)
    n = surface.normal().reshape((-1, 3))
    nmins = n[mins]
    gammamins = gammas[mins]

    # Now that we have found the closest node, we approximate the surface with
    # a plane through that node with the appropriate normal and then compute
    # the distance from the point to that plane
    # https://stackoverflow.com/questions/55189333/how-to-get-distance-from-point-to-plane-in-3d
    mindist = np.sum((xyz-gammamins) * nmins, axis=1)

    a_point_in_the_surface = np.mean(surface.gamma()[0, :, :], axis=0)
    sign_of_interiorpoint = np.sign(np.sum((a_point_in_the_surface-gammas[0, :])*n[0, :]))

    signed_dists = mindist * sign_of_interiorpoint
    return signed_dists


class SurfaceClassifier():

    def __init__(self, surface, p=1, h=0.05):
        gammas = surface.gamma()
        r = np.linalg.norm(gammas[:, :, :2], axis=2)
        z = gammas[:, :, 2]
        rmin = max(np.min(r) - 0.1, 0.)
        rmax = np.max(r) + 0.1
        zmin = np.min(z) - 0.1
        zmax = np.max(z) + 0.1

        self.zrange = (zmin, zmax)
        self.rrange = (rmin, rmax)

        nr = int((self.rrange[1]-self.rrange[0])/h)
        nphi = int(2*np.pi/h)
        nz = int((self.zrange[1]-self.zrange[0])/h)

        def fbatch(rs, phis, zs):
            xyz = np.zeros((len(rs), 3))
            xyz[:, 0] = rs * np.cos(phis)
            xyz[:, 1] = rs * np.sin(phis)
            xyz[:, 2] = zs
            return list(signed_distance_from_surface(xyz, surface))

        rule = sgpp.UniformInterpolationRule(p)
        self.dist = sgpp.RegularGridInterpolant3D(
            rule, [rmin, rmax, nr], [0., 2*np.pi, nphi], [zmin, zmax, nz], 1, True)
        self.dist.interpolate_batch(fbatch)

    def evaluate(self, xyz):
        rphiz = np.zeros_like(xyz)
        rphiz[:, 0] = np.linalg.norm(xyz[:, :2], axis=1)
        rphiz[:, 1] = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2*np.pi)
        rphiz[:, 2] = xyz[:, 2]
        d = np.zeros((xyz.shape[0], 1))
        self.dist.evaluate_batch(rphiz, d)
        return d

    def to_vtk(self, filename, h=0.01):
        from pyevtk.hl import gridToVTK

        nr = int((self.rrange[1]-self.rrange[0])/h)
        nphi = int(2*np.pi/h)
        nz = int((self.zrange[1]-self.zrange[0])/h)
        rs = np.linspace(self.rrange[0], self.rrange[1], nr)
        phis = np.linspace(0, 2*np.pi, nphi)
        zs = np.linspace(self.zrange[0], self.zrange[1], nz)

        R, Phi, Z = np.meshgrid(rs, phis, zs)
        X = R * np.cos(Phi)
        Y = R * np.sin(Phi)
        Z = Z

        RPhiZ = np.zeros((R.size, 3))
        RPhiZ[:, 0] = R.flatten()
        RPhiZ[:, 1] = Phi.flatten()
        RPhiZ[:, 2] = Z.flatten()
        vals = np.zeros((R.size, 1))
        self.dist.evaluate_batch(RPhiZ, vals)
        vals = vals.reshape(R.shape)
        gridToVTK(filename, X, Y, Z, pointData={"levelset": vals})


class LevelsetStoppingCriterion(sgpp.LevelsetStoppingCriterion):
    pass
