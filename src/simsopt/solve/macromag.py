from __future__ import annotations
import math
import numpy as np
from coilpy import rotation_angle
from simsopt.field import Coil, BiotSavart
from simsopt.util.permanent_magnet_helper_functions import read_focus_coils
from scipy.constants import mu_0
from scipy.sparse.linalg import LinearOperator, minres, gmres
from numpy.linalg import norm
from numba import njit, prange
from pathlib import Path
from typing import Optional

__all__ = ["MacroMag"]


_INV4PI  = 1.0 / (4.0 * math.pi)
_MACHEPS = np.finfo(np.float64).eps


@njit(inline='always')
def _f_3D(a, b, c, x, y, z):
    half_a, half_b, half_c = a, b, c
    eps_h, eps_l = 1.0001, 0.9999

    dx = half_a - x
    if dx == 0.0:
        dxh, dxl = half_a - x*eps_h, half_a - x*eps_l
        r_h = math.sqrt(dxh*dxh + (half_b - y)**2 + (half_c - z)**2)
        r_l = math.sqrt(dxl*dxl + (half_b - y)**2 + (half_c - z)**2)
        val_h = (half_b - y)*(half_c - z)/(dxh * r_h)
        val_l = (half_b - y)*(half_c - z)/(dxl * r_l)
        return 0.5*(val_h + val_l)
    else:
        r = math.sqrt(dx*dx + (half_b - y)**2 + (half_c - z)**2)
        return (half_b - y)*(half_c - z)/(dx * r)

@njit(inline='always')
def _g_3D(a, b, c, x, y, z):
    half_a, half_b, half_c = a, b, c
    eps_h, eps_l = 1.0001, 0.9999

    dy = half_b - y
    if dy == 0.0:
        dyh, dyl = half_b - y*eps_h, half_b - y*eps_l
        r_h = math.sqrt((half_a - x)**2 + dyh*dyh + (half_c - z)**2)
        r_l = math.sqrt((half_a - x)**2 + dyl*dyl + (half_c - z)**2)
        val_h = (half_a - x)*(half_c - z)/(dyh * r_h)
        val_l = (half_a - x)*(half_c - z)/(dyl * r_l)
        return 0.5*(val_h + val_l)
    else:
        r = math.sqrt((half_a - x)**2 + dy*dy + (half_c - z)**2)
        return (half_a - x)*(half_c - z)/(dy * r)


@njit(inline='always')
def _h_3D(a, b, c, x, y, z):
    half_a, half_b, half_c = a, b, c
    eps_h, eps_l = 1.0001, 0.9999

    dz = half_c - z
    if dz == 0.0:
        dzh, dzl = half_c - z*eps_h, half_c - z*eps_l
        r_h = math.sqrt((half_a - x)**2 + (half_b - y)**2 + dzh*dzh)
        r_l = math.sqrt((half_a - x)**2 + (half_b - y)**2 + dzl*dzl)
        val_h = (half_a - x)*(half_b - y)/(dzh * r_h)
        val_l = (half_a - x)*(half_b - y)/(dzl * r_l)
        return 0.5*(val_h + val_l)
    else:
        r = math.sqrt((half_a - x)**2 + (half_b - y)**2 + dz*dz)
        return (half_a - x)*(half_b - y)/(dz * r)
    
@njit(inline='always')
def _FF_3D(a, b, c, x, y, z):
    ha, hb, hc = a, b, c
    return (hc - z) + math.sqrt((ha - x)**2 + (hb - y)**2 + (hc - z)**2)

@njit(inline='always')
def _GG_3D(a, b, c, x, y, z):
    ha, hb, hc = a, b, c
    return (ha - x) + math.sqrt((ha - x)**2 + (hb - y)**2 + (hc - z)**2)

@njit(inline='always')
def _HH_3D(a, b, c, x, y, z):
    ha, hb, hc = a, b, c
    return (hb - y) + math.sqrt((ha - x)**2 + (hb - y)**2 + (hc - z)**2)


@njit(inline='always')
def getF_limit(a, b, c, x, y, z, func):
    lim_h, lim_l = 1.0001, 0.9999
    xl, yl, zl = x * lim_l, y * lim_l, z * lim_l
    xh, yh, zh = x * lim_h, y * lim_h, z * lim_h

    # flip only the half dim signs, keep coords fixed
    nom_l = ( func( +a, +b, +c, xl, yl, zl )
            * func( -a, -b, +c, xl, yl, zl )
            * func( +a, -b, -c, xl, yl, zl )
            * func( -a, +b, -c, xl, yl, zl ) )

    denom_l = ( func( +a, -b, +c, xl, yl, zl )
              * func( -a, +b, +c, xl, yl, zl )
              * func( +a, +b, -c, xl, yl, zl )
              * func( -a, -b, -c, xl, yl, zl ) )

    nom_h = ( func( +a, +b, +c, xh, yh, zh )
            * func( -a, -b, +c, xh, yh, zh )
            * func( +a, -b, -c, xh, yh, zh )
            * func( -a, +b, -c, xh, yh, zh ) )

    denom_h = ( func( +a, -b, +c, xh, yh, zh )
              * func( -a, +b, +c, xh, yh, zh )
              * func( +a, +b, -c, xh, yh, zh )
              * func( -a, -b, -c, xh, yh, zh ) )

    return (nom_l + nom_h) / (denom_l + denom_h)

@njit(cache=False)
def _prism_N_local_nb(a, b, c, x0, y0, z0):
    N = np.empty((3,3), dtype=np.float64)
    sum_f = sum_g = sum_h = 0.0
    for si in (1.0, -1.0):
        for sj in (1.0, -1.0):
            for sk in (1.0, -1.0):
                xi, yi, zi = si*x0, sj*y0, sk*z0
                sum_f += math.atan(_f_3D(a, b, c, xi, yi, zi))
                sum_g += math.atan(_g_3D(a, b, c, xi, yi, zi))
                sum_h += math.atan(_h_3D(a, b, c, xi, yi, zi))
    N[0,0] = _INV4PI * sum_f
    N[1,1] = _INV4PI * sum_g
    N[2,2] = _INV4PI * sum_h

    def _off_diag(func, idx1, idx2):
        nom = ( func( +a, +b, +c, x0, y0, z0 )
            * func( -a, -b, +c, x0, y0, z0 )
            * func( +a, -b, -c, x0, y0, z0 )
            * func( -a, +b, -c, x0, y0, z0 ) )
        denom = ( func( +a, -b, +c, x0, y0, z0 )
                * func( -a, +b, +c, x0, y0, z0 )
                * func( +a, +b, -c, x0, y0, z0 )
                * func( -a, -b, -c, x0, y0, z0 ) )

        if nom == 0 or denom == 0:
            val = -_INV4PI * math.log( getF_limit(a,b,c,x0,y0,z0, func) )
        else:
            val = -_INV4PI * math.log(nom/denom)
        if denom and abs((nom-denom)/denom) < 10*_MACHEPS:
            val = 0.0
        N[idx1, idx2] = N[idx2, idx1] = val

    _off_diag(_FF_3D, 0, 1)
    _off_diag(_GG_3D, 1, 2)
    _off_diag(_HH_3D, 0, 2)

    return -N

@njit(parallel=True, cache=False)
def _assemble_tensor_local_nb(centres, half, Rg2l):
    n = centres.shape[0]
    Nloc = np.zeros((n, n, 3, 3), dtype=np.float64)
    for i in prange(n):
        a, b, c = half[i]
        Rgl = Rg2l[i]
        for j in range(n):
            # vector from source j to target i in global coords
            dx = centres[i,0] - centres[j,0]
            dy = centres[i,1] - centres[j,1]
            dz = centres[i,2] - centres[j,2]
            # rotate into local frame...
            x_loc = Rgl[0,0]*dx + Rgl[0,1]*dy + Rgl[0,2]*dz
            y_loc = Rgl[1,0]*dx + Rgl[1,1]*dy + Rgl[1,2]*dz
            z_loc = Rgl[2,0]*dx + Rgl[2,1]*dy + Rgl[2,2]*dz
            # exact local prism tensor
            Nloc[i,j] = _prism_N_local_nb(a, b, c, x_loc, y_loc, z_loc)
    return Nloc
class MacroMag:
    """Compute demag tensor for rectangular prisms."""

    _INV4PI = _INV4PI
    _MACHEPS = _MACHEPS

    def __init__(self, tiles):
        self.tiles = tiles
        self.n = tiles.n
        self.centres = tiles.offset.copy()
        self.half    = (tiles.size * 0.5).astype(np.float64)
        self.Rg2l    = np.zeros((self.n, 3, 3), dtype=np.float64)
        for k in range(self.n):
            self.Rg2l[k], _ = MacroMag.get_rotation_matrices(*tiles.rot[k])

    @staticmethod
    def _f_3D(a, b, c, x, y, z):
        return _f_3D(a, b, c, x, y, z)

    @staticmethod
    def _g_3D(a, b, c, x, y, z):
        return _g_3D(a, b, c, x, y, z)

    @staticmethod
    def _h_3D(a, b, c, x, y, z):
        return _h_3D(a, b, c, x, y, z)

    @staticmethod
    def _FF_3D(a, b, c, x, y, z):
        return _FF_3D(a, b, c, x, y, z)

    @staticmethod
    def _GG_3D(a, b, c, x, y, z):
        return _GG_3D(a, b, c, x, y, z)

    @staticmethod
    def _HH_3D(a, b, c, x, y, z):
        return _HH_3D(a, b, c, x, y, z)

    @staticmethod
    def _prism_N_local_nb(a, b, c, x0, y0, z0):
        return _prism_N_local_nb(a, b, c, x0, y0, z0)

    @staticmethod
    def get_rotation_matrices(phi_x, phi_y, phi_z):
        ax, ay, az = -phi_x, -phi_y, -phi_z
        RotX = np.array([[1,0,0],
                       [0,math.cos(ax),-math.sin(ax)],
                       [0,math.sin(ax), math.cos(ax)]])
        RotY = np.array([[math.cos(ay),0,math.sin(ay)],
                       [0,1,0],
                       [-math.sin(ay),0,math.cos(ay)]])
        RotZ = np.array([[math.cos(az),-math.sin(az),0],
                       [math.sin(az), math.cos(az),0],
                       [0,0,1]])
        Rg2l = RotZ @ RotY @ RotX    # global -> local
        Rl2g = RotX.T @ RotY.T @ RotZ.T  # local -> global

        return Rg2l, Rl2g
    
    def fast_get_demag_tensor(self, cache: bool = True):
        # Reuse a cached full tensor if it matches the current size
        if cache and hasattr(self, "_N_full") and self._N_full is not None and self._N_full.shape[:2] == (self.n, self.n):
            return self._N_full
        N = _assemble_tensor_local_nb(self.centres, self.half, self.Rg2l)
        self._N_full = N
        return N
    
    def load_coils(self, focus_file: Path) -> None:
        """Load and cache a BiotSavart coil model for this instance."""
        curves, currents, ncoils = read_focus_coils(str(focus_file))
        coils = [Coil(curves[i], currents[i]) for i in range(ncoils)]
        self._bs_coil = BiotSavart(coils)

    def _anisotropy_field(self, M: np.ndarray, K: float, Ms: float) -> np.ndarray:
        """Compute uniaxial anisotropy for our tiles."""
        u = self.tiles.u_ea
        proj = np.einsum('ij,ij->i', M, u)
        return (2*K/(mu_0*Ms**2)) * proj[:, None] * u

    def _coil_field_at(self, pts: np.ndarray, const_H: tuple[float,float,float], use_coils: bool) -> np.ndarray:
        """Return H at pts, either uniform or via Biot–Savart."""
        if use_coils and getattr(self, '_bs_coil', None) is not None:
            self._bs_coil.set_points(pts)
            return self._bs_coil.B() / mu_0
        N = pts.shape[0]
        return np.broadcast_to(const_H, (N,3)).copy()
    
    def _demag_field_at(self, pts: np.ndarray, demag_tensor: np.ndarray) -> np.ndarray:
        """
        Compute the demagnetizing field at a set of points for rectangular prisms.

        Parameters
        ----------
        pts : ndarray of shape (n_pts, 3)
            Coordinates of the evaluation points in global frame.
        demag_tensor : ndarray of shape (n_tiles, n_pts, 3, 3)
            Precomputed local demagnetization tensor for each tile and point.

        Returns
        -------
        H : ndarray of shape (n_pts, 3)
            Demagnetizing field (A/m) at each evaluation point in global frame.
        """
        
        n, n_pts = self.tiles.n, pts.shape[0]
        H = np.zeros((n_pts,3), dtype=np.float64)

        # Precompute rotations and local M once
        Rg2l = np.zeros((n,3,3), dtype=np.float64)
        Rl2g = np.zeros((n,3,3), dtype=np.float64)
        M_loc = np.zeros((n,3), dtype=np.float64)
        for i in range(n):
            Rg2l[i], Rl2g[i] = self.get_rotation_matrices(*self.tiles.rot[i])
            M_loc[i] = Rg2l[i] @ self.tiles.M[i]

        # Now accumulate H
        for i in range(n):
            # demag_local[i] is shape (n_pts,3,3)
            # multiply each (3×3) by the same M_loc[i] -> yields (n_pts,3)
            H_loc = np.einsum('pqr,r->pq', demag_tensor[i], M_loc[i])
            # rotate back to global
            H    += (Rl2g[i] @ H_loc.T).T

        return H

        
        
    def sor_solve(
        self,
        Ms: float,
        K: float,
        omega: float,
        max_it: int,
        tol: float,
        const_H: tuple[float,float,float],
        use_coils: bool = False,
        coil_path: Optional[Path] = None,
        demag_only: bool = False,
        log_every: int = 10,
    ) -> "MacroMag":
        """Run the SOR‐based macromag solve in place."""
        # build coil field object if requested
        bs = self.load_coils(coil_path) if use_coils and coil_path else None
        def Hcoil_func(pts):
            return self._coil_field_at(pts, const_H, use_coils)

        tiles   = self.tiles
        centres = self.centres
        demag_tensor = self.fast_get_demag_tensor()

        lambda_ = 1.0
        H_prev  = np.zeros((tiles.n, 3), dtype=np.float64)
        M_prev  = None
        maxDiff = [0.0, 0.0, 0.0, 0.0]

        for it in range(1, max_it + 1):
            # 1) update M from H_prev
            u      = tiles.u_ea
            H_par  = (H_prev * u).sum(axis=1)[:, None] * u
            H_perp = H_prev - H_par
            chi_par  = (tiles.mu_r_ea - 1)[:, None]
            chi_perp = (tiles.mu_r_oa - 1)[:, None]
            M_rem   = tiles.M_rem[:, None] * u
            tiles.M = M_rem + chi_par * H_par + chi_perp * H_perp

            # 2) compute field terms
            H_demag = self._demag_field_at(centres, demag_tensor).astype(np.float64)
            H_anis  = np.zeros_like(H_demag)
            # H_anis = anisotropy_field(tiles.M, u, K, Ms) if not demag_only else np.zeros_like(H_demag)
            H_coil  = (
                Hcoil_func(centres).astype(np.float64)
                if not demag_only
                else np.zeros_like(H_demag)
            )
            H_eff = H_demag + H_anis + H_coil

            # 3) SOR update
            H_prev = H_prev + lambda_ * (H_eff - H_prev)

            # 4) check convergence on |M|
            if M_prev is None:
                err = float("inf")
            else:
                delta_m = np.abs(
                    np.linalg.norm(tiles.M, axis=1)
                    - np.linalg.norm(M_prev, axis=1)
                )
                rel = delta_m / np.maximum(np.linalg.norm(M_prev, axis=1), 1e-30)
                err = rel.max()

            # 5) adapt relaxation parameter
            maxDiff = [err] + maxDiff[:3]
            if it > 3:
                c1 = maxDiff[0] > maxDiff[1] < maxDiff[2] > maxDiff[3]
                c2 = maxDiff[3] > maxDiff[2] > maxDiff[1] > maxDiff[0]
                c3 = maxDiff[3] > maxDiff[2] > maxDiff[1] < maxDiff[0]
                if c1 or c2 or c3:
                    lambda_ *= 0.5

            # 6) logging
            m_hat = tiles.M / np.linalg.norm(tiles.M, axis=1, keepdims=True)
            h_hat = H_eff / np.maximum(
                np.linalg.norm(H_eff, axis=1, keepdims=True), 1e-30
            )
            max_angle_MH = np.degrees(
                np.arccos(
                    np.clip((m_hat * h_hat).sum(axis=1), -1, 1)
                )
            ).max()

            if it == 1 or it % log_every == 0:
                print(
                    f"  iter {it:3d}: lambda={lambda_:.3f} "
                    f"| max‖ΔM‖/‖M‖={err:.2e} max∠MH={max_angle_MH:6.2f}° "
                    f"| max|H_demag|={np.linalg.norm(H_demag, axis=1).max():.2e} "
                    f"| max|H_anis|={np.linalg.norm(H_anis, axis=1).max():.2e} "
                    f"| max|H_coil|={np.linalg.norm(H_coil, axis=1).max():.2e} "
                    f"| max|H_eff|={np.linalg.norm(H_eff, axis=1).max():.2e}"
                )

            if err < tol * lambda_ and it > 3:
                print(f"  converged in {it} iterations (err={err:.2e})")
                break

            M_prev = tiles.M.copy()

        return self

    def direct_solve(self,
        Ms: float,
        K: float,
        const_H: tuple[float,float,float],
        use_coils: bool = False,
        coil_path: Optional[Path] = None,
        demag_only: bool = False,
        krylov_tol: float = 1e-3,
        krylov_it: int = 200,
        demag_tensor: np.ndarray | None = None, 
        print_progress: bool = False,
        x0: np.ndarray | None = None) -> "MacroMag":
        
        """Run the Krylov‐based direct magstatics solve in place."""
        if use_coils and coil_path:
            self.load_coils(coil_path)

        bs = self.load_coil_BiotSavart(coil_path) if use_coils and coil_path else None
        def Hcoil_func(pts):
            return self._coil_field_at(pts, const_H, use_coils)

        tiles   = self.tiles
        centres = self.centres
        n       = tiles.n
        u       = tiles.u_ea
        m_rem   = tiles.M_rem
        chi_pa  = tiles.mu_r_ea - 1
        chi_pe  = tiles.mu_r_oa - 1

        # build local chi tensors
        chi = np.empty((n, 3, 3), dtype=np.float64)
        for i in range(n):
            uuT   = np.outer(u[i], u[i])
            chi[i] = chi_pa[i]*uuT + chi_pe[i]*(np.eye(3)-uuT)

        # assemble b
        H_a = (
            Hcoil_func(centres).astype(np.float64)
            if not demag_only
            else np.zeros((n, 3))
        )
        b = np.vstack([m_rem[i]*u[i] + chi[i] @ H_a[i] for i in range(n)]).ravel()

        # assemble A
        N = demag_tensor if demag_tensor is not None else self.fast_get_demag_tensor()

        A = np.zeros((3*n, 3*n), dtype=np.float64)
        for i in range(n):
            mats = -np.einsum('pr,jrq->jpq', chi[i], N[i], optimize=True)
            mats[i] += np.eye(3)
            r = slice(3*i, 3*i+3)
            for j in range(n):
                c = slice(3*j, 3*j+3)
                A[r, c] = mats[j]

        sym = np.allclose(A, A.T, atol=1e-12, rtol=1e-10)
        if print_progress:
            print(f"symmetry check: {sym}")

        linop = LinearOperator(A.shape,
            matvec = lambda v: A @ v,
            rmatvec= lambda v: A.T @ v
        )
        
        if sym:
            x, info = minres(linop, b, rtol=krylov_tol, maxiter=krylov_it, x0=x0)
            if print_progress: print(f"MINRES   res  {norm(A@x - b)/norm(b):.2e}, info {info}")
        else:
            x, info = gmres(linop, b, rtol=krylov_tol, maxiter=krylov_it, x0=x0)
            if print_progress: print(f"GMRES    res  {norm(A@x - b)/norm(b):.2e}, info {info}")
        
        tiles.M = x.reshape(n, 3)
        return self



"""
MagTense “Tile” and “Tiles” classes
-----------------------------------

This code is adapted from the MagTense framework to eliminate its
GPU dependency and allow pure-CPU use in SimsOpt:

    - GitHub:  https://github.com/cmt-dtu-energy/MagTense  
    - Paper:   R. Bjørk, E. B. Poulsen, K. K. Nielsen & A. R. Insinga,
               “MagTense: A micromagnetic framework using the analytical
               demagnetization tensor,” Journal of Magnetism and Magnetic Materials,
               vol. 535, 168057 (2021).  
               https://doi.org/10.1016/j.jmmm.2021.168057

MagTense is authored and maintained by the DTU CMT group (Rasmus Bjørk et al.).
It provides fully analytic demagnetization‐tensor calculations for uniformly
magnetized tiles (rectangular prisms, cylinders, tetrahedra, etc.) in
micromagnetic simulations.

Here we re­implement the core Tile/Tiles functionality in pure Python/NumPy
to avoid the original MagTense GPU requirement.
"""


class Tile:
    """
    This is a proxy class to enable Tiles to be accessed as if they were a list of Tile
    objects. So, for example,

    t = Tiles(5)
    print(t[0].center_pos)  # [0, 0, 0]
    t[0].center_pos = [1, 1, 1]
    print(t[0].center_pos)  # [1, 1, 1]
    """

    def __init__(self, parent, index) -> None:
        self._parent = parent
        self._index = index

    def __getattr__(self, name) -> np.float64 | np.ndarray:
        # Get the attribute from the parent
        attr = getattr(self._parent, name)
        # If the attribute is a np.ndarray, return the element at our index
        if isinstance(attr, np.ndarray):
            return attr[self._index]
        # Otherwise, just return the attribute itself
        return attr

    def __setattr__(self, name, value) -> None:
        if name in ["_parent", "_index"]:
            # For these attributes, set them normally
            super().__setattr__(name, value)
        else:
            # For all other attributes, set the value in the parent's attribute array
            attr = getattr(self._parent, name)
            if isinstance(attr, np.ndarray):
                attr[self._index] = value


class Tiles:
    """
    Input to Fortran derived type MagTile.

    Args:
        center_pos: r0, theta0, z0
        dev_center: dr, dtheta, dz
        size: a, b, c
        vertices: v1, v2, v3, v4 as column vectors.
        M: Mx, My, Mz
        u_ea: Easy axis.
        u_oa: Other axis.
        mu_r_ea: Relative permeability in easy axis.
        mu_r_oa: Relative permeability in other axis.
        M_rem: Remanent magnetization.
        tile_type: 1 = cylinder, 2 = prism, 3 = circ_piece, 4 = circ_piece_inv,
                   5 = tetrahedron, 6 = sphere, 7 = spheroid, 10 = ellipsoid
        offset: Offset of global coordinates.
        rot: Rotation in local coordinate system.
        color: Color in visualization.
        magnet_type: 1 = hard magnet, 2 = soft magnet, 3 = soft + constant mu_r
        stfcn_index: default index into the state function.
        incl_it: If equal to zero the tile is not included in the iteration.
        use_sym: Whether to exploit symmetry.
        sym_op: 1 for symmetry and -1 for anti-symmetry respectively to the planes.
        M_rel: Change in magnetization during last iteration of iterate_magnetization().
        n: Number of tiles in the simulation.
    """

    def __init__(
        self,
        n: int,
        center_pos: list[float] | None = None,
        dev_center: list[float] | None = None,
        size: list[float] | None = None,
        vertices: list[float] | None = None,
        M_rem: float | list[float] | None = None,
        easy_axis: list[float] | None = None,
        mag_angle: list[float] | None = None,
        mu_r_ea: float | list[float] | None = None,
        mu_r_oa: float | list[float] | None = None,
        tile_type: int | list[int] | None = None,
        offset: list[float] | None = None,
        rot: list[float] | None = None,
        color: list[float] | None = None,
        magnet_type: list[int] | None = None,
    ) -> None:
        self.n = n
        self.center_pos = center_pos
        self.dev_center = dev_center
        self.size = size
        self.vertices = vertices
        self.M_rem = M_rem

        if easy_axis is not None:
            self.u_ea = easy_axis
        else:
            self.set_easy_axis(mag_angle)

        self.mu_r_ea = mu_r_ea
        self.mu_r_oa = mu_r_oa
        self.tile_type = tile_type
        self.offset = offset
        self.rot = rot

        self.color = color
        self.magnet_type = magnet_type

        # Initialize other attributes
        self.stfcn_index = None
        self.incl_it = None
        self.M_rel = None
        self._use_sym = np.zeros(shape=(self.n), dtype=np.int32, order="F")
        self._sym_op = np.ones(shape=(self.n, 3), dtype=np.float64, order="F")

    def __str__(self) -> str:
        res = ""
        for i in range(self.n):
            res += f"Tile_{i} with coordinates {self.offset[i]}.\n"
        return res

    def __getitem__(self, index) -> Tile:
        if index < 0 or index >= self.n:
            idx_err = f"Index {index} out of bounds"
            raise IndexError(idx_err)
        return Tile(self, index)

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, val: int) -> None:
        self._n = val

    @property
    def center_pos(self) -> np.ndarray:
        return self._center_pos

    @center_pos.setter
    def center_pos(self, val) -> None:
        if not hasattr(self, "_center_pos"):
            self._center_pos = np.zeros(shape=(self.n, 3), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._center_pos[val[1]] = np.asarray(val[0])
        elif isinstance(val[0], (int, float)):
            self._center_pos[:] = np.asarray(val)
        elif len(val) == self.n:
            self._center_pos = np.asarray(val)

    @property
    def dev_center(self) -> np.ndarray:
        return self._dev_center

    @dev_center.setter
    def dev_center(self, val) -> None:
        if not hasattr(self, "_dev_center"):
            self._dev_center = np.zeros(shape=(self.n, 3), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._dev_center[val[1]] = np.asarray(val[0])
        elif isinstance(val[0], (int, float)):
            self._dev_center[:] = np.asarray(val)
        elif len(val) == self.n:
            self._dev_center = np.asarray(val)

    @property
    def size(self) -> np.ndarray:
        return self._size

    @size.setter
    def size(self, val) -> None:
        if not hasattr(self, "_size"):
            self._size = np.zeros(shape=(self.n, 3), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._size[val[1]] = np.asarray(val[0])
        elif isinstance(val[0], (int, float)):
            self._size[:] = np.asarray(val)
        elif len(val) == self.n:
            self._size = np.asarray(val)

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices

    @vertices.setter
    def vertices(self, val) -> None:
        if not hasattr(self, "_vertices"):
            self._vertices = np.zeros(shape=(self.n, 3, 4), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            vert = np.asarray(val[0])
            if vert.shape == (3, 4):
                self._vertices[val[1]] = vert
            elif vert.shape == (4, 3):
                self._vertices[val[1]] = vert.T
            else:
                value_err = "Four 3-D vertices have to be defined!"
                raise ValueError(value_err)
        else:
            vert = np.asarray(val)
            if vert.shape == (3, 4):
                self._vertices[:] = vert
            elif vert.shape == (4, 3):
                self._vertices[:] = vert.T
            else:
                assert vert[0].shape == (3, 4)
                self._vertices = vert

    @property
    def tile_type(self) -> np.ndarray:
        return self._tile_type

    @tile_type.setter
    def tile_type(self, val) -> None:
        if not hasattr(self, "_tile_type"):
            self._tile_type = np.ones(shape=(self.n), dtype=np.int32, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._tile_type[val[1]] = val[0]
        elif isinstance(val, (int, float)):
            self._tile_type = np.asarray([val for _ in range(self.n)], dtype=np.int32)
        elif len(val) == self.n:
            self._tile_type = np.asarray(val, dtype=np.int32)

    @property
    def offset(self) -> np.ndarray:
        return self._offset

    @offset.setter
    def offset(self, val) -> None:
        if not hasattr(self, "_offset"):
            self._offset = np.zeros(shape=(self.n, 3), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._offset[val[1]] = np.asarray(val[0])
        elif isinstance(val[0], (int, float)):
            self._offset[:] = np.asarray(val)
        elif len(val) == self.n:
            self._offset = np.asarray(val)

    @property
    def rot(self) -> np.ndarray:
        return self._rot

    @rot.setter
    def rot(self, val) -> None:
        if not hasattr(self, "_rot"):
            self._rot = np.zeros(shape=(self.n, 3), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._rot[val[1]] = (
                _euler_to_rot_axis(val[0])
                if self.tile_type[val[1]] == 7
                else np.asarray(val[0])
            )
        elif isinstance(val[0], (list, np.ndarray)):
            for i in range(self.n):
                self._rot[i] = (
                    _euler_to_rot_axis(val[i])
                    if self.tile_type[i] == 7
                    else np.asarray(val[i])
                )
        else:
            self._rot[0] = (
                _euler_to_rot_axis(val) if self.tile_type[0] == 7 else np.asarray(val)
            )

    @property
    def M(self) -> np.ndarray:
        return self._M

    @M.setter
    def M(self, val) -> None:
        if not hasattr(self, "_M"):
            self._M = np.zeros(shape=(self.n, 3), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._M[val[1]] = val[0]
        elif isinstance(val[0], (int, float)):
            assert len(val) == 3
            self._M = np.asarray([val for _ in range(self.n)])
        elif len(val) == self.n:
            assert len(val[0]) == 3
            self._M = np.asarray(val)

    @property
    def u_ea(self) -> np.ndarray:
        return self._u_ea

    @u_ea.setter
    def u_ea(self, val) -> None:
        if not hasattr(self, "_u_ea"):
            self._u_ea = np.zeros(shape=(self.n, 3), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._u_ea[val[1]] = np.around(val[0] / np.linalg.norm(val[0]), decimals=9)
            self.M = (self.M_rem[val[1]] * self.u_ea[val[1]], val[1])
        else:
            if isinstance(val[0], (int, float)):
                val = [val for _ in range(self.n)]
            for i, ea in enumerate(val):
                self._u_ea[i] = np.around(ea / np.linalg.norm(ea), decimals=9)
                self.M = (self.M_rem[i] * self.u_ea[i], i)
                if ea[1] != 0 or ea[2] != 0:
                    w = np.array([1, 0, 0])
                else:
                    w = np.array([0, 1, 0])
                self.u_oa1 = (np.around(np.cross(self.u_ea[i], w), decimals=9), i)
                self.u_oa2 = (
                    np.around(np.cross(self.u_ea[i], self.u_oa1[i]), decimals=9),
                    i,
                )

    @property
    def u_oa1(self) -> np.ndarray:
        return self._u_oa1

    @u_oa1.setter
    def u_oa1(self, val) -> None:
        if not hasattr(self, "_u_oa1"):
            self._u_oa1 = np.zeros(shape=(self.n, 3), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._u_oa1[val[1]] = val[0]
        elif isinstance(val[0], (int, float)):
            assert len(val) == 3
            self._u_oa1 = np.asarray([val for _ in range(self.n)])
        elif len(val) == self.n:
            assert len(val[0]) == 3
            self._u_oa1 = np.asarray(val)

    @property
    def u_oa2(self) -> np.ndarray:
        return self._u_oa2

    @u_oa2.setter
    def u_oa2(self, val) -> None:
        if not hasattr(self, "_u_oa2"):
            self._u_oa2 = np.zeros(shape=(self.n, 3), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._u_oa2[val[1]] = val[0]
        elif isinstance(val[0], (int, float)):
            assert len(val) == 3
            self._u_oa2 = np.asarray([val for _ in range(self.n)])
        elif len(val) == self.n:
            assert len(val[0]) == 3
            self._u_oa2 = np.asarray(val)

    @property
    def mu_r_ea(self) -> np.ndarray:
        return self._mu_r_ea

    @mu_r_ea.setter
    def mu_r_ea(self, val) -> None:
        if not hasattr(self, "_mu_r_ea"):
            self._mu_r_ea = np.ones(shape=(self.n), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._mu_r_ea[val[1]] = val[0]
        elif isinstance(val, (int, float)):
            self._mu_r_ea = np.asarray([val for _ in range(self.n)])
        elif len(val) == self.n:
            self._mu_r_ea = np.asarray(val)

    @property
    def mu_r_oa(self) -> np.ndarray:
        return self._mu_r_oa

    @mu_r_oa.setter
    def mu_r_oa(self, val) -> None:
        if not hasattr(self, "_mu_r_oa"):
            self._mu_r_oa = np.ones(shape=(self.n), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._mu_r_oa[val[1]] = val[0]
        elif isinstance(val, (int, float)):
            self._mu_r_oa = np.asarray([val for _ in range(self.n)])
        elif len(val) == self.n:
            self._mu_r_oa = np.asarray(val)

    @property
    def M_rem(self) -> np.ndarray:
        return self._M_rem

    @M_rem.setter
    def M_rem(self, val) -> None:
        if not hasattr(self, "_M_rem"): 
            self._M_rem = np.zeros(shape=(self.n), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._M_rem[val[1]] = val[0]
        elif isinstance(val, (int, float)):
            self._M_rem = np.asarray([val for _ in range(self.n)])
        elif len(val) == self.n:
            self._M_rem = np.asarray(val)

    @property
    def color(self) -> np.ndarray:
        return self._color

    @color.setter
    def color(self, val) -> None:
        if not hasattr(self, "_color"):
            self._color = np.zeros(shape=(self.n, 3), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._color[val[1]] = np.asarray(val[0])
        elif isinstance(val[0], (int, float)):
            self._color[:] = np.asarray(val)
        elif len(val) == self.n:
            self._color = np.asarray(val)

    @property
    def magnet_type(self) -> np.ndarray:
        return self._magnet_type

    @magnet_type.setter
    def magnet_type(self, val) -> None:
        if not hasattr(self, "_magnet_type"):
            self._magnet_type = np.ones(shape=(self.n), dtype=np.int32, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._magnet_type[val[1]] = val[0]
        elif isinstance(val, (int, float)):
            self._magnet_type = np.asarray([val for _ in range(self.n)])
        elif len(val) == self.n:
            self._magnet_type = np.asarray(val)

    @property
    def stfcn_index(self) -> np.ndarray:
        return self._stfcn_index

    @stfcn_index.setter
    def stfcn_index(self, val) -> None:
        if not hasattr(self, "_stfcn_index"):
            self._stfcn_index = np.ones(shape=(self.n), dtype=np.int32, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._stfcn_index[val[1]] = val[0]
        elif isinstance(val, (int, float)):
            self._stfcn_index = np.asarray([val for _ in range(self.n)])
        elif len(val) == self.n:
            self._stfcn_index = np.asarray(val)

    @property
    def incl_it(self) -> np.ndarray:
        return self._incl_it

    @incl_it.setter
    def incl_it(self, val) -> None:
        if not hasattr(self, "_incl_it"):
            self._incl_it = np.ones(shape=(self.n), dtype=np.int32, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._incl_it[val[1]] = val[0]
        elif isinstance(val, (int, float)):
            self._incl_it = np.asarray([val for _ in range(self.n)])
        elif len(val) == self.n:
            self._incl_it = np.asarray(val)

    @property
    def M_rel(self) -> np.ndarray:
        return self._M_rel

    @M_rel.setter
    def M_rel(self, val) -> None:
        if not hasattr(self, "_M_rel"):
            self._M_rel = np.zeros(shape=(self.n), dtype=np.float64, order="F")

        if val is None:
            return
        elif isinstance(val, tuple):
            self._M_rel[val[1]] = val[0]
        elif isinstance(val, (int, float)):
            self._M_rel = [val for _ in range(self.n)]
        elif len(val) == self.n:
            self._M_rel = val

    @property
    def use_sym(self) -> np.ndarray:
        return self._use_sym

    @property
    def sym_op(self) -> np.ndarray:
        return self._sym_op

    def set_easy_axis(
        self, val: list | None = None, idx: int | None = None, seed: int = 42
    ) -> None:
        """
        polar angle [0, pi], azimuth [0, 2*pi]
        """
        if val is None:
            rng = np.random.default_rng(seed)
            rand = rng.random(size=(self.n, 2))

        if idx is None:
            for i in range(self.n):
                if val is None:
                    self._set_ea_i([np.pi, 2 * np.pi] * rand[i], i)
                elif isinstance(val[0], (int, float)):
                    self._set_ea_i(val, i)
                else:
                    self._set_ea_i(val[i], i)
        else:
            i_val = [np.pi, 2 * np.pi] * rand[idx] if val is None else val
            self._set_ea_i(i_val, idx)

    def _set_ea_i(self, val, i) -> None:
        if isinstance(val, (int, float)):
            value_err = "Both spherical angles have to be set!"
            raise TypeError(value_err)
        else:
            polar_angle, azimuth = val
            self.u_ea = (
                np.array(
                    [
                        np.sin(polar_angle) * np.cos(azimuth),
                        np.sin(polar_angle) * np.sin(azimuth),
                        np.cos(polar_angle),
                    ]
                ),
                i,
            )
            self.u_oa1 = (
                np.array(
                    [
                        np.sin(polar_angle) * np.sin(azimuth),
                        np.sin(polar_angle) * (-np.cos(azimuth)),
                        0,
                    ]
                ),
                i,
            )
            self.u_oa2 = (
                np.array(
                    [
                        0.5 * np.sin(2 * polar_angle) * np.cos(azimuth),
                        0.5 * np.sin(2 * polar_angle) * np.sin(azimuth),
                        -1 * np.sin(polar_angle) ** 2,
                    ]
                ),
                i,
            )

    def _add_tiles(self, n: int) -> None:
        self._center_pos = np.append(
            self._center_pos,
            np.zeros(shape=(n, 3), dtype=np.float64, order="F"),
            axis=0,
        )
        self._dev_center = np.append(
            self._dev_center,
            np.zeros(shape=(n, 3), dtype=np.float64, order="F"),
            axis=0,
        )
        self._size = np.append(
            self._size, np.zeros(shape=(n, 3), dtype=np.float64, order="F"), axis=0
        )
        self._vertices = np.append(
            self._vertices,
            np.zeros(shape=(n, 3, 4), dtype=np.float64, order="F"),
            axis=0,
        )
        self._M = np.append(
            self._M, np.zeros(shape=(n, 3), dtype=np.float64, order="F"), axis=0
        )
        self._u_ea = np.append(
            self._u_ea, np.zeros(shape=(n, 3), dtype=np.float64, order="F"), axis=0
        )
        self._u_oa1 = np.append(
            self._u_oa1, np.zeros(shape=(n, 3), dtype=np.float64, order="F"), axis=0
        )
        self._u_oa2 = np.append(
            self._u_oa2, np.zeros(shape=(n, 3), dtype=np.float64, order="F"), axis=0
        )
        self._mu_r_ea = np.append(
            self._mu_r_ea, np.ones(shape=(n), dtype=np.float64, order="F"), axis=0
        )
        self._mu_r_oa = np.append(
            self._mu_r_oa, np.ones(shape=(n), dtype=np.float64, order="F"), axis=0
        )
        self._M_rem = np.append(
            self._M_rem, np.zeros(shape=(n), dtype=np.float64, order="F"), axis=0
        )
        self._tile_type = np.append(
            self._tile_type, np.ones(n, dtype=np.int32, order="F"), axis=0
        )
        self._offset = np.append(
            self._offset, np.zeros(shape=(n, 3), dtype=np.float64, order="F"), axis=0
        )
        self._rot = np.append(
            self._rot, np.zeros(shape=(n, 3), dtype=np.float64, order="F"), axis=0
        )
        self._color = np.append(
            self._color, np.zeros(shape=(n, 3), dtype=np.float64, order="F"), axis=0
        )
        self._magnet_type = np.append(
            self._magnet_type, np.ones(n, dtype=np.int32, order="F"), axis=0
        )
        self._stfcn_index = np.append(
            self._stfcn_index, np.ones(shape=(n), dtype=np.int32, order="F"), axis=0
        )
        self._incl_it = np.append(
            self._incl_it, np.ones(shape=(n), dtype=np.int32, order="F"), axis=0
        )
        self._use_sym = np.append(
            self._use_sym, np.zeros(shape=(n), dtype=np.int32, order="F"), axis=0
        )
        self._sym_op = np.append(
            self._sym_op, np.ones(shape=(n, 3), dtype=np.float64, order="F"), axis=0
        )
        self._M_rel = np.append(
            self._M_rel, np.zeros(shape=(n), dtype=np.float64, order="F"), axis=0
        )
        self._n += n

    def refine_prism(self, idx: int | float | list, mat: list) -> None:
        if isinstance(idx, (int, float)):
            self._refine_prism_i(idx, mat)
        else:
            for i in idx:
                self._refine_prism_i(i, mat)

    def _refine_prism_i(self, i: int | float, mat: list) -> None:
        assert self.tile_type[i] == 2
        old_n = self.n
        self._add_tiles(np.prod(mat) - 1)
        R_mat = get_rotmat(self.rot[i])

        x_off, y_off, z_off = self.size[i] / mat
        c_0 = -self.size[i] / 2 + (self.size[i] / mat) / 2
        c_pts = [
            c_0 + np.array([x_off * j, y_off * k, z_off * m])
            for j in range(mat[0])
            for k in range(mat[1])
            for m in range(mat[2])
        ]
        ver_cube = (np.dot(R_mat, np.array(c_pts).T)).T + self.offset[i]

        for j in range(mat[0]):
            for k in range(mat[1]):
                for m in range(mat[2]):
                    cube_idx = j * mat[1] * mat[2] + k * mat[2] + m
                    idx = cube_idx + old_n
                    if idx == self.n:
                        self._offset[i] = ver_cube[cube_idx]
                        self._size[i] = self.size[i] / mat
                    else:
                        self._size[idx] = self.size[i] / mat
                        self._M[idx] = self.M[i]
                        self._M_rel[idx] = self.M_rel[i]
                        self._color[idx] = self.color[i]
                        self._magnet_type[idx] = self.magnet_type[i]
                        self._mu_r_ea[idx] = self.mu_r_ea[i]
                        self._mu_r_oa[idx] = self.mu_r_oa[i]
                        self._rot[idx] = self.rot[i]
                        self._tile_type[idx] = self.tile_type[i]
                        self._u_ea[idx] = self.u_ea[i]
                        self._u_oa1[idx] = self.u_oa1[i]
                        self._u_oa2[idx] = self.u_oa2[i]
                        self._offset[idx] = ver_cube[cube_idx]
                        self._incl_it[idx] = self.incl_it[i]



def get_rotmat(rot: np.ndarray) -> np.ndarray:
    """
    G to L in local coordinate system
    TODO: Check rotation from local to global: (1) Rot_X, (2) Rot_Y, (3) Rot_Z
    """
    rot_x = np.asarray(
        (
            [1, 0, 0],
            [0, np.cos(rot[0]), -np.sin(rot[0])],
            [0, np.sin(rot[0]), np.cos(rot[0])],
        )
    )
    rot_y = np.asarray(
        (
            [np.cos(rot[1]), 0, np.sin(rot[1])],
            [0, 1, 0],
            [-np.sin(rot[1]), 0, np.cos(rot[1])],
        )
    )
    rot_z = np.asarray(
        (
            [np.cos(rot[2]), -np.sin(rot[2]), 0],
            [np.sin(rot[2]), np.cos(rot[2]), 0],
            [0, 0, 1],
        )
    )
    return rot_x @ rot_y @ rot_z


def _euler_to_rot_axis(euler: np.ndarray) -> np.ndarray:
    """
    Converting Euler angles to rotation axis.
    For spheroids, the rotation axis has to be set rather than Euler angles.
    Rotation in MagTense performed in local coordinate system:
    Euler - (1) Rot_X_L, (2) Rot_Y_L', (3) Rot_Z_L''
    """
    # symm-axis of geometry points in the direction of z-axis of L
    # Rotates given rotation axis with pi/2 around y_L''
    # Moves x-axis of L'' to z-axis
    # ax[0] = ax[2], ax[1] = ax[1], ax[2] = -ax[0]
    R_y = np.array(
        [
            [np.cos(np.pi / 2), 0, np.sin(np.pi / 2)],
            [0, 1, 0],
            [-np.sin(np.pi / 2), 0, np.cos(np.pi / 2)],
        ]
    )
    ax = np.dot(R_y, np.asarray(euler).T)

    # Calculate the spherical coordinates: yaw and pitch
    # x_L'' has to match ax
    # Perform negative yaw around x_L and pitch around y_L'
    # The azimuthal angle is offset by pi/2 (zero position of x_L'')
    rot_x = -np.arctan2(ax[1], ax[0])
    rot_y = np.arccos(ax[2] / np.sqrt(ax[0] ** 2 + ax[1] ** 2 + ax[2] ** 2)) - np.pi / 2

    return np.array([rot_x, rot_y, 0])


def get_rotmat(rot: np.ndarray) -> np.ndarray:
    """
    G to L in local coordinate system
    TODO: Check rotation from local to global: (1) Rot_X, (2) Rot_Y, (3) Rot_Z
    """
    rot_x = np.asarray(
        (
            [1, 0, 0],
            [0, np.cos(rot[0]), -np.sin(rot[0])],
            [0, np.sin(rot[0]), np.cos(rot[0])],
        )
    )
    rot_y = np.asarray(
        (
            [np.cos(rot[1]), 0, np.sin(rot[1])],
            [0, 1, 0],
            [-np.sin(rot[1]), 0, np.cos(rot[1])],
        )
    )
    rot_z = np.asarray(
        (
            [np.cos(rot[2]), -np.sin(rot[2]), 0],
            [np.sin(rot[2]), np.cos(rot[2]), 0],
            [0, 0, 1],
        )
    )
    return rot_x @ rot_y @ rot_z

"""
CoilPy‐derived muse2tiles → local Tiles adapter
------------------------------------------------

This module re‐implements CoilPy’s `muse2magntense` + `build_prism` workflow
to populate our local `Tiles` container, removing the original MagTense‐GPU
dependency.

Adapted from:
  - CoilPy GitHub: https://github.com/zhucaoxiang/CoilPy  
    (see `coilpy.magtense_interface.muse2magntense`)  
  - CoilPy authors: Zhu, Cao et al.
"""


def muse2tiles(muse_file, mu=(1.05, 1.05), magnetization=1.16e6, **kwargs):
    data = np.loadtxt(muse_file, skiprows=1, delimiter=",")
    nmag = len(data)
    print(f"{nmag} magnets are identified in {muse_file!r}.")

    center = np.zeros((nmag, 3))
    rot    = np.zeros((nmag, 3))
    lwh    = np.zeros((nmag, 3))
    ang    = np.zeros((nmag, 2))
    mu_arr = np.repeat([mu], nmag, axis=0)
    Br     = np.zeros(nmag)

    for i in range(nmag):
        center[i] = data[i, 0:3]
        lwh[i]    = data[i, 3:6]
        n1, n2, n3 = data[i, 6:9], data[i, 9:12], data[i, 12:15]

        # build normalized basis → Euler angles
        rot_mat = np.vstack([
            n1 / np.linalg.norm(n1),
            n2 / np.linalg.norm(n2),
            n3 / np.linalg.norm(n3),
        ])
        rot[i] = rotation_angle(rot_mat.T, xyz=True)

        # magnetization angles
        mxyz = data[i, 15:18]
        mp   = math.atan2(mxyz[1], mxyz[0])
        mt   = math.acos(mxyz[2] / np.linalg.norm(mxyz))
        ang[i] = (mt, mp)

        Br[i] = magnetization

    return build_prism(lwh, center, rot, ang, mu_arr, Br)


def build_prism(lwh, center, rot, mag_angle, mu, remanence):
    """
    Construct a prism in our local Tiles container (no bind helper needed).

    Args:
        lwh        (n x 3): half dims (i.e. half lengths)
        center     (n x 3): centers
        rot        (n x 3): Euler angles
        mag_angle  (n x 2): (theta,phi)
        mu         (n x 2): (mu_r_ea, mu_r_oa)
        remanence  (n,) : M_rem
    """
    lwh       = np.atleast_2d(lwh)
    nmag      = lwh.shape[0]
    center    = np.atleast_2d(center)
    rot       = np.atleast_2d(rot)
    mag_angle = np.atleast_2d(mag_angle)
    mu        = np.atleast_2d(mu)
    rem       = np.atleast_1d(remanence)

    prism = Tiles(nmag)

    # set broadcast properties
    prism.tile_type = 2         # all prisms

    for i in range(nmag):
        # tuple‐setter form writes into the i-th row of each array
        prism.size     = (lwh[i],      i)
        prism.offset   = (center[i],   i)
        prism.rot      = (rot[i],      i)
        prism.M_rem    = (rem[i],      i)
        prism.mu_r_ea  = (mu[i,0],     i)
        prism.mu_r_oa  = (mu[i,1],     i)
        prism.set_easy_axis(val=mag_angle[i], idx=i)
        prism.color    = ([0, 0, (i+1)/nmag], i)

    return prism