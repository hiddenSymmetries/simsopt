import warnings
import numpy as np
from simsopt.geo import CurveRZFourier
from scipy.integrate import solve_ivp

__all__ = ['compute_on_axis_iota', 'compute_on_axis_tangent_map']


def compute_on_axis_tangent_map(axis, magnetic_field):
    """
    Computes the tangent map of the field-line map over one field period, linearized about the magnetic axis, by
    integrating the variational equation along the axis (equation (13) of Greene, Journal of Mathematical Physics 20,
    1183 (1979); doi: 10.1063/1.524170). Its trace says what kind of closed field line the axis is: elliptic for
    ``|trace| < 2`` (field lines nearby turn around it, and the axis has a rotational transform), hyperbolic for
    ``trace > 2``, and reflection-hyperbolic for ``trace < -2`` (field lines nearby land on the opposite side of the axis
    every period and move away from it; the nearby field lines lie on Klein bottles; Smiet, Phys. Plasmas 32,
    102503 (2025)). NOTE: this function does not check that the provided axis is actually a magnetic axis of the
    input magnetic_field.

    Args:
        axis: a CurveRZFourier corresponding to the magnetic axis of the magnetic field.
        magnetic_field: the magnetic field in which the axis lies.

    Returns:
        M: the 2x2 tangent map over one field period (its transpose, which has the same trace and eigenvalues).
    """
    assert type(axis) is CurveRZFourier

    def tangent_map(phi, x):
        ind = np.array(phi)
        out = np.zeros((1, 3))
        axis.gamma_impl(out, ind)
        magnetic_field.set_points(out)

        out = out.flatten()
        B = magnetic_field.B().flatten()
        dB_by_dX = magnetic_field.dB_by_dX().reshape((3, 3))
        B1 = B[0]
        B2 = B[1]
        B3 = B[2]

        dB1_dx = dB_by_dX[0, 0]
        dB1_dy = dB_by_dX[0, 1]
        dB1_dz = dB_by_dX[0, 2]

        dB2_dx = dB_by_dX[1, 0]
        dB2_dy = dB_by_dX[1, 1]
        dB2_dz = dB_by_dX[1, 2]

        dB3_dx = dB_by_dX[2, 0]
        dB3_dy = dB_by_dX[2, 1]
        dB3_dz = dB_by_dX[2, 2]

        c = np.cos(2*np.pi*phi)
        s = np.sin(2*np.pi*phi)

        R = np.sqrt(out[0]**2 + out[1]**2)
        dx_dR = c
        dy_dR = s

        BR = c*B1 + s*B2
        Bphi = -s*B1 + c*B2
        BZ = B3

        dB1_dR = dB1_dx * dx_dR + dB1_dy * dy_dR
        dB2_dR = dB2_dx * dx_dR + dB2_dy * dy_dR
        dB3_dR = dB3_dx * dx_dR + dB3_dy * dy_dR

        dBR_dR = c*dB1_dR + s*dB2_dR
        dBphi_dR = -s*dB1_dR + c*dB2_dR
        dBZ_dR = dB3_dR

        dBR_dZ = c*dB1_dz + s*dB2_dz
        dBphi_dZ = -s*dB1_dz + c*dB2_dz
        dBZ_dZ = dB3_dz

        Bphi_R = Bphi/R
        d_Bphi_R_dR = (dBphi_dR * R - Bphi)/R**2
        d_Bphi_R_dZ = dBphi_dZ/R

        A11 = (dBR_dR - (BR / Bphi_R) * d_Bphi_R_dR) / Bphi_R
        A21 = (dBZ_dR - (BZ / Bphi_R) * d_Bphi_R_dR) / Bphi_R
        A12 = (dBR_dZ - (BR / Bphi_R) * d_Bphi_R_dZ) / Bphi_R
        A22 = (dBZ_dZ - (BZ / Bphi_R) * d_Bphi_R_dZ) / Bphi_R
        A = np.array([[A11, A12], [A21, A22]])
        return 2*np.pi*np.array([A@x[:2], A@x[2:]]).flatten()

    t_span = [0, 1/axis.nfp]
    t_eval = t_span

    y0 = np.array([1, 0, 0, 1])
    results = solve_ivp(tangent_map, t_span, y0, t_eval=t_eval, rtol=1e-12, atol=1e-12, method='RK45')
    return results.y[:, -1].reshape((2, 2))


def compute_on_axis_iota(axis, magnetic_field):
    """
    Computes the rotational transform on the magnetic axis of a device using a method based on
    equation (13) of Greene, Journal of Mathematical Physics 20, 1183 (1979); doi: 10.1063/1.524170.
    This method was shared by Stuart Hudson and Matt Landreman.  NOTE: this function does not check that the provided
    axis is actually a magnetic axis of the input magnetic_field.

    The rotational transform is defined only on an elliptic axis (``|trace| < 2`` for the tangent map over one field
    period, see :func:`compute_on_axis_tangent_map`). On a reflection-hyperbolic axis (trace below -2) this function
    returns ``nfp/2``, and on a hyperbolic axis (trace above 2) it returns 0; in both cases it warns, since the value
    returned is then not a measurement.

    Args:
        axis: a CurveRZFourier corresponding to the magnetic axis of the magnetic field.
        magnetic_field: the magnetic field in which the axis lies.

    Returns:
        iota: the rotational transform on the given axis.
    """
    M = compute_on_axis_tangent_map(axis, magnetic_field)
    trace = np.trace(M)
    if abs(trace) >= 2:
        kind = 'reflection-hyperbolic' if trace < 0 else 'hyperbolic'
        warnings.warn(f"compute_on_axis_iota: the axis is {kind} (trace of its one-period tangent map {trace:.8f}, "
                      f"{'below -2' if trace < 0 else 'above 2'}): it has no rotational transform, and the value returned "
                      f"({axis.nfp / 2 if trace < 0 else 0}) is not a measurement.", RuntimeWarning, stacklevel=2)
    evals, evecs = np.linalg.eig(M)
    iota = np.arctan2(np.imag(evals[0]), np.real(evals[0])) * axis.nfp/(2*np.pi)
    return iota
