import numpy as np
from scipy.linalg import lu
from scipy.optimize import minimize, least_squares
import simsoptpp as sopp

from .surfaceobjectives import boozer_surface_residual, boozer_surface_dexactresidual_dcoils_dcurrents_vjp, boozer_surface_dlsqgrad_dcoils_vjp, MU0, finite_beta_boozer_residual, finite_beta_boozer_residual_dsurface, finite_beta_boozer_residual_dparameters, finite_beta_boozer_surface_field, finite_beta_sheet_current, finite_beta_virtual_casing_residual
from .._core.optimizable import Optimizable
from functools import partial

__all__ = ['BoozerSurface', 'FiniteBetaFieldProvider', 'SurfaceCurrentFieldProvider', 'FiniteBetaBoozerSurface']


class FiniteBetaFieldProvider:
    """
    Base class for finite-beta magnetic-field providers.

    Providers can reevaluate the interface fields from the current nonlinear
    state, enabling no-VC closures that depend on the evolving surface current.
    """

    depends_on_state = True

    def evaluate(self, surface, iota, G, I, current_potential,
                 biotsavart=None, finite_beta_surface=None):
        raise NotImplementedError()


class SurfaceCurrentFieldProvider(FiniteBetaFieldProvider):
    r"""
    Evaluate the exterior finite-beta interface field from a surface-current
    Biot-Savart model while keeping the interior field on the Boozer branch.

    The interior field is taken directly from the single-surface Boozer model,
    so the scalar state variables ``iota``, ``G``, and ``I`` immediately affect
    ``B_in``. The exterior field is obtained from the coil field plus a direct
    principal-value boundary integral of the sheet current and the exact
    one-sided jump

    .. math::
        B_\text{out} - B_\text{in} = \mu_0 K \times \hat n.

    The coil field is evaluated directly on the interface, since it is
    continuous across the surface.
    """

    depends_on_state = True

    def __init__(self, offset_distance=None, offset_scale=0.25,
                 min_offset=1e-5, chunk_size=None, include_biotsavart=True):
        self.offset_distance = offset_distance
        self.offset_scale = float(offset_scale)
        self.min_offset = float(min_offset)
        self.chunk_size = chunk_size
        self.include_biotsavart = bool(include_biotsavart)

    def _quadrature_weights(self, surface):
        if len(surface.quadpoints_phi) < 2 or len(surface.quadpoints_theta) < 2:
            raise ValueError('SurfaceCurrentFieldProvider requires at least 2 quadrature points in each direction.')

        dphi = np.diff(surface.quadpoints_phi)
        dtheta = np.diff(surface.quadpoints_theta)
        if not np.allclose(dphi, dphi[0]):
            raise ValueError('SurfaceCurrentFieldProvider requires a uniformly spaced toroidal quadrature grid.')
        if not np.allclose(dtheta, dtheta[0]):
            raise ValueError('SurfaceCurrentFieldProvider requires a uniformly spaced poloidal quadrature grid.')

        return np.linalg.norm(surface.normal(), axis=2) * float(dphi[0]) * float(dtheta[0])

    def _effective_offset(self, surface):
        if self.offset_distance is not None:
            return float(self.offset_distance)

        gamma = surface.gamma()
        phi_spacing = np.mean(np.linalg.norm(np.roll(gamma, -1, axis=0) - gamma, axis=2))
        theta_spacing = np.mean(np.linalg.norm(np.roll(gamma, -1, axis=1) - gamma, axis=2))
        spacing = min(float(phi_spacing), float(theta_spacing))
        if not np.isfinite(spacing) or spacing <= 0.0:
            spacing = 1.0
        return max(self.min_offset, self.offset_scale * spacing)

    def _sheet_field(self, source_points, sheet_current, weights, evaluation_points):
        factor = MU0 / (4.0 * np.pi)
        target_count = evaluation_points.shape[0]
        source_points = source_points.reshape((-1, 3))
        sheet_current = sheet_current.reshape((-1, 3))
        weights = weights.reshape((-1,))

        if self.chunk_size is None or self.chunk_size <= 0:
            chunk_size = target_count
        else:
            chunk_size = int(self.chunk_size)

        result = np.zeros((target_count, 3))
        for start in range(0, target_count, chunk_size):
            stop = min(start + chunk_size, target_count)
            delta = evaluation_points[start:stop, None, :] - source_points[None, :, :]
            distance_sq = np.sum(delta**2, axis=2)
            distance_sq = np.maximum(distance_sq, 1e-30)
            distance_cubed = distance_sq * np.sqrt(distance_sq)
            kernel = np.cross(sheet_current[None, :, :], delta, axis=2) / distance_cubed[:, :, None]
            result[start:stop, :] = factor * np.sum(kernel * weights[None, :, None], axis=1)
        return result

    def _principal_value_sheet_field(self, source_points, sheet_current, weights,
                                     evaluation_points, self_source_indices=None):
        factor = MU0 / (4.0 * np.pi)
        target_count = evaluation_points.shape[0]
        source_points = source_points.reshape((-1, 3))
        sheet_current = sheet_current.reshape((-1, 3))
        weights = weights.reshape((-1,))

        if self.chunk_size is None or self.chunk_size <= 0:
            chunk_size = target_count
        else:
            chunk_size = int(self.chunk_size)

        result = np.zeros((target_count, 3))
        for start in range(0, target_count, chunk_size):
            stop = min(start + chunk_size, target_count)
            delta = evaluation_points[start:stop, None, :] - source_points[None, :, :]
            distance_sq = np.sum(delta**2, axis=2)
            if self_source_indices is not None:
                local_self = np.asarray(self_source_indices[start:stop], dtype=int)
                distance_sq[np.arange(stop - start), local_self] = np.inf
            distance_sq = np.maximum(distance_sq, 1e-30)
            distance_cubed = distance_sq * np.sqrt(distance_sq)
            kernel = np.cross(sheet_current[None, :, :], delta, axis=2) / distance_cubed[:, :, None]
            result[start:stop, :] = factor * np.sum(kernel * weights[None, :, None], axis=1)
        return result

    def _symmetry_matrix(self, angle, flip):
        rotmat = np.asarray(
            [[np.cos(angle), -np.sin(angle), 0.0],
             [np.sin(angle),  np.cos(angle), 0.0],
             [0.0,            0.0,           1.0]],
            dtype=float,
        ).T
        if flip:
            rotmat = rotmat @ np.asarray(
                [[1.0, 0.0, 0.0],
                 [0.0, -1.0, 0.0],
                 [0.0, 0.0, -1.0]],
                dtype=float,
            )
        return rotmat

    def _expand_sheet_sources(self, surface, source_points, sheet_current, weights):
        nfp = int(getattr(surface, 'nfp', 1))

        expanded_points = []
        expanded_currents = []
        expanded_weights = []

        for k in range(nfp):
            angle = 2.0 * np.pi * k / nfp
            rotmat = self._symmetry_matrix(angle, False)
            transformed_points = source_points @ rotmat
            transformed_current = sheet_current @ rotmat
            expanded_points.append(transformed_points)
            expanded_currents.append(transformed_current)
            expanded_weights.append(weights)

        return (
            np.concatenate(expanded_points, axis=0),
            np.concatenate(expanded_currents, axis=0),
            np.concatenate(expanded_weights, axis=0),
        )

    def _coil_field(self, biotsavart, points, shape):
        if biotsavart is None or not self.include_biotsavart:
            return np.zeros(shape)

        biotsavart.set_points(points)
        biotsavart.compute(0)
        return biotsavart.B().reshape(shape)

    def evaluate(self, surface, iota, G, I, current_potential,
                 biotsavart=None, finite_beta_surface=None):
        del finite_beta_surface

        potential = np.zeros(surface.gamma().shape[:2]) if current_potential is None else np.asarray(current_potential, dtype=float)
        if potential.shape != surface.gamma().shape[:2]:
            raise ValueError(f'current_potential must have shape {surface.gamma().shape[:2]}.')

        gamma = surface.gamma()
        unitnormal = surface.unitnormal()
        weights = self._quadrature_weights(surface)
        sheet_current = finite_beta_sheet_current(surface, current_potential=potential)
        source_points, source_sheet_current, source_weights = self._expand_sheet_sources(
            surface,
            gamma.reshape((-1, 3)),
            sheet_current.reshape((-1, 3)),
            weights.reshape((-1,)),
        )
        B_sheet_pv = self._principal_value_sheet_field(
            source_points,
            source_sheet_current,
            source_weights,
            gamma.reshape((-1, 3)),
            self_source_indices=np.arange(gamma.shape[0] * gamma.shape[1]),
        ).reshape(gamma.shape)
        half_jump = 0.5 * MU0 * np.cross(sheet_current, unitnormal)
        coil_field = self._coil_field(biotsavart, gamma.reshape((-1, 3)), gamma.shape)
        B_in = finite_beta_boozer_surface_field(surface, iota=iota, G=G, I=I)
        B_out = coil_field + B_sheet_pv + half_jump
        return B_in, B_out


class FiniteBetaBoozerSurface(Optimizable):
    r"""
    Entry point for the upcoming finite-beta Boozer-surface workflow.

    The finite-beta formulation extends the vacuum Boozer-surface solve with a
    pressure-jump model, an internal current scalar ``I``, and a sheet-current
    potential ``Phi`` on the target surface. The full residual assembly and solve
    are introduced incrementally, so this class presently provides the validated
    constructor-level API, option handling, and pressure-jump resolution logic
    needed by the later implementation phases.

    Args:
        biotsavart (:obj:`~simsopt.field.BiotSavart`): Coil field used to seed the
            finite-beta solve.
        surface (:obj:`~simsopt.geo.SurfaceXYZFourier`,
            :obj:`~simsopt.geo.SurfaceXYZTensorFourier`): Target surface.
        label (:obj:`~simsopt._core.optimizable.Optimizable`): Surface label
            evaluator, e.g. :obj:`~simsopt.geo.Volume` or
            :obj:`~simsopt.geo.Area`.
        targetlabel (float): Target value of the label on the surface.
        pressure_jump (float or callable, optional): Pressure jump on the target
            surface. If callable, it may accept either no arguments or the
            surface object and must return a scalar.
        virtual_casing (optional): Virtual-casing helper or cached data object to
            be consumed by the finite-beta solve.
        constraint_weight (float, optional): Residual weight used by the planned
            least-squares solve.
        options (dict, optional): Solver options for the finite-beta workflow.
    """

    def __init__(self, biotsavart, surface, label, targetlabel,
                 pressure_jump=None, virtual_casing=None,
                 constraint_weight=1.0, options=None):
        depends_on = [biotsavart] if biotsavart is not None else []
        super().__init__(depends_on=depends_on)

        from simsopt.geo import SurfaceXYZFourier, SurfaceXYZTensorFourier
        if not isinstance(surface, SurfaceXYZTensorFourier) and not isinstance(surface, SurfaceXYZFourier):
            raise Exception("The input surface must be a SurfaceXYZTensorFourier or SurfaceXYZFourier.")

        self.biotsavart = biotsavart
        self.surface = surface
        self.label = label
        self.targetlabel = targetlabel
        self.pressure_jump = pressure_jump
        self.virtual_casing = virtual_casing
        self.constraint_weight = constraint_weight
        self.solve_type = 'ls'
        self.need_to_run_code = True
        self.res = None

        if options is None:
            options = {}
        if 'verbose' not in options:
            options['verbose'] = True
        if 'newton_tol' not in options:
            options['newton_tol'] = 1e-11
        if 'newton_maxiter' not in options:
            options['newton_maxiter'] = 40
        if 'weight_inv_modB' not in options:
            options['weight_inv_modB'] = True
        self.options = options

    def _default_G(self):
        if self.biotsavart is None:
            return 0.0
        return 2. * np.pi * np.sum([np.abs(c.current.get_value()) for c in self.biotsavart.coils]) * (4 * np.pi * 10**(-7) / (2 * np.pi))

    def _normalize_current_potential(self, current_potential=None):
        shape = self.surface.gamma().shape[:2]
        if current_potential is None:
            potential = np.zeros(shape)
        else:
            potential = np.asarray(current_potential, dtype=float)
            if potential.shape != shape:
                raise ValueError(f"current_potential must have shape {shape}.")
            potential = potential.copy()
        potential -= potential[0, 0]
        return potential

    def _pack_state(self, iota, G, I, current_potential,
                    optimize_iota, optimize_G, optimize_I, optimize_current_potential,
                    optimize_surface=False, surface_dofs=None):
        x0 = []
        if optimize_surface:
            x0.extend(surface_dofs)
        if optimize_iota:
            x0.append(float(iota))
        if optimize_G:
            x0.append(float(G))
        if optimize_I:
            x0.append(float(I))
        if optimize_current_potential:
            x0.extend(current_potential.reshape((-1,))[1:])
        return np.asarray(x0, dtype=float)

    def _unpack_state(self, vector, iota, G, I, current_potential,
                      optimize_iota, optimize_G, optimize_I, optimize_current_potential,
                      optimize_surface=False, nsurfdofs=0):
        cursor = 0
        surface_dofs_val = None
        iota_val = float(iota)
        G_val = float(G)
        I_val = float(I)
        current_potential_val = current_potential.copy()

        if optimize_surface:
            surface_dofs_val = np.asarray(vector[cursor:cursor + nsurfdofs], dtype=float)
            cursor += nsurfdofs

        if optimize_iota:
            iota_val = float(vector[cursor])
            cursor += 1
        if optimize_G:
            G_val = float(vector[cursor])
            cursor += 1
        if optimize_I:
            I_val = float(vector[cursor])
            cursor += 1
        if optimize_current_potential:
            flattened = current_potential_val.reshape((-1,))
            flattened[1:] = vector[cursor:cursor + flattened.size - 1]
            cursor += flattened.size - 1
            current_potential_val = flattened.reshape(current_potential_val.shape)
        current_potential_val -= current_potential_val[0, 0]

        return surface_dofs_val, iota_val, G_val, I_val, current_potential_val

    def _weighted_residual_vector(self, blocks):
        weights = self.options.get('block_weights', {})
        return np.concatenate([
            weights.get('boozer', 1.0) * blocks['boozer'].reshape((-1,)),
            weights.get('normal', 1.0) * blocks['normal'].reshape((-1,)),
            weights.get('pressure', 1.0) * blocks['pressure'].reshape((-1,)),
            weights.get('jump', 1.0) * blocks['jump'].reshape((-1,)),
        ])

    def _weighted_parameter_jacobian(self, jacobian):
        weights = self.options.get('block_weights', {})
        nphi, ntheta = self.surface.gamma().shape[:2]
        row_sizes = [
            ('boozer', nphi * ntheta * 3),
            ('normal', nphi * ntheta),
            ('pressure', nphi * ntheta),
            ('jump', nphi * ntheta * 3),
        ]
        pieces = []
        cursor = 0
        for name, size in row_sizes:
            pieces.append(weights.get(name, 1.0) * jacobian[cursor:cursor + size, :])
            cursor += size
        return np.concatenate(pieces, axis=0)

    def _weighted_surface_jacobian(self, jacobian):
        return self._weighted_parameter_jacobian(jacobian)

    def _vc_block_weights(self, pressure_jump=0.0, B_coils=None):
        weights = self.options.get('vc_block_weights')
        if weights is not None:
            return {
                'coil_match': float(weights.get('coil_match', 1.0)),
                'normal': float(weights.get('normal', 1.0)),
                'pressure': float(weights.get('pressure', 1.0)),
            }

        if B_coils is None:
            B_scale = 1.0
        else:
            B_scale = np.sqrt(np.mean(np.sum(np.asarray(B_coils)**2, axis=2)))
            B_scale = max(float(B_scale), 1e-12)

        pressure_scale = max(B_scale**2, 2.0 * MU0 * abs(float(pressure_jump)), 1e-12)
        return {
            'coil_match': 1.0 / B_scale,
            'normal': 1.0 / B_scale,
            'pressure': 1.0 / pressure_scale,
        }

    def _weighted_vc_residual_vector(self, blocks, pressure_jump=0.0, B_coils=None):
        weights = self._vc_block_weights(pressure_jump=pressure_jump, B_coils=B_coils)
        return np.concatenate([
            weights['coil_match'] * blocks['coil_match'].reshape((-1,)),
            weights['normal'] * blocks['normal'].reshape((-1,)),
            weights['pressure'] * blocks['pressure'].reshape((-1,)),
        ])

    def _weighted_vc_parameter_jacobian(self, jacobian_blocks, pressure_jump=0.0, B_coils=None):
        weights = self._vc_block_weights(pressure_jump=pressure_jump, B_coils=B_coils)
        return np.concatenate([
            weights['coil_match'] * jacobian_blocks['coil_match'].reshape((-1, jacobian_blocks['coil_match'].shape[-1])),
            weights['normal'] * jacobian_blocks['normal'].reshape((-1, jacobian_blocks['normal'].shape[-1])),
            weights['pressure'] * jacobian_blocks['pressure'].reshape((-1, jacobian_blocks['pressure'].shape[-1])),
        ], axis=0)

    def single_surface_vc_parameter_jacobian(self, iota, G=None, I=0.0, lambda_current=None,
                                             pressure_jump=None, B_coils=None, vc_digits=None,
                                             optimize_iota=True, optimize_lambda_current=True,
                                             virtual_casing=None, B_total=None, B_external=None):
        from ..mhd.virtual_casing import VirtualCasing

        if G is None:
            G = self._default_G()
        if lambda_current is None:
            lambda_current = float(G) + float(iota) * float(I)
        if pressure_jump is None:
            pressure_jump = self.resolve_pressure_jump()
        if pressure_jump is None:
            pressure_jump = 0.0
        if vc_digits is None:
            vc_digits = self.options.get('vc_digits', 6)

        if B_coils is None:
            if self.biotsavart is None:
                raise ValueError('biotsavart is required for the single-surface virtual-casing Jacobian.')
            x = self.surface.gamma().reshape((-1, 3))
            self.biotsavart.set_points(x)
            self.biotsavart.compute(0)
            B_coils = self.biotsavart.B().reshape(self.surface.gamma().shape)
        else:
            B_coils = np.asarray(B_coils)

        xphi = self.surface.gammadash1()
        xtheta = self.surface.gammadash2()
        tang = xphi + float(iota) * xtheta
        tang_norm_sq = np.sum(tang**2, axis=2)
        tang_dot_xtheta = np.sum(tang * xtheta, axis=2)

        if B_total is None:
            B_total = finite_beta_boozer_surface_field(
                self.surface,
                iota=iota,
                lambda_current=lambda_current,
            )
        if virtual_casing is None:
            vc = VirtualCasing.from_surface(self.surface, B_total, digits=vc_digits)
        else:
            vc = virtual_casing
        if B_external is None:
            B_external = vc.B_external
        unit_normal = self.surface.unitnormal()

        dB_total_diota = float(lambda_current) * (
            xtheta / tang_norm_sq[:, :, None]
            - 2.0 * tang * tang_dot_xtheta[:, :, None] / tang_norm_sq[:, :, None]**2
        )
        dB_total_dlambda = tang / tang_norm_sq[:, :, None]

        columns = []
        names = []

        if optimize_iota:
            dB_external_diota = vc.compute_external_B(dB_total_diota)
            columns.append((dB_total_diota, dB_external_diota))
            names.append('iota')
        if optimize_lambda_current:
            dB_external_dlambda = vc.compute_external_B(dB_total_dlambda)
            columns.append((dB_total_dlambda, dB_external_dlambda))
            names.append('lambda_current')

        if len(columns) == 0:
            reference = self.self_consistent_single_surface_residual(
                iota=iota,
                G=G,
                I=I,
                lambda_current=lambda_current,
                pressure_jump=pressure_jump,
                B_coils=B_coils,
                vc_digits=vc_digits,
            )
            return np.zeros((self._weighted_vc_residual_vector(reference['blocks'], pressure_jump=pressure_jump, B_coils=B_coils).size, 0))

        jacobian_blocks = {'coil_match': [], 'normal': [], 'pressure': []}
        B_total_sq = np.sum(B_total**2, axis=2)
        B_external_sq = np.sum(B_external**2, axis=2)
        _ = (B_total_sq, B_external_sq)
        for dB_total, dB_external in columns:
            jacobian_blocks['coil_match'].append(dB_external)
            jacobian_blocks['normal'].append(np.sum(dB_external * unit_normal, axis=2))
            jacobian_blocks['pressure'].append(
                2.0 * np.sum(B_external * dB_external, axis=2)
                - 2.0 * np.sum(B_total * dB_total, axis=2)
            )

        stacked = {
            'coil_match': np.stack(jacobian_blocks['coil_match'], axis=-1),
            'normal': np.stack(jacobian_blocks['normal'], axis=-1),
            'pressure': np.stack(jacobian_blocks['pressure'], axis=-1),
        }
        return self._weighted_vc_parameter_jacobian(stacked, pressure_jump=pressure_jump, B_coils=B_coils)

    def single_surface_vc_surface_jacobian(self, iota, G=None, I=0.0, lambda_current=None,
                                           pressure_jump=None, B_coils=None, vc_digits=None,
                                           baseline_result=None, baseline_weighted_residual=None,
                                           fd_rel_step=None):
        """
        Approximate the weighted VC residual Jacobian with respect to the surface
        dofs while keeping the exact parameter columns for ``iota`` and
        ``lambda_current`` available separately.

        The underlying virtual-casing extension does not expose derivatives of
        the operator with respect to the source surface geometry, so this method
        computes a structured forward-difference surface block around the current
        geometry. Using this block together with exact scalar-parameter columns
        is significantly more informative than a full black-box finite-difference
        Jacobian on all variables.
        """
        if G is None:
            G = self._default_G()
        if lambda_current is None:
            lambda_current = float(G) + float(iota) * float(I)
        if pressure_jump is None:
            pressure_jump = self.resolve_pressure_jump()
        if pressure_jump is None:
            pressure_jump = 0.0
        if vc_digits is None:
            vc_digits = self.options.get('vc_digits', 6)
        if fd_rel_step is None:
            fd_rel_step = self.options.get('vc_surface_fd_rel_step', 1e-7)

        surface_dofs0 = self.surface.x.copy()
        if baseline_result is None:
            baseline_result = self.self_consistent_single_surface_residual(
                iota=iota,
                G=G,
                I=I,
                lambda_current=lambda_current,
                pressure_jump=pressure_jump,
                B_coils=B_coils,
                vc_digits=vc_digits,
            )
        if baseline_weighted_residual is None:
            baseline_weighted_residual = self._weighted_vc_residual_vector(
                baseline_result['blocks'],
                pressure_jump=pressure_jump,
                B_coils=baseline_result['B_coils'],
            )

        jacobian = np.zeros((baseline_weighted_residual.size, surface_dofs0.size))
        steps = np.maximum(np.abs(surface_dofs0), 1.0) * float(fd_rel_step)
        steps = np.where(steps > 0.0, steps, float(fd_rel_step))

        try:
            for idx, step in enumerate(steps):
                perturbed = surface_dofs0.copy()
                perturbed[idx] += step
                self.surface.x = perturbed
                perturbed_result = self.self_consistent_single_surface_residual(
                    iota=iota,
                    G=G,
                    I=I,
                    lambda_current=lambda_current,
                    pressure_jump=pressure_jump,
                    B_coils=B_coils,
                    vc_digits=vc_digits,
                )
                perturbed_weighted = self._weighted_vc_residual_vector(
                    perturbed_result['blocks'],
                    pressure_jump=pressure_jump,
                    B_coils=perturbed_result['B_coils'],
                )
                jacobian[:, idx] = (perturbed_weighted - baseline_weighted_residual) / step
        finally:
            self.surface.x = surface_dofs0

        return jacobian

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True

    def resolve_pressure_jump(self):
        """
        Evaluate the configured pressure jump as a finite scalar.
        """
        if self.pressure_jump is None:
            return None

        if callable(self.pressure_jump):
            try:
                value = self.pressure_jump(self.surface)
            except TypeError:
                value = self.pressure_jump()
        else:
            value = self.pressure_jump

        value = float(value)
        if not np.isfinite(value):
            raise ValueError("pressure_jump must evaluate to a finite scalar.")
        return value

    def _field_provider_depends_on_state(self, field_provider):
        return field_provider is not None and bool(getattr(field_provider, 'depends_on_state', True))

    def _evaluate_field_provider(self, field_provider, iota, G, I, current_potential):
        if hasattr(field_provider, 'evaluate'):
            fields = field_provider.evaluate(
                self.surface,
                iota=iota,
                G=G,
                I=I,
                current_potential=current_potential,
                biotsavart=self.biotsavart,
                finite_beta_surface=self,
            )
        elif callable(field_provider):
            fields = field_provider(
                self.surface,
                iota=iota,
                G=G,
                I=I,
                current_potential=current_potential,
                biotsavart=self.biotsavart,
                finite_beta_surface=self,
            )
        else:
            raise ValueError('field_provider must be callable or implement an evaluate(...) method.')

        if not isinstance(fields, (tuple, list)) or len(fields) != 2:
            raise ValueError('field_provider must return a pair (B_in, B_out).')
        return np.asarray(fields[0]), np.asarray(fields[1])

    def resolve_field_components(self, B_in=None, B_out=None, field_provider=None,
                                 iota=None, G=None, I=0.0, current_potential=None):
        """
        Return interior and exterior magnetic fields on the surface grid.

        Explicit arrays take precedence. If they are omitted and a virtual casing
        object with ``B_total`` and ``B_external`` is attached, those arrays are
        used. Otherwise, the Biot-Savart field is used on both sides, giving the
        vacuum limit.
        """
        if field_provider is not None and (B_in is not None or B_out is not None):
            raise ValueError('field_provider cannot be combined with explicit B_in/B_out inputs.')

        if (B_in is None) != (B_out is None):
            raise ValueError("B_in and B_out must be provided together.")

        expected_shape = self.surface.gamma().shape
        if field_provider is not None:
            if iota is None or G is None:
                raise ValueError('field_provider evaluation requires iota and G.')
            B_in_eval, B_out_eval = self._evaluate_field_provider(field_provider, iota, G, I, current_potential)
            if B_in_eval.shape != expected_shape or B_out_eval.shape != expected_shape:
                raise ValueError(f'field_provider must return arrays with shape {expected_shape}.')
            return B_in_eval, B_out_eval

        if callable(B_in) or callable(B_out):
            if not callable(B_in) or not callable(B_out):
                raise ValueError("B_in and B_out must either both be callables or both be arrays.")
            B_in = np.asarray(B_in(self.surface))
            B_out = np.asarray(B_out(self.surface))
            if B_in.shape != expected_shape or B_out.shape != expected_shape:
                raise ValueError(f"Callable B_in and B_out must return arrays with shape {expected_shape}.")
            return B_in, B_out

        if B_in is not None:
            B_in = np.asarray(B_in)
            B_out = np.asarray(B_out)
            if B_in.shape != expected_shape or B_out.shape != expected_shape:
                raise ValueError(f"B_in and B_out must have shape {expected_shape}.")
            return B_in, B_out

        vc = self.virtual_casing
        if vc is not None and hasattr(vc, 'B_total') and hasattr(vc, 'B_external'):
            B_total = np.asarray(vc.B_total)
            B_external = np.asarray(vc.B_external)
            if B_total.shape != expected_shape or B_external.shape != expected_shape:
                raise ValueError(
                    f"virtual_casing.B_total and virtual_casing.B_external must have shape {expected_shape}."
                )
            return B_total - B_external, B_external

        if self.biotsavart is None:
            raise ValueError("biotsavart is required when explicit fields or virtual_casing data are not provided.")

        x = self.surface.gamma().reshape((-1, 3))
        self.biotsavart.set_points(x)
        self.biotsavart.compute(0)
        B = self.biotsavart.B().reshape(expected_shape)
        return B, B

    def residual_blocks(self, iota, G, I=0.0, pressure_jump=None,
                        current_potential=None, current_potential_derivatives=None,
                        B_in=None, B_out=None, field_provider=None):
        resolved_pressure_jump = self.resolve_pressure_jump() if pressure_jump is None else pressure_jump
        if resolved_pressure_jump is None:
            resolved_pressure_jump = 0.0

        resolved_B_in, resolved_B_out = self.resolve_field_components(
            B_in=B_in,
            B_out=B_out,
            field_provider=field_provider,
            iota=iota,
            G=G,
            I=I,
            current_potential=current_potential,
        )
        return finite_beta_boozer_residual(
            self.surface,
            iota,
            G,
            resolved_B_in,
            resolved_B_out,
            I=I,
            pressure_jump=resolved_pressure_jump,
            current_potential=current_potential,
            current_potential_derivatives=current_potential_derivatives,
        )

    def residual_parameter_jacobian(self, iota, G, I=0.0,
                                    B_in=None, B_out=None, field_provider=None,
                                    optimize_iota=True, optimize_G=False,
                                    optimize_I=True, optimize_current_potential=True):
        if self._field_provider_depends_on_state(field_provider):
            raise ValueError('Analytic parameter Jacobians are not available for state-dependent field providers.')

        resolved_B_in, resolved_B_out = self.resolve_field_components(
            B_in=B_in,
            B_out=B_out,
            field_provider=field_provider,
            iota=iota,
            G=G,
            I=I,
        )
        derivatives = finite_beta_boozer_residual_dparameters(
            self.surface,
            iota,
            G,
            resolved_B_in,
            resolved_B_out,
            I=I,
        )

        columns = []
        if optimize_iota:
            columns.append(derivatives['iota'][:, None])
        if optimize_G:
            columns.append(derivatives['G'][:, None])
        if optimize_I:
            columns.append(derivatives['I'][:, None])
        if optimize_current_potential:
            columns.append(derivatives['current_potential'][:, 1:])

        if len(columns) == 0:
            reference = self.residual_blocks(iota=iota, G=G, I=I, B_in=resolved_B_in, B_out=resolved_B_out)
            return np.zeros((self._weighted_residual_vector(reference['blocks']).size, 0))

        return self._weighted_parameter_jacobian(np.concatenate(columns, axis=1))

    def residual_surface_jacobian(self, iota, G, I=0.0,
                                  current_potential=None,
                                  current_potential_derivatives=None,
                                  B_in=None, B_out=None, field_provider=None):
        if self._field_provider_depends_on_state(field_provider):
            raise ValueError('Analytic surface Jacobians are not available for state-dependent field providers.')

        resolved_B_in, resolved_B_out = self.resolve_field_components(
            B_in=B_in,
            B_out=B_out,
            field_provider=field_provider,
            iota=iota,
            G=G,
            I=I,
            current_potential=current_potential,
        )
        derivatives = finite_beta_boozer_residual_dsurface(
            self.surface,
            iota,
            G,
            resolved_B_in,
            resolved_B_out,
            I=I,
            current_potential=current_potential,
            current_potential_derivatives=current_potential_derivatives,
        )
        return self._weighted_surface_jacobian(derivatives['residual'])

    def self_consistent_single_surface_residual(self, iota, G=None, I=0.0, lambda_current=None,
                                                pressure_jump=None, B_coils=None, vc_digits=6):
        """
        Evaluate the single-surface VC-closed finite-beta interface residual.

        This helper is "self-consistent" only in the limited sense that the
        total Boozer field and the exterior field inferred by the virtual-casing
        operator are solved on the same surface. It is not a pure no-operator
        finite-beta closure.
        """
        from ..mhd.virtual_casing import VirtualCasing

        if G is None:
            G = self._default_G()
        if pressure_jump is None:
            pressure_jump = self.resolve_pressure_jump()
        if pressure_jump is None:
            pressure_jump = 0.0

        if B_coils is None:
            if self.biotsavart is None:
                raise ValueError('biotsavart is required for the single-surface virtual-casing closure.')
            x = self.surface.gamma().reshape((-1, 3))
            self.biotsavart.set_points(x)
            self.biotsavart.compute(0)
            B_coils = self.biotsavart.B().reshape(self.surface.gamma().shape)
        else:
            B_coils = np.asarray(B_coils)

        B_total = finite_beta_boozer_surface_field(
            self.surface,
            iota=iota,
            G=G,
            I=I,
            lambda_current=lambda_current,
        )
        vc = VirtualCasing.from_surface(
            self.surface,
            B_total,
            digits=vc_digits,
        )
        result = finite_beta_virtual_casing_residual(
            self.surface,
            iota=iota,
            lambda_current=(float(G) + float(iota) * float(I)) if lambda_current is None else float(lambda_current),
            B_external=vc.B_external,
            B_coils=B_coils,
            pressure_jump=pressure_jump,
        )
        result['B_coils'] = B_coils
        result['virtual_casing'] = vc
        return result

    def frozen_state_closure_diagnostics(self, iota, G=None, I=0.0, lambda_current=None,
                                         current_potential=None, field_provider=None,
                                         pressure_jump=None, B_coils=None, vc_digits=6):
        """
        Compare the direct field-provider closure and the full virtual-casing
        closure on the same frozen surface state.
        """
        from ..mhd.virtual_casing import VirtualCasing

        if field_provider is None:
            raise ValueError('frozen_state_closure_diagnostics requires a field_provider.')

        if G is None:
            G = self._default_G()
        if lambda_current is None:
            lambda_current = float(G) + float(iota) * float(I)
        if pressure_jump is None:
            pressure_jump = self.resolve_pressure_jump()
        if pressure_jump is None:
            pressure_jump = 0.0

        current_potential = self._normalize_current_potential(current_potential)
        expected_shape = self.surface.gamma().shape

        if B_coils is None:
            if self.biotsavart is None:
                raise ValueError('biotsavart is required when B_coils is not supplied.')
            x = self.surface.gamma().reshape((-1, 3))
            self.biotsavart.set_points(x)
            self.biotsavart.compute(0)
            B_coils = self.biotsavart.B().reshape(expected_shape)
        else:
            B_coils = np.asarray(B_coils)
            if B_coils.shape != expected_shape:
                raise ValueError(f'B_coils must have shape {expected_shape}.')

        direct_residual = self.residual_blocks(
            iota=iota,
            G=G,
            I=I,
            pressure_jump=pressure_jump,
            current_potential=current_potential,
            field_provider=field_provider,
        )
        B_in_direct, B_out_direct = self.resolve_field_components(
            field_provider=field_provider,
            iota=iota,
            G=G,
            I=I,
            current_potential=current_potential,
        )
        direct_vc_style = finite_beta_virtual_casing_residual(
            self.surface,
            iota=iota,
            lambda_current=lambda_current,
            B_external=B_out_direct,
            B_coils=B_coils,
            pressure_jump=pressure_jump,
        )

        vc = VirtualCasing.from_surface(self.surface, B_in_direct, digits=vc_digits)
        vc_result = finite_beta_virtual_casing_residual(
            self.surface,
            iota=iota,
            lambda_current=lambda_current,
            B_external=vc.B_external,
            B_coils=B_coils,
            pressure_jump=pressure_jump,
        )
        vc_result['virtual_casing'] = vc

        direct_coil_match = direct_vc_style['blocks']['coil_match']
        vc_coil_match = vc_result['blocks']['coil_match']
        direct_normal = direct_vc_style['blocks']['normal']
        vc_normal = vc_result['blocks']['normal']
        direct_pressure = direct_vc_style['blocks']['pressure']
        vc_pressure = vc_result['blocks']['pressure']
        direct_sheet_current = direct_vc_style['blocks']['sheet_current']
        vc_sheet_current = vc_result['blocks']['sheet_current']
        B_external_difference = B_out_direct - vc.B_external
        normal_difference = np.sum(B_external_difference * self.surface.unitnormal(), axis=2)

        def relative_norm(delta, reference):
            return float(np.linalg.norm(delta) / max(float(np.linalg.norm(reference)), 1e-30))

        return {
            'pressure_jump': float(pressure_jump),
            'lambda_current': float(lambda_current),
            'B_coils': B_coils,
            'direct': {
                'B_in': B_in_direct,
                'B_external': B_out_direct,
                'residual_blocks': direct_residual['blocks'],
                'vc_style': direct_vc_style,
                'current_potential': current_potential,
            },
            'virtual_casing': vc_result,
            'differences': {
                'B_external': B_external_difference,
                'B_external_normal': normal_difference,
                'coil_match': direct_coil_match - vc_coil_match,
                'normal': direct_normal - vc_normal,
                'pressure': direct_pressure - vc_pressure,
                'sheet_current': direct_sheet_current - vc_sheet_current,
                'B_external_rel_norm': relative_norm(B_external_difference, vc.B_external),
                'coil_match_rel_norm': relative_norm(direct_coil_match - vc_coil_match, vc_coil_match),
                'normal_rel_norm': relative_norm(direct_normal - vc_normal, vc_normal),
                'pressure_rel_norm': relative_norm(direct_pressure - vc_pressure, vc_pressure),
                'sheet_current_rel_norm': relative_norm(direct_sheet_current - vc_sheet_current, vc_sheet_current),
            },
        }

    def _surface_constraint_residual(self):
        label_residual = float(self.constraint_weight) * (self.label.J() - self.targetlabel)
        anchor_residual = self.surface.gamma()[0, 0, 2]
        return np.asarray([label_residual, anchor_residual])

    def _surface_constraint_jacobian(self):
        label_jacobian = float(self.constraint_weight) * self.label.dJ(partials=True)(self.surface)
        anchor_jacobian = self.surface.dgamma_by_dcoeff()[0, 0, 2, :]
        return np.vstack([label_jacobian, anchor_jacobian])

    def run_code(self, iota, G=None, I=0.0, current_potential=None,
                 B_in=None, B_out=None, field_provider=None, optimize_iota=True,
                 optimize_G=False, optimize_I=True,
                 optimize_current_potential=True, optimize_surface=False):
        """
        Run a finite-beta least-squares solve.

        The current implementation optimizes Boozer parameters and the
        sheet-current potential on the existing surface quadrature grid. Surface
        geometry can also be optimized when either fixed field samples are
        supplied on that grid or the field providers can be reevaluated on the
        moved surface.
        """
        if G is None:
            G = self._default_G()

        current_potential0 = self._normalize_current_potential(current_potential)
        surface_dofs0 = self.surface.x.copy()
        vc = self.virtual_casing
        explicit_fields = (B_in is not None and B_out is not None and not callable(B_in) and not callable(B_out))
        callable_fields = callable(B_in) and callable(B_out)
        provider_fields = field_provider is not None
        state_dependent_fields = self._field_provider_depends_on_state(field_provider)
        virtual_casing_fields = (
            B_in is None and B_out is None and vc is not None
            and hasattr(vc, 'B_total') and hasattr(vc, 'B_external')
        )
        biotsavart_fields = B_in is None and B_out is None and not virtual_casing_fields and self.biotsavart is not None
        analytic_surface_jacobian = optimize_surface and (explicit_fields or virtual_casing_fields)

        if optimize_surface and not (explicit_fields or callable_fields or provider_fields or virtual_casing_fields or biotsavart_fields):
            raise ValueError(
                "optimize_surface requires explicit fixed fields, virtual_casing data, callable B_in/B_out providers, a field_provider, or a biotsavart object."
            )

        if optimize_surface and not analytic_surface_jacobian:
            resolved_B_in = None
            resolved_B_out = None
        else:
            resolved_B_in, resolved_B_out = self.resolve_field_components(
                B_in=B_in,
                B_out=B_out,
                field_provider=field_provider,
                iota=iota,
                G=G,
                I=I,
                current_potential=current_potential0,
            )
        x0 = self._pack_state(iota, G, I, current_potential0,
                              optimize_iota, optimize_G, optimize_I, optimize_current_potential,
                              optimize_surface=optimize_surface, surface_dofs=surface_dofs0)

        if x0.size == 0:
            result = self.residual_blocks(
                iota=iota,
                G=G,
                I=I,
                current_potential=current_potential0,
                B_in=resolved_B_in,
                B_out=resolved_B_out,
                field_provider=None if resolved_B_in is not None else field_provider,
            )
            residual = self._weighted_residual_vector(result['blocks'])
            self.res = {
                'success': True,
                'iter': 0,
                'iota': float(iota),
                'G': float(G),
                'I': float(I),
                'current_potential': current_potential0,
                'residual': residual,
                'residual_norm': np.linalg.norm(residual),
                'block_norms': {name: np.linalg.norm(values) for name, values in result['blocks'].items()},
                'message': 'No free variables selected.',
            }
            self.need_to_run_code = False
            return self.res

        def objective(vector):
            surface_dofs_val, iota_val, G_val, I_val, current_potential_val = self._unpack_state(
                vector, iota, G, I, current_potential0,
                optimize_iota, optimize_G, optimize_I, optimize_current_potential,
                optimize_surface=optimize_surface, nsurfdofs=surface_dofs0.size,
            )
            if optimize_surface:
                self.surface.x = surface_dofs_val
                if analytic_surface_jacobian:
                    resolved_B_in_local = resolved_B_in
                    resolved_B_out_local = resolved_B_out
                else:
                    resolved_B_in_local, resolved_B_out_local = self.resolve_field_components(
                        B_in=B_in,
                        B_out=B_out,
                        field_provider=field_provider,
                        iota=iota_val,
                        G=G_val,
                        I=I_val,
                        current_potential=current_potential_val,
                    )
            elif state_dependent_fields:
                resolved_B_in_local, resolved_B_out_local = self.resolve_field_components(
                    field_provider=field_provider,
                    iota=iota_val,
                    G=G_val,
                    I=I_val,
                    current_potential=current_potential_val,
                )
            else:
                resolved_B_in_local = resolved_B_in
                resolved_B_out_local = resolved_B_out
            result = self.residual_blocks(
                iota=iota_val,
                G=G_val,
                I=I_val,
                current_potential=current_potential_val,
                B_in=resolved_B_in_local,
                B_out=resolved_B_out_local,
            )
            residual = self._weighted_residual_vector(result['blocks'])
            if optimize_surface:
                residual = np.concatenate([residual, self._surface_constraint_residual()])
            return residual

        def jacobian(vector):
            surface_dofs_val, iota_val, G_val, I_val, current_potential_val = self._unpack_state(
                vector, iota, G, I, current_potential0,
                optimize_iota, optimize_G, optimize_I, optimize_current_potential,
                optimize_surface=optimize_surface, nsurfdofs=surface_dofs0.size,
            )
            if optimize_surface:
                self.surface.x = surface_dofs_val

            parameter_jacobian = self.residual_parameter_jacobian(
                iota=iota_val,
                G=G_val,
                I=I_val,
                B_in=resolved_B_in,
                B_out=resolved_B_out,
                field_provider=field_provider,
                optimize_iota=optimize_iota,
                optimize_G=optimize_G,
                optimize_I=optimize_I,
                optimize_current_potential=optimize_current_potential,
            )

            if not optimize_surface:
                return parameter_jacobian

            surface_jacobian = self.residual_surface_jacobian(
                iota=iota_val,
                G=G_val,
                I=I_val,
                current_potential=current_potential_val,
                B_in=resolved_B_in,
                B_out=resolved_B_out,
                field_provider=field_provider,
            )
            constraint_jacobian = self._surface_constraint_jacobian()

            if parameter_jacobian.shape[1] == 0:
                residual_jacobian = surface_jacobian
                full_constraint_jacobian = constraint_jacobian
            else:
                residual_jacobian = np.concatenate([surface_jacobian, parameter_jacobian], axis=1)
                full_constraint_jacobian = np.concatenate(
                    [constraint_jacobian, np.zeros((constraint_jacobian.shape[0], parameter_jacobian.shape[1]))],
                    axis=1,
                )
            return np.concatenate([residual_jacobian, full_constraint_jacobian], axis=0)

        tol = self.options.get('ls_tol', self.options.get('newton_tol', 1e-11))
        max_nfev = self.options.get('ls_max_nfev', 100)
        use_analytic_jacobian = (not state_dependent_fields) and (not optimize_surface or analytic_surface_jacobian)
        lsq = least_squares(
            objective,
            x0,
            jac=jacobian if use_analytic_jacobian else '2-point',
            method=self.options.get('ls_method', 'trf'),
            ftol=tol,
            xtol=tol,
            gtol=tol,
            max_nfev=max_nfev,
            verbose=2 if self.options.get('verbose', True) else 0,
        )

        surface_dofs_val, iota_val, G_val, I_val, current_potential_val = self._unpack_state(
            lsq.x, iota, G, I, current_potential0,
            optimize_iota, optimize_G, optimize_I, optimize_current_potential,
            optimize_surface=optimize_surface, nsurfdofs=surface_dofs0.size,
        )
        if optimize_surface:
            self.surface.x = surface_dofs_val
        if field_provider is not None and (optimize_surface or state_dependent_fields):
            result = self.residual_blocks(
                iota=iota_val,
                G=G_val,
                I=I_val,
                current_potential=current_potential_val,
                field_provider=field_provider,
            )
        else:
            result = self.residual_blocks(
                iota=iota_val,
                G=G_val,
                I=I_val,
                current_potential=current_potential_val,
                B_in=B_in if optimize_surface else resolved_B_in,
                B_out=B_out if optimize_surface else resolved_B_out,
            )
        residual = self._weighted_residual_vector(result['blocks'])
        if optimize_surface:
            residual = np.concatenate([residual, self._surface_constraint_residual()])

        self.res = {
            'success': bool(lsq.success),
            'iter': int(lsq.nfev),
            'surface': self.surface,
            'iota': iota_val,
            'G': G_val,
            'I': I_val,
            'current_potential': current_potential_val,
            'residual': residual,
            'residual_norm': np.linalg.norm(residual),
            'block_norms': {name: np.linalg.norm(values) for name, values in result['blocks'].items()},
            'message': lsq.message,
            'least_squares_result': lsq,
        }
        self.need_to_run_code = False
        return self.res

    def run_code_single_surface_vc(self, iota, G=None, I=0.0,
                                   optimize_iota=True, optimize_lambda_current=True,
                                   optimize_surface=False):
        """
        Solve the single-surface VC-closed finite-beta validation problem.

        The surface field is generated from the Boozer relation itself, the
        virtual-casing operator extracts the contribution due to currents outside
        the surface, and that exterior field is matched to the coil field on the
        same surface. This is a practical no-VMEC surrogate, not a pure no-VC
        self-consistent finite-beta closure.
        """
        if self.biotsavart is None:
            raise ValueError('run_code_single_surface_vc requires a biotsavart object.')

        if G is None:
            G = self._default_G()

        target_pressure_jump = self.resolve_pressure_jump()
        if target_pressure_jump is None:
            target_pressure_jump = 0.0

        surface_dofs0 = self.surface.x.copy()
        lambda0 = float(G) + float(iota) * float(I)
        x0 = []
        if optimize_surface:
            x0.extend(surface_dofs0)
        if optimize_iota:
            x0.append(float(iota))
        if optimize_lambda_current:
            x0.append(float(lambda0))
        x0 = np.asarray(x0, dtype=float)

        def unpack(vector):
            cursor = 0
            surface_dofs_val = None
            iota_val = float(iota)
            lambda_val = float(lambda0)
            if optimize_surface:
                surface_dofs_val = np.asarray(vector[cursor:cursor + surface_dofs0.size], dtype=float)
                cursor += surface_dofs0.size
            if optimize_iota:
                iota_val = float(vector[cursor])
                cursor += 1
            if optimize_lambda_current:
                lambda_val = float(vector[cursor])
            return surface_dofs_val, iota_val, lambda_val

        evaluation_cache = {
            'vector': None,
            'pressure_jump': None,
            'result': None,
            'weighted_residual': None,
            'full_residual': None,
        }

        def evaluate(vector, pressure_jump_value):
            surface_dofs_val, iota_val, lambda_val = unpack(vector)
            if (
                evaluation_cache['vector'] is not None
                and np.array_equal(evaluation_cache['vector'], vector)
                and evaluation_cache['pressure_jump'] == float(pressure_jump_value)
            ):
                return evaluation_cache

            if optimize_surface:
                self.surface.x = surface_dofs_val
            result = self.self_consistent_single_surface_residual(
                iota=iota_val,
                G=G,
                lambda_current=lambda_val,
                pressure_jump=pressure_jump_value,
                vc_digits=self.options.get('vc_digits', 6),
            )
            weighted_residual = self._weighted_vc_residual_vector(
                result['blocks'],
                pressure_jump=pressure_jump_value,
                B_coils=result['B_coils'],
            )
            if optimize_surface:
                full_residual = np.concatenate([weighted_residual, self._surface_constraint_residual()])
            else:
                full_residual = weighted_residual
            evaluation_cache.update({
                'vector': vector.copy(),
                'pressure_jump': float(pressure_jump_value),
                'result': result,
                'weighted_residual': weighted_residual,
                'full_residual': full_residual,
            })
            return evaluation_cache

        def objective(vector, pressure_jump_value):
            return evaluate(vector, pressure_jump_value)['full_residual']

        def jacobian(vector, pressure_jump_value):
            surface_dofs_val, iota_val, lambda_val = unpack(vector)
            cached = evaluate(vector, pressure_jump_value)
            if optimize_surface:
                self.surface.x = surface_dofs_val
            parameter_jacobian = self.single_surface_vc_parameter_jacobian(
                iota=iota_val,
                G=G,
                lambda_current=lambda_val,
                pressure_jump=pressure_jump_value,
                B_coils=cached['result']['B_coils'],
                vc_digits=self.options.get('vc_digits', 6),
                optimize_iota=optimize_iota,
                optimize_lambda_current=optimize_lambda_current,
                virtual_casing=cached['result']['virtual_casing'],
                B_total=cached['result']['B_total'],
                B_external=cached['result']['B_external'],
            )
            if not optimize_surface:
                return parameter_jacobian

            surface_jacobian = self.single_surface_vc_surface_jacobian(
                iota=iota_val,
                G=G,
                lambda_current=lambda_val,
                pressure_jump=pressure_jump_value,
                B_coils=cached['result']['B_coils'],
                vc_digits=self.options.get('vc_digits', 6),
                baseline_result=cached['result'],
                baseline_weighted_residual=cached['weighted_residual'],
            )
            constraint_jacobian = self._surface_constraint_jacobian()

            if parameter_jacobian.shape[1] == 0:
                residual_jacobian = surface_jacobian
                full_constraint_jacobian = constraint_jacobian
            else:
                residual_jacobian = np.concatenate([surface_jacobian, parameter_jacobian], axis=1)
                full_constraint_jacobian = np.concatenate(
                    [constraint_jacobian, np.zeros((constraint_jacobian.shape[0], parameter_jacobian.shape[1]))],
                    axis=1,
                )
            return np.concatenate([residual_jacobian, full_constraint_jacobian], axis=0)

        continuation_steps = int(self.options.get('vc_pressure_continuation_steps', 1 if abs(target_pressure_jump) == 0.0 else 4))
        continuation_steps = max(1, continuation_steps)
        pressure_schedule = np.linspace(0.0, float(target_pressure_jump), continuation_steps)
        if continuation_steps == 1:
            pressure_schedule = np.asarray([float(target_pressure_jump)])

        if x0.size == 0:
            result = self.self_consistent_single_surface_residual(
                iota=iota,
                G=G,
                lambda_current=lambda0,
                pressure_jump=target_pressure_jump,
                vc_digits=self.options.get('vc_digits', 6),
            )
            residual = self._weighted_vc_residual_vector(
                result['blocks'],
                pressure_jump=target_pressure_jump,
                B_coils=result['B_coils'],
            )
            self.res = {
                'success': True,
                'iter': 0,
                'surface': self.surface,
                'iota': float(iota),
                'G': float(G),
                'I': float(I),
                'lambda_current': float(lambda0),
                'B_total': result['B_total'],
                'B_external': result['B_external'],
                'B_coils': result['B_coils'],
                'virtual_casing': result['virtual_casing'],
                'residual': residual,
                'residual_norm': np.linalg.norm(residual),
                'raw_residual': result['residual'],
                'raw_residual_norm': np.linalg.norm(result['residual']),
                'block_norms': {name: np.linalg.norm(values) for name, values in result['blocks'].items()},
                'message': 'No free variables selected.',
                'continuation_history': [],
            }
            self.need_to_run_code = False
            return self.res

        tol = self.options.get('ls_tol', self.options.get('newton_tol', 1e-11))
        max_nfev = self.options.get('ls_max_nfev', 100)
        lsq = None
        stage_history = []
        current_x = x0.copy()
        for pressure_jump_value in pressure_schedule:
            if optimize_surface and not self.options.get('vc_surface_hybrid_jacobian', True):
                jac = '2-point'
            else:
                jac = lambda vector, pressure_jump_value=pressure_jump_value: jacobian(vector, pressure_jump_value)

            lsq = least_squares(
                lambda vector, pressure_jump_value=pressure_jump_value: objective(vector, pressure_jump_value),
                current_x,
                jac=jac,
                method=self.options.get('ls_method', 'trf'),
                x_scale=self.options.get('vc_x_scale', 'jac'),
                ftol=tol,
                xtol=tol,
                gtol=tol,
                max_nfev=max_nfev,
                verbose=2 if self.options.get('verbose', True) else 0,
            )
            current_x = lsq.x.copy()
            stage_eval = evaluate(current_x, pressure_jump_value)
            stage_history.append({
                'pressure_jump': float(pressure_jump_value),
                'success': bool(lsq.success),
                'nfev': int(lsq.nfev),
                'cost': float(lsq.cost),
                'message': lsq.message,
                'weighted_residual_norm': float(np.linalg.norm(stage_eval['weighted_residual'])),
                'raw_residual_norm': float(np.linalg.norm(stage_eval['result']['residual'])),
                'coil_match_norm': float(np.linalg.norm(stage_eval['result']['blocks']['coil_match'])),
                'normal_norm': float(np.linalg.norm(stage_eval['result']['blocks']['normal'])),
                'pressure_norm': float(np.linalg.norm(stage_eval['result']['blocks']['pressure'])),
            })

        surface_dofs_val, iota_val, lambda_val = unpack(current_x)
        if optimize_surface:
            self.surface.x = surface_dofs_val
        result = self.self_consistent_single_surface_residual(
            iota=iota_val,
            G=G,
            lambda_current=lambda_val,
            pressure_jump=target_pressure_jump,
            vc_digits=self.options.get('vc_digits', 6),
        )
        residual = self._weighted_vc_residual_vector(
            result['blocks'],
            pressure_jump=target_pressure_jump,
            B_coils=result['B_coils'],
        )
        if optimize_surface:
            residual = np.concatenate([residual, self._surface_constraint_residual()])

        if abs(iota_val) > 1e-12:
            I_val = (lambda_val - float(G)) / iota_val
        else:
            I_val = 0.0

        self.res = {
            'success': bool(lsq.success),
            'iter': int(lsq.nfev),
            'surface': self.surface,
            'iota': iota_val,
            'G': float(G),
            'I': I_val,
            'lambda_current': lambda_val,
            'B_total': result['B_total'],
            'B_external': result['B_external'],
            'B_coils': result['B_coils'],
            'virtual_casing': result['virtual_casing'],
            'residual': residual,
            'residual_norm': np.linalg.norm(residual),
            'raw_residual': result['residual'],
            'raw_residual_norm': np.linalg.norm(result['residual']),
            'block_norms': {name: np.linalg.norm(values) for name, values in result['blocks'].items()},
            'message': lsq.message,
            'least_squares_result': lsq,
            'continuation_history': stage_history,
        }
        self.need_to_run_code = False
        return self.res


class BoozerSurface(Optimizable):
    r"""
    The BoozerSurface class computes a flux surface of a BiotSavart magnetic field where the angles
    of the surface are Boozer angles [1,2]. The class takes as input a Surface representation 
    (:obj:`~simsopt.geo.SurfaceXYZFourier` or :obj:`~simsopt.geo.SurfaceXYZTensorFourier`), 
    a BiotSavart magnetic field, a flux surface label evaluator, and a target value of the label.

    The Boozer angles are computed by solving a constrained least squares problem,

        .. math::

            \min_x J(x) = \frac{1}{2} \mathbf r^T(x) \mathbf r(x)

    subject to

        .. math::
            
            l(x) = l_0

            z(\varphi=0,\theta=0) = 0

    where :math:`\mathbf r` is a vector of residuals computed by :obj:`~simsopt.geo.boozer_surface_residual`, 
    :math:`l` is a surface label function with target value :math:`l_0`. The degrees of freedom are the
    surface coefficients, the rotational transform, :math:`\iota`, and the value of Boozer's :math:`G` on the surface.
    This objective is zero when the surface corresponds to a magnetic surface of the field, :math:`(\phi,\theta)` 
    that parametrize the surface correspond to Boozer angles, and the constraints are satisfied.

    The recommended approach to finding the Boozer angles is to use the :mod:`run_code` method,
        
        :obj:`~simsopt.geo.BoozerSurface.run_code(iota_guess, G=G_guess)`.
    
    Depending on how the class is initialized, :mod:`run_code`, will use either the BoozerLS [2] or BoozerExact [1] approach
    to finding the flux surface. The BoozerLS approach finds the flux surface by solving the constrained least squares
    problem mentioned above. The methods

        #. :obj:`~simsopt.geo.BoozerSurface.minimize_boozer_penalty_constraints_LBFGS`
        #. :obj:`~simsopt.geo.BoozerSurface.minimize_boozer_penalty_constraints_newton`
        #. :obj:`~simsopt.geo.BoozerSurface.minimize_boozer_penalty_constraints_ls`

    scalarize the constrained problem using a quadratic penalty method, 
    and apply L-BFGS, Newton, or :mod:`scipy.optimize.least_squares` to solve the penalty problem.
    Alternatively, the constraints can be enforced exactly (not with a penalty) using,

        :obj:`~simsopt.geo.BoozerSurface.minimize_boozer_exact_constraints_newton`

    In this approach, Newton's method is used to solve the first order necessary conditions for optimality. Note
    that this differs from the BoozerExact approach.The BoozerExact approach solves the residual equations directly
    at a specific set of colocation points on the surface,

        .. math::

            \mathbf r(x) = 0

            l(x) = l_0

            z(\varphi=0,\theta=0) = 0

    The colocation points are chosen such that the number of colocation points is equal to the number of unknowns
    in on the surface, so that the resulting nonlinear system of equations can be solved using
    Newton's method. The BoozerExact approach is implemented in
        
         :obj:`~simsopt.geo.BoozerSurface.solve_residual_equation_exactly_newton`
    
    Generally, the BoozerExact approach is faster than the BoozerLS approach, but it is less robust. Note that there 
    are specific requirements on the set of colocation points, i.e. :mod:`surface.quadpoints_phi` and 
    :mod:`surface.quadpoints_theta`, for stellarator symmetric BoozerExact surfaces. See the class method 
    :obj:`~simsopt.geo.BoozerSurface.solve_residual_equation_exactly_newton` and :obj:`~simsopt.geo.SurfaceXYZTensorFourier.get_stellsym_mask()`
    for more information.

    *[1]: Giuliani A, Wechsung F, Stadler G, Cerfon A, Landreman M. Direct computation of magnetic surfaces in Boozer coordinates and coil optimization for quasisymmetry. Journal of Plasma Physics. 2022;88(4):905880401. doi:10.1017/S0022377822000563*
    
    *[2]: Giuliani, A., Wechsung, F., Cerfon, A., Landreman, M., & Stadler, G. (2023). Direct stellarator coil optimization for nested magnetic surfaces with precise quasi-symmetry. Physics of Plasmas, 30(4).*
    """

    def __init__(self, biotsavart, surface, label, targetlabel, constraint_weight=None, options=None):
        """
        Args:
            biotsavart (:obj:`~simsopt.field.BiotSavart`): BiotSavart object.
            surface (:obj:`~simsopt.geo.SurfaceXYZFourier`, :obj:`~simsopt.geo.SurfaceXYZTensorFourier`): Surface object.
            label (:obj:`~simsopt._core.optimizable.Optimizable`): A method that computes a flux surface label for the surface, such as  
                :obj:`~simsopt.geo.Volume`, :obj:`~simsopt.geo.Area`, or :obj:`~simsopt.geo.ToroidalFlux`.
            targetlabel (float): The target value of the label on the surface.
            constraint_weight (float, Optional): The weight of the label constraint used when solving Boozer least squares. 
                If None, then Boozer Exact is used in the :mod:`run_code` method.
            options (dict, Optional): A dictionary of solver options. If a keyword is not specified, then a default
                value is used. Possible keywords are:

                - `verbose` (bool): display convergence information. Defaults to True.
                - `newton_tol` (float): tolerance for newton solver. Defaults to 1e-13 for BoozerExact and 1e-11 for BoozerLS.
                - `bfgs_tol` (float): tolerance for bfgs solver. Defaults to 1e-10.
                - `newton_maxiter` (int): maximum number of iterations for Newton solver. Defaults to 40.
                - `bfgs_maxiter` (int): maximum number of iterations for BFGS solver. Defaults to 1500.
                - `limited_memory` (bool): True if L-BFGS solver is desired, False if the BFGS solver otherwise. Defaults to False.
                - `weight_inv_modB` (float): for BoozerLS surfaces, weight the residual by modB so that it does not scale with coil currents.  Defaults to True.
        """
        super().__init__(depends_on=[biotsavart])

        from simsopt.geo import SurfaceXYZFourier, SurfaceXYZTensorFourier
        if not isinstance(surface, SurfaceXYZTensorFourier) and not isinstance(surface, SurfaceXYZFourier):
            raise Exception("The input surface must be a SurfaceXYZTensorFourier or SurfaceXYZFourier.")

        self.biotsavart = biotsavart
        self.surface = surface
        self.label = label
        self.targetlabel = targetlabel
        self.constraint_weight = constraint_weight
        self.boozer_type = 'ls' if constraint_weight else 'exact'
        self.need_to_run_code = True

        if options is None:
            options = {}

        # set the default options now
        if 'verbose' not in options:
            options['verbose'] = True

        # default solver options for the BoozerExact and BoozerLS solvers
        if self.boozer_type == 'exact':
            if 'newton_tol' not in options:
                options['newton_tol'] = 1e-13
            if 'newton_maxiter' not in options:
                options['newton_maxiter'] = 40
        elif self.boozer_type == 'ls':
            if 'bfgs_tol' not in options:
                options['bfgs_tol'] = 1e-10
            if 'newton_tol' not in options:
                options['newton_tol'] = 1e-11
            if 'newton_maxiter' not in options:
                options['newton_maxiter'] = 40
            if 'bfgs_maxiter' not in options:
                options['bfgs_maxiter'] = 1500
            if 'limited_memory' not in options:
                options['limited_memory'] = False
            if 'weight_inv_modB' not in options:
                options['weight_inv_modB'] = True
        self.options = options

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True

    def run_code(self, iota, G=None):
        """
        Run the default solvers, i.e., run Newton's method directly if you are computing a BoozerExact surface,
        and run BFGS followed by Newton if you are computing a BoozerLS surface.

        Args:
            iota (float): Guess for value of rotational transform on the surface.
            G (float, Optional): Guess for value of G on surface, defaults to None. Note that if None is used, then the coil currents must be fixed.
        
        Returns:
            dict: A dictionary containing the results of the optimization. The dictionary contains the following keys in addition
            to others:

                - `"residual"`: the residual of the optimization problem
                - `"iter"`: the number of iterations taken to converge
                - `"success"`: True if the optimization converged, False otherwise
                - `"G"`: the value of G on the surface
                - `"s"`: the surface object
                - `"iota"`: the value of iota on the surface
                - `"PLU"`: the LU decomposition of the hessian

        """
        if not self.need_to_run_code:
            return

        # for coil optimizations, the gradient calculations of the objective assume that the coil currents are fixed when G is None.
        if G is None:
            assert np.all([c.current.dofs.all_fixed() for c in self.biotsavart.coils])

        # BoozerExact default solver
        if self.boozer_type == 'exact':
            res = self.solve_residual_equation_exactly_newton(iota=iota, G=G, tol=self.options['newton_tol'], maxiter=self.options['newton_maxiter'], verbose=self.options['verbose'])
            return res

        # BoozerLS default solver
        elif self.boozer_type == 'ls':
            # you need a label constraint for a BoozerLS surface
            assert self.constraint_weight is not None

            # first try BFGS.  You could also try L-BFGS by setting limited_memory=True in the options dictionary, which might be faster.  However, BFGS appears
            # to generally result in solutions closer to optimality.
            res = self.minimize_boozer_penalty_constraints_LBFGS(constraint_weight=self.constraint_weight, iota=iota, G=G,
                                                                 tol=self.options['bfgs_tol'], maxiter=self.options['bfgs_maxiter'], verbose=self.options['verbose'], limited_memory=self.options['limited_memory'],
                                                                 weight_inv_modB=self.options['weight_inv_modB'])
            iota, G = res['iota'], res['G']

            ## polish off using Newton's method
            self.need_to_run_code = True
            res = self.minimize_boozer_penalty_constraints_newton(constraint_weight=self.constraint_weight, iota=iota, G=G,
                                                                  verbose=self.options['verbose'], tol=self.options['newton_tol'], maxiter=self.options['newton_maxiter'],
                                                                  weight_inv_modB=self.options['weight_inv_modB'])
            return res

    def boozer_penalty_constraints_vectorized(self, dofs, derivatives=0, constraint_weight=1., optimize_G=False, weight_inv_modB=True):
        """
        Replacement for the previous `boozer_penalty_constraints` function, which has issues on ubuntu.  It
        is much faster and uses less memory since it calls a vectorized implementation in cpp. This is
        especially true when `derivatives=2`, i.e., when the Hessian is requested.

        Args:
            dofs (ndarray): The degrees of freedom of the Surface object, followed by the value of iota and G.
                e.g. ``[surface.x, iota, G]`` or ``[surface.x, iota]`` if ``optimize_G=False``.
            derivatives (int, Optional): 0 if no derivatives are requested, 1 if first derivatives are requested.
            constraint_weight (float, Optional): The weight of the label constraint used when solving Boozer least squares.
            optimize_G (bool, Optional): True if G is a variable in the optimization problem, False otherwise.
            weight_inv_modB (bool, Optional): If True, weight the residual by modB so that it does not scale with coil currents. Defaults to True.
        
        Returns:
            tuple: ``(r, J, H)`` The residual vector, the Jacobian of the optimization problem, and the Hessian of the optimization problem.
            If ``derivatives=0``, then ``J`` and ``H`` are None. If ``derivatives=1``, then ``H`` is None.
        """

        assert derivatives in [0, 1, 2]
        if optimize_G:
            sdofs = dofs[:-2]
            iota = dofs[-2]
            G = dofs[-1]
        else:
            sdofs = dofs[:-1]
            iota = dofs[-1]
            G = 2. * np.pi * np.sum(np.abs([coil.current.get_value() for coil in self.biotsavart._coils])) * (4 * np.pi * 10**(-7) / (2 * np.pi))

        s = self.surface
        nphi = s.quadpoints_phi.size
        ntheta = s.quadpoints_theta.size
        nsurfdofs = sdofs.size

        s.set_dofs(sdofs)

        surface = self.surface
        biotsavart = self.biotsavart
        x = surface.gamma()
        xphi = surface.gammadash1()
        xtheta = surface.gammadash2()
        nphi = x.shape[0]
        ntheta = x.shape[1]

        xsemiflat = x.reshape((x.size//3, 3)).copy()
        biotsavart.set_points(xsemiflat)
        biotsavart.compute(derivatives)
        B = biotsavart.B().reshape((nphi, ntheta, 3))

        if derivatives >= 1:
            dx_dc = surface.dgamma_by_dcoeff()
            dxphi_dc = surface.dgammadash1_by_dcoeff()
            dxtheta_dc = surface.dgammadash2_by_dcoeff()
            dB_dx = biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))

        if derivatives == 2:
            d2B_by_dXdX = biotsavart.d2B_by_dXdX().reshape((nphi, ntheta, 3, 3, 3))

        num_res = 3 * s.quadpoints_phi.size * s.quadpoints_theta.size
        if derivatives == 0:
            val = sopp.boozer_residual(G, iota, xphi, xtheta, B, weight_inv_modB)
            boozer = val,
        elif derivatives == 1:
            val, dval = sopp.boozer_residual_ds(G, iota, B, dB_dx, xphi, xtheta, dx_dc, dxphi_dc, dxtheta_dc, weight_inv_modB)
            boozer = val, dval
        elif derivatives == 2:
            val, dval, d2val = sopp.boozer_residual_ds2(G, iota, B, dB_dx, d2B_by_dXdX, xphi, xtheta, dx_dc, dxphi_dc, dxtheta_dc, weight_inv_modB)
            boozer = val, dval, d2val

        # normalizing the residuals here
        boozer = tuple([b/num_res for b in boozer])

        lab = self.label.J()

        rnl = boozer[0]
        rl = np.sqrt(constraint_weight) * (lab-self.targetlabel)
        rz = np.sqrt(constraint_weight) * (s.gamma()[0, 0, 2] - 0.)
        r = rnl + 0.5*rl**2 + 0.5*rz**2

        if derivatives == 0:
            return r

        dl = np.zeros(dofs.shape)
        drz = np.zeros(dofs.shape)
        dl[:nsurfdofs] = self.label.dJ(partials=True)(s)
        drz[:nsurfdofs] = s.dgamma_by_dcoeff()[0, 0, 2, :]

        Jnl = boozer[1]
        if not optimize_G:
            Jnl = Jnl[:-1]

        drl = np.sqrt(constraint_weight) * dl
        drz = np.sqrt(constraint_weight) * drz
        J = Jnl + rl * drl + rz * drz

        if derivatives == 1:
            return r, J

        Hnl = boozer[2]
        if not optimize_G:
            Hnl = Hnl[:-1, :-1]

        d2rl = np.zeros((dofs.shape[0], dofs.shape[0]))
        d2rl[:nsurfdofs, :nsurfdofs] = np.sqrt(constraint_weight)*self.label.d2J_by_dsurfacecoefficientsdsurfacecoefficients()
        H = Hnl + drl[:, None] @ drl[None, :] + drz[:, None] @ drz[None, :] + rl * d2rl

        return r, J, H

    def boozer_exact_constraints(self, xl, derivatives=0, optimize_G=True):
        r"""
        This function returns the optimality conditions corresponding to the minimization problem

        .. math::
            \text{min}_x ~J(x)

        subject to 

        .. math::
            l - l_0 &= 0 \\
            z(\varphi=0,\theta=0) - 0 &= 0

        The function can additionally return the first derivatives of these optimality conditions.

        Args:
            xl (ndarray): The degrees of freedom of the Surface object, followed by the value of iota and G.
                e.g. ``[surface.x, iota, G]`` or ``[surface.x, iota]`` if ``optimize_G=False``.
            derivatives (int, Optional): 0 if no derivatives are requested, 1 if first derivatives are requested.
            optimize_G (bool, Optional): True if G is a variable in the optimization problem, False otherwise.

        Returns:
            If ``derivatives=0``, return ``res`` the residual of the optimization problem.
            If ``derivatives=1``, return ``(res, dres)`` the residual and the Jacobian of the optimization problem.
        """
        assert derivatives in [0, 1]
        if optimize_G:
            sdofs = xl[:-4]
            iota = xl[-4]
            G = xl[-3]
        else:
            sdofs = xl[:-3]
            iota = xl[-3]
            G = None
        lm = xl[-2:]
        s = self.surface
        biotsavart = self.biotsavart
        s.set_dofs(sdofs)
        nsurfdofs = sdofs.size

        boozer = boozer_surface_residual(s, iota, G, biotsavart, derivatives=derivatives+1)
        r, J = boozer[0:2]

        dl = np.zeros((xl.shape[0]-2,))

        l = self.label.J()
        dl[:nsurfdofs] = self.label.dJ(partials=True)(s)
        drz = np.zeros((xl.shape[0]-2,))
        g = [l-self.targetlabel]
        rz = (s.gamma()[0, 0, 2] - 0.)
        drz[:nsurfdofs] = s.dgamma_by_dcoeff()[0, 0, 2, :]

        res = np.zeros(xl.shape)
        res[:-2] = np.sum(r[:, None]*J, axis=0) - lm[-2] * dl - lm[-1] * drz
        res[-2] = g[0]
        res[-1] = rz
        if derivatives == 0:
            return res

        H = boozer[2]

        d2l = np.zeros((xl.shape[0]-2, xl.shape[0]-2))
        d2l[:nsurfdofs, :nsurfdofs] = self.label.d2J_by_dsurfacecoefficientsdsurfacecoefficients()

        dres = np.zeros((xl.shape[0], xl.shape[0]))
        dres[:-2, :-2] = J.T @ J + np.sum(r[:, None, None] * H, axis=0) - lm[-2]*d2l
        dres[:-2, -2] = -dl
        dres[:-2, -1] = -drz

        dres[-2, :-2] = dl
        dres[-1, :-2] = drz
        return res, dres

    def minimize_boozer_penalty_constraints_LBFGS(self, tol=1e-3, maxiter=1000, constraint_weight=1., iota=0., G=None, limited_memory=True, weight_inv_modB=True, verbose=False):
        r"""
        This function uses L-BFGS to find the surface that approximately solves

        .. math::
            \text{min}_x ~J(x) + \frac{1}{2} w_c (l - l_0)^2
                                 + \frac{1}{2} w_c (z(\varphi=0, \theta=0) - 0)^2

        where :math:`J(x) = \frac{1}{2}\mathbf r(x)^T \mathbf r(x)`, and :math:`\mathbf r(x)` contains
        the Boozer residuals at quadrature points :math:`1,\dots,n`.

        Args:
            tol (float, Optional): The tolerance for the optimization. Defaults to 1e-3.
            maxiter (int, Optional): The maximum number of iterations for the optimization. Defaults to 1000.
            constraint_weight (float, Optional): The weight of the label constraint used when solving Boozer least squares.
            iota (float, Optional): The initial guess for the value of the rotational transform on the surface. Defaults to 0.
            G (float, Optional): The initial guess for the value of G on the surface. Defaults to None.
            limited_memory (bool, Optional): If True, use the limited memory version of L-BFGS. Defaults to True.
            weight_inv_modB (bool, Optional): If True, weight the residual by modB so that it does not scale with coil currents. Defaults to True.
            verbose (bool, Optional): If True, print the optimization progress. Defaults to False.
        
        Returns:
            res (dict): A dictionary containing the results of the optimization. The dictionary contains the following keys in addition
            to others:

                - 'fun': the value of the objective function at the solution
                - 'gradient': the gradient of the objective function at the solution
                - 'iter': the number of iterations taken to converge
                - 'info': the optimization result
                - 'success': True if the optimization converged, False otherwise
                - 'G': the value of G on the surface
                - 's': the surface object
                - 'iota': the value of iota on the surface
                - 'weight_inv_modB': the value of weight_inv_modB used in the optimization
                - 'type': the type of optimization used

        """

        if not self.need_to_run_code:
            return self.res

        s = self.surface
        if G is None:
            x = np.concatenate((s.get_dofs(), [iota]))
        else:
            x = np.concatenate((s.get_dofs(), [iota, G]))

        def fun(x): return self.boozer_penalty_constraints_vectorized(x, derivatives=1, constraint_weight=constraint_weight, optimize_G=G is not None, weight_inv_modB=weight_inv_modB)

        method = 'L-BFGS-B' if limited_memory else 'BFGS'
        options = {'maxiter': maxiter, 'gtol': tol}
        if limited_memory:
            options['maxcor'] = 200
            options['ftol'] = tol

        res = minimize(
            fun, x, jac=True, method=method,
            options=options)

        resdict = {
            "fun": res.fun, "gradient": res.jac, "iter": res.nit, "info": res, "success": res.success, "G": None, 'weight_inv_modB': weight_inv_modB, 'type': 'ls'
        }
        if G is None:
            s.set_dofs(res.x[:-1])
            iota = res.x[-1]
        else:
            s.set_dofs(res.x[:-2])
            iota = res.x[-2]
            G = res.x[-1]
            resdict['G'] = G
        resdict['s'] = s
        resdict['iota'] = iota

        self.res = resdict
        self.need_to_run_code = False

        if verbose:
            print(f"{method} solve - {resdict['success']}  iter={resdict['iter']}, iota={resdict['iota']:.16f}, ||grad||_inf = {np.linalg.norm(resdict['gradient'], ord=np.inf):.3e}", flush=True)

        return resdict

    def minimize_boozer_penalty_constraints_newton(self, tol=1e-12, maxiter=10, constraint_weight=1., iota=0., G=None, stab=0., weight_inv_modB=True, verbose=False):
        """
        This function does the same as :mod:`minimize_boozer_penalty_constraints_LBFGS`, but instead of LBFGS it uses
        Newton's method.

        Args:
            tol (float, Optional): The tolerance for the optimization. Defaults to 1e-12.
            maxiter (int, Optional): The maximum number of iterations for the optimization. Defaults to 10.
            constraint_weight (float, Optional): The weight of the label constraint used when solving Boozer least squares.
            iota (float, Optional): The initial guess for the value of the rotational transform on the surface. Defaults to 0.
            G (float, Optional): The initial guess for the value of G on the surface. Defaults to None.
            stab (float, Optional): The stabilization parameter for the Newton method. Defaults to 0.
            weight_inv_modB (bool, Optional): If True, weight the residual by modB so that it does not scale with coil currents. Defaults to True.
            verbose (bool, Optional): If True, print the optimization progress. Defaults to False.
        
        Returns:
            dict: A dictionary containing the results of the optimization. The dictionary contains the following keys in addition
            to others:

                - 'residual': the value of the residual at the solution
                - 'jacobian': the value of the Jacobian at the solution
                - 'hessian': the value of the Hessian at the solution
                - 'iter': the number of iterations taken to converge
                - 'success': True if the optimization converged, False otherwise
                - 'G': the value of G on the surface
                - 'iota': the value of iota on the surface
                - 'PLU': the LU decomposition of the hessian
                - 'type': 'ls'.
                - 'weight_inv_modB': the value of weight_inv_modB used in the optimization
        """
        if not self.need_to_run_code:
            return self.res

        s = self.surface
        if G is None:
            x = np.concatenate((s.get_dofs(), [iota]))
        else:
            x = np.concatenate((s.get_dofs(), [iota, G]))
        i = 0

        val, dval, d2val = self.boozer_penalty_constraints_vectorized(x, derivatives=2, constraint_weight=constraint_weight, optimize_G=G is not None, weight_inv_modB=weight_inv_modB)

        norm = np.linalg.norm(dval)
        while i < maxiter and norm > tol:
            d2val += stab*np.identity(d2val.shape[0])
            dx = np.linalg.solve(d2val, dval)
            if norm < 1e-9:
                dx += np.linalg.solve(d2val, dval - d2val@dx)
            x = x - dx
            val, dval, d2val = self.boozer_penalty_constraints_vectorized(x, derivatives=2, constraint_weight=constraint_weight, optimize_G=G is not None, weight_inv_modB=weight_inv_modB)
            norm = np.linalg.norm(dval)
            i = i+1

        # Get residual for output - vectorized version returns scalar objective
        # We use the gradient norm as a proxy for the residual norm
        r = dval  # Use gradient as residual representation

        P, L, U = lu(d2val)
        res = {
            "residual": r, "jacobian": dval, "hessian": d2val, "iter": i, "success": norm <= tol, "G": None,
            "PLU": (P, L, U), "vjp": partial(boozer_surface_dlsqgrad_dcoils_vjp, weight_inv_modB=weight_inv_modB),
            "type": "ls", "weight_inv_modB": weight_inv_modB
        }
        if G is None:
            s.set_dofs(x[:-1])
            iota = x[-1]
        else:
            s.set_dofs(x[:-2])
            iota = x[-2]
            G = x[-1]
            res['G'] = G
        res['iota'] = iota

        self.res = res
        self.need_to_run_code = False

        if verbose:
            print(f"NEWTON solve - {res['success']}  iter={res['iter']}, iota={res['iota']:.16f}, ||grad||_inf = {np.linalg.norm(res['jacobian'], ord=np.inf):.3e}", flush=True)

        return res

    def minimize_boozer_penalty_constraints_ls(self, tol=1e-12, maxiter=10, constraint_weight=1., iota=0., G=None, method='lm', weight_inv_modB=True):
        """
        This function does the same as :mod:`minimize_boozer_penalty_constraints_LBFGS`, but instead of LBFGS it
        uses a nonlinear least squares algorithm when ``method='lm'``.  Options for the method 
        are the same as for :mod:`scipy.optimize.least_squares`. If ``method='manual'``, then a 
        damped Gauss-Newton method is used.

        Args:
            tol (float, Optional): The tolerance for the optimization. Defaults to 1e-12.
            maxiter (int, Optional): The maximum number of iterations for the optimization. Defaults to 10.
            constraint_weight (float, Optional): The weight of the label constraint used when solving Boozer least squares.
            iota (float, Optional): The initial guess for the value of the rotational transform on the surface. Defaults to 0.
            G (float, Optional): The initial guess for the value of G on the surface. Defaults to None.
            method (str, Optional): The method to use for the optimization. Defaults to 'lm'.
            weight_inv_modB (bool, Optional): If True, weight the residual by modB so that it does not scale with coil currents. Defaults to True.

        Returns:
            res (dict): A dictionary containing the results of the optimization. The dictionary contains the following keys in addition
            to others:

                - 'residual': the value of the residual at the solution
                - 'gradient': the value of the gradient at the solution
                - 'jacobian': the value of the jacobian at the solution
                - 'success': True if the optimization converged, False otherwise
                - 'G': the value of G on the surface
                - 's': the surface object
                - 'iota': the value of iota on the surface
        """

        if not self.need_to_run_code:
            return self.res

        s = self.surface
        if G is None:
            x = np.concatenate((s.get_dofs(), [iota]))
        else:
            x = np.concatenate((s.get_dofs(), [iota, G]))
        norm = 1e10
        if method == 'manual':
            i = 0
            lam = 1.
            r, J = self._get_residual_vector_and_jacobian(
                x, constraint_weight, G is not None, weight_inv_modB)
            b = J.T@r
            JTJ = J.T@J
            norm = np.linalg.norm(b)
            while i < maxiter and norm > tol:
                dx = np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ)), b)
                x -= dx
                r, J = self._get_residual_vector_and_jacobian(
                    x, constraint_weight, G is not None, weight_inv_modB)
                b = J.T@r
                JTJ = J.T@J
                norm = np.linalg.norm(b)
                lam *= 1/3
                i += 1
            resdict = {
                "residual": r, "gradient": b, "jacobian": JTJ, "success": norm <= tol
            }
            if G is None:
                s.set_dofs(x[:-1])
                iota = x[-1]
            else:
                s.set_dofs(x[:-2])
                iota = x[-2]
                G = x[-1]
                resdict['G'] = G
            resdict['s'] = s
            resdict['iota'] = iota
            return resdict

        def fun(x): 
            return self._get_residual_vector_and_jacobian(
                x, constraint_weight, G is not None, weight_inv_modB)[0]

        def jac(x): 
            return self._get_residual_vector_and_jacobian(
                x, constraint_weight, G is not None, weight_inv_modB)[1]

        res = least_squares(fun, x, jac=jac, method=method, ftol=tol, xtol=tol, gtol=tol, x_scale=1.0, max_nfev=maxiter)
        resdict = {
            "info": res, "residual": res.fun, "gradient": res.grad, "jacobian": res.jac, "success": res.status > 0,
            "G": None,
        }
        if G is None:
            s.set_dofs(res.x[:-1])
            iota = res.x[-1]
        else:
            s.set_dofs(res.x[:-2])
            iota = res.x[-2]
            G = res.x[-1]
            resdict['G'] = G
        resdict['s'] = s
        resdict['iota'] = iota

        self.res = resdict
        self.need_to_run_code = False
        return resdict

    def _get_residual_vector_and_jacobian(self, x, constraint_weight, optimize_G, weight_inv_modB):
        """Helper function to get residual vector and Jacobian for least_squares"""
        if optimize_G:
            sdofs = x[:-2]
            iota = x[-2]
            G = x[-1]
        else:
            sdofs = x[:-1]
            iota = x[-1]
            G = None
        nsurfdofs = sdofs.size
        s = self.surface
        num_res = 3 * s.quadpoints_phi.size * s.quadpoints_theta.size

        s.set_dofs(sdofs)
        # When G=None, boozer_surface_residual returns J without G column
        # When G is provided, it returns J with G column
        boozer = boozer_surface_residual(s, iota, G, self.biotsavart, derivatives=1, weight_inv_modB=weight_inv_modB)
        r = boozer[0] / np.sqrt(num_res)
        J = boozer[1] / np.sqrt(num_res)

        l = self.label.J()
        rl = np.sqrt(constraint_weight) * (l - self.targetlabel)
        rz = np.sqrt(constraint_weight) * (s.gamma()[0, 0, 2] - 0.)
        r = np.concatenate((r, [rl, rz]))

        # Get constraint derivatives - shape should match J shape (which already has correct number of columns)
        dl = np.zeros(J.shape[1])
        drz = np.zeros(J.shape[1])
        dl[:nsurfdofs] = self.label.dJ(partials=True)(s)
        drz[:nsurfdofs] = s.dgamma_by_dcoeff()[0, 0, 2, :]

        J = np.vstack((J, np.sqrt(constraint_weight) * dl[None, :], np.sqrt(constraint_weight) * drz[None, :]))
        return r, J

    def minimize_boozer_exact_constraints_newton(self, tol=1e-12, maxiter=10, iota=0., G=None, lm=[0., 0.]):
        r"""
        This function solves the constrained optimization problem

        .. math::
            \text{min}_x ~ J(x)

        subject to

        .. math::
            l - l_0 &= 0 \\
            z(\varphi=0,\theta=0) - 0 &= 0

        using the method of Lagrange multipliers and applying Newton's method. In the above,
        :math:`J(x) = \frac{1}{2}\mathbf r(x)^T \mathbf r(x)`, and :math:`\mathbf r(x)` contains
        the Boozer residuals at quadrature points :math:`1,\dots,n`.

        The final constraint is not necessary for stellarator symmetric surfaces as it is automatically
        satisfied by the stellarator symmetric surface parametrization.

        Args:
            tol (float, Optional): The tolerance for the optimization. Defaults to 1e-12.
            maxiter (int, Optional): The maximum number of iterations for the optimization. Defaults to 10.
            iota (float, Optional): The initial guess for the value of the rotational transform on the surface. Defaults to 0.
            G (float, Optional): The initial guess for the value of G on the surface. Defaults to None.
            lm (list, Optional): The initial guesses for the Lagrange multipliers. Defaults to [0., 0.].

        Returns:
            dict: A dictionary containing the results of the optimization. The dictionary contains the following keys in addition
            to others:

                - 'residual': the value of the residual at the solution
                - 'jacobian': the value of the jacobian at the solution
                - 'iter': the number of iterations taken to converge
                - 'success': True if the optimization converged, False otherwise
                - 'G': the value of G on the surface
                - 'lm': the value of the Lagrange multipliers
        """

        if not self.need_to_run_code:
            return self.res

        s = self.surface
        if G is not None:
            xl = np.concatenate((s.get_dofs(), [iota, G], lm))
        else:
            xl = np.concatenate((s.get_dofs(), [iota], lm))
        val, dval = self.boozer_exact_constraints(xl, derivatives=1, optimize_G=G is not None)
        norm = np.linalg.norm(val)
        i = 0
        while i < maxiter and norm > tol:
            if s.stellsym:
                A = dval[:-1, :-1]
                b = val[:-1]
                dx = np.linalg.solve(A, b)
                if norm < 1e-9:  # iterative refinement for higher accuracy. TODO: cache LU factorisation
                    dx += np.linalg.solve(A, b-A@dx)
                xl[:-1] = xl[:-1] - dx
            else:
                dx = np.linalg.solve(dval, val)
                if norm < 1e-9:  # iterative refinement for higher accuracy. TODO: cache LU factorisation
                    dx += np.linalg.solve(dval, val-dval@dx)
                xl = xl - dx
            val, dval = self.boozer_exact_constraints(xl, derivatives=1, optimize_G=G is not None)
            norm = np.linalg.norm(val)
            i = i + 1

        if s.stellsym:
            lm = xl[-2]
        else:
            lm = xl[-2:]

        res = {
            "residual": val, "jacobian": dval, "iter": i, "success": norm <= tol, "lm": lm, "G": None,
        }
        if G is not None:
            s.set_dofs(xl[:-4])
            iota = xl[-4]
            G = xl[-3]
            res['G'] = G
        else:
            s.set_dofs(xl[:-3])
            iota = xl[-3]
        res['s'] = s
        res['iota'] = iota

        self.res = res
        self.need_to_run_code = False
        return res

    def solve_residual_equation_exactly_newton(self, tol=1e-10, maxiter=10, iota=0., G=None, verbose=False):
        """
        The function implements the BoozerExact approach by solving residual equation exactly using Newtons 
        method.  
        
        For Newton's method to be applied, we need the right balance of quadrature points, degrees
        of freedom and constraints.  For this reason, this function is only implemented for
        surfaces of type :obj:`~simsopt.geo.SurfaceXYZTensorFourier` right now.

        Given ``ntor``, ``mpol``, ``nfp`` and ``stellsym``, the surface is expected to be
        created in the following way::

            phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
            thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
            s = SurfaceXYZTensorFourier(
                mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
                quadpoints_phi=phis, quadpoints_theta=thetas)

        Or the following two are also possible in the stellsym case::

            phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
            thetas = np.linspace(0, 0.5, mpol+1, endpoint=False)

        or::

            phis = np.linspace(0, 1/(2*nfp), ntor+1, endpoint=False)
            thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)

        and then::

            s = SurfaceXYZTensorFourier(
                mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
                quadpoints_phi=phis, quadpoints_theta=thetas)

        For the stellarator symmetric case, there is some redundancy between DOFs.  This is
        taken care of inside this function.

        In the non-stellarator-symmetric case, the surface has
        ``(2*ntor+1)*(2*mpol+1)`` many quadrature points and
        ``3*(2*ntor+1)*(2*mpol+1)`` many dofs.

        Equations:
            - Boozer residual in x, y, and z at all quadrature points
            - z(0, 0) = 0
            - label constraint (e.g. volume or flux)

        Unknowns:
            - Surface dofs
            - iota
            - G

        So we end up having ``3*(2*ntor+1)*(2*mpol+1) + 2`` equations and the
        same number of unknowns.

        In the stellarator-symmetric case, we have
        ``D = (ntor+1)*(mpol+1)+ ntor*mpol + 2*(ntor+1)*mpol + 2*ntor*(mpol+1)
        = 6*ntor*mpol + 3*ntor + 3*mpol + 1``
        many dofs in the surface. After calling ``surface.get_stellsym_mask()`` we have kicked out
        ``2*ntor*mpol + ntor + mpol``
        quadrature points, i.e. we have
        ``2*ntor*mpol + ntor + mpol + 1``
        quadrature points remaining. In addition we know that the x coordinate of the
        residual at phi=0=theta is also always satisfied. In total this
        leaves us with
        ``3*(2*ntor*mpol + ntor + mpol) + 2`` equations for the boozer residual, plus
        1 equation for the label,
        which is the same as the number of surface dofs + 2 extra unknowns
        given by iota and G.

        Args:
            tol (float, Optional): The tolerance for the optimization. Defaults to 1e-10.
            maxiter (int, Optional): The maximum number of iterations for the optimization. Defaults to 10.
            iota (float, Optional): The initial guess for the value of the rotational transform on the surface. Defaults to 0.
            G (float, Optional): The initial guess for the value of G on the surface. Defaults to None.
            verbose (bool, Optional): If True, print the optimization progress. Defaults to False.
        
        Returns:
            dict: A dictionary containing the results of the optimization. The dictionary contains the following keys in addition
            to others:

                - 'residual': the value of the residual at the solution
                - 'jacobian': the value of the jacobian at the solution
                - 'iter': the number of iterations taken to converge
                - 'success': True if the optimization converged, False otherwise
                - 'G': the value of G on the surface
                - 's': the surface object
                - 'iota': the value of iota on the surface
                - 'PLU': the LU decomposition of the jacobian
                - 'mask': a mask for the residuals that are not used in the optimization
                - 'type': 'exact'.
                - 'vjp': the vector-Jacobian product for the optimization
        """
        if not self.need_to_run_code:
            return self.res

        from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
        s = self.surface
        if not isinstance(s, SurfaceXYZTensorFourier):
            raise RuntimeError('Exact solution of Boozer Surfaces only supported for SurfaceXYZTensorFourier')

        # In the case of stellarator symmetry, some of the information is
        # redundant, since the coordinates at (-phi, -theta) are the same (up
        # to sign changes) to those at (phi, theta). In addition, for stellsym
        # surfaces and stellsym magnetic fields, the residual in the x
        # component is always satisfied at phi=theta=0, so we ignore that one
        # too. The mask object below is True for those parts of the residual
        # that we need to keep, and False for those that we ignore.
        m = s.get_stellsym_mask()
        mask = np.concatenate((m[..., None], m[..., None], m[..., None]), axis=2)
        if s.stellsym:
            mask[0, 0, 0] = False
        mask = mask.flatten()

        label = self.label
        if G is None:
            G = 2. * np.pi * np.sum(np.abs([c.current.get_value() for c in self.biotsavart.coils])) * (4 * np.pi * 10**(-7) / (2 * np.pi))
        x = np.concatenate((s.get_dofs(), [iota, G]))
        i = 0
        r, J = boozer_surface_residual(s, iota, G, self.biotsavart, derivatives=1)
        norm = 1e6
        while i < maxiter:
            if s.stellsym:
                b = np.concatenate((r[mask], [(label.J()-self.targetlabel)]))
            else:
                b = np.concatenate((r[mask], [(label.J()-self.targetlabel), s.gamma()[0, 0, 2]]))
            norm = np.linalg.norm(b)
            if norm <= tol:
                break
            if s.stellsym:
                J = np.vstack((
                    J[mask, :],
                    np.concatenate((label.dJ(partials=True)(s), [0., 0.])),
                ))
            else:
                J = np.vstack((
                    J[mask, :],
                    np.concatenate((label.dJ(partials=True)(s), [0., 0.])),
                    np.concatenate((s.dgamma_by_dcoeff()[0, 0, 2, :], [0., 0.]))
                ))
            dx = np.linalg.solve(J, b)
            dx += np.linalg.solve(J, b-J@dx)
            x -= dx
            s.set_dofs(x[:-2])
            iota = x[-2]
            G = x[-1]
            i += 1
            r, J = boozer_surface_residual(s, iota, G, self.biotsavart, derivatives=1)

        if s.stellsym:
            J = np.vstack((
                J[mask, :],
                np.concatenate((label.dJ(partials=True)(s), [0., 0.])),
            ))
        else:
            J = np.vstack((
                J[mask, :],
                np.concatenate((label.dJ(partials=True)(s), [0., 0.])),
                np.concatenate((s.dgamma_by_dcoeff()[0, 0, 2, :], [0., 0.]))
            ))

        P, L, U = lu(J)
        res = {
            "residual": r, "jacobian": J, "iter": i, "success": norm <= tol, "G": G, "s": s, "iota": iota, "PLU": (P, L, U),
            "mask": mask, 'type': 'exact', "vjp": boozer_surface_dexactresidual_dcoils_dcurrents_vjp
        }

        if verbose:
            print(f"NEWTON solve - {res['success']}  iter={res['iter']}, iota={res['iota']:.16f}, ||residual||_inf = {np.linalg.norm(res['residual'], ord=np.inf):.3e}", flush=True)

        self.res = res
        self.need_to_run_code = False
        return res
