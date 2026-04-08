from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TYPE_MAP = {
    'Float32': np.dtype('<f4'),
    'Float64': np.dtype('<f8'),
    'Int32': np.dtype('<i4'),
    'UInt8': np.dtype('u1'),
    'UInt64': np.dtype('<u8'),
}


def load_appended_vtk_xml(path: Path) -> tuple[ET.Element, bytes]:
    raw = path.read_bytes()
    tag = b'<AppendedData'
    start = raw.index(tag)
    payload_start = raw.index(b'>', start) + 1
    payload_end = raw.index(b'</AppendedData>', payload_start)
    header = raw[:start].decode('utf-8') + '</VTKFile>'
    root = ET.fromstring(header)
    payload = raw[payload_start:payload_end].lstrip()
    if not payload.startswith(b'_'):
        raise ValueError(f'{path} does not use raw appended VTK data.')
    return root, payload[1:]


def read_data_array(payload: bytes, header_dtype: np.dtype, data_array: ET.Element) -> np.ndarray:
    offset = int(data_array.attrib['offset'])
    block_size = int(np.frombuffer(payload, dtype=header_dtype, count=1, offset=offset)[0])
    start = offset + header_dtype.itemsize
    stop = start + block_size
    dtype = TYPE_MAP[data_array.attrib['type']]
    return np.frombuffer(payload[start:stop], dtype=dtype).copy()


def squeeze_structured_shape(array: np.ndarray, dims: tuple[int, int, int], ncomp: int) -> np.ndarray:
    shaped = array.reshape(dims + (ncomp,), order='C')
    for axis in range(3):
        if shaped.shape[axis] == 1:
            shaped = np.take(shaped, 0, axis=axis)
            break
    if ncomp == 1:
        shaped = shaped[..., 0]
    return shaped


def load_structured_grid(path: Path) -> dict[str, np.ndarray]:
    root, payload = load_appended_vtk_xml(path)
    piece = root.find('.//Piece')
    if piece is None:
        raise ValueError(f'No Piece element found in {path}.')
    extent = [int(value) for value in piece.attrib['Extent'].split()]
    dims = (
        extent[1] - extent[0] + 1,
        extent[3] - extent[2] + 1,
        extent[5] - extent[4] + 1,
    )
    header_dtype = TYPE_MAP[root.attrib.get('header_type', 'UInt64')]
    arrays: dict[str, np.ndarray] = {}
    for data_array in root.findall('.//DataArray'):
        raw = read_data_array(payload, header_dtype, data_array)
        name = data_array.attrib['Name']
        ncomp = int(data_array.attrib.get('NumberOfComponents', '1'))
        arrays[name] = squeeze_structured_shape(raw, dims, ncomp)
    return arrays


def load_poly_lines(path: Path) -> list[np.ndarray]:
    root, payload = load_appended_vtk_xml(path)
    piece = root.find('.//Piece')
    if piece is None:
        raise ValueError(f'No Piece element found in {path}.')
    header_dtype = TYPE_MAP[root.attrib.get('header_type', 'UInt64')]
    arrays: dict[str, np.ndarray] = {}
    for data_array in root.findall('.//DataArray'):
        arrays[data_array.attrib['Name']] = read_data_array(payload, header_dtype, data_array)
    points = arrays['points'].reshape((-1, 3))
    connectivity = arrays['connectivity'].astype(int)
    offsets = arrays['offsets'].astype(int)
    lines = []
    start = 0
    for stop in offsets:
        lines.append(points[connectivity[start:stop]])
        start = stop
    return lines


def set_equal_3d_limits(ax: plt.Axes, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def surface_metrics(arrays: dict[str, np.ndarray]) -> dict[str, float | np.ndarray]:
    dS = np.linalg.norm(arrays['dphi x dtheta'], axis=2)
    b_total = arrays['B_total_mag']
    b_qs = np.mean(b_total * dS, axis=0) / np.mean(dS, axis=0)
    b_qs_2d = b_qs[None, :]
    qs_defect = b_total - b_qs_2d

    weights = dS / np.mean(dS)
    def weighted_rms(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(weights * values**2)))

    def weighted_mean(values: np.ndarray) -> float:
        return float(np.mean(weights * values))

    points = arrays['points']
    major_radius = np.sqrt(points[..., 0] ** 2 + points[..., 1] ** 2)
    nonqs_ratio = float(np.mean(dS * qs_defect**2) / np.mean(dS * b_qs_2d**2))

    return {
        'nonqs_ratio': nonqs_ratio,
        'normal_rms': weighted_rms(arrays['normal_residual']),
        'pressure_rms': weighted_rms(arrays['pressure_balance']),
        'coil_match_rms': weighted_rms(arrays['coil_match_mag']),
        'sheet_current_mean': weighted_mean(arrays['sheet_current_mag']),
        'major_radius_mean': weighted_mean(major_radius),
        'B_total_mean': weighted_mean(b_total),
        'B_external_mean': weighted_mean(arrays['B_external_mag']),
        'qs_defect': qs_defect,
        'magnetic_pressure_jump': arrays['magnetic_pressure_jump'],
    }


def plot_geometry(output_dir: Path, init_surface: dict[str, np.ndarray], final_surface: dict[str, np.ndarray], init_curves: list[np.ndarray], final_curves: list[np.ndarray]) -> None:
    fig = plt.figure(figsize=(12, 5.5), constrained_layout=True)
    axes = [fig.add_subplot(1, 2, idx + 1, projection='3d') for idx in range(2)]
    panels = [
        ('Initial coils and surface', axes[0], init_surface['points'], init_curves, '#8ecae6', '#023047'),
        ('Final coils and surface', axes[1], final_surface['points'], final_curves, '#90be6d', '#bc4749'),
    ]
    for title, ax, surface_points, curves, surface_color, curve_color in panels:
        x = surface_points[..., 0]
        y = surface_points[..., 1]
        z = surface_points[..., 2]
        ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0.35, edgecolor='white', alpha=0.75, color=surface_color)
        for curve in curves:
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=curve_color, linewidth=0.8, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        set_equal_3d_limits(ax, surface_points.reshape((-1, 3)))
        ax.view_init(elev=24, azim=35)
    fig.savefig(output_dir / 'boozerQA_finitebeta_geometry_comparison.png', dpi=220)
    plt.close(fig)


def plot_state_comparison(output_dir: Path, init_metrics: dict[str, float | np.ndarray], final_metrics: dict[str, float | np.ndarray]) -> None:
    qs_init = np.asarray(init_metrics['qs_defect'])
    qs_final = np.asarray(final_metrics['qs_defect'])
    mp_init = np.asarray(init_metrics['magnetic_pressure_jump'])
    mp_final = np.asarray(final_metrics['magnetic_pressure_jump'])

    qs_scale = np.max(np.abs(np.concatenate([qs_init.ravel(), qs_final.ravel()])))
    mp_scale = np.max(np.abs(np.concatenate([mp_init.ravel(), mp_final.ravel()])))
    qs_scale = qs_scale if qs_scale > 0 else 1.0
    mp_scale = mp_scale if mp_scale > 0 else 1.0

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    panels = [
        (axes[0, 0], qs_init, 'Initial QA defect |B|-<|B|>_phi', 'coolwarm', -qs_scale, qs_scale),
        (axes[0, 1], qs_final, 'Final QA defect |B|-<|B|>_phi', 'coolwarm', -qs_scale, qs_scale),
        (axes[1, 0], mp_init, 'Initial magnetic pressure jump', 'coolwarm', -mp_scale, mp_scale),
        (axes[1, 1], mp_final, 'Final magnetic pressure jump', 'coolwarm', -mp_scale, mp_scale),
    ]
    for ax, field, title, cmap, vmin, vmax in panels:
        image = ax.imshow(field.T, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('Toroidal index')
        ax.set_ylabel('Poloidal index')
        fig.colorbar(image, ax=ax, shrink=0.82)
    fig.savefig(output_dir / 'boozerQA_finitebeta_state_comparison.png', dpi=220)
    plt.close(fig)


def plot_objective_comparison(output_dir: Path, init_metrics: dict[str, float | np.ndarray], final_metrics: dict[str, float | np.ndarray]) -> None:
    residual_labels = ['QA ratio', 'Coil-match RMS', 'Normal RMS', 'Pressure RMS']
    residual_init = [
        float(init_metrics['nonqs_ratio']),
        float(init_metrics['coil_match_rms']),
        float(init_metrics['normal_rms']),
        float(init_metrics['pressure_rms']),
    ]
    residual_final = [
        float(final_metrics['nonqs_ratio']),
        float(final_metrics['coil_match_rms']),
        float(final_metrics['normal_rms']),
        float(final_metrics['pressure_rms']),
    ]

    state_labels = ['Mean |K|', 'Mean R', 'Mean |B_total|', 'Mean |B_external|']
    state_init = [
        float(init_metrics['sheet_current_mean']),
        float(init_metrics['major_radius_mean']),
        float(init_metrics['B_total_mean']),
        float(init_metrics['B_external_mean']),
    ]
    state_final = [
        float(final_metrics['sheet_current_mean']),
        float(final_metrics['major_radius_mean']),
        float(final_metrics['B_total_mean']),
        float(final_metrics['B_external_mean']),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    width = 0.36
    for ax, labels, values_init, values_final, title, logy in [
        (axes[0], residual_labels, residual_init, residual_final, 'Residual and QS objectives', True),
        (axes[1], state_labels, state_init, state_final, 'State and diagnostic summaries', True),
    ]:
        xpos = np.arange(len(labels))
        ax.bar(xpos - width / 2, values_init, width=width, label='Initial', color='#457b9d')
        ax.bar(xpos + width / 2, values_final, width=width, label='Final', color='#e76f51')
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=20, ha='right')
        ax.set_title(title)
        if logy:
            ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.25)
    axes[0].legend(loc='upper right')
    fig.savefig(output_dir / 'boozerQA_finitebeta_objective_comparison.png', dpi=220)
    plt.close(fig)

    with (output_dir / 'boozerQA_finitebeta_objective_summary.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['metric', 'initial', 'final'])
        for label, init_value, final_value in zip(residual_labels + state_labels, residual_init + state_init, residual_final + state_final):
            writer.writerow([label, f'{init_value:.16e}', f'{final_value:.16e}'])


def load_history(path: Path) -> list[dict[str, float]]:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        return [{key: float(value) for key, value in row.items()} for row in reader]


def write_history_summary(output_dir: Path, history_rows: list[dict[str, float]], pressure_rows: list[dict[str, float]]) -> None:
    summary_path = output_dir / 'boozerQA_finitebeta_report_summary.txt'
    with summary_path.open('w') as handle:
        if history_rows:
            first = history_rows[0]
            best = min(history_rows, key=lambda row: row['J'])
            handle.write('Outer QA history\n')
            handle.write(f"  evaluations: {len(history_rows)}\n")
            handle.write(f"  initial J: {first['J']:.16e}\n")
            handle.write(f"  best J: {best['J']:.16e}\n")
            handle.write(f"  best iota: {best['iota']:.16e}\n")
            handle.write(f"  best major radius: {best['major_radius']:.16e}\n")
        if pressure_rows:
            final = pressure_rows[-1]
            handle.write('\nPressure scan\n')
            handle.write(f"  samples: {len(pressure_rows)}\n")
            handle.write(f"  final pressure jump: {final['pressure_jump']:.16e}\n")
            handle.write(f"  final beta jump: {final['beta_jump']:.16e}\n")
            handle.write(f"  final iota: {final['iota']:.16e}\n")
            handle.write(f"  final lambda_current: {final['lambda_current']:.16e}\n")
            handle.write(f"  final nonqs: {final['nonqs']:.16e}\n")


def main() -> None:
    output_dir = Path(__file__).resolve().parent / 'output'
    init_surface = load_structured_grid(output_dir / 'surf_init.vts')
    final_surface = load_structured_grid(output_dir / 'surf_opt.vts')
    init_boundary = load_structured_grid(output_dir / 'boozerQA_finitebeta_boundary_init.vts')
    final_boundary = load_structured_grid(output_dir / 'boozerQA_finitebeta_boundary.vts')
    init_curves = load_poly_lines(output_dir / 'curves_init.vtu')
    final_curves = load_poly_lines(output_dir / 'curves_opt.vtu')

    init_metrics = surface_metrics(init_boundary)
    final_metrics = surface_metrics(final_boundary)
    history_rows = load_history(output_dir / 'boozerQA_finitebeta_history.csv')
    pressure_rows = load_history(output_dir / 'boozerQA_finitebeta_pressure_scan.csv')

    plot_geometry(output_dir, init_surface, final_surface, init_curves, final_curves)
    plot_state_comparison(output_dir, init_metrics, final_metrics)
    plot_objective_comparison(output_dir, init_metrics, final_metrics)
    write_history_summary(output_dir, history_rows, pressure_rows)


if __name__ == '__main__':
    main()