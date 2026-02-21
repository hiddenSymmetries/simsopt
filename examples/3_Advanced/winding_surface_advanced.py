#!/usr/bin/env python3
"""
Cut Coils: Extract discrete coils from a current potential on a winding surface.

This example demonstrates how to extract coil contours from a regularized current
potential (from REGCOIL or simsopt-regcoil) and convert them into simsopt Coil
objects. The workflow includes:

1. Loading current potential data from a NetCDF file (simsopt-regcoil or legacy REGCOIL)
2. Computing the current potential φ and |∇φ| on a (θ, ζ) grid
3. Selecting contours by:
   - Interactive double-click selection
   - Specifying (θ, ζ) points that lie on desired contours
   - Specifying contour level values
   - Choosing N contours per field period
4. Classifying contours as window-pane (closed), modular, or helical
5. Computing currents for each contour
6. Converting contours to 3D curves and Coil objects
7. Applying stellarator symmetry to get the full coil set

Usage
-----
From the simsopt root directory:

    python winding_surface_advanced.py --surface regcoil_out.hsx.nc

With custom options:

    python winding_surface_advanced.py \\
        --surface /path/to/simsopt_regcoil_out.nc \\
        --ilambda 2 \\
        --points "0.5,1.0" "1.0,0.5" \\
        --output my_output_dir

For interactive mode (double-click to select contours):

    python winding_surface_advanced.py --surface file.nc --interactive

Requirements
------------
- A simsopt-regcoil or REGCOIL NetCDF output file
- For legacy REGCOIL: a surface file (nescin format) for 3D mapping
- matplotlib for contour visualization
"""

import argparse
from pathlib import Path

# Add parent to path for standalone run
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simsopt.util import run_cut_coils

# Default test file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
DEFAULT_REGCOIL = TEST_DIR / "regcoil_out.hsx.nc"


def main():
    parser = argparse.ArgumentParser(
        description="Extract coils from a current potential (REGCOIL / simsopt-regcoil output).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--surface",
        type=Path,
        default=DEFAULT_REGCOIL,
        help="REGCOIL or simsopt-regcoil NetCDF file (relative paths are resolved against tests/test_files).",
    )
    parser.add_argument(
        "--ilambda",
        type=int,
        default=-1,
        help="Lambda index (0-based) to use.",
    )
    parser.add_argument(
        "--points",
        nargs="*",
        type=str,
        default=None,
        help='(θ,ζ) points as "theta,zeta" for contour selection.',
    )
    parser.add_argument(
        "--no-sv",
        action="store_false",
        help="Use full (multi-valued) current potential with net toroidal/poloidal currents.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Use interactive double-click contour selection.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for VTK and JSON (default: winding_surface_<surface_stem>/).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not show final coil plot.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save coils to JSON file.",
    )
    args = parser.parse_args()

    points = None
    if args.points:
        points = []
        for s in args.points:
            parts = s.split(",")
            if len(parts) != 2:
                parser.error(f"Invalid point format: {s}. Use theta,zeta")
            points.append([float(parts[0]), float(parts[1])])

    surface_filename = args.surface if args.surface.is_absolute() else TEST_DIR / args.surface
    output_path = args.output or Path(f"winding_surface_{surface_filename.stem}")
    run_cut_coils(
        surface_filename=surface_filename,
        ilambda=args.ilambda,
        points=points,
        single_valued=not args.no_sv,
        interactive=args.interactive,
        show_final_coilset=not args.no_plot,
        write_coils_to_file=args.save,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
