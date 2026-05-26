#!/usr/bin/env python
"""
Generate precomputed Hamiltonian HDF5 caches (requires PySCF).

Example:
    python generate_hamiltonians.py --molecule h4_rectangle --grid 0.6 1.1 2.0
    python generate_hamiltonians.py --molecule h2o --hoh-angle-deg 104.5
    python generate_hamiltonians.py --all-default
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hamiltonian_generation import generate_and_save, generate_scan
from hamiltonian_geometry import SUPPORTED_MOLECULES, default_grid_for_molecule


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Hamiltonian HDF5 cache files.")
    parser.add_argument("--molecule", choices=sorted(SUPPORTED_MOLECULES))
    parser.add_argument(
        "--grid",
        nargs="*",
        type=float,
        help="Geometry parameter values (default: molecule-specific grid).",
    )
    parser.add_argument(
        "--cache-dir",
        default="hamiltonian_cache",
        help="Output directory for .h5 files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--hoh-angle-deg",
        type=float,
        default=104.5,
        help="H2O H-O-H angle in degrees.",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=float,
        default=1.5,
        help="H4 rectangle aspect ratio (long/short).",
    )
    parser.add_argument(
        "--all-default",
        action="store_true",
        help="Generate default grids for every supported molecule.",
    )
    parser.add_argument(
        "--x",
        type=float,
        help="Single geometry parameter (alternative to --grid).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cache_dir = Path(args.cache_dir)
    geom_extras = {
        "hoh_angle_deg": args.hoh_angle_deg,
        "aspect_ratio": args.aspect_ratio,
    }

    if args.all_default:
        for mol in sorted(SUPPORTED_MOLECULES):
            print(f"\n=== {mol} ===")
            generate_scan(
                mol,
                grid=default_grid_for_molecule(mol),
                cache_dir=cache_dir,
                overwrite=args.overwrite,
                **geom_extras,
            )
        return

    if args.molecule is None:
        raise SystemExit("Specify --molecule or --all-default.")

    if args.x is not None:
        generate_and_save(
            args.molecule,
            args.x,
            cache_dir=cache_dir,
            overwrite=args.overwrite,
            **geom_extras,
        )
        return

    grid = args.grid
    if grid is None:
        grid = default_grid_for_molecule(args.molecule)

    generate_scan(
        args.molecule,
        grid=grid,
        cache_dir=cache_dir,
        overwrite=args.overwrite,
        **geom_extras,
    )


if __name__ == "__main__":
    main()
