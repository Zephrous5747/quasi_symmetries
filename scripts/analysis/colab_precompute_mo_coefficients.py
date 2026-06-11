#!/usr/bin/env python
"""
Colab workflow: append canonical HF MO coefficients to existing Hamiltonian caches.

Run this on Google Colab after uploading the repo (or cloning) and the hamiltonian_cache/
folder. PySCF is required; local machines without PySCF can consume the patched .h5 files.

Example (Colab cell):

    !pip -q install pyscf openfermion openfermionpyscf
    %cd /content/quasi_symmetries
    !python scripts/colab_precompute_mo_coefficients.py --molecule h2o --x 1.6433333333333333

Then download the updated cache file, e.g. hamiltonian_cache/h2o_1643_1045.h5.
"""

from __future__ import annotations

import argparse

from quasi_symmetries.hamiltonian.generation import append_mo_coefficients_to_cache
from quasi_symmetries.hamiltonian.geometry import iter_scan_points


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append MO coefficients to Hamiltonian HDF5 caches.")
    parser.add_argument("--molecule", default="h2o")
    parser.add_argument("--x", type=float, help="Single geometry parameter.")
    parser.add_argument("--grid", nargs="*", type=float, help="Optional geometry grid.")
    parser.add_argument("--cache-dir", default="hamiltonian_cache")
    parser.add_argument("--hoh-angle-deg", type=float, default=104.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    geom_kwargs = {"hoh_angle_deg": args.hoh_angle_deg}

    if args.x is not None:
        append_mo_coefficients_to_cache(
            args.molecule,
            args.x,
            cache_dir=args.cache_dir,
            overwrite=args.overwrite,
            **geom_kwargs,
        )
        return

    grid = args.grid
    if grid is None:
        grid = [x for x, _ in iter_scan_points(args.molecule, **geom_kwargs)]

    for x, point_kwargs in iter_scan_points(args.molecule, grid=grid, **geom_kwargs):
        append_mo_coefficients_to_cache(
            args.molecule,
            x,
            cache_dir=args.cache_dir,
            overwrite=args.overwrite,
            **point_kwargs,
        )


if __name__ == "__main__":
    main()
