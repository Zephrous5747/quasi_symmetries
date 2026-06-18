"""Run H2O fixed-ABC seniority optimization and store rotation artifacts locally."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quasi_symmetries.config import LEGACY_ABC_OPT_RESULTS_DIR
from quasi_symmetries_abc.workflows import abc as ow


def main() -> None:
    csv_path = LEGACY_ABC_OPT_RESULTS_DIR / "h2o_quasi_symmetry_fixed_abc.csv"
    print("[h2o-seniority] starting fixed_abc scan", flush=True)
    rows = ow.main(
        workflow=ow.WORKFLOW_FIXED_ABC,
        molecule="h2o",
        csv_filename=str(csv_path),
    )
    print(f"[h2o-seniority] completed {len(rows)} geometries", flush=True)

    out_dir = LEGACY_ABC_OPT_RESULTS_DIR / "h2o_seniority_rotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        x = float(row["Geometry_Param"])
        thetas = np.asarray(json.loads(row["Thetas"]), dtype=float)
        u = np.asarray(json.loads(row["U_Spatial"]), dtype=np.complex128)
        pairs = np.asarray(json.loads(row["Pairs"]), dtype=int)
        operator_angles = np.asarray(json.loads(row["Operator_Angles"]), dtype=float)
        tag = f"{x:.6g}".replace(".", "p")
        out = out_dir / f"h2o_{tag}.npz"
        np.savez(
            out,
            molecule="h2o",
            geometry_param=x,
            workflow="fixed_abc",
            thetas=thetas,
            operator_angles=operator_angles,
            u_spatial=u,
            pairs=pairs,
            v_identity=float(row["V_Identity"]),
            v_optimized=float(row["V_Optimized"]),
            a=float(row["a"]),
            b=float(row["b"]),
            c=float(row["c"]),
            e_hf=float(row["E_HF"]),
            e_fci=float(row["E_FCI"]),
            n_spatial=int(row["n_spatial"]),
            n_electrons=int(row["n_electrons"]),
        )
        print(
            f"[h2o-seniority] wrote {out.name} V_opt={float(row['V_Optimized']):.6g}",
            flush=True,
        )

    manifest = {
        "workflow": "fixed_abc",
        "molecule": "h2o",
        "csv": str(csv_path),
        "rotation_dir": str(out_dir),
        "geometries": [float(r["Geometry_Param"]) for r in rows],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[h2o-seniority] wrote {out_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
