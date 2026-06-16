"""Merge sparse action diagnostics back into N2 summary CSV tables."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from quasi_symmetries.config import CACHE_DIR, IMAGES_DIR, OPT_RESULTS_DIR, TABLES_DIR

ACTION_COLS = [
    "Build_Seconds",
    "Operator_Count",
    "Sum_CommSq_Action",
    "Sum_Expectation_Action",
    "Sum_Variance_Action",
    "Coarse_Entropy_Action",
    "NumSectors_Action",
    "Expectations_Action_JSON",
    "Variances_Action_JSON",
    "CommSq_Action_JSON",
    "Energy_Seconds",
    "Edec_Identity",
    "Edec_Optimized",
    "Ecoupled_Identity",
    "Ecoupled_Optimized",
    "Kcoupled_Identity",
    "Kcoupled_Optimized",
    "Coupled_Converged_Identity",
    "Coupled_Converged_Optimized",
]
MERGE_FIELDS = {
    "Sum_CommSq_Optimized": "Sum_CommSq_Action",
    "Sum_Sexp_Optimized": "Sum_Expectation_Action",
    "Coarse_Entropy_Optimized": "Coarse_Entropy_Action",
    "NumSectors_Optimized": "NumSectors_Action",
    "Edec_Identity": "Edec_Identity",
    "Edec_Optimized": "Edec_Optimized",
    "Ecoupled_Identity": "Ecoupled_Identity",
    "Ecoupled_Optimized": "Ecoupled_Optimized",
    "Kcoupled_Identity": "Kcoupled_Identity",
    "Kcoupled_Optimized": "Kcoupled_Optimized",
}


def _is_missing(value: str | None) -> bool:
    if value is None or value == "":
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _row_key(row: dict[str, str], *, quartet: bool) -> tuple[str, ...]:
    if quartet:
        return (row["Baseline"], row["Geometry_Param"])
    return (row["Geometry_Param"],)


def _load_action_lookup(path: Path, *, quartet: bool) -> dict[tuple[str, ...], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {_row_key(row, quartet=quartet): row for row in rows}


def merge_tables(
    base_path: Path,
    action_path: Path,
    out_path: Path,
    *,
    quartet: bool,
) -> None:
    action_lookup = _load_action_lookup(action_path, quartet=quartet)
    with base_path.open(newline="", encoding="utf-8") as handle:
        base_rows = list(csv.DictReader(handle))
    if not base_rows:
        raise ValueError(f"No rows found in {base_path}")

    fieldnames = list(base_rows[0].keys())
    for column in ACTION_COLS:
        if column not in fieldnames:
            fieldnames.append(column)

    merged: list[dict[str, str]] = []
    for row in base_rows:
        key = _row_key(row, quartet=quartet)
        action = action_lookup.get(key)
        if action is None:
            raise KeyError(f"Missing action diagnostics for key {key}")

        out = dict(row)
        dense_skipped = str(out.get("DenseDiagnosticsSkipped", "")).lower() == "true"
        for dst, src in MERGE_FIELDS.items():
            action_value = action.get(src)
            if _is_missing(action_value):
                continue
            if _is_missing(out.get(dst)) or dense_skipped:
                out[dst] = action_value
        for column in ACTION_COLS:
            if column in action:
                out[column] = action[column]
        merged.append(out)

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)
    print(f"Wrote {len(merged)} rows to {out_path}")


def main() -> None:
    tables = TABLES_DIR
    merge_tables(
        tables / "n2_quartet_variance_summary.csv",
        tables / "n2_quartet_action_diagnostics.csv",
        tables / "n2_quartet_baseline_summary.csv",
        quartet=True,
    )
    merge_tables(
        tables / "n2_quasi_symmetry_fixed_abc.csv",
        tables / "n2_fixed_abc_action_diagnostics.csv",
        tables / "n2_quasi_symmetry_fixed_abc.csv",
        quartet=False,
    )
    merge_tables(
        tables / "n2_mixed_pool_summary.csv",
        tables / "n2_mixed_pool_action_diagnostics.csv",
        tables / "n2_mixed_pool_summary.csv",
        quartet=False,
    )
    merge_tables(
        tables / "n2_parity_seniority_summary.csv",
        tables / "n2_parity_seniority_action_diagnostics.csv",
        tables / "n2_parity_seniority_summary.csv",
        quartet=False,
    )


if __name__ == "__main__":
    main()
