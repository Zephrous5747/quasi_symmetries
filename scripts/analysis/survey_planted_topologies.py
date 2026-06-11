"""Survey Eq. (9)-(10) parent Hamiltonians on H4 and LiH after random-unitary confusion."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.helpers import (
    H4_GEOMETRY,
    H4_MOLECULE,
    LIH_GEOMETRY,
    LIH_MOLECULE,
    PARENT_PROTOCOL_NAMES,
    PLANTED_SEED,
    load_h4_reference,
    load_lih_reference,
    run_parent_hamiltonian_survey,
)


def print_survey(label: str, rows: list[dict[str, object]]) -> None:
    print(f"\n{'=' * 78}")
    print(label)
    print(f"{'=' * 78}")
    header = (
        f"{'Parent':14} {'Optimize':14} {'E0':>12} "
        f"{'K_pre':>6} {'K_post':>7} {'Var_post':>10}"
    )
    print(header)
    print("-" * len(header))

    for parent_protocol in PARENT_PROTOCOL_NAMES:
        block = [row for row in rows if row["ParentProtocol"] == parent_protocol]
        construction = block[0]["Construction"]
        energy = block[0]["EnergyTarget"]
        print(f"[{parent_protocol}, {construction}, E0={float(energy):.6f}]")
        for row in block:
            print(
                f"{'':14} {row['OptimizationProtocol']:14} "
                f"{'':12} "
                f"{row['K_Before']:6d} {row['K_After']:7d} "
                f"{row['VarianceAfter']:10.4f}"
            )
        print()


def main() -> None:
    systems = [
        (f"{H4_MOLECULE} @ {H4_GEOMETRY} A", load_h4_reference()),
        (f"{LIH_MOLECULE} @ {LIH_GEOMETRY} A", load_lih_reference()),
    ]

    for system_label, ref in systems:
        rows = run_parent_hamiltonian_survey(
            ref,
            apply_confusion=True,
            confusion_seed=PLANTED_SEED,
        )
        dim = len(ref["basis_bitstrings"])
        print_survey(
            f"{system_label}  (dim={dim}, n_spatial={ref['n_spatial']}, seed={PLANTED_SEED})",
            rows,
        )


if __name__ == "__main__":
    main()
