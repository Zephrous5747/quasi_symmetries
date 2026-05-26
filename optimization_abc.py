"""Compatibility wrapper for the unified workflow module."""

from typing import Any, Iterable

from optimization_workflow import (
    WORKFLOW_SHARED_ABC,
    evaluate_single_point as _workflow_evaluate_single_point,
    main as _workflow_main,
    run_scan as _workflow_run_scan,
)


def evaluate_single_point(molecule: str, x: float, **kwargs: Any) -> dict[str, Any]:
    return _workflow_evaluate_single_point(
        workflow=WORKFLOW_SHARED_ABC,
        molecule=molecule,
        x=x,
        **kwargs,
    )


def run_scan(
    molecule: str,
    grid: Iterable[float],
    csv_filename: str | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return _workflow_run_scan(
        workflow=WORKFLOW_SHARED_ABC,
        molecule=molecule,
        grid=grid,
        csv_filename=csv_filename,
        **kwargs,
    )


def main(
    molecule: str = "lih",
    grid: Iterable[float] | None = None,
    csv_filename: str | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return _workflow_main(
        workflow=WORKFLOW_SHARED_ABC,
        molecule=molecule,
        grid=grid,
        csv_filename=csv_filename,
        **kwargs,
    )
