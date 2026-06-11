#!/usr/bin/env python
"""Run the parity-parent test suite (PySCF tests skipped when unavailable)."""

from __future__ import annotations

import argparse
import sys
import unittest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose unittest output.",
    )
    parser.add_argument(
        "-k",
        dest="pattern",
        default=None,
        help="Only run tests matching this substring (unittest -k).",
    )
    args = parser.parse_args(argv)

    loader = unittest.TestLoader()
    if args.pattern:
        suite = loader.loadTestsFromName(f"tests.{args.pattern}")
        if suite.countTestCases() == 0:
            suite = unittest.TestSuite()
            for module_suite in loader.discover("tests", pattern="test_*.py"):
                for test_group in module_suite:
                    for test_case in test_group:
                        if args.pattern in test_case.id():
                            suite.addTest(test_case)
    else:
        suite = loader.discover("tests", pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
