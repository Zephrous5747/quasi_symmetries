"""Backward-compatible entry point; see plot_orbital_heatmaps.py."""

from plot_orbital_heatmaps import main

if __name__ == "__main__":
    import sys

    if "--molecule" not in sys.argv:
        sys.argv.extend(["--molecule", "h2o", "--x", "1.6433333333333333"])
    main()
