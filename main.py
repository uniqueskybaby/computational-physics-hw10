"""Run the week 10 point-source lattice experiments from the repository root."""

import argparse
import json
from pathlib import Path

from sor import PointSource, solve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="run the N=31 task using the standard library")
    mode.add_argument("--validate", action="store_true", help="run the independent small-grid checks")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--no-plots", action="store_true", help="generate only the numerical results")
    args = parser.parse_args()

    if args.smoke:
        problem = PointSource(31, 0.5)
        result = solve(problem, omega=1.5)
        center_x, center_y = problem.source_xy
        center_index = center_y * problem.width + center_x
        assert result.converged and result.residual < 1e-6
        assert result.field.index(max(result.field)) == center_index
        print(json.dumps({"N": 31, "m": 0.5, "omega": 1.5,
                          "sweeps": result.sweeps, "residual_inf": result.residual,
                          "source_response": result.field[center_index]}, indent=2))
    elif args.validate:
        from reference import validation_examples
        print(json.dumps(validation_examples(), ensure_ascii=False, indent=2))
    else:
        from experiments import run
        run(args.output)
        if not args.no_plots:
            from plot_results import draw_all
            draw_all(args.output)


if __name__ == "__main__":
    main()
