"""Compare a fresh experiment run with the reference data in this repository."""

import argparse
import csv
import json
from math import isclose
from pathlib import Path


def check_equal(actual, expected, location):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"{location}: different fields")
        for key, value in expected.items():
            check_equal(actual[key], value, f"{location}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{location}: different number of values")
        for index, (current, value) in enumerate(zip(actual, expected)):
            check_equal(current, value, f"{location}[{index}]")
    elif isinstance(expected, float):
        if not isinstance(actual, (float, int)) or not isclose(actual, expected, rel_tol=2e-9, abs_tol=5e-13):
            raise ValueError(f"{location}: {actual!r} differs from {expected!r}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{location}: {actual!r} differs from {expected!r}")


def parse_cell(cell):
    try:
        return float(cell)
    except ValueError:
        return cell


def rows(path):
    with path.open(encoding="utf-8", newline="") as source:
        return [[parse_cell(cell) for cell in row] for row in csv.reader(source)]


def verify(fresh, recorded):
    summary = json.loads((fresh / "summary.json").read_text(encoding="utf-8"))
    expected = json.loads((recorded / "summary.json").read_text(encoding="utf-8"))
    check_equal(summary, expected, "summary")

    data_files = sorted(path.name for path in (recorded / "data").glob("*.csv"))
    check_equal(sorted(path.name for path in (fresh / "data").glob("*.csv")), data_files, "CSV file names")
    for name in data_files:
        check_equal(rows(fresh / "data" / name), rows(recorded / "data" / name), name)

    formats = {".pdf": b"%PDF-", ".png": b"\x89PNG\r\n\x1a\n"}
    figure_files = sorted(path.name for path in (recorded / "figures").iterdir() if path.suffix in formats)
    check_equal(sorted(path.name for path in (fresh / "figures").iterdir() if path.suffix in formats),
                figure_files, "figure file names")
    for name in figure_files:
        path = fresh / "figures" / name
        with path.open("rb") as figure:
            if not figure.read(8).startswith(formats[path.suffix]):
                raise ValueError(f"{name}: invalid image format")

    center = summary["tasks"]["task1"]
    small_grid = summary["validation"]["direct_comparison"]
    return {
        "center_source": {key: center[key] for key in ("sweeps", "residual", "peak", "symmetry_errors")},
        "small_grid_direct_check": [
            {key: row[key] for key in ("tolerance", "sweeps", "residual", "max_error", "symmetry_errors")}
            for row in small_grid
        ],
        "verified_scan_records": summary["scan_row_count"],
        "verified_data_tables": len(data_files),
        "regenerated_figures": len(figure_files) // len(formats),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated", type=Path)
    args = parser.parse_args()
    reference = Path(__file__).resolve().parent / "results"
    print(json.dumps(verify(args.generated, reference), indent=2))
