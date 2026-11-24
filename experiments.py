"""Generate numerical evidence for the four tasks and the extensions."""

import csv
from functools import lru_cache
import json
from math import exp
from pathlib import Path

import numpy as np

from reference import as_array, sine_solution, validation_examples, verify_solution
from sor import PointSource, solve, theory

TOLERANCE = 1e-6
MAX_SWEEPS = 5000
MASSES = (0.2, 0.5, 1.0)
COARSE = (*range(1000, 1901, 100), 1950)
NEAR_TWO = (1975, 1990)


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_grid(path, field):
    np.savetxt(path, field, delimiter=",", fmt="%.17g")


def first_threshold_crossing(normalized, threshold):
    """First integer site strictly below the threshold, plus linear interpolation."""
    distance = next(k for k, value in enumerate(normalized) if value < threshold)
    hi = normalized[distance - 1]
    lo = normalized[distance]
    between = distance - 1 + (hi - threshold) / (hi - lo)
    return distance, float(between)


def solve_checked(problem, omega=1.5, tolerance=TOLERANCE, method="sor"):
    result = solve(problem, omega, tolerance, MAX_SWEEPS, method)
    if not result.converged:
        raise RuntimeError(f"N={problem.n}, m={problem.mass}, omega={omega}: {result.status}")
    reference = sine_solution(problem)
    metrics = verify_solution(problem, result, tolerance, reference)
    return result, reference, metrics


def scan_mass(mass):
    problem = PointSource(63, mass)
    rho_j, omega_th = theory(problem.n, mass)

    @lru_cache(maxsize=None)
    def calculate(omega, tolerance=TOLERANCE):
        return solve(problem, omega, tolerance, MAX_SWEEPS)

    def row(omega, stage, tolerance=TOLERANCE):
        result = calculate(omega, tolerance)
        assert 0 < result.sweeps <= MAX_SWEEPS
        if result.converged:
            assert result.residual < tolerance <= result.residuals[-2]
            assert all(r >= tolerance for r in result.residuals[1:-1])
        return {"n": problem.n, "mass": mass, "omega": omega, "stage": stage,
                "tolerance": tolerance, "max_sweeps": MAX_SWEEPS,
                "sweeps": result.sweeps, "previous_residual": result.residuals[-2],
                "residual": result.residual, "converged": result.converged,
                "status": result.status}

    def sweep(ticks, stage, tolerance=TOLERANCE):
        return [row(tick / 1000, stage, tolerance) for tick in ticks]

    def minima(rows):
        count = min(r["sweeps"] for r in rows if r["converged"])
        return sorted({r["omega"] for r in rows if r["converged"] and r["sweeps"] == count}), count

    coarse = sweep(COARSE, "coarse_0.1")
    coarse_winners, coarse_minimum = minima(coarse)
    left = max(1000, round(1000 * min(coarse_winners)) - 100)
    right = min(1950, round(1000 * max(coarse_winners)) + 100)
    fine = sweep(range(left, right + 1, 5), "fine_0.005")
    fine_winners, fine_minimum = minima(fine)
    local_left = max(left, round(1000 * min(fine_winners)) - 10)
    local_right = min(right, round(1000 * max(fine_winners)) + 10)
    fine_final = sweep(range(local_left, local_right + 1), "refined_0.001")
    best, minimum = minima(coarse + fine + fine_final)
    final_omega = min(best)
    gs = calculate(1.0, TOLERANCE)
    near_two = sweep(NEAR_TWO, "near_2")
    theoretical = row(omega_th, "theory")
    sampled = coarse + fine + fine_final + near_two
    fast = sorted({r["omega"] for r in sampled
                   if r["converged"] and r["sweeps"] <= minimum + 5})

    # Reuse exactly the empirical scan points at a second stopping threshold.
    common_omegas = sorted({r["omega"] for r in coarse + fine + fine_final})
    tight = [row(omega, "same_points_tight", 1e-8) for omega in common_omegas]
    tight_best, tight_k = minima(tight)
    summary = {
        "mass": mass, "source": problem.source_xy,
        "coarse_best_omegas": coarse_winners, "coarse_min_sweeps": coarse_minimum,
        "fine_best_omegas": fine_winners, "fine_min_sweeps": fine_minimum,
        "fine_interval": [left / 1000, right / 1000], "fine_step": 0.005,
        "refined_interval": [local_left / 1000, local_right / 1000], "refined_step": 0.001,
        "best_omegas": best, "omega_experimental": final_omega, "min_sweeps": minimum,
        "winner_final_residual": calculate(final_omega, TOLERANCE).residual,
        "gs_sweeps": gs.sweeps, "gs_residual": gs.residual,
        "sweep_ratio": gs.sweeps / minimum, "sweep_reduction": gs.sweeps - minimum,
        "reduction_percent": 100 * (1 - minimum / gs.sweeps),
        "fast_threshold_sweeps": minimum + 5, "fast_omegas": fast,
        "near_two": near_two, "rho_jacobi": rho_j, "omega_theory": omega_th,
        "theory_sweeps": theoretical["sweeps"], "rho_sor_theory": theory(63, mass, omega_th),
        "tight_best_omegas": tight_best, "tight_min_sweeps": tight_k,
        "tight_gs_sweeps": calculate(1.0, 1e-8).sweeps,
    }
    all_rows = sampled + [theoretical] + tight
    assert all(r["converged"] for r in all_rows)
    assert fast[0] >= 1 and best[0] > 1 and best[-1] < 2
    return summary, all_rows


def run(output):
    output = Path(output)
    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)
    validation = validation_examples()
    print("Independent checks: dense vs. SOR, sine vs. dense, hand sweep, cap, Jacobi, spectrum.", flush=True)

    problem = PointSource(31, 0.5)
    baseline, _, checks = solve_checked(problem)
    field = as_array(problem, baseline)
    max_y, max_x = np.unravel_index(np.argmax(field), field.shape)
    x0, y0 = problem.source_xy
    assert (max_x, max_y) == (x0, y0)
    assert np.all(np.diff(field[y0, x0:]) < 0)
    assert np.all(np.diff(field[y0, :x0 + 1]) > 0)
    task1 = dict(n=31, mass=0.5, omega=1.5, sweeps=baseline.sweeps,
                 residual=baseline.residual, previous_residual=baseline.residuals[-2],
                 peak_xy=[int(max_x), int(max_y)], peak=float(field[y0, x0]),
                 horizontal_first_neighbor=float(field[y0, x0 + 1]), **checks)
    write_grid(data / "task1_field_n31.csv", field)

    profiles, propagation = [], []
    for mass in MASSES:
        problem = PointSource(63, mass)
        result, reference, checks = solve_checked(problem)
        field = as_array(problem, result)
        x0, y0 = problem.source_xy
        center = field[y0, x0]
        row = field[y0] / center
        reference_row = reference[y0] / reference[y0, x0]
        right, right_linear = first_threshold_crossing(row[x0:], exp(-1))
        left, left_linear = first_threshold_crossing(row[:x0 + 1][::-1], exp(-1))
        assert right == left
        assert abs(right_linear - left_linear) < 1e-4
        assert first_threshold_crossing(reference_row[x0:], exp(-1))[0] == right
        distant, _ = first_threshold_crossing(row[x0:], 0.01)
        profiles.extend({"mass": mass, "x_offset": x - x0, "u": field[y0, x],
                         "normalized": row[x], "reference_normalized": reference_row[x]}
                        for x in range(problem.width))
        propagation.append(dict(mass=mass, n=63, omega=1.5, sweeps=result.sweeps,
                                residual=result.residual, center=float(center),
                                first_below_exp_minus_one=right, left_first_below=left,
                                interpolated_range=right_linear, left_interpolated_range=left_linear,
                                first_below_0_01=distant, **checks))
    write_rows(data / "normalized_profiles.csv", profiles)

    scans, scan_summary = [], []
    for mass in MASSES:
        summary, rows = scan_mass(mass)
        scans.extend(rows)
        scan_summary.append(summary)
        print(f"N=63, m={mass:g}: GS {summary['gs_sweeps']}, "
              f"SOR {summary['min_sweeps']} at omega={summary['omega_experimental']:.3f}, "
              f"theory {summary['omega_theory']:.6f}", flush=True)
    write_rows(data / "omega_scan.csv", scans)

    middle = scan_summary[1]
    problem = PointSource(63, 0.5)
    convergence_rows = []
    method_rows = []
    chosen = [("Jacobi", "jacobi", 1.0), ("GS", "sor", 1.0),
              ("SOR coarse", "sor", min(middle["coarse_best_omegas"])),
              ("SOR fine", "sor", middle["omega_experimental"]),
              ("SOR theory", "sor", middle["omega_theory"]),
              ("SOR 1.90", "sor", 1.9), ("SOR 1.99", "sor", 1.99)]
    for label, method, omega in chosen:
        result, _, _ = solve_checked(problem, omega, method=method)
        method_rows.append(dict(label=label, method=method, omega=omega,
                                sweeps=result.sweeps, residual=result.residual))
        convergence_rows.extend(dict(label=label, method=method, omega=omega,
                                     sweep=k, residual=residual)
                                for k, residual in enumerate(result.residuals))
    write_rows(data / "residual_history.csv", convergence_rows)

    sizes, boundary_profiles = [], []
    for n in (31, 63, 127):
        problem = PointSource(n, 0.2)
        result, reference, checks = solve_checked(problem, 1.75)
        x0, y0 = problem.source_xy
        profile = reference[y0, x0:] / reference[y0, x0]
        sizes.append(dict(n=n, mass=0.2, omega=1.75, boundary_distance=n + 1 - x0,
                          unknowns=n * n, sweeps=result.sweeps, residual=result.residual,
                          reference_center=float(reference[y0, x0]),
                          normalized_at_8=float(profile[8]), normalized_at_15=float(profile[15]),
                          normalized_at_31=None if n < 63 else float(profile[31]),
                          **checks))
        boundary_profiles.extend({"n": n, "distance": k, "normalized": value}
                                 for k, value in enumerate(profile))
    write_rows(data / "boundary_profiles.csv", boundary_profiles)

    offset = PointSource(63, 0.5, (5, 32))
    shifted, shifted_ref, shifted_checks = solve_checked(offset)
    shifted_field = as_array(offset, shifted)
    write_grid(data / "offset_source_field.csv", shifted_field)
    x0, y0 = offset.source_xy
    off_center = dict(n=63, mass=0.5, omega=1.5, source=offset.source_xy,
                      sweeps=shifted.sweeps, residual=shifted.residual,
                      left_at_4=float(shifted_ref[y0, x0 - 4] / shifted_ref[y0, x0]),
                      right_at_4=float(shifted_ref[y0, x0 + 4] / shifted_ref[y0, x0]),
                      peak=float(shifted_field[y0, x0]), **shifted_checks)
    summary = dict(tolerance=TOLERANCE, max_sweeps=MAX_SWEEPS, spacing=1,
                   ordering="lexicographic y then x; x grows within each row",
                   boundary="homogeneous Dirichlet", initial="all zero",
                   tasks=dict(task1=task1, propagation=propagation, omega_scan=scan_summary),
                   validation=validation, method_comparison=method_rows,
                   grid_size_experiment=sizes, off_center_source=off_center,
                   scan_row_count=len(scans))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                    allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(scans)} scan records and independent validation to {output}.", flush=True)
