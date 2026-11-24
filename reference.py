"""Independent direct reference solvers and checks."""

import numpy as np

from direct import assemble_system
from sor import PointSource, solve, theory


def dense_system(problem):
    """Assemble A and b on interior nodes, with 0 Dirichlet values eliminated."""
    matrix, rhs = assemble_system(problem)
    return np.asarray(matrix), np.asarray(rhs)


def sine_solution(problem):
    """Diagonalize the same finite 2-D Dirichlet operator by sine modes."""
    n = problem.n
    theta = np.pi * np.arange(1, n + 1) / (n + 1)
    basis = np.sin(np.arange(1, n + 1)[:, None] * theta[None, :])
    basis *= np.sqrt(2.0 / (n + 1))
    eigenvalues_1d = 2.0 - 2.0 * np.cos(theta)
    spectrum = problem.mass ** 2 + eigenvalues_1d[:, None] + eigenvalues_1d[None, :]
    sx, sy = problem.source_xy
    coefficients = problem.strength * np.outer(basis[sy - 1], basis[sx - 1]) / spectrum
    y_transformed = np.dot(basis, coefficients)
    return np.pad(np.dot(y_transformed, basis.T), 1)


def as_array(problem, result):
    return np.asarray(result.field).reshape(problem.width, problem.width)


def independent_residual(problem, field):
    """Stencil residual via a separate array-slice implementation."""
    b = np.zeros_like(field[1:-1, 1:-1])
    x, y = problem.source_xy
    b[y - 1, x - 1] = problem.strength
    residual = b - (4.0 + problem.mass ** 2) * field[1:-1, 1:-1]
    residual += field[1:-1, :-2] + field[1:-1, 2:]
    residual += field[:-2, 1:-1] + field[2:, 1:-1]
    return float(np.max(np.abs(residual)))


def symmetry_errors(field):
    return {
        "x_reflection": float(np.max(np.abs(field - field[:, ::-1]))),
        "y_reflection": float(np.max(np.abs(field - field[::-1, :]))),
        "transpose": float(np.max(np.abs(field - field.T))),
        "rotation_90": float(np.max(np.abs(field - np.rot90(field)))),
    }


def verify_solution(problem, result, tolerance, reference=None):
    """Audit boundaries, the stopping event, residual and a reference error bound."""
    field = as_array(problem, result)
    assert np.isfinite(field).all()
    assert np.max(np.abs(field[[0, -1], :])) == 0
    assert np.max(np.abs(field[:, [0, -1]])) == 0
    residual = independent_residual(problem, field)
    assert abs(residual - result.residual) <= 1e-14
    if result.converged:
        assert result.residual < tolerance
        assert all(r >= tolerance for r in result.residuals[1:-1])
    else:
        assert min(result.residuals[1:]) >= tolerance
    metrics = {"independent_residual": residual, "symmetry_errors": symmetry_errors(field)}
    if reference is not None:
        reference_residual = independent_residual(problem, reference)
        max_error = float(np.max(np.abs(field - reference)))
        bound = (residual + reference_residual) / problem.mass ** 2
        assert reference_residual < 1e-12
        assert max_error <= bound + 5e-14
        metrics.update(max_error=max_error, residual_error_bound=bound,
                       reference_residual=reference_residual)
    return metrics


def validation_examples():
    problem = PointSource(7, 0.5)
    matrix, rhs = dense_system(problem)
    direct = np.linalg.solve(matrix, rhs)
    padded = np.pad(direct.reshape(problem.n, problem.n), 1)
    dense_residual = float(np.max(np.abs(np.dot(matrix, direct) - rhs)))
    assert dense_residual < 1e-12
    assert np.max(np.abs(sine_solution(problem) - padded)) < 1e-13
    rows = []
    for tolerance in (1e-6, 1e-8):
        result = solve(problem, 1.5, tolerance)
        row = verify_solution(problem, result, tolerance, padded)
        row.update(n=7, mass=0.5, omega=1.5, tolerance=tolerance,
                   sweeps=result.sweeps, residual=result.residual,
                   direct_algebraic_residual=dense_residual,
                   direct_center=float(padded[4, 4]), sor_center=float(as_array(problem, result)[4, 4]),
                   min_eigenvalue=float(np.linalg.eigvalsh(matrix)[0]))
        rows.append(row)

    # N=3 first sweep can be worked out from north/west updates by hand.
    hand = PointSource(3, 0.5)
    first = solve(hand, 1.5, max_sweeps=1)
    actual = as_array(hand, first)
    a = 1.5 / hand.diagonal
    expected = np.zeros((5, 5))
    expected[2, 2] = a
    expected[2, 3] = expected[3, 2] = a ** 2
    expected[3, 3] = 2 * a ** 3
    assert np.max(np.abs(actual - expected)) < 2e-16
    assert first.sweeps == 1 and not first.converged
    assert first.status == "在最大迭代次数内未收敛"

    # Jacobi propagates only to the source in the first sweep.
    jacobi = as_array(hand, solve(hand, max_sweeps=1, method="jacobi"))
    assert np.count_nonzero(jacobi) == 1
    assert abs(jacobi[2, 2] - 1 / hand.diagonal) < 2e-16

    # Verify Young's spectral factor on an explicit small iteration matrix.
    omega_theory = theory(7, 0.5)[1]
    diagonal = np.diag(np.diag(matrix))
    lower = -np.tril(matrix, -1)
    upper = -np.triu(matrix, 1)
    spectral = []
    for omega in (1.0, 1.3, omega_theory - 0.002, omega_theory + 0.002, 1.9):
        iteration = np.linalg.solve(diagonal - omega * lower,
                                    (1.0 - omega) * diagonal + omega * upper)
        numerical = float(np.max(np.abs(np.linalg.eigvals(iteration))))
        predicted = theory(7, 0.5, omega)
        assert abs(numerical - predicted) < 2e-11
        spectral.append({"omega": omega, "rho_matrix": numerical, "rho_formula": predicted})

    return {
        "direct_comparison": rows,
        "hand_first_sweep": {
            "n": 3, "mass": 0.5, "omega": 1.5, "status": first.status,
            "residual": first.residual, "max_difference": float(np.max(np.abs(actual - expected))),
        },
        "jacobi_one_sweep": "one nonzero node, at the source",
        "sor_spectral_check": spectral,
    }
