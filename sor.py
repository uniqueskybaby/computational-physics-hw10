"""Pointwise relaxation of the square Dirichlet lattice, with h = 1.

Grid coordinates are 1-based (x, y); a padded array holds the zero boundary.
The fixed scan visits y = 1..N, and x = 1..N within each row.
"""

from dataclasses import dataclass
from math import cos, isfinite, pi, sqrt
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class PointSource:
    n: int
    mass: float
    source: Optional[Tuple[int, int]] = None
    strength: float = 1.0

    def __post_init__(self):
        if type(self.n) is not int or self.n < 1:
            raise ValueError("N must be a positive integer")
        if not (self.mass > 0 and isfinite(self.diagonal)):
            raise ValueError("m must be positive and give a finite diagonal")
        if not (isfinite(self.strength) and self.strength > 0):
            raise ValueError("source strength must be finite and positive")
        x, y = self.source_xy
        if any(type(c) is not int or not 1 <= c <= self.n for c in (x, y)):
            raise ValueError("source coordinates must be interior integers")

    @property
    def width(self):
        return self.n + 2

    @property
    def diagonal(self):
        return 4.0 + self.mass * self.mass

    @property
    def source_xy(self):
        center = (self.n + 1) // 2
        return (center, center) if self.source is None else self.source

    def rhs(self):
        values = [0.0] * (self.width * self.width)
        x, y = self.source_xy
        values[y * self.width + x] = self.strength
        return values


@dataclass
class Solution:
    field: List[float]
    residuals: List[float]
    converged: bool

    @property
    def sweeps(self):
        return len(self.residuals) - 1

    @property
    def residual(self):
        return self.residuals[-1]

    @property
    def status(self):
        if self.converged:
            return "已收敛"
        return "在最大迭代次数内未收敛"


def interior_indices(problem):
    width = problem.width
    return tuple(y * width + x for y in range(1, problem.n + 1)
                 for x in range(1, problem.n + 1))


def residual_norm(field, rhs, width, diagonal, indices):
    """Evaluate ||s-Au||∞ on the completed iterate."""
    largest = 0.0
    for k in indices:
        neighbors = field[k - 1] + field[k + 1] + field[k - width] + field[k + width]
        absolute = abs(rhs[k] - (diagonal * field[k] - neighbors))
        if absolute > largest:
            largest = absolute
    return largest


def solve(problem, omega=1.5, tolerance=1e-6, max_sweeps=5000, method="sor"):
    """Start from zero and stop at the first full sweep with residual < tolerance.

    For Gauss–Seidel use method='sor', omega=1.
    Jacobi takes all four neighbors from the preceding sweep.
    """
    if method not in {"sor", "jacobi"}:
        raise ValueError("method must be 'sor' or 'jacobi'")
    if not 0 < omega < 2:
        raise ValueError("omega must lie strictly between zero and two")
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if type(max_sweeps) is not int or max_sweeps < 1:
        raise ValueError("max_sweeps must be a positive integer")

    width = problem.width
    diagonal = problem.diagonal
    inverse_diagonal = 1.0 / diagonal
    old_weight = 1.0 - omega
    rhs = problem.rhs()
    indices = interior_indices(problem)
    field = [0.0] * len(rhs)
    previous = field.copy() if method == "jacobi" else None
    history = [problem.strength]
    for _ in range(max_sweeps):
        if method == "jacobi":
            field, previous = previous, field
            for k in indices:
                neighbors = previous[k - 1] + previous[k + 1]
                neighbors += previous[k - width] + previous[k + width]
                field[k] = (rhs[k] + neighbors) * inverse_diagonal
        else:
            for k in indices:
                neighbors = field[k - 1] + field[k + 1]
                neighbors += field[k - width] + field[k + width]
                gs_candidate = (rhs[k] + neighbors) * inverse_diagonal
                field[k] = old_weight * field[k] + omega * gs_candidate
        residual = residual_norm(field, rhs, width, diagonal, indices)
        history.append(residual)
        if residual < tolerance:
            return Solution(field, history, True)
    return Solution(field, history, False)


def theory(n, mass, omega=None):
    """Young's spectral result for this lexicographically ordered stencil."""
    rho_j = 4.0 * cos(pi / (n + 1)) / (4.0 + mass * mass)
    omega_theory = 2.0 / (1.0 + sqrt(1.0 - rho_j * rho_j))
    if omega is None:
        return rho_j, omega_theory
    if omega >= omega_theory:
        return omega - 1.0
    coupling = omega * rho_j
    return ((coupling + sqrt(max(0.0, coupling * coupling - 4.0 * (omega - 1.0)))) / 2.0) ** 2
