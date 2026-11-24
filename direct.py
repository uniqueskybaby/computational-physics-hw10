"""Dense small-grid reference for the five-point Dirichlet operator."""

from math import fsum, sqrt


def assemble_system(problem):
    """Build the interior linear system with the zero boundary eliminated."""
    n = problem.n
    size = n * n
    matrix = [[0.0] * size for _ in range(size)]
    rhs = [0.0] * size
    source_x, source_y = problem.source_xy
    rhs[(source_y - 1) * n + source_x - 1] = problem.strength
    for y in range(n):
        for x in range(n):
            row = y * n + x
            matrix[row][row] = 4.0 + problem.mass ** 2
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor_x, neighbor_y = x + dx, y + dy
                if 0 <= neighbor_x < n and 0 <= neighbor_y < n:
                    matrix[row][neighbor_y * n + neighbor_x] = -1.0
    return matrix, rhs


def cholesky_solve(matrix, rhs):
    """Solve a small symmetric positive definite system using A = L Lᵀ."""
    size = len(rhs)
    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            cross = fsum(lower[row][j] * lower[column][j] for j in range(column))
            remaining = matrix[row][column] - cross
            lower[row][column] = sqrt(remaining) if row == column else remaining / lower[column][column]

    intermediate = [0.0] * size
    for row in range(size):
        known = fsum(lower[row][j] * intermediate[j] for j in range(row))
        intermediate[row] = (rhs[row] - known) / lower[row][row]

    field = [0.0] * size
    for row in reversed(range(size)):
        known = fsum(lower[j][row] * field[j] for j in range(row + 1, size))
        field[row] = (intermediate[row] - known) / lower[row][row]
    return field
