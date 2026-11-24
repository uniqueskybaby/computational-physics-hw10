"""Check the required center-source and direct-solve cases without dependencies."""

from math import fsum
import unittest

from direct import assemble_system, cholesky_solve
from sor import PointSource, solve


def dense_residual(matrix, field, rhs):
    return max(abs(value - fsum(a * u for a, u in zip(row, field)))
               for row, value in zip(matrix, rhs))


class TestSOR(unittest.TestCase):
    def test_center_source_and_first_residual_crossing(self):
        problem = PointSource(31, 0.5)
        result = solve(problem, omega=1.5)
        x, y = problem.source_xy
        width = problem.width
        self.assertTrue(result.converged)
        self.assertEqual(result.sweeps, 33)
        self.assertLess(result.residual, 1e-6)
        self.assertGreaterEqual(min(result.residuals[:-1]), 1e-6)
        self.assertEqual(result.field.index(max(result.field)), y * width + x)
        self.assertEqual(max(map(abs, result.field[:width] + result.field[-width:])), 0.0)
        self.assertEqual(max(map(abs, result.field[::width] + result.field[width - 1::width])), 0.0)

    def test_small_grid_matches_dense_cholesky(self):
        problem = PointSource(7, 0.5)
        matrix, rhs = assemble_system(problem)
        reference = cholesky_solve(matrix, rhs)
        reference_residual = dense_residual(matrix, reference, rhs)
        self.assertLess(reference_residual, 1e-14)

        for tolerance in (1e-6, 1e-8):
            with self.subTest(tolerance=tolerance):
                result = solve(problem, omega=1.5, tolerance=tolerance)
                field = [result.field[y * problem.width + x]
                         for y in range(1, problem.n + 1)
                         for x in range(1, problem.n + 1)]
                self.assertTrue(result.converged)
                residual = dense_residual(matrix, field, rhs)
                self.assertLess(residual, tolerance)
                self.assertAlmostEqual(residual, result.residual, delta=1e-14)
                bound = (residual + reference_residual) / problem.mass ** 2
                error = max(abs(actual - expected) for actual, expected in zip(field, reference))
                self.assertLessEqual(error, bound + 1e-14)
