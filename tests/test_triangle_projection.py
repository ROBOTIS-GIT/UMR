import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import triangle_projection as projection

class TriangleTests(unittest.TestCase):
    def test_interior(self):
        triangle = np.array([[0,0,0],[1,0,0],[0,1,0.]])
        point, weights = projection.project([.2,.3,1], triangle)
        np.testing.assert_allclose(point, [.2,.3,0])
        np.testing.assert_allclose(weights @ triangle, point)
    def test_collinear(self):
        triangle = np.array([[0,0,0],[1,0,0],[2,0,0.]])
        point, weights = projection.project([1.5,1,0], triangle)
        np.testing.assert_allclose(point, [1.5,0,0])
        np.testing.assert_allclose(weights @ triangle, point)
    def test_point(self):
        point, weights = projection.project([1,2,3], np.zeros((3,3)))
        np.testing.assert_allclose(point, [0,0,0]); self.assertEqual(weights.sum(), 1)
    def test_only_nonfinite_rows_replaced(self):
        triangles = np.zeros((2,3,3)); points = np.ones((2,3))
        def implementation(t,p): return np.array([[5.,6,7],[np.nan]*3])
        result = projection.finite_closest_points(triangles, points, implementation)
        np.testing.assert_array_equal(result[0], [5,6,7])
        np.testing.assert_array_equal(result[1], [0,0,0])

class TrimeshCompatibilityTests(unittest.TestCase):
    def test_regular_triangles_match_trimesh(self):
        try:
            import trimesh
        except ImportError:
            self.skipTest('trimesh not installed')
        rng = np.random.default_rng(42)
        triangles = rng.normal(size=(100,3,3))
        points = rng.normal(size=(100,3))
        expected = trimesh.triangles.closest_point(triangles, points)
        actual = projection.finite_closest_points(triangles, points, trimesh.triangles.closest_point)
        np.testing.assert_array_equal(actual, expected)
    def test_degenerate_trimesh_outputs_finite(self):
        try:
            import trimesh
        except ImportError:
            self.skipTest('trimesh not installed')
        triangles = np.array([[[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[1,0,0],[2,0,0.]]])
        points = np.array([[1,2,3],[1.5,1,0.]])
        actual = projection.finite_closest_points(triangles, points, trimesh.triangles.closest_point)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, [[0,0,0],[1.5,0,0]])
