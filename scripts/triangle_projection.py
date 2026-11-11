"""Finite projection for degenerate triangles without discarding mesh faces."""
import numpy as np


def closest_edge(point, triangle):
    candidates = []
    for left, right in ((0, 1), (1, 2), (2, 0)):
        delta = triangle[right] - triangle[left]
        length = float(delta @ delta)
        fraction = float(np.clip((point - triangle[left]) @ delta / length, 0, 1)) if length > 0 else 0.
        weights = np.zeros(3)
        weights[left], weights[right] = 1 - fraction, fraction
        projected = triangle[left] + fraction * delta
        candidates.append((float(np.sum((projected - point)**2)), projected, weights))
    _, projected, weights = min(candidates, key=lambda item: item[0])
    return projected, weights


def project(point, triangle):
    point, triangle = np.asarray(point, dtype=np.float64), np.asarray(triangle, dtype=np.float64)
    if not np.isfinite(point).all() or not np.isfinite(triangle).all():
        raise ValueError('Projection requires finite geometry')
    a, b, c = triangle
    normal = np.cross(b-a, c-a)
    area = float(normal @ normal)
    if area > np.finfo(np.float64).tiny:
        projected = point - float((point-a) @ normal / area) * normal
        v = float(np.cross(projected-a, c-a) @ normal / area)
        w = float(np.cross(b-a, projected-a) @ normal / area)
        weights = np.array([1-v-w, v, w])
        if weights.min() >= 0:
            return projected, weights
    return closest_edge(point, triangle)


def finite_closest_points(triangles, points, implementation):
    triangles, points = np.asarray(triangles, dtype=np.float64), np.asarray(points, dtype=np.float64)
    if not np.isfinite(triangles).all() or not np.isfinite(points).all():
        raise ValueError('Projection requires finite geometry')
    with np.errstate(divide='ignore', invalid='ignore'):
        result = implementation(triangles, points)
    for index in np.flatnonzero(~np.isfinite(result).all(axis=1)):
        result[index], _ = project(points[index], triangles[index])
    if not np.isfinite(result).all():
        raise ValueError('Triangle projection produced non-finite values')
    return result
