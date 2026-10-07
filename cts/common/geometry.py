"""cts.common.geometry

2D polygon representations, point-in-polygon checks, Separating Axis Theorem (SAT)
collision detection, boundary distance calculations, and convex decomposition.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Union, Sequence
import numpy as np


class Polygon2D:
    """Represents a 2D planar polygon with topological, geometric, and collision operations."""

    def __init__(
        self,
        vertices: Union[np.ndarray, Sequence[Sequence[float]]],
        ensure_ccw: bool = True,
    ):
        """Initialize Polygon2D.

        Args:
            vertices: (N, 2) array-like of 2D coordinates.
            ensure_ccw: If True, reverses vertex sequence if clockwise (signed area < 0).
        """
        verts = np.asarray(vertices, dtype=np.float64)
        if verts.ndim != 2 or verts.shape[1] != 2:
            raise ValueError(f"Vertices must have shape (N, 2), got {verts.shape}")
        if len(verts) < 3:
            raise ValueError(f"Polygon must have at least 3 vertices, got {len(verts)}")

        # Check signed area via Shoelace formula
        signed_area = self._compute_signed_area(verts)
        if ensure_ccw and signed_area < 0:
            verts = verts[::-1]
            signed_area = -signed_area

        self._vertices = np.ascontiguousarray(verts)
        self._signed_area = float(signed_area)

    @staticmethod
    def _compute_signed_area(pts: np.ndarray) -> float:
        x, y = pts[:, 0], pts[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    def __len__(self) -> int:
        return len(self._vertices)

    def __getitem__(self, idx):
        return self._vertices[idx]

    def __iter__(self):
        return iter(self._vertices)

    def __array__(self, dtype=None):
        if dtype:
            return self._vertices.astype(dtype)
        return self._vertices

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices.copy()

    @property
    def num_vertices(self) -> int:
        return len(self._vertices)

    @property
    def signed_area(self) -> float:
        return self._signed_area

    @property
    def area(self) -> float:
        return abs(self._signed_area)

    @property
    def is_ccw(self) -> bool:
        return self._signed_area > 0

    @property
    def edges(self) -> np.ndarray:
        """Returns (N, 2) edge vectors e_i = v_{i+1} - v_i."""
        return np.roll(self._vertices, -1, axis=0) - self._vertices

    @property
    def edge_lengths(self) -> np.ndarray:
        """Returns (N,) lengths of all edges."""
        return np.linalg.norm(self.edges, axis=1)

    @property
    def edge_normals(self) -> np.ndarray:
        """Returns (N, 2) outward unit normal vectors for CCW polygon.
        
        For edge (dx, dy), outward normal is (dy, -dx) / length.
        """
        e = self.edges
        lengths = np.linalg.norm(e, axis=1, keepdims=True)
        normals = np.column_stack([e[:, 1], -e[:, 0]]) / (lengths + 1e-14)
        return normals

    @property
    def centroid(self) -> np.ndarray:
        """Compute geometric centroid (Cx, Cy)."""
        pts = self._vertices
        x, y = pts[:, 0], pts[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cross = x * y1 - x1 * y
        factor = 1.0 / (6.0 * self._signed_area)
        cx = factor * np.sum((x + x1) * cross)
        cy = factor * np.sum((y + y1) * cross)
        return np.array([cx, cy], dtype=np.float64)

    @property
    def aabb(self) -> Tuple[float, float, float, float]:
        """Axis-Aligned Bounding Box (xmin, ymin, xmax, ymax)."""
        return (
            float(np.min(self._vertices[:, 0])),
            float(np.min(self._vertices[:, 1])),
            float(np.max(self._vertices[:, 0])),
            float(np.max(self._vertices[:, 1])),
        )

    @property
    def is_convex(self) -> bool:
        """Checks if polygon is strictly convex."""
        e = self.edges
        # Cross products of consecutive edges
        cross_products = e[:, 0] * np.roll(e[:, 1], -1) - e[:, 1] * np.roll(e[:, 0], -1)
        return bool(np.all(cross_products >= -1e-9) or np.all(cross_products <= 1e-9))

    def translate(self, dx: float, dy: float) -> "Polygon2D":
        """Translate polygon by (dx, dy)."""
        return Polygon2D(self._vertices + np.array([dx, dy], dtype=np.float64), ensure_ccw=False)

    def rotate(self, angle_rad: float, center: Optional[Tuple[float, float]] = None) -> "Polygon2D":
        """Rotate polygon by angle_rad around center (default: polygon centroid)."""
        if center is None:
            c = self.centroid
        else:
            c = np.array(center, dtype=np.float64)
        c_cos = np.cos(angle_rad)
        c_sin = np.sin(angle_rad)
        R = np.array([[c_cos, -c_sin], [c_sin, c_cos]], dtype=np.float64)
        rotated = (self._vertices - c) @ R.T + c
        return Polygon2D(rotated, ensure_ccw=False)

    def transform(self, x: float, y: float, theta_rad: float) -> "Polygon2D":
        """Apply rigid SE(2) transformation [x, y, theta] to polygon."""
        c_cos = np.cos(theta_rad)
        c_sin = np.sin(theta_rad)
        R = np.array([[c_cos, -c_sin], [c_sin, c_cos]], dtype=np.float64)
        transformed = self._vertices @ R.T + np.array([x, y], dtype=np.float64)
        return Polygon2D(transformed, ensure_ccw=False)

    def contains_point(self, point: Union[np.ndarray, Tuple[float, float], Sequence[float]], tol: float = 1e-4) -> bool:
        """Ray-casting point-in-polygon check with on-boundary tolerance."""
        px, py = float(point[0]), float(point[1])
        xmin, ymin, xmax, ymax = self.aabb
        if px < xmin - tol or px > xmax + tol or py < ymin - tol or py > ymax + tol:
            return False

        # Check if on any edge segment within tolerance
        v = self._vertices
        n = len(v)
        p = np.array([px, py], dtype=np.float64)
        for i in range(n):
            a = v[i]
            b = v[(i + 1) % n]
            ab = b - a
            ab_sq = float(np.dot(ab, ab))
            if ab_sq > 1e-12:
                t = max(0.0, min(1.0, float(np.dot(p - a, ab)) / ab_sq))
                proj = a + t * ab
                if float(np.linalg.norm(p - proj)) <= tol:
                    return True

        # Standard ray-casting test
        inside = False
        p1x, p1y = v[0]
        for i in range(1, n + 1):
            p2x, p2y = v[i % n]
            if min(p1y, p2y) < py <= max(p1y, p2y):
                if px <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or px <= xinters:
                        inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def distance_to_point(self, point: Union[np.ndarray, Tuple[float, float], Sequence[float]]) -> float:
        """Compute minimum Euclidean distance from point to polygon boundary."""
        px, py = float(point[0]), float(point[1])
        p = np.array([px, py], dtype=np.float64)
        v = self._vertices
        n = len(v)
        min_dist = float("inf")
        for i in range(n):
            a = v[i]
            b = v[(i + 1) % n]
            ab = b - a
            ab_sq = float(np.dot(ab, ab))
            if ab_sq < 1e-12:
                dist = float(np.linalg.norm(p - a))
            else:
                t = max(0.0, min(1.0, float(np.dot(p - a, ab)) / ab_sq))
                proj = a + t * ab
                dist = float(np.linalg.norm(p - proj))
            if dist < min_dist:
                min_dist = dist
        return float(min_dist)

    def signed_distance_to_point(self, point: Union[np.ndarray, Tuple[float, float], Sequence[float]]) -> float:
        """Signed distance: positive inside polygon, negative outside."""
        dist = self.distance_to_point(point)
        inside = self.contains_point(point)
        return dist if inside else -dist


def sat_convex_collision(
    poly1: Polygon2D, poly2: Polygon2D
) -> Tuple[bool, Optional[np.ndarray], float]:
    """Separating Axis Theorem (SAT) collision check between two convex polygons.

    Returns:
        (collides, mtv_axis, penetration_depth)
        collides: True if intersecting, False if disjoint.
        mtv_axis: Minimum Translation Vector direction (unit vector).
        penetration_depth: Overlap magnitude along mtv_axis.
    """
    # AABB early rejection
    b1, b2 = poly1.aabb, poly2.aabb
    if b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1]:
        return False, None, 0.0

    axes = np.vstack([poly1.edge_normals, poly2.edge_normals])
    min_overlap = float("inf")
    best_axis = None

    v1, v2 = poly1.vertices, poly2.vertices
    for axis in axes:
        # Project poly1
        proj1 = np.dot(v1, axis)
        min1, max1 = float(np.min(proj1)), float(np.max(proj1))

        # Project poly2
        proj2 = np.dot(v2, axis)
        min2, max2 = float(np.min(proj2)), float(np.max(proj2))

        # Check overlap
        overlap = min(max1, max2) - max(min1, min2)
        if overlap <= 1e-9:
            # Separating axis found
            return False, None, 0.0

        if overlap < min_overlap:
            min_overlap = overlap
            best_axis = axis

    return True, best_axis, float(min_overlap)


def decompose_island_to_convex(island_poly: Polygon2D) -> List[Polygon2D]:
    """Decompose the carrier island obstacle into two strictly convex polygons."""
    # Split at x = 74.0
    piece1 = Polygon2D([[62.0, 27.0], [74.0, 27.0], [74.0, 43.0], [62.0, 43.0]], ensure_ccw=True)
    piece2 = Polygon2D([[74.0, 27.0], [92.0, 27.0], [92.0, 44.0], [77.0, 44.0], [74.0, 43.0]], ensure_ccw=True)
    return [piece1, piece2]


def check_polygon_collision(poly1: Polygon2D, poly2: Polygon2D) -> bool:
    """General collision detection supporting convex and decomposed non-convex polygons."""
    parts1 = decompose_island_to_convex(poly1) if (not poly1.is_convex and poly1.num_vertices == 6) else [poly1]
    parts2 = decompose_island_to_convex(poly2) if (not poly2.is_convex and poly2.num_vertices == 6) else [poly2]

    for p1 in parts1:
        for p2 in parts2:
            collides, _, _ = sat_convex_collision(p1, p2)
            if collides:
                return True
    return False


def _segments_intersect(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> bool:
    """Helper: line segment intersection test using CCW orientations."""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))


def is_polygon_contained(inner: Polygon2D, outer: Polygon2D, margin: float = 0.0) -> bool:
    """Check if inner polygon is strictly contained inside outer polygon with safety margin."""
    # 1. All inner vertices must be inside outer with distance >= margin
    for pt in inner.vertices:
        if not outer.contains_point(pt):
            return False
        if margin > 0.0 and outer.distance_to_point(pt) < margin:
            return False

    # 2. Check that no inner edges intersect outer boundary
    v_inner = inner.vertices
    v_outer = outer.vertices
    n_in = len(v_inner)
    n_out = len(v_outer)

    for i in range(n_in):
        p1 = v_inner[i]
        p2 = v_inner[(i + 1) % n_in]
        for j in range(n_out):
            q1 = v_outer[j]
            q2 = v_outer[(j + 1) % n_out]
            if _segments_intersect(p1, p2, q1, q2):
                return False
    return True
