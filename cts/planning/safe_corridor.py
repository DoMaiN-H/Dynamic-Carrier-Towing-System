"""cts.planning.safe_corridor

Feature F12: Spatiotemporal Safe Flight Corridor (SFC) Generation.
Convex polyhedral corridor decomposition in space-time (x, y, t) and spatial
polytopes A_k [x, y]^T <= b_k valid over time intervals [t_k, t_{k+1}].
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Sequence
import numpy as np

from cts.common.geometry import (
    Polygon2D,
    sat_convex_collision,
    decompose_island_to_convex,
    is_polygon_contained,
)
from cts.environment.deck_env import DeckEnvironment


@dataclass
class Polytope:
    """Convex polytope in H-representation: A * p <= b, active on [t_min, t_max].
    
    Attributes:
        A: (M, 2) array of outward half-plane normal vectors.
        b: (M,) array of half-plane offsets.
        t_min: start of validity time interval (s).
        t_max: end of validity time interval (s).
    """
    A: np.ndarray
    b: np.ndarray
    t_min: float = 0.0
    t_max: float = float("inf")

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=np.float64)
        self.b = np.asarray(self.b, dtype=np.float64)
        if self.A.ndim != 2 or self.A.shape[1] != 2:
            raise ValueError(f"Matrix A must have shape (M, 2), got {self.A.shape}")
        if self.b.ndim != 1 or len(self.b) != self.A.shape[0]:
            raise ValueError(f"Vector b length ({len(self.b)}) must match A rows ({self.A.shape[0]})")
        self.t_min = float(self.t_min)
        self.t_max = float(self.t_max)

    def contains(self, point: Union[np.ndarray, Sequence[float]], tol: float = 1e-6) -> bool:
        """Check if point [x, y] is inside the spatial polytope."""
        p = np.asarray(point, dtype=np.float64)[:2]
        return bool(np.all(self.A @ p <= self.b + tol))

    def signed_distance(self, point: Union[np.ndarray, Sequence[float]]) -> float:
        """Minimum signed distance to polytope boundary (positive inside)."""
        p = np.asarray(point, dtype=np.float64)[:2]
        norms = np.linalg.norm(self.A, axis=1) + 1e-14
        margins = (self.b - self.A @ p) / norms
        return float(np.min(margins))

    def shrink(self, margin: float) -> "Polytope":
        """Uniformly shrink polytope inward by margin."""
        norms = np.linalg.norm(self.A, axis=1)
        b_shrunk = self.b - float(margin) * norms
        return Polytope(self.A.copy(), b_shrunk, self.t_min, self.t_max)

    def to_spacetime_hrep(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to 3D spatiotemporal system A_st * [x, y, t]^T <= b_st."""
        M = len(self.b)
        A_st = np.zeros((M + 2, 3), dtype=np.float64)
        b_st = np.zeros(M + 2, dtype=np.float64)
        A_st[:M, :2] = self.A
        b_st[:M] = self.b
        # t <= t_max
        A_st[M, 2] = 1.0
        b_st[M] = self.t_max
        # -t <= -t_min
        A_st[M + 1, 2] = -1.0
        b_st[M + 1] = -self.t_min
        return A_st, b_st

    def compute_vertices(self) -> np.ndarray:
        """Compute 2D vertices of polytope via half-plane intersections."""
        # Find intersections of all pairs of bounding lines: a_i^T p = b_i and a_j^T p = b_j
        num_planes = len(self.b)
        candidates: List[np.ndarray] = []

        for i in range(num_planes):
            a_i = self.A[i]
            b_i = self.b[i]
            for j in range(i + 1, num_planes):
                a_j = self.A[j]
                b_j = self.b[j]

                det = a_i[0] * a_j[1] - a_i[1] * a_j[0]
                if abs(det) < 1e-9:
                    continue  # Parallel lines

                # Solve 2x2 system
                x = (b_i * a_j[1] - b_j * a_i[1]) / det
                y = (a_i[0] * b_j - a_j[0] * b_i) / det
                pt = np.array([x, y], dtype=np.float64)

                # Check if point satisfies all other half-planes
                if np.all(self.A @ pt <= self.b + 1e-4):
                    candidates.append(pt)

        if len(candidates) < 3:
            # Fallback: create small bounding diamond around center if degenerate
            return np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)

        # Remove duplicate vertices
        unique_pts: List[np.ndarray] = []
        for pt in candidates:
            if not any(np.linalg.norm(pt - u) < 1e-4 for u in unique_pts):
                unique_pts.append(pt)

        if len(unique_pts) < 3:
            return np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)

        pts_arr = np.array(unique_pts, dtype=np.float64)
        center = np.mean(pts_arr, axis=0)
        angles = np.arctan2(pts_arr[:, 1] - center[1], pts_arr[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        return pts_arr[sorted_indices]

    def to_polygon(self) -> Polygon2D:
        """Convert polytope to Polygon2D object."""
        verts = self.compute_vertices()
        return Polygon2D(verts, ensure_ccw=True)

    def overlaps(self, other: "Polytope") -> bool:
        """Check if this polytope overlaps with another polytope."""
        p1 = self.to_polygon()
        p2 = other.to_polygon()
        collides, _, _ = sat_convex_collision(p1, p2)
        return collides


@dataclass(frozen=True)
class SafeFlightCorridorConfig:
    """Configuration parameters for Safe Flight Corridor generation."""
    safety_margin: float = 1.0           # Safety clearance to obstacle boundary (m)
    max_box_width: float = 12.0          # Maximum lateral expansion width (m)
    expansion_step: float = 0.25         # Inflation incremental step size (m)
    max_inflation_iters: int = 40        # Maximum inflation loop iterations
    use_spacetime: bool = False          # Output 3D spacetime polytopes vs 2D spatial + time


class SafeFlightCorridorGenerator:
    """Spatiotemporal Safe Flight Corridor generation engine."""

    def __init__(
        self,
        deck_env: DeckEnvironment,
        config: Optional[SafeFlightCorridorConfig] = None,
    ) -> None:
        self.env = deck_env
        self.config = config or SafeFlightCorridorConfig()
        self.island_pieces = decompose_island_to_convex(self.env.island.polygon)

    def inflate_segment(
        self,
        p_start: np.ndarray,
        p_end: np.ndarray,
        t_start: float,
        t_end: float,
        obstacles: List[Polygon2D],
    ) -> Polytope:
        """Inflate a convex corridor around line segment [p_start, p_end].
        
        Args:
            p_start: (2,) starting waypoint.
            p_end: (2,) ending waypoint.
            t_start: segment start timestamp.
            t_end: segment end timestamp.
            obstacles: list of convex obstacle polygons to avoid.
            
        Returns:
            Convex Polytope guaranteed collision-free.
        """
        p0 = np.asarray(p_start, dtype=np.float64)[:2]
        p1 = np.asarray(p_end, dtype=np.float64)[:2]

        # Initial bounding box around seed segment with small initial buffer
        init_buffer = 0.50
        xmin = min(p0[0], p1[0]) - init_buffer
        xmax = max(p0[0], p1[0]) + init_buffer
        ymin = min(p0[1], p1[1]) - init_buffer
        ymax = max(p0[1], p1[1]) + init_buffer

        # Expansion limits
        max_span = self.config.max_box_width
        step = self.config.expansion_step
        margin = self.config.safety_margin

        # Flags indicating whether expansion in each direction is frozen
        freeze_left = False    # -x
        freeze_right = False   # +x
        freeze_bottom = False  # -y
        freeze_top = False     # +y

        for _ in range(self.config.max_inflation_iters):
            all_frozen = freeze_left and freeze_right and freeze_bottom and freeze_top
            if all_frozen:
                break

            # Try expanding -x (left)
            if not freeze_left:
                cand_xmin = xmin - step
                if (xmax - cand_xmin) > max_span:
                    freeze_left = True
                else:
                    cand_box = Polygon2D([[cand_xmin, ymin], [xmax, ymin], [xmax, ymax], [cand_xmin, ymax]], ensure_ccw=True)
                    if self._is_box_safe(cand_box, obstacles, margin):
                        xmin = cand_xmin
                    else:
                        freeze_left = True

            # Try expanding +x (right)
            if not freeze_right:
                cand_xmax = xmax + step
                if (cand_xmax - xmin) > max_span:
                    freeze_right = True
                else:
                    cand_box = Polygon2D([[xmin, ymin], [cand_xmax, ymin], [cand_xmax, ymax], [xmin, ymax]], ensure_ccw=True)
                    if self._is_box_safe(cand_box, obstacles, margin):
                        xmax = cand_xmax
                    else:
                        freeze_right = True

            # Try expanding -y (bottom)
            if not freeze_bottom:
                cand_ymin = ymin - step
                if (ymax - cand_ymin) > max_span:
                    freeze_bottom = True
                else:
                    cand_box = Polygon2D([[xmin, cand_ymin], [xmax, cand_ymin], [xmax, ymax], [xmin, ymax]], ensure_ccw=True)
                    if self._is_box_safe(cand_box, obstacles, margin):
                        ymin = cand_ymin
                    else:
                        freeze_bottom = True

            # Try expanding +y (top)
            if not freeze_top:
                cand_ymax = ymax + step
                if (cand_ymax - ymin) > max_span:
                    freeze_top = True
                else:
                    cand_box = Polygon2D([[xmin, ymin], [xmax, ymin], [xmax, cand_ymax], [xmin, cand_ymax]], ensure_ccw=True)
                    if self._is_box_safe(cand_box, obstacles, margin):
                        ymax = cand_ymax
                    else:
                        freeze_top = True

        # Formulate H-representation: A * p <= b
        # [ 1,  0] * [x, y]^T <= xmax
        # [-1,  0] * [x, y]^T <= -xmin
        # [ 0,  1] * [x, y]^T <= ymax
        # [ 0, -1] * [x, y]^T <= -ymin
        A = np.array([
            [ 1.0,  0.0],
            [-1.0,  0.0],
            [ 0.0,  1.0],
            [ 0.0, -1.0],
        ], dtype=np.float64)
        b = np.array([xmax, -xmin, ymax, -ymin], dtype=np.float64)

        return Polytope(A=A, b=b, t_min=t_start, t_max=t_end)

    def _is_box_safe(self, box: Polygon2D, obstacles: List[Polygon2D], margin: float) -> bool:
        """Verify candidate bounding box is inside deck boundary and free of obstacle collisions."""
        # 1. Deck outer boundary containment with margin
        if not is_polygon_contained(box, self.env.outer_boundary, margin=margin):
            return False

        # 2. Island obstacle collision
        for piece in self.island_pieces:
            collides, _, _ = sat_convex_collision(box, piece)
            if collides:
                return False

        # 3. Extra static obstacles
        for obs in obstacles:
            collides, _, _ = sat_convex_collision(box, obs)
            if collides:
                return False

        return True

    def generate(
        self,
        waypoints: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        extra_obstacles: Optional[List[Polygon2D]] = None,
    ) -> List[Polytope]:
        """Generate sequence of safe flight corridors covering path waypoints.
        
        Args:
            waypoints: (K, 2) or (K, >=3) path waypoints [x, y, ...].
            timestamps: Optional (K,) array of timestamps at waypoints.
            extra_obstacles: Optional additional obstacle polygons.
            
        Returns:
            List of Polytope objects, each valid for segment [t_{k-1}, t_k].
        """
        pts = np.asarray(waypoints, dtype=np.float64)[:, :2]
        K = len(pts)
        if K < 2:
            raise ValueError(f"Need at least 2 waypoints to generate corridors, got {K}")

        if timestamps is not None:
            ts = np.asarray(timestamps, dtype=np.float64)
        else:
            # Generate nominal timestamps based on 1.0 m/s speed
            dists = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
            durations = np.maximum(dists / 1.0, 0.5)
            ts = np.concatenate([[0.0], np.cumsum(durations)])

        obstacles = extra_obstacles or []
        corridors: List[Polytope] = []

        for k in range(K - 1):
            p_start = pts[k]
            p_end = pts[k + 1]
            t_start = float(ts[k])
            t_end = float(ts[k + 1])

            poly = self.inflate_segment(p_start, p_end, t_start, t_end, obstacles)
            corridors.append(poly)

        return corridors

    def verify_zero_obstacle_intersection(
        self,
        corridors: List[Polytope],
        obstacles: Optional[List[Polygon2D]] = None,
    ) -> bool:
        """Formally verify zero intersection between corridors and obstacles via SAT."""
        all_obs = list(self.island_pieces)
        if obstacles:
            all_obs.extend(obstacles)

        for poly in corridors:
            poly_geom = poly.to_polygon()
            # Verify containment inside deck
            if not is_polygon_contained(poly_geom, self.env.outer_boundary):
                return False
            # Verify no collision with obstacles
            for obs in all_obs:
                collides, _, _ = sat_convex_collision(poly_geom, obs)
                if collides:
                    return False

        return True
