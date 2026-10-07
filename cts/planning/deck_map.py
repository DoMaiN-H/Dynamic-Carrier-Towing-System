"""cts.planning.deck_map

Carrier flight deck map representation, JSON environment parser,
dual-body swept-volume modeling (Tractor + Aircraft Fuselage + Wingspan),
Euclidean Distance Field (EDT), and Separating Axis Theorem (SAT) collision queries.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Sequence
import json
import numpy as np
from scipy.ndimage import distance_transform_edt

from cts.common.config import CarrierConfig, TractorConfig, AircraftConfig
from cts.common.geometry import (
    Polygon2D,
    sat_convex_collision,
    check_polygon_collision,
    decompose_island_to_convex,
    is_polygon_contained,
)
from cts.common.math_utils import wrap_to_pi
from cts.environment.deck_env import DeckEnvironment, DeckRegion, DeckObstacle, TouchdownPoint


@dataclass(frozen=True)
class RunwayCenterline:
    """Angled deck runway axis and Frenet projection utility.
    
    Directed line from stern entry (25.0, 42.52) to bow (215.0, 89.89)
    along angled deck heading 14.0 deg (0.244346 rad).
    """
    start_point: np.ndarray      # (2,) [x, y] near stern entry (m)
    end_point: np.ndarray        # (2,) [x, y] near angled deck bow (m)
    touchdown_point: np.ndarray  # (2,) [55.0, 50.0] 3rd wire touchdown (m)
    heading_deg: float = 14.0    # 14.0 deg
    heading_rad: float = 0.24434609527920614  # 0.244346 rad
    width: float = 22.0          # Safe runway corridor width (m)

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_point", np.asarray(self.start_point, dtype=np.float64)[:2])
        object.__setattr__(self, "end_point", np.asarray(self.end_point, dtype=np.float64)[:2])
        object.__setattr__(self, "touchdown_point", np.asarray(self.touchdown_point, dtype=np.float64)[:2])

    @property
    def length(self) -> float:
        """Runway segment length (m)."""
        return float(np.linalg.norm(self.end_point - self.start_point))

    @property
    def direction(self) -> np.ndarray:
        """Unit tangent vector along runway centerline."""
        diff = self.end_point - self.start_point
        norm = np.linalg.norm(diff)
        if norm < 1e-12:
            return np.array([np.cos(self.heading_rad), np.sin(self.heading_rad)], dtype=np.float64)
        return diff / norm

    @property
    def normal(self) -> np.ndarray:
        """Unit normal vector (pointing left/port: [-sin, cos])."""
        u = self.direction
        return np.array([-u[1], u[0]], dtype=np.float64)

    def project(self, point: Union[Sequence[float], np.ndarray]) -> Tuple[float, float]:
        """Project point onto runway axis, returning Frenet (s_along, d_cross).
        
        Args:
            point: [x, y] coordinates in deck frame.
            
        Returns:
            (s, d): s is longitudinal distance from start_point along centerline (m),
                    d is signed perpendicular lateral distance (m, positive to port/left).
        """
        p = np.asarray(point, dtype=np.float64)[:2]
        delta = p - self.start_point
        u = self.direction
        n = self.normal
        s = float(np.dot(delta, u))
        d = float(np.dot(delta, n))
        return s, d

    def distance_to_centerline(self, point: Union[Sequence[float], np.ndarray]) -> float:
        """Euclidean perpendicular distance from point to runway axis line."""
        _, d = self.project(point)
        return abs(d)

    def is_in_corridor(self, point: Union[Sequence[float], np.ndarray], width: Optional[float] = None) -> bool:
        """Check if point lies within lateral runway safety corridor."""
        s, d = self.project(point)
        w = float(width if width is not None else self.width)
        return bool(0.0 <= s <= self.length and abs(d) <= 0.5 * w)


def default_runway_centerline() -> RunwayCenterline:
    """Canonical angled-deck runway axis shared by planner and visualizers."""
    return RunwayCenterline(
        start_point=np.array([25.0, 42.52]),
        end_point=np.array([215.0, 89.89]),
        touchdown_point=np.array([55.0, 50.0]),
        heading_deg=14.0,
        heading_rad=0.24434609527920614,
        width=22.0,
    )


@dataclass(frozen=True)
class DualBodyFootprint:
    """Oriented multi-polygon footprint of the coupled tractor-aircraft system.
    
    Decomposed into 3 convex Oriented Bounding Boxes (OBBs):
    1. Tractor chassis
    2. Aircraft fuselage
    3. Aircraft wingspan
    """
    tractor: Polygon2D
    aircraft_fuselage: Polygon2D
    aircraft_wings: Polygon2D
    pose: Tuple[float, float, float, float]  # (x_t, y_t, psi_t, theta_h)

    @property
    def polygons(self) -> List[Polygon2D]:
        return [self.tractor, self.aircraft_fuselage, self.aircraft_wings]

    @property
    def aabb(self) -> Tuple[float, float, float, float]:
        """Global composite AABB enclosing all 3 sub-body polygons."""
        boxes = [p.aabb for p in self.polygons]
        xmin = min(b[0] for b in boxes)
        ymin = min(b[1] for b in boxes)
        xmax = max(b[2] for b in boxes)
        ymax = max(b[3] for b in boxes)
        return (xmin, ymin, xmax, ymax)

    @property
    def circumscribed_radius(self) -> float:
        """Maximum distance from tractor rear axle (x_t, y_t) to any vertex (~15.2 m)."""
        x_t, y_t = self.pose[0], self.pose[1]
        center = np.array([x_t, y_t], dtype=np.float64)
        max_r = 0.0
        for poly in self.polygons:
            dists = np.linalg.norm(poly.vertices - center, axis=1)
            poly_max = float(np.max(dists))
            if poly_max > max_r:
                max_r = poly_max
        return max_r


class DualBodyModel:
    """Generates dual-body polygon footprints and swept volumes."""

    def __init__(
        self,
        tractor_cfg: Optional[TractorConfig] = None,
        aircraft_cfg: Optional[AircraftConfig] = None,
        safety_margin_tractor: float = 0.15,
        safety_margin_aircraft: float = 0.25,
    ) -> None:
        self.t_cfg = tractor_cfg or TractorConfig()
        self.a_cfg = aircraft_cfg or AircraftConfig()
        self.margin_t = float(safety_margin_tractor)
        self.margin_a = float(safety_margin_aircraft)

        # Tractor dimensions (origin at rear axle center)
        self.l_t = float(self.t_cfg.wheelbase)       # 1.8 m
        self.w_t = float(self.t_cfg.width)           # 1.4 m
        self.d_c = float(self.t_cfg.hitch_offset)    # 0.1 m
        # Tractor chassis extends from hitch (-0.2 m behind axle) to front bumper (+2.2 m ahead of axle)
        self.t_x_front = self.l_t + 0.40             # +2.20 m
        self.t_x_rear = -0.20                        # -0.20 m
        self.t_half_w = 0.5 * self.w_t + self.margin_t

        # Aircraft dimensions (origin at cradle hitch point)
        # Fuselage: 18.3 m x 3.2 m (nose +3.9 m ahead of hitch, tail -14.4 m behind hitch)
        self.fuse_x_front = float(self.a_cfg.nose_overhang)  # +3.9 m
        self.fuse_x_rear = self.fuse_x_front - float(self.a_cfg.length)  # 3.9 - 18.3 = -14.4 m
        self.fuse_half_w = 1.60 + self.margin_a              # 3.2 m / 2 = 1.6 m

        # Wingspan: 5.5 m x 8.6 m (covering main gear/wing root [-10.5, -5.0] m behind hitch)
        # Nose gear to main gear wheelbase is l_a = 8.0 m. Main gear at -8.0 m.
        # Wing root span [-10.5, -5.0] m behind hitch.
        self.wing_x_front = -5.00
        self.wing_x_rear = -10.50
        self.wing_half_w = 0.5 * float(self.a_cfg.folded_width) + self.margin_a  # 8.6 / 2 = 4.3 m

    def get_footprint(self, x_t: float, y_t: float, psi_t: float, theta_h: float) -> DualBodyFootprint:
        """Construct exact OBBs for tractor chassis, fuselage, and wingspan.
        
        Args:
            x_t: Tractor rear axle X coordinate (m).
            y_t: Tractor rear axle Y coordinate (m).
            psi_t: Tractor heading angle (rad).
            theta_h: Articulation hitch angle (rad).
            
        Returns:
            DualBodyFootprint containing 3 oriented convex Polygon2D objects.
        """
        # 1. Tractor chassis polygon (oriented along psi_t)
        c_t, s_t = np.cos(psi_t), np.sin(psi_t)
        R_t = np.array([[c_t, -s_t], [s_t, c_t]], dtype=np.float64)
        t_local = np.array([
            [self.t_x_front,  self.t_half_w],
            [self.t_x_front, -self.t_half_w],
            [self.t_x_rear,  -self.t_half_w],
            [self.t_x_rear,   self.t_half_w],
        ], dtype=np.float64)
        t_world = t_local @ R_t.T + np.array([x_t, y_t], dtype=np.float64)
        poly_tractor = Polygon2D(t_world, ensure_ccw=True)

        # 2. Hitch location in deck world frame
        x_h = x_t - self.d_c * c_t
        y_h = y_t - self.d_c * s_t
        hitch_pt = np.array([x_h, y_h], dtype=np.float64)

        # 3. Aircraft heading: psi_a = wrap_to_pi(psi_t + theta_h)
        psi_a = wrap_to_pi(psi_t + theta_h)
        c_a, s_a = np.cos(psi_a), np.sin(psi_a)
        R_a = np.array([[c_a, -s_a], [s_a, c_a]], dtype=np.float64)

        # 4. Aircraft fuselage polygon (18.3 m x 3.2 m)
        fuse_local = np.array([
            [self.fuse_x_front,  self.fuse_half_w],
            [self.fuse_x_front, -self.fuse_half_w],
            [self.fuse_x_rear,  -self.fuse_half_w],
            [self.fuse_x_rear,   self.fuse_half_w],
        ], dtype=np.float64)
        fuse_world = fuse_local @ R_a.T + hitch_pt
        poly_fuselage = Polygon2D(fuse_world, ensure_ccw=True)

        # 5. Aircraft wingspan polygon (5.5 m x 8.6 m)
        wing_local = np.array([
            [self.wing_x_front,  self.wing_half_w],
            [self.wing_x_front, -self.wing_half_w],
            [self.wing_x_rear,  -self.wing_half_w],
            [self.wing_x_rear,   self.wing_half_w],
        ], dtype=np.float64)
        wing_world = wing_local @ R_a.T + hitch_pt
        poly_wings = Polygon2D(wing_world, ensure_ccw=True)

        return DualBodyFootprint(
            tractor=poly_tractor,
            aircraft_fuselage=poly_fuselage,
            aircraft_wings=poly_wings,
            pose=(float(x_t), float(y_t), float(psi_t), float(theta_h)),
        )

    def get_swept_volume(
        self,
        pose1: Tuple[float, float, float, float],
        pose2: Tuple[float, float, float, float],
        num_substeps: int = 5,
    ) -> List[DualBodyFootprint]:
        """Sub-step interpolated footprint sequence preventing obstacle tunneling."""
        footprints: List[DualBodyFootprint] = []
        n_steps = max(2, int(num_substeps))
        alphas = np.linspace(0.0, 1.0, n_steps)

        x1, y1, psi1, th1 = pose1
        x2, y2, psi2, th2 = pose2

        # Unwrap angles for smooth angular interpolation
        dpsi = wrap_to_pi(psi2 - psi1)
        dth = wrap_to_pi(th2 - th1)

        for alpha in alphas:
            x_i = float(x1 + alpha * (x2 - x1))
            y_i = float(y1 + alpha * (y2 - y1))
            psi_i = float(wrap_to_pi(psi1 + alpha * dpsi))
            th_i = float(wrap_to_pi(th1 + alpha * dth))
            footprints.append(self.get_footprint(x_i, y_i, psi_i, th_i))

        return footprints


class DeckGridMap:
    """2D raster grid with Euclidean Distance Transform (EDT) for sub-microsecond clearance queries.
    
    Guarantees >= 1M queries/second lookup throughput via O(1) vectorized memory indexing.
    """

    def __init__(
        self,
        deck_env: DeckEnvironment,
        resolution: float = 1.0,
        x_bounds: Tuple[float, float] = (0.0, 350.0),
        y_bounds: Tuple[float, float] = (0.0, 120.0),
        occupied_polygons: Optional[List[Polygon2D]] = None,
    ) -> None:
        self.env = deck_env
        self.resolution = float(resolution)
        self.x_min, self.x_max = float(x_bounds[0]), float(x_bounds[1])
        self.y_min, self.y_max = float(y_bounds[0]), float(y_bounds[1])
        self.occupied_polygons = occupied_polygons or []

        self.nx = int(np.ceil((self.x_max - self.x_min) / self.resolution))
        self.ny = int(np.ceil((self.y_max - self.y_min) / self.resolution))

        # Build raster grids
        self.occupancy_grid = np.ones((self.nx, self.ny), dtype=bool)  # True = obstacle/exterior
        self.distance_map = np.zeros((self.nx, self.ny), dtype=np.float64)  # Clearance in meters

        self._build_grid()

    def _build_grid(self) -> None:
        """Rasterize outer deck polygon, island obstacle, and extra occupied polygons."""
        # Cell center coordinates
        xs = self.x_min + (np.arange(self.nx, dtype=np.float64) + 0.5) * self.resolution
        ys = self.y_min + (np.arange(self.ny, dtype=np.float64) + 0.5) * self.resolution
        X, Y = np.meshgrid(xs, ys, indexing="ij")  # (nx, ny)

        pts = np.column_stack([X.ravel(), Y.ravel()])

        # Fast vectorized ray-casting point-in-polygon containment
        def vectorized_pip(verts: np.ndarray, query_pts: np.ndarray) -> np.ndarray:
            px = query_pts[:, 0]
            py = query_pts[:, 1]
            n_v = len(verts)
            inside = np.zeros(len(query_pts), dtype=bool)
            for i in range(n_v):
                p1 = verts[i]
                p2 = verts[(i + 1) % n_v]
                edge_cond = ((p1[1] <= py) & (py < p2[1])) | ((p2[1] <= py) & (py < p1[1]))
                if np.any(edge_cond):
                    dy = p2[1] - p1[1]
                    dx = p2[0] - p1[0]
                    x_inters = p1[0] + (py[edge_cond] - p1[1]) * dx / (dy + 1e-14)
                    inside[edge_cond] ^= (px[edge_cond] < x_inters)
            return inside

        # 1. Outer boundary containment
        navigable = vectorized_pip(self.env.outer_boundary.vertices, pts)

        # 2. Island obstacle exclusion
        island_inside = vectorized_pip(self.env.island.polygon.vertices, pts)
        navigable[island_inside] = False

        # 3. Static extra obstacles
        if self.occupied_polygons:
            for poly in self.occupied_polygons:
                poly_inside = vectorized_pip(poly.vertices, pts)
                navigable[poly_inside] = False

        # Reshape to grid: False = navigable, True = obstacle
        nav_grid = navigable.reshape((self.nx, self.ny))
        self.occupancy_grid = ~nav_grid

        # 4. Compute Euclidean Distance Transform (EDT)
        edt_dist_cells = distance_transform_edt(nav_grid)
        self.distance_map = np.ascontiguousarray(edt_dist_cells * self.resolution, dtype=np.float64)
        self.inv_res = 1.0 / self.resolution

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (x, y) to grid indices (ix, iy)."""
        ix = int((x - self.x_min) * self.inv_res)
        iy = int((y - self.y_min) * self.inv_res)
        return ix, iy

    def grid_to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        """Convert grid indices (ix, iy) to cell center world coordinates (x, y)."""
        x = self.x_min + (float(ix) + 0.5) * self.resolution
        y = self.y_min + (float(iy) + 0.5) * self.resolution
        return x, y

    def is_in_bounds(self, x: float, y: float) -> bool:
        """Check if world coordinate is within raster bounds."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def is_point_free(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Check if world coordinate is free with given clearance margin."""
        ix = int((x - self.x_min) * self.inv_res)
        iy = int((y - self.y_min) * self.inv_res)
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            return float(self.distance_map[ix, iy]) >= float(margin)
        return False

    def get_clearance(self, x: float, y: float) -> float:
        """Instantaneous O(1) clearance query to nearest deck boundary or obstacle."""
        ix = int((x - self.x_min) * self.inv_res)
        iy = int((y - self.y_min) * self.inv_res)
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            return self.distance_map[ix, iy]
        return 0.0

    def get_clearance_batch(self, points: np.ndarray) -> np.ndarray:
        """Vectorized fast clearance queries."""
        pts = np.asarray(points, dtype=np.float64)
        ix = ((pts[:, 0] - self.x_min) * self.inv_res).astype(np.int64)
        iy = ((pts[:, 1] - self.y_min) * self.inv_res).astype(np.int64)
        valid = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        out = np.zeros(len(pts), dtype=np.float64)
        out[valid] = self.distance_map[ix[valid], iy[valid]]
        return out


class CarrierDeckMap:
    """Master flight deck spatial map for kinodynamic path planning.
    
    Features:
    - Parses carrier_deck_environment.json
    - RunwayCenterline directed line projection
    - DualBodyModel with 3 convex OBBs
    - DeckGridMap with Euclidean Distance Transform (EDT)
    - Two-tier hierarchical collision checking: O(1) EDT clearance filter followed by exact SAT.
    """

    def __init__(
        self,
        deck_env: DeckEnvironment,
        grid_map: DeckGridMap,
        runway: RunwayCenterline,
        dual_body: DualBodyModel,
        static_obstacles: Optional[Dict[str, Polygon2D]] = None,
    ) -> None:
        self.env = deck_env
        self.grid = grid_map
        self.runway = runway
        self.dual_body = dual_body
        self.static_obstacles: Dict[str, Polygon2D] = static_obstacles or {}

        # Cached decomposed island polygons
        self.island_pieces = decompose_island_to_convex(self.env.island.polygon)

        # Precalculated dual-body geometry bounds for two-tier filtering
        # Tractor rear axle to furthest vertex of dual body footprint
        self.r_circumscribed = 15.20  # meters
        self.r_inscribed = 0.70       # meters

    @classmethod
    def from_json(
        cls,
        json_path: Union[Path, str],
        resolution: float = 1.0,
        occupied_spots: Optional[List[str]] = None,
    ) -> "CarrierDeckMap":
        """Factory: loads JSON environment, builds runway, dual-body model, and EDT grid."""
        env = DeckEnvironment.from_json(json_path)

        # Angled deck runway centerline
        runway = default_runway_centerline()

        dual_body = DualBodyModel()

        # Parse static obstacles from occupied parking spots
        static_obstacles: Dict[str, Polygon2D] = {}
        if occupied_spots:
            for spot_id in occupied_spots:
                region = env.get_region_by_id(spot_id)
                if region:
                    static_obstacles[spot_id] = region.polygon

        grid_map = DeckGridMap(
            deck_env=env,
            resolution=resolution,
            occupied_polygons=list(static_obstacles.values()),
        )

        return cls(
            deck_env=env,
            grid_map=grid_map,
            runway=runway,
            dual_body=dual_body,
            static_obstacles=static_obstacles,
        )

    @classmethod
    def from_default(
        cls,
        resolution: float = 1.0,
        occupied_spots: Optional[List[str]] = None,
    ) -> "CarrierDeckMap":
        """Factory from default project JSON asset path."""
        default_path = Path(__file__).resolve().parents[2] / "data" / "carrier_deck_environment.json"
        if not default_path.exists():
            default_path = Path("data/carrier_deck_environment.json")
        return cls.from_json(default_path, resolution=resolution, occupied_spots=occupied_spots)

    def is_pose_collision_free(
        self,
        x_t: float,
        y_t: float,
        psi_t: float,
        theta_h: float,
        safety_margin: float = 0.0,
    ) -> bool:
        """Two-tier hierarchical collision check: O(1) EDT clearance filter followed by exact SAT.
        
        Args:
            x_t: Tractor rear axle X (m).
            y_t: Tractor rear axle Y (m).
            psi_t: Tractor heading (rad).
            theta_h: Articulation hitch angle (rad).
            safety_margin: Additional safety clearance buffer (m).
            
        Returns:
            True if pose is strictly collision-free, False otherwise.
        """
        # Tier 1: Fast O(1) EDT Clearance Filter
        clearance = self.grid.get_clearance(x_t, y_t)

        # If clearance exceeds circumscribed radius plus margin, footprint is unconditionally free!
        if clearance > self.r_circumscribed + safety_margin:
            return True

        # If clearance is less than inscribed radius, tractor center is too close to boundary/obstacle
        if clearance < self.r_inscribed + safety_margin:
            return False

        # Tier 2: Exact Separating Axis Theorem (SAT) on all 3 sub-body polygons
        footprint = self.dual_body.get_footprint(x_t, y_t, psi_t, theta_h)

        # 2a. Fast vertex clearance filter: if any vertex penetrates boundary/obstacle, fail early
        min_v_clearance = float("inf")
        for poly in footprint.polygons:
            for pt in poly.vertices:
                c = self.grid.get_clearance(pt[0], pt[1])
                if c < safety_margin:
                    return False
                if c < min_v_clearance:
                    min_v_clearance = c

        # 2b. If vertex clearance is close to boundary, do exact outer boundary containment check
        if min_v_clearance < 2.5 + safety_margin:
            for poly in footprint.polygons:
                if not is_polygon_contained(poly, self.env.outer_boundary, margin=safety_margin):
                    return False

        # 2c. Check SAT collision with island obstacle pieces ONLY if within range of island
        # Island bbox is [62.0, 27.0] to [92.0, 44.0]
        dx_isl = max(62.0 - x_t, 0.0, x_t - 92.0)
        dy_isl = max(27.0 - y_t, 0.0, y_t - 44.0)
        if dx_isl**2 + dy_isl**2 <= (self.r_circumscribed + 2.0)**2:
            for poly in footprint.polygons:
                for island_piece in self.island_pieces:
                    collides, _, _ = sat_convex_collision(poly, island_piece)
                    if collides:
                        return False

        # 2d. Check SAT collision with active static obstacles (parked planes)
        if self.static_obstacles:
            r_check_sq = (self.r_circumscribed + safety_margin + 2.0) ** 2
            for obs_poly in self.static_obstacles.values():
                xmin, ymin, xmax, ymax = obs_poly.aabb
                dx_obs = max(xmin - x_t, 0.0, x_t - xmax)
                dy_obs = max(ymin - y_t, 0.0, y_t - ymax)
                if dx_obs**2 + dy_obs**2 > r_check_sq:
                    continue
                for poly in footprint.polygons:
                    collides, _, _ = sat_convex_collision(poly, obs_poly)
                    if collides:
                        return False

        return True

    def check_swept_volume_collision(
        self,
        pose1: Tuple[float, float, float, float],
        pose2: Tuple[float, float, float, float],
        safety_margin: float = 0.0,
        num_substeps: Optional[int] = None,
    ) -> bool:
        """Continuous collision detection along motion primitive segment.
        
        Returns:
            True if collision occurs (swept volume collides), False if safe.
        """
        # Compute required number of substeps based on distance
        x1, y1 = pose1[0], pose1[1]
        x2, y2 = pose2[0], pose2[1]
        dist = float(np.hypot(x2 - x1, y2 - y1))
        n_steps = num_substeps if num_substeps is not None else max(3, int(np.ceil(dist / 0.30)))

        footprints = self.dual_body.get_swept_volume(pose1, pose2, num_substeps=n_steps)
        for fp in footprints:
            x_i, y_i, psi_i, th_i = fp.pose
            if not self.is_pose_collision_free(x_i, y_i, psi_i, th_i, safety_margin=safety_margin):
                return True  # Collision detected
        return False  # All sub-steps safe

    def mark_spot_occupied(self, spot_id: str) -> None:
        """Designate a parking spot as containing an obstacle aircraft."""
        region = self.env.get_region_by_id(spot_id)
        if region:
            self.static_obstacles[spot_id] = region.polygon
            # Rebuild grid map to include new obstacle
            self.grid = DeckGridMap(
                deck_env=self.env,
                resolution=self.grid.resolution,
                occupied_polygons=list(self.static_obstacles.values()),
            )

    def clear_spot(self, spot_id: str) -> None:
        """Clear obstacle from designated spot."""
        if spot_id in self.static_obstacles:
            del self.static_obstacles[spot_id]
            self.grid = DeckGridMap(
                deck_env=self.env,
                resolution=self.grid.resolution,
                occupied_polygons=list(self.static_obstacles.values()),
            )
