"""cts.planning.hybrid_a_star

Feature F11: 5D Spatiotemporal Hybrid A* Search.
Nonholonomic kinodynamic planning for towbarless carrier tractor-aircraft system
under dynamic 6-DOF deck motion, dynamic admissibility gating, and dual-body collision avoidance.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
from collections import deque
import heapq
import time
import numpy as np

from cts.common.config import CTSConfig, TractorConfig, AircraftConfig, CarrierConfig, TireConfig
from cts.common.geometry import Polygon2D, sat_convex_collision, is_polygon_contained
from cts.common.math_utils import wrap_to_pi, wrap_to_2pi
from cts.environment.deck_env import DeckEnvironment
from cts.environment.deck_motion import DeckMotionSimulator, DeckState
from cts.dynamics.multibody_dynamics import VehicleState
from cts.planning.deck_map import CarrierDeckMap, DualBodyModel, DualBodyFootprint
from cts.planning.kinodynamic_limits import KinodynamicLimits, KinodynamicBound


@dataclass(frozen=True)
class HybridAStarConfig:
    """Configuration hyperparameters for 5D Spatiotemporal Hybrid A* Search."""
    # 5D Discretization resolutions for closed set
    xy_resolution: float = 0.8           # Spatial grid cell size (m)
    yaw_resolution_deg: float = 5.0      # Heading resolution (deg)
    hitch_resolution_deg: float = 10.0   # Articulation angle resolution (deg)
    time_resolution: float = 0.5         # Spatiotemporal bin (s)
    
    # Motion Primitive Generation
    v_levels: Tuple[float, ...] = (1.8,)                     # Forward cruise speed (m/s)
    enable_reverse: bool = True                              # Allow low-speed reverse primitives
    v_reverse: float = -0.8                                  # Reverse speed level (m/s)
    delta_levels_deg: Tuple[float, ...] = (                  # Steering angles (deg)
        -35.0, -18.0, 0.0, 18.0, 35.0
    )
    dt_prim: float = 1.5                 # Primitive duration (s)
    dt_sub: float = 0.25                 # Numerical integration sub-step (s)
    
    # Goal Tolerances
    xy_goal_tol: float = 0.8             # Euclidean distance tolerance to goal (m)
    yaw_goal_tol_deg: float = 8.0        # Heading tolerance to goal (deg)
    hitch_goal_tol_deg: float = 70.0     # Articulation tolerance to goal (deg)
    
    # Analytic Expansion
    analytic_interval: int = 1           # Attempt analytic expansion every N nodes
    analytic_distance_thresh: float = 200.0  # Max distance to goal to trigger analytic check (m)
    
    # Cost Weights
    w_dist: float = 1.0                  # Weight on path length
    w_time: float = 0.2                  # Weight on transit duration
    w_steer: float = 0.1                 # Weight on steering magnitude
    w_steer_change: float = 0.3          # Weight on steering angle rate/chatter
    w_hitch: float = 0.5                 # Weight on articulation angle magnitude
    w_reverse: float = 5.0               # Penalty on reverse motion
    w_tilt: float = 2.0                  # Penalty on moving during high deck roll tilt
    
    # Search Budget & Limits
    max_iterations: int = 15000          # Max node expansions
    timeout_s: float = 2.0               # Strict search timeout (s)
    safety_margin_tractor: float = 0.15  # Extra safety buffer for tractor (m)
    safety_margin_aircraft: float = 0.25 # Extra safety buffer for aircraft (m)


@dataclass
class HybridAStarNode:
    """Spatiotemporal search tree node."""
    # 5D State Vector
    x_t: float
    y_t: float
    psi_t: float
    theta_h: float
    t: float
    
    # Preceding Control
    v_t: float = 0.0
    delta_f: float = 0.0
    
    # Costs
    g_cost: float = 0.0
    h_cost: float = 0.0
    f_cost: float = 0.0
    
    # Tree Links
    parent: Optional["HybridAStarNode"] = None
    intermediate_states: List[Tuple[float, float, float, float, float]] = field(default_factory=list)
    
    # Unique tie-breaker ID
    node_id: int = 0

    def __lt__(self, other: "HybridAStarNode") -> bool:
        if abs(self.f_cost - other.f_cost) > 1e-6:
            return self.f_cost < other.f_cost
        return self.node_id < other.node_id


@dataclass(frozen=True)
class HybridAStarResult:
    """Output container for Spatiotemporal Hybrid A* planning result."""
    success: bool
    timestamps: np.ndarray       # (K,) Array of timestamps [t_0, ..., t_K]
    x: np.ndarray                # (K,) Tractor X coordinates (m)
    y: np.ndarray                # (K,) Tractor Y coordinates (m)
    psi_t: np.ndarray            # (K,) Tractor heading angles (rad)
    theta_h: np.ndarray          # (K,) Articulation hitch angles (rad)
    psi_a: np.ndarray            # (K,) Aircraft heading angles (rad)
    v: np.ndarray                # (K,) Forward velocity (m/s)
    delta_f: np.ndarray          # (K,) Steering angles (rad)
    nodes_expanded: int
    primitives_evaluated: int
    search_time_s: float
    total_cost: float
    failure_reason: str = ""


class ObstacleDistanceHeuristic:
    """Precomputes 2D grid BFS distance transform from goal position."""

    def __init__(self, deck_map: CarrierDeckMap, resolution: float = 2.0) -> None:
        self.deck_map = deck_map
        self.res = float(resolution)
        self.inv_res = 1.0 / self.res
        self.x_min, self.x_max = 0.0, 350.0
        self.y_min, self.y_max = 0.0, 120.0
        self.nx = int(np.ceil((self.x_max - self.x_min) * self.inv_res))
        self.ny = int(np.ceil((self.y_max - self.y_min) * self.inv_res))
        self.dist_grid = np.full((self.nx, self.ny), np.inf, dtype=np.float32)
        self.goal_pos: Optional[Tuple[float, float]] = None

    def compute(self, goal_x: float, goal_y: float) -> None:
        """Run fast 8-connected BFS backward from (goal_x, goal_y)."""
        self.goal_pos = (float(goal_x), float(goal_y))
        self.dist_grid.fill(np.inf)

        gx_idx = int((goal_x - self.x_min) * self.inv_res)
        gy_idx = int((goal_y - self.y_min) * self.inv_res)
        if not (0 <= gx_idx < self.nx and 0 <= gy_idx < self.ny):
            return

        self.dist_grid[gx_idx, gy_idx] = 0.0
        queue = deque([(gx_idx, gy_idx)])

        moves = [
            (1, 0, self.res),
            (-1, 0, self.res),
            (0, 1, self.res),
            (0, -1, self.res),
            (1, 1, 1.4142 * self.res),
            (1, -1, 1.4142 * self.res),
            (-1, 1, 1.4142 * self.res),
            (-1, -1, 1.4142 * self.res),
        ]

        while queue:
            ix, iy = queue.popleft()
            d = self.dist_grid[ix, iy]

            for dx, dy, step_cost in moves:
                nx_idx = ix + dx
                ny_idx = iy + dy
                if 0 <= nx_idx < self.nx and 0 <= ny_idx < self.ny:
                    new_d = d + step_cost
                    if new_d < self.dist_grid[nx_idx, ny_idx]:
                        wx = self.x_min + (nx_idx + 0.5) * self.res
                        wy = self.y_min + (ny_idx + 0.5) * self.res
                        if self.deck_map.grid.get_clearance(wx, wy) >= 1.0:
                            self.dist_grid[nx_idx, ny_idx] = new_d
                            queue.append((nx_idx, ny_idx))

    def get_distance(self, x: float, y: float) -> float:
        """Query 2D obstacle-aware distance in O(1)."""
        ix = int((x - self.x_min) * self.inv_res)
        iy = int((y - self.y_min) * self.inv_res)
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            val = float(self.dist_grid[ix, iy])
            if np.isfinite(val):
                return val
        if self.goal_pos is not None:
            return float(np.hypot(x - self.goal_pos[0], y - self.goal_pos[1]))
        return 0.0


class DubinsReedsSheppHeuristic:
    """Evaluates analytical shortest nonholonomic distance between SE(2) poses."""

    def __init__(self, r_min: float) -> None:
        self.r_min = max(0.5, float(r_min))

    def distance(self, x1: float, y1: float, psi1: float, x2: float, y2: float, psi2: float) -> float:
        """Analytical lower bound on nonholonomic path distance."""
        dx = x2 - x1
        dy = y2 - y1
        d_euclid = float(np.hypot(dx, dy))

        # Angular alignment difference
        dpsi = abs(float(wrap_to_pi(psi2 - psi1)))

        # Analytical lower bound on turning arc length
        # When heading change is dpsi, minimum turning distance is r_min * dpsi
        d_turn = self.r_min * dpsi

        # Combined admissible lower bound: hypot(d_euclid, d_turn) or max(d_euclid, d_turn)
        return float(max(d_euclid, d_turn * 0.8))


class SpatiotemporalHybridAStar:
    """Feature F11: 5D Spatiotemporal Hybrid A* Planner."""

    def __init__(
        self,
        deck_map: CarrierDeckMap,
        sea_state: DeckMotionSimulator,
        config: Optional[HybridAStarConfig] = None,
        tractor_cfg: Optional[TractorConfig] = None,
        aircraft_cfg: Optional[AircraftConfig] = None,
        tire_cfg: Optional[TireConfig] = None,
    ) -> None:
        self.deck_map = deck_map
        self.sea_state = sea_state
        self.cfg = config or HybridAStarConfig()
        self.t_cfg = tractor_cfg or TractorConfig()
        self.a_cfg = aircraft_cfg or AircraftConfig()
        self.tire_cfg = tire_cfg or TireConfig()

        # Tractor-aircraft kinematics constants
        self.l_t = float(self.t_cfg.wheelbase)       # 1.80 m
        self.l_a = float(self.a_cfg.wheelbase)       # 8.00 m
        self.d_c = float(self.t_cfg.hitch_offset)    # 0.10 m
        self.delta_max = float(self.t_cfg.delta_max_rad)
        self.theta_h_max = float(self.t_cfg.beta_max_rad)
        self.r_min = float(self.l_t / np.tan(self.delta_max))

        # Dynamic limits evaluator
        self.kin_limits = KinodynamicLimits(
            tractor_cfg=self.t_cfg,
            aircraft_cfg=self.a_cfg,
            tire_cfg=self.tire_cfg,
            carrier_cfg=CarrierConfig(),
        )

        # Dual heuristic evaluators
        self.obs_heuristic = ObstacleDistanceHeuristic(self.deck_map, resolution=self.cfg.xy_resolution)
        self.rs_heuristic = DubinsReedsSheppHeuristic(self.r_min)

        # Discretization angles in radians
        self.res_yaw = float(np.deg2rad(self.cfg.yaw_resolution_deg))
        self.res_hitch = float(np.deg2rad(self.cfg.hitch_resolution_deg))

    def _state_to_key(self, x: float, y: float, psi: float, theta_h: float, t: float) -> Tuple[int, int, int, int]:
        """Discretize kinematic state into discrete hash key for closed set."""
        ix = int(x / self.cfg.xy_resolution)
        iy = int(y / self.cfg.xy_resolution)
        ipsi = int(wrap_to_2pi(psi) / self.res_yaw)
        ith = int((theta_h + np.pi) / self.res_hitch)
        return (ix, iy, ipsi, ith)

    def _evaluate_heuristic(self, x: float, y: float, psi: float, t: float, goal_pose: Tuple[float, float, float]) -> float:
        """Combined admissible dual-heuristic h(s) = max(h_obs, h_rs) + h_tilt."""
        gx, gy, gpsi = goal_pose
        h_obs = self.obs_heuristic.get_distance(x, y)
        h_rs = self.rs_heuristic.distance(x, y, psi, gx, gy, gpsi)
        h_geom = max(h_obs, h_rs)

        # Deck roll tilt penalty
        try:
            deck_st = self.sea_state.get_deck_state(t)
            phi = float(deck_st.rpy_rad[0])
            h_tilt = self.cfg.w_tilt * abs(np.sin(phi))
        except Exception:
            h_tilt = 0.0

        return self.cfg.w_dist * h_geom + h_tilt

    def _forward_integrate_primitive(
        self,
        start_state: Tuple[float, float, float, float, float],
        v_t: float,
        delta_f: float,
        delta_f_prev: float,
    ) -> Optional[Tuple[Tuple[float, float, float, float, float], List[Tuple[float, float, float, float, float]], float]]:
        """RK4 forward integration of motion primitive with 4-tier admissibility gating.
        
        Returns:
            (end_state, intermediate_states, step_cost) or None if inadmissible/colliding.
        """
        x_0, y_0, psi_0, th_0, t_0 = start_state
        dt_sub = self.cfg.dt_sub
        n_steps = int(np.round(self.cfg.dt_prim / dt_sub))

        # Gate 1: Anti-Jackknife Initial Check
        if abs(th_0) > self.theta_h_max:
            return None

        # Gate 4: Steering Slew Rate Limit
        steer_rate = abs(delta_f - delta_f_prev) / self.cfg.dt_prim
        if steer_rate > self.kin_limits.dot_delta_max + 1e-4:
            return None

        curr_x, curr_y, curr_psi, curr_th, curr_t = x_0, y_0, psi_0, th_0, t_0
        intermediates: List[Tuple[float, float, float, float, float]] = []

        tan_delta = np.tan(delta_f)
        steer_term = (v_t / self.l_t) * tan_delta
        cos_corr = 1.0 + (self.d_c / self.l_a) * np.cos(curr_th)

        # Gates 2 & 3: Dynamic Lateral Acceleration and Rollover Bounds
        # a_y_req is constant for the primitive: v_t^2 * |tan(delta_f)| / l_t
        a_y_req = (v_t ** 2) * abs(tan_delta / self.l_t)
        bounds = self.kin_limits.compute_limits(
            t_0, self.sea_state, v_t=abs(v_t), tractor_pos=np.array([x_0, y_0, 0.0]), psi_t=psi_0
        )
        if a_y_req > bounds.a_y_crit_slip or a_y_req > bounds.a_y_crit_rollover:
            return None

        for _ in range(n_steps):
            # RK4 Integration of (x, y, psi, theta_h)
            def ode_deriv(x, y, psi, th):
                dx = v_t * np.cos(psi)
                dy = v_t * np.sin(psi)
                dpsi = (v_t / self.l_t) * tan_delta
                dth = -(v_t / self.l_a) * np.sin(th) - steer_term * (1.0 + (self.d_c / self.l_a) * np.cos(th))
                return dx, dy, dpsi, dth

            k1_x, k1_y, k1_psi, k1_th = ode_deriv(curr_x, curr_y, curr_psi, curr_th)
            k2_x, k2_y, k2_psi, k2_th = ode_deriv(
                curr_x + 0.5 * dt_sub * k1_x,
                curr_y + 0.5 * dt_sub * k1_y,
                curr_psi + 0.5 * dt_sub * k1_psi,
                curr_th + 0.5 * dt_sub * k1_th,
            )
            k3_x, k3_y, k3_psi, k3_th = ode_deriv(
                curr_x + 0.5 * dt_sub * k2_x,
                curr_y + 0.5 * dt_sub * k2_y,
                curr_psi + 0.5 * dt_sub * k2_psi,
                curr_th + 0.5 * dt_sub * k2_th,
            )
            k4_x, k4_y, k4_psi, k4_th = ode_deriv(
                curr_x + dt_sub * k3_x,
                curr_y + dt_sub * k3_y,
                curr_psi + dt_sub * k3_psi,
                curr_th + dt_sub * k3_th,
            )

            curr_x += (dt_sub / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
            curr_y += (dt_sub / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
            curr_psi = wrap_to_pi(curr_psi + (dt_sub / 6.0) * (k1_psi + 2.0 * k2_psi + 2.0 * k3_psi + k4_psi))
            curr_th = wrap_to_pi(curr_th + (dt_sub / 6.0) * (k1_th + 2.0 * k2_th + 2.0 * k3_th + k4_th))
            curr_t += dt_sub

            # Gate 1: Anti-Jackknife Travel Limit
            if abs(curr_th) > self.theta_h_max:
                return None

            # Collision Check at sub-step
            if not self.deck_map.is_pose_collision_free(curr_x, curr_y, curr_psi, curr_th):
                return None

            intermediates.append((float(curr_x), float(curr_y), float(curr_psi), float(curr_th), float(curr_t)))

        # Primitive cost calculation
        step_dist = abs(v_t) * self.cfg.dt_prim
        cost = (
            self.cfg.w_dist * step_dist +
            self.cfg.w_time * self.cfg.dt_prim +
            self.cfg.w_steer * abs(delta_f) +
            self.cfg.w_steer_change * steer_rate +
            self.cfg.w_hitch * abs(curr_th)
        )
        if v_t < 0.0:
            cost *= self.cfg.w_reverse

        end_state = (float(curr_x), float(curr_y), float(curr_psi), float(curr_th), float(curr_t))
        return end_state, intermediates, float(cost)

    def _attempt_analytic_expansion(
        self,
        node: HybridAStarNode,
        goal_state: Tuple[float, float, float, float],
    ) -> Optional[List[Tuple[float, float, float, float, float, float, float]]]:
        """Attempt analytical straight / arc connection to goal pose."""
        gx, gy, gpsi, gth = goal_state
        dx = gx - node.x_t
        dy = gy - node.y_t
        dist = float(np.hypot(dx, dy))

        if dist > self.cfg.analytic_distance_thresh or dist < 0.1:
            return None

        # Check orientation alignment to target bearing
        target_bearing = float(np.arctan2(dy, dx))
        heading_err = abs(float(wrap_to_pi(node.psi_t - target_bearing)))

        # Only attempt if oriented within 85 degrees of target bearing
        if heading_err > np.deg2rad(85.0):
            return None

        # Generate smooth Dubins/straight trajectory segment to goal
        n_steps = max(5, int(np.ceil(dist / 0.80)))
        step_dist = dist / n_steps
        v_speed = min(1.0, float(self.t_cfg.v_max))  # nominal connection speed (strictly <= v_max)
        sub_dt = step_dist / v_speed                 # physically consistent step duration (v = step_dist / sub_dt)

        pts: List[Tuple[float, float, float, float, float, float, float]] = []
        curr_x, curr_y, curr_psi, curr_th, curr_t = node.x_t, node.y_t, node.psi_t, node.theta_h, node.t

        alphas = np.linspace(0.0, 1.0, n_steps + 1)[1:]
        dpsi = float(wrap_to_pi(gpsi - node.psi_t))

        for alpha in alphas:
            t_x = node.x_t + alpha * dx
            t_y = node.y_t + alpha * dy
            t_psi = wrap_to_pi(node.psi_t + alpha * dpsi)

            # Steering angle required for turn
            curv = dpsi / max(dist, 1.0)
            steer = np.arctan(self.l_t * curv)
            if abs(steer) > self.delta_max:
                return None

            # Integrate articulation angle
            dth = -(v_speed / self.l_a) * np.sin(curr_th) - (v_speed / self.l_t) * np.tan(steer) * (1.0 + (self.d_c / self.l_a) * np.cos(curr_th))
            curr_th = wrap_to_pi(curr_th + dth * sub_dt)
            curr_t += sub_dt

            if abs(curr_th) > self.theta_h_max:
                return None

            if not self.deck_map.is_pose_collision_free(t_x, t_y, t_psi, curr_th):
                return None

            pts.append((float(t_x), float(t_y), float(t_psi), float(curr_th), float(curr_t), float(v_speed), float(steer)))

        # Verify final articulation within tolerance
        if abs(wrap_to_pi(curr_th - gth)) > np.deg2rad(self.cfg.hitch_goal_tol_deg):
            return None

        return pts

    def search(
        self,
        start_state: VehicleState,
        goal_state: VehicleState,
        t0: float = 0.0,
    ) -> HybridAStarResult:
        """Execute 5D Spatiotemporal Hybrid A* search.
        
        Args:
            start_state: Initial tractor-aircraft VehicleState.
            goal_state: Target tractor-aircraft VehicleState.
            t0: Starting timestamp (s).
            
        Returns:
            HybridAStarResult containing discrete path waypoints and metadata.
        """
        start_time = time.perf_counter()

        # 1. Early Infeasibility Rejection (< 0.01 ms): Verify start pose collision-free & within bounds FIRST
        if not self.deck_map.is_pose_collision_free(start_state.x_t, start_state.y_t, start_state.psi_t, start_state.theta_h):
            return HybridAStarResult(
                success=False,
                timestamps=np.array([t0]),
                x=np.array([start_state.x_t]),
                y=np.array([start_state.y_t]),
                psi_t=np.array([start_state.psi_t]),
                theta_h=np.array([start_state.theta_h]),
                psi_a=np.array([wrap_to_pi(start_state.psi_t + start_state.theta_h)]),
                v=np.array([0.0]),
                delta_f=np.array([0.0]),
                nodes_expanded=0,
                primitives_evaluated=0,
                search_time_s=time.perf_counter() - start_time,
                total_cost=float("inf"),
                failure_reason="Start state is in collision or out of bounds",
            )

        # 2. Early Infeasibility Rejection (< 0.01 ms): Verify goal pose collision-free & within bounds
        if not self.deck_map.is_pose_collision_free(goal_state.x_t, goal_state.y_t, goal_state.psi_t, goal_state.theta_h):
            return HybridAStarResult(
                success=False,
                timestamps=np.array([t0]),
                x=np.array([start_state.x_t]),
                y=np.array([start_state.y_t]),
                psi_t=np.array([start_state.psi_t]),
                theta_h=np.array([start_state.theta_h]),
                psi_a=np.array([wrap_to_pi(start_state.psi_t + start_state.theta_h)]),
                v=np.array([0.0]),
                delta_f=np.array([0.0]),
                nodes_expanded=0,
                primitives_evaluated=0,
                search_time_s=time.perf_counter() - start_time,
                total_cost=float("inf"),
                failure_reason="Goal state is in collision or out of bounds",
            )

        goal_pose_3d = (goal_state.x_t, goal_state.y_t, goal_state.psi_t)
        goal_full = (goal_state.x_t, goal_state.y_t, goal_state.psi_t, goal_state.theta_h)

        # 3. Precompute 2D obstacle distance transform from goal ONLY after endpoints are verified
        self.obs_heuristic.compute(goal_state.x_t, goal_state.y_t)

        # Initialize root node
        h0 = self._evaluate_heuristic(start_state.x_t, start_state.y_t, start_state.psi_t, t0, goal_pose_3d)
        root = HybridAStarNode(
            x_t=start_state.x_t,
            y_t=start_state.y_t,
            psi_t=start_state.psi_t,
            theta_h=start_state.theta_h,
            t=float(t0),
            v_t=start_state.v_t,
            delta_f=start_state.delta_f,
            g_cost=0.0,
            h_cost=h0,
            f_cost=h0,
            node_id=0,
        )

        open_set: List[HybridAStarNode] = [root]
        closed_set: Dict[Tuple[int, int, int, int, int], float] = {}

        node_id_counter = 1
        nodes_expanded = 0
        primitives_evaluated = 0

        # Steering angle options in radians
        steer_options = [float(np.deg2rad(deg)) for deg in self.cfg.delta_levels_deg]
        speed_options = list(self.cfg.v_levels)
        if self.cfg.enable_reverse:
            speed_options.append(self.cfg.v_reverse)

        best_node: Optional[HybridAStarNode] = None
        analytic_solution: Optional[List[Tuple[float, float, float, float, float, float, float]]] = None

        while open_set and nodes_expanded < self.cfg.max_iterations:
            # Check timeout
            elapsed = time.perf_counter() - start_time
            if elapsed > self.cfg.timeout_s:
                break

            current = heapq.heappop(open_set)

            # Check if cell already visited with lower cost
            c_key = self._state_to_key(current.x_t, current.y_t, current.psi_t, current.theta_h, current.t)
            if c_key in closed_set and closed_set[c_key] <= current.g_cost:
                continue
            closed_set[c_key] = current.g_cost
            nodes_expanded += 1

            # Check Goal Condition
            dist_to_goal = np.hypot(current.x_t - goal_state.x_t, current.y_t - goal_state.y_t)
            yaw_err = abs(wrap_to_pi(current.psi_t - goal_state.psi_t))
            hitch_err = abs(wrap_to_pi(current.theta_h - goal_state.theta_h))

            if (dist_to_goal <= self.cfg.xy_goal_tol and
                yaw_err <= np.deg2rad(self.cfg.yaw_goal_tol_deg) and
                hitch_err <= np.deg2rad(self.cfg.hitch_goal_tol_deg)):
                best_node = current
                break

            # Attempt Analytic Expansion
            if nodes_expanded % self.cfg.analytic_interval == 0:
                sol = self._attempt_analytic_expansion(current, goal_full)
                if sol is not None:
                    best_node = current
                    analytic_solution = sol
                    break

            # Expand Motion Primitives
            for v in speed_options:
                for delta in steer_options:
                    # Do not turn sharply in reverse
                    if v < 0.0 and abs(delta) > np.deg2rad(15.0):
                        continue

                    primitives_evaluated += 1
                    sim_result = self._forward_integrate_primitive(
                        (current.x_t, current.y_t, current.psi_t, current.theta_h, current.t),
                        v, delta, current.delta_f
                    )
                    if sim_result is None:
                        continue

                    next_state, intermediates, step_cost = sim_result
                    nx, ny, npsi, nth, nt = next_state
                    new_key = self._state_to_key(nx, ny, npsi, nth, nt)

                    new_g = current.g_cost + step_cost
                    if new_key in closed_set and closed_set[new_key] <= new_g:
                        continue

                    new_h = self._evaluate_heuristic(nx, ny, npsi, nt, goal_pose_3d)
                    new_f = new_g + new_h

                    child_node = HybridAStarNode(
                        x_t=nx,
                        y_t=ny,
                        psi_t=npsi,
                        theta_h=nth,
                        t=nt,
                        v_t=v,
                        delta_f=delta,
                        g_cost=new_g,
                        h_cost=new_h,
                        f_cost=new_f,
                        parent=current,
                        intermediate_states=intermediates,
                        node_id=node_id_counter,
                    )
                    node_id_counter += 1
                    heapq.heappush(open_set, child_node)

        total_time = time.perf_counter() - start_time

        # If goal was reached or analytic solution found, reconstruct trajectory
        if best_node is not None:
            # 1. Backtrack tree path backward from best_node to root
            tree_pts: List[Tuple[float, float, float, float, float, float, float]] = []
            curr: Optional[HybridAStarNode] = best_node
            while curr is not None and curr.parent is not None:
                # Include intermediate sub-steps in reverse order during backtracking
                for pt in reversed(curr.intermediate_states):
                    tree_pts.append((pt[0], pt[1], pt[2], pt[3], pt[4], curr.v_t, curr.delta_f))
                curr = curr.parent

            # Add root state
            if curr is not None:
                tree_pts.append((curr.x_t, curr.y_t, curr.psi_t, curr.theta_h, curr.t, curr.v_t, curr.delta_f))

            # 2. Reverse tree path so it flows forward chronologically from root to best_node
            tree_pts.reverse()

            # 3. Append forward analytic solution (which already flows forward from best_node to goal)
            if analytic_solution:
                tree_pts.extend(analytic_solution)

            # Convert to arrays
            arr = np.array(tree_pts, dtype=np.float64)
            ts = arr[:, 4].copy()

            # 4. Strict timestamp monotonicity guard
            for k in range(1, len(ts)):
                if ts[k] <= ts[k - 1]:
                    ds = float(np.hypot(arr[k, 0] - arr[k - 1, 0], arr[k, 1] - arr[k - 1, 1]))
                    v_nom = max(abs(float(arr[k, 5])), 0.5)
                    dt_k = max(ds / v_nom, 0.05)
                    ts[k] = ts[k - 1] + dt_k

            xs = arr[:, 0]
            ys = arr[:, 1]
            psis = arr[:, 2]
            ths = arr[:, 3]
            vs = arr[:, 5]
            deltas = arr[:, 6]
            psi_a = np.array([wrap_to_pi(psis[i] + ths[i]) for i in range(len(psis))])

            return HybridAStarResult(
                success=True,
                timestamps=ts,
                x=xs,
                y=ys,
                psi_t=psis,
                theta_h=ths,
                psi_a=psi_a,
                v=vs,
                delta_f=deltas,
                nodes_expanded=nodes_expanded,
                primitives_evaluated=primitives_evaluated,
                search_time_s=total_time,
                total_cost=best_node.g_cost,
            )

        # Search failed
        return HybridAStarResult(
            success=False,
            timestamps=np.array([t0]),
            x=np.array([start_state.x_t]),
            y=np.array([start_state.y_t]),
            psi_t=np.array([start_state.psi_t]),
            theta_h=np.array([start_state.theta_h]),
            psi_a=np.array([wrap_to_pi(start_state.psi_t + start_state.theta_h)]),
            v=np.array([0.0]),
            delta_f=np.array([0.0]),
            nodes_expanded=nodes_expanded,
            primitives_evaluated=primitives_evaluated,
            search_time_s=total_time,
            total_cost=float("inf"),
            failure_reason=f"Search exhausted {nodes_expanded} nodes without reaching goal within timeout ({total_time:.2f}s)",
        )
