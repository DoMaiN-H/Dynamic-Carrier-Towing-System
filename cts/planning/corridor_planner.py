"""cts.planning.corridor_planner

Master Spatiotemporal Corridor Planner integrating:
1. 5D Spatiotemporal Hybrid A* Search (Feature F11)
2. Safe Flight Corridor Generation (Feature F12)
3. Smooth C^2 MINCO Trajectory Optimization (Feature F13)
4. Dynamic Speed Profiler coupled with 6-DOF non-inertial deck acceleration field
conforming strictly to the PROJECT.md interface contract.
"""

from __future__ import annotations
import time
from typing import List, Tuple, Optional, Union
import numpy as np

from cts.common.config import CTSConfig, TractorConfig, AircraftConfig, CarrierConfig, TireConfig
from cts.common.math_utils import wrap_to_pi
from cts.dynamics.multibody_dynamics import VehicleState
from cts.environment.deck_env import DeckEnvironment
from cts.environment.deck_motion import DeckMotionSimulator
from cts.planning.deck_map import CarrierDeckMap, RunwayCenterline, DualBodyModel, DeckGridMap
from cts.planning.kinodynamic_limits import KinodynamicLimits
from cts.planning.hybrid_a_star import SpatiotemporalHybridAStar, HybridAStarConfig, HybridAStarResult
from cts.planning.safe_corridor import SafeFlightCorridorGenerator, SafeFlightCorridorConfig, Polytope
from cts.planning.minco_optimizer import MincoTrajectoryOptimizer, MincoTrajectoryConfig, PlannedTrajectory


class DynamicSpeedProfiler:
    """Profiles and smooths forward speed along geometric path p(s) subject to 6-DOF deck dynamics."""

    def __init__(
        self,
        kin_limits: KinodynamicLimits,
        v_max: float = 2.0,
        a_max: float = 1.0,
        mu: Optional[float] = None,
    ) -> None:
        self.limits = kin_limits
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.mu = mu

    def profile_speed(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        psis: np.ndarray,
        kappas: np.ndarray,
        t0: float,
        sea_state: DeckMotionSimulator,
        v_start: float,
        v_goal: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform forward-backward reachability speed profiling coupled to 6-DOF deck acceleration.
        
        Returns:
            timestamps: (N,) updated monotonically increasing timestamps.
            vs: (N,) dynamically safe smoothed speeds.
            accs: (N,) tangential longitudinal accelerations.
        """
        N = len(xs)
        dx = np.diff(xs)
        dy = np.diff(ys)
        ds = np.hypot(dx, dy)
        ds = np.maximum(ds, 1e-4)

        # Initial time parameterization guess (~1.0 m/s)
        timestamps = np.zeros(N, dtype=np.float64)
        timestamps[0] = t0
        for k in range(N - 1):
            timestamps[k + 1] = timestamps[k] + ds[k] / 1.0

        v_profile = np.full(N, self.v_max, dtype=np.float64)

        # 4-iteration Picard contraction for time-coupled acceleration queries with 50% under-relaxation
        for _ in range(4):
            # 1. Compute dynamic velocity ceiling at each point
            v_ceiling = np.zeros(N, dtype=np.float64)
            for k in range(N):
                tk = timestamps[k]
                pos = np.array([xs[k], ys[k], 0.0], dtype=np.float64)
                vel = np.array([v_profile[k] * np.cos(psis[k]), v_profile[k] * np.sin(psis[k]), 0.0], dtype=np.float64)

                apparent = sea_state.compute_non_inertial_acceleration(pos, vel, tk)
                geff_z = float(-apparent.a_apparent_total[2])
                a_deck_y = float(apparent.a_apparent_total[1])

                crit_roll = self.limits.compute_critical_rollover_acceleration(geff_z)
                crit_slip = self.limits.compute_critical_slip_acceleration(geff_z, a_deck_y, mu=self.mu)

                # Available lateral turning acceleration
                a_y_max = max(0.0, min(crit_roll, crit_slip))
                kap = abs(kappas[k])

                if geff_z <= 0.0:
                    # True heave decompression: zero traction normal force
                    v_ceiling[k] = 0.0
                elif a_y_max <= 1e-4:
                    if kap > 1e-4:
                        v_ceiling[k] = 0.0
                    else:
                        v_ceiling[k] = min(self.v_max, 0.10)
                else:
                    # Conservative turning speed ceiling
                    v_lat = np.sqrt(0.95 * a_y_max / max(kap, 1e-4))
                    v_ceiling[k] = min(self.v_max, v_lat)

                # Continuous rollover ceiling for heave decompression
                if kap > 1e-4 and geff_z < 1.0:
                    if geff_z <= 0.05:
                        v_ceiling[k] = 0.0
                    else:
                        crit_rollover = self.limits.k_rollover * self.limits.ltr_limit * max(geff_z, 0.0)
                        v_ceiling[k] = min(v_ceiling[k], np.sqrt(0.95 * crit_rollover / kap))

            # Enforce boundary speeds
            v_ceiling[0] = min(v_ceiling[0], max(v_start, 0.0))
            v_ceiling[-1] = min(v_ceiling[-1], max(v_goal, 0.0))

            # 2. Backward Pass (Braking reachability)
            v_bwd = np.copy(v_ceiling)
            for k in range(N - 2, -1, -1):
                v_max_brake = np.sqrt(v_bwd[k + 1]**2 + 2.0 * self.a_max * ds[k])
                v_bwd[k] = min(v_bwd[k], v_max_brake)

            # 3. Forward Pass (Traction reachability)
            v_fwd = np.copy(v_bwd)
            for k in range(0, N - 1):
                v_max_acc = np.sqrt(v_fwd[k]**2 + 2.0 * self.a_max * ds[k])
                v_fwd[k + 1] = min(v_fwd[k + 1], v_max_acc)

            v_profile = v_fwd

            # 4. Update timestamps with 50% under-relaxation damping
            timestamps_prev = np.copy(timestamps)
            timestamps_new = np.zeros(N, dtype=np.float64)
            timestamps_new[0] = t0
            for k in range(N - 1):
                v_avg = max(0.5 * (v_profile[k] + v_profile[k + 1]), 0.05)
                timestamps_new[k + 1] = timestamps_new[k] + ds[k] / v_avg
            timestamps = 0.5 * timestamps_prev + 0.5 * timestamps_new

        # 5. Final pass: Multi-pass decompression verification, reachability and timestamp sync
        for _ in range(2):
            for k in range(N - 1):
                v_avg = max(0.5 * (v_profile[k] + v_profile[k + 1]), 0.05)
                timestamps[k + 1] = timestamps[k] + ds[k] / v_avg

            for k in range(N):
                tk = timestamps[k]
                pos = np.array([xs[k], ys[k], 0.0], dtype=np.float64)
                vel = np.array([v_profile[k] * np.cos(psis[k]), v_profile[k] * np.sin(psis[k]), 0.0], dtype=np.float64)
                app = sea_state.compute_non_inertial_acceleration(pos, vel, tk)
                geff_z = float(-app.a_apparent_total[2])
                kap = abs(kappas[k])

                if geff_z <= 0.0:
                    v_profile[k] = 0.0
                elif kap > 1e-4 and geff_z < 1.0:
                    if geff_z <= 0.05:
                        v_profile[k] = 0.0
                    else:
                        crit_rollover = self.limits.k_rollover * self.limits.ltr_limit * max(geff_z, 0.0)
                        v_profile[k] = min(v_profile[k], np.sqrt(0.95 * crit_rollover / kap))

            # Re-run backward-forward reachability pass after applying decompression clamping
            for k in range(N - 2, -1, -1):
                v_profile[k] = min(v_profile[k], np.sqrt(v_profile[k + 1]**2 + 2.0 * self.a_max * ds[k]))
            for k in range(0, N - 1):
                v_profile[k + 1] = min(v_profile[k + 1], np.sqrt(v_profile[k]**2 + 2.0 * self.a_max * ds[k]))

        for k in range(N - 1):
            v_avg = max(0.5 * (v_profile[k] + v_profile[k + 1]), 0.05)
            timestamps[k + 1] = timestamps[k] + ds[k] / v_avg

        # Tangential accelerations
        accs = np.zeros(N, dtype=np.float64)
        for k in range(N - 1):
            accs[k] = (v_profile[k + 1]**2 - v_profile[k]**2) / (2.0 * ds[k])
        accs[-1] = accs[-2] if N > 1 else 0.0

        return timestamps, v_profile, accs


class SpatiotemporalCorridorPlanner:
    """Master planner matching PROJECT.md interface contract:
    
    plan(start_state: VehicleState, goal_state: VehicleState, deck_env: DeckEnvironment, sea_state: DeckMotionSimulator, t0: float) -> PlannedTrajectory
    """

    def __init__(
        self,
        config: Optional[CTSConfig] = None,
        hybrid_a_star_cfg: Optional[HybridAStarConfig] = None,
        corridor_cfg: Optional[SafeFlightCorridorConfig] = None,
        minco_cfg: Optional[MincoTrajectoryConfig] = None,
    ) -> None:
        self.config = config or CTSConfig()
        self.t_cfg = self.config.tractor
        self.a_cfg = self.config.aircraft
        self.tire_cfg = self.config.tire
        self.c_cfg = self.config.carrier

        self.h_cfg = hybrid_a_star_cfg or HybridAStarConfig()
        self.sfc_cfg = corridor_cfg or SafeFlightCorridorConfig()
        self.minco_cfg = minco_cfg or MincoTrajectoryConfig()

        self.optimizer = MincoTrajectoryOptimizer(
            config=self.minco_cfg,
            tractor_cfg=self.t_cfg,
            aircraft_cfg=self.a_cfg,
        )

        self.kin_limits = KinodynamicLimits(
            tractor_cfg=self.t_cfg,
            aircraft_cfg=self.a_cfg,
            tire_cfg=self.tire_cfg,
            carrier_cfg=self.c_cfg,
            a_y_min_floor=0.0,
        )

        self.speed_profiler = DynamicSpeedProfiler(
            kin_limits=self.kin_limits,
            v_max=self.t_cfg.v_max,
            a_max=self.t_cfg.a_max,
            mu=self.tire_cfg.mu_dry,
        )

    def plan(
        self,
        start_state: VehicleState,
        goal_state: VehicleState,
        deck_env: DeckEnvironment,
        sea_state: DeckMotionSimulator,
        t0: float = 0.0,
    ) -> PlannedTrajectory:
        """Plan a kinodynamically admissible, collision-free, C^2-smooth reference trajectory."""
        t_start = time.perf_counter()

        # 1. Early Boundary & Infeasibility Check
        goal_pos = np.array([goal_state.x_t, goal_state.y_t], dtype=np.float64)
        if deck_env.island.polygon.contains_point(goal_pos) or not deck_env.outer_boundary.contains_point(goal_pos):
            return self._build_stationary_holding_trajectory(start_state, t0)

        start_pos = np.array([start_state.x_t, start_state.y_t], dtype=np.float64)
        if deck_env.island.polygon.contains_point(start_pos) or not deck_env.outer_boundary.contains_point(start_pos):
            return self._build_stationary_holding_trajectory(start_state, t0)

        # 2. Build CarrierDeckMap with EDT spatial index
        grid_map = DeckGridMap(deck_env=deck_env, resolution=self.h_cfg.xy_resolution)
        runway = RunwayCenterline(
            start_point=np.array([25.0, 42.52]),
            end_point=np.array([215.0, 89.89]),
            touchdown_point=np.array([55.0, 50.0]),
        )
        dual_body = DualBodyModel(self.t_cfg, self.a_cfg)
        deck_map = CarrierDeckMap(
            deck_env=deck_env,
            grid_map=grid_map,
            runway=runway,
            dual_body=dual_body,
        )

        # Check collision on start and goal with dual-body footprint
        if not deck_map.is_pose_collision_free(start_state.x_t, start_state.y_t, start_state.psi_t, start_state.theta_h):
            return self._build_stationary_holding_trajectory(start_state, t0)
        if not deck_map.is_pose_collision_free(goal_state.x_t, goal_state.y_t, goal_state.psi_t, goal_state.theta_h):
            return self._build_stationary_holding_trajectory(start_state, t0)

        # 3. Execute 5D Spatiotemporal Hybrid A* Search
        searcher = SpatiotemporalHybridAStar(
            deck_map=deck_map,
            sea_state=sea_state,
            config=self.h_cfg,
            tractor_cfg=self.t_cfg,
            aircraft_cfg=self.a_cfg,
            tire_cfg=self.tire_cfg,
        )
        search_res = searcher.search(start_state, goal_state, t0=t0)

        if not search_res.success:
            # Fallback direct path only if strictly collision-free
            waypoints, timestamps = self._fallback_waypoints(start_state, goal_state, deck_map, t0)
            if waypoints is None:
                return self._build_stationary_holding_trajectory(start_state, t0)
        else:
            waypoints, timestamps = self._subsample_path(search_res)

        # 4. Generate Safe Flight Corridors
        sfc_gen = SafeFlightCorridorGenerator(deck_env=deck_env, config=self.sfc_cfg)
        extra_obs = [reg.polygon for reg in deck_env.parking_regions.values()] + [reg.polygon for reg in deck_env.support_regions.values()]
        corridors = sfc_gen.generate(waypoints, timestamps=timestamps, extra_obstacles=extra_obs)

        segment_durations = np.diff(timestamps)
        segment_durations = np.maximum(segment_durations, 0.5)

        # 5. Solve C^2 MINCO Trajectory Optimization
        raw_traj = self.optimizer.optimize(
            waypoints=waypoints,
            times=segment_durations,
            corridors=corridors,
            start_state=start_state,
            goal_state=goal_state,
            dynamic_limits=self.kin_limits,
            deck_motion=sea_state,
            t0=t0,
        )

        # 6. Apply Dynamic Speed Profiling Coupled with 6-DOF Deck Motion
        new_ts, new_vs, new_as = self.speed_profiler.profile_speed(
            xs=raw_traj.x,
            ys=raw_traj.y,
            psis=raw_traj.psi_t,
            kappas=raw_traj.kappa,
            t0=t0,
            sea_state=sea_state,
            v_start=start_state.v_t,
            v_goal=goal_state.v_t,
        )

        # 7. Forward-Integrate Articulation Hitch Angle Without Artificial Clipping
        l_a = float(self.a_cfg.wheelbase)
        l_t = float(self.t_cfg.wheelbase)
        d_c = float(self.t_cfg.hitch_offset)
        N_pts = len(new_ts)
        new_ths = np.zeros(N_pts, dtype=np.float64)
        new_ths[0] = float(start_state.theta_h)

        for k in range(N_pts - 1):
            h = float(new_ts[k + 1] - new_ts[k])
            vk = new_vs[k]
            kapk = raw_traj.kappa[k]
            tan_delta = l_t * float(np.clip(kapk, -self.minco_cfg.kappa_max, self.minco_cfg.kappa_max))

            def dth_dt(th_val):
                trailer = -(vk / l_a) * np.sin(th_val)
                steer = (vk / l_t) * tan_delta * (1.0 + (d_c / l_a) * np.cos(th_val))
                return trailer - steer

            th_c = new_ths[k]
            k1 = dth_dt(th_c)
            k2 = dth_dt(th_c + 0.5 * h * k1)
            k3 = dth_dt(th_c + 0.5 * h * k2)
            k4 = dth_dt(th_c + h * k3)
            # Natural physical integration: no artificial clipping!
            new_ths[k + 1] = th_c + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

        return PlannedTrajectory(
            timestamps=new_ts - t0,  # relative trajectory timestamps matching PROJECT.md
            x=raw_traj.x,
            y=raw_traj.y,
            psi_t=raw_traj.psi_t,
            theta_h=new_ths,
            v=new_vs,
            a=new_as,
            kappa=raw_traj.kappa,
            corridors=raw_traj.corridors,
        )

    def _build_stationary_holding_trajectory(
        self,
        state: VehicleState,
        t0: float,
        duration: float = 3.0,
        num_pts: int = 21,
    ) -> PlannedTrajectory:
        """Construct safe zero-velocity holding trajectory for infeasible/emergency requests."""
        ts = np.linspace(0.0, duration, num_pts)
        dummy_poly = Polytope(
            A=np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
            b=np.array([state.x_t + 10.0, -state.x_t + 10.0, state.y_t + 10.0, -state.y_t + 10.0]),
        )
        return PlannedTrajectory(
            timestamps=ts,
            x=np.full(num_pts, state.x_t),
            y=np.full(num_pts, state.y_t),
            psi_t=np.full(num_pts, state.psi_t),
            theta_h=np.full(num_pts, state.theta_h),
            v=np.zeros(num_pts),
            a=np.zeros(num_pts),
            kappa=np.zeros(num_pts),
            corridors=[dummy_poly],
        )

    def _subsample_path(
        self,
        search_res: HybridAStarResult,
        target_step_dist: float = 2.50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Subsample dense Hybrid A* path into clean waypoints for polynomial optimization."""
        xs = search_res.x
        ys = search_res.y
        ts = search_res.timestamps
        N = len(xs)

        if N <= 3:
            return np.column_stack([xs, ys]), ts

        selected_indices = [0]
        accum_dist = 0.0

        for i in range(1, N - 1):
            d = float(np.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))
            accum_dist += d
            if accum_dist >= target_step_dist:
                selected_indices.append(i)
                accum_dist = 0.0

        if selected_indices[-1] != N - 1:
            selected_indices.append(N - 1)

        sub_x = xs[selected_indices]
        sub_y = ys[selected_indices]
        sub_t = ts[selected_indices]

        for k in range(1, len(sub_t)):
            if sub_t[k] <= sub_t[k - 1] + 0.2:
                sub_t[k] = sub_t[k - 1] + 0.5

        return np.column_stack([sub_x, sub_y]), sub_t

    def _fallback_waypoints(
        self,
        start_state: VehicleState,
        goal_state: VehicleState,
        deck_map: CarrierDeckMap,
        t0: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Fallback direct connection when search terminates without hitting exact tolerance."""
        p_start = np.array([start_state.x_t, start_state.y_t], dtype=np.float64)
        p_goal = np.array([goal_state.x_t, goal_state.y_t], dtype=np.float64)
        dist = float(np.linalg.norm(p_goal - p_start))

        num_segs = max(2, int(np.ceil(dist / 4.0)))
        alphas = np.linspace(0.0, 1.0, num_segs + 1)
        pts = np.array([(1.0 - a) * p_start + a * p_goal for a in alphas], dtype=np.float64)

        # Collision check each fallback waypoint
        for pt in pts:
            if not deck_map.is_pose_collision_free(pt[0], pt[1], start_state.psi_t, 0.0):
                return None, None

        seg_times = np.maximum(dist / (num_segs * 1.0), 1.0)
        ts = t0 + np.cumsum(np.concatenate([[0.0], np.full(num_segs, seg_times)]))
        return pts, ts
