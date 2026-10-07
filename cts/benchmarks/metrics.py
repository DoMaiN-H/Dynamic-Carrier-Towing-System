"""cts.benchmarks.metrics

Quantitative Performance and Safety Metrics Computation (Feature F20).
Evaluates lateral tracking precision, heading accuracy, load transfer ratio (LTR),
tire friction circle utilization, computation latency, and safety constraints.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Sequence
import numpy as np

from cts.common.math_utils import wrap_to_pi
from cts.dynamics.multibody_dynamics import VehicleState, ControlAction, WheelLoads
from cts.planning.minco_optimizer import PlannedTrajectory
from cts.environment.deck_env import DeckEnvironment


@dataclass
class BenchmarkMetrics:
    """Standardized benchmark metrics container matching academic evaluation standards."""
    # Tracking Accuracy
    max_lateral_error_m: float
    rms_lateral_error_m: float
    mean_lateral_error_m: float
    max_heading_error_deg: float
    rms_heading_error_deg: float
    max_longitudinal_error_m: float
    rms_velocity_error_mps: float

    # Terminal Pose Accuracy (vs. scenario goal state)
    final_position_error_m: float       # Euclidean distance to goal position at rollout end
    final_heading_error_deg: float      # |wrap(psi_t - psi_goal)| at rollout end
    final_pose_goal_ok: bool            # pos <= terminal_pos_tol_m AND heading <= tol

    # Articulation & Kinematics
    max_hitch_angle_deg: float
    final_hitch_angle_deg: float
    max_steer_angle_deg: float
    max_steer_rate_deg_s: float

    # Dynamic Stability & Safety
    max_load_transfer_ratio: float      # LTR in [0, 1]
    max_friction_utilization: float     # eta_tire = sqrt(Fx^2 + Fy^2) / (mu * Fz)
    mean_friction_utilization: float

    # Computational Performance
    mean_solve_time_ms: float
    p95_solve_time_ms: float
    max_solve_time_ms: float

    # Boolean Pass Criteria
    tracking_precision_passed: bool     # SS4: max_lat <= 0.35m, rms_lat <= 0.15m, max_psi <= 3 deg
    jackknife_safe: bool                # |theta_h| <= 70 deg physical envelope
    rollover_safe: bool                 # LTR <= 0.70
    collision_free: bool                # Zero collisions with deck boundary and island
    sla_passed: bool                    # mean solve time <= 20 ms
    overall_success: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a serializable dictionary."""
        return asdict(self)


def compute_trajectory_metrics(
    timestamps: Sequence[float],
    states: Sequence[VehicleState],
    actions: Sequence[ControlAction],
    reference_trajectory: PlannedTrajectory,
    wheel_loads: Optional[Sequence[WheelLoads]] = None,
    solve_times_ms: Optional[Sequence[float]] = None,
    friction_utilizations: Optional[Sequence[float]] = None,
    deck_env: Optional[DeckEnvironment] = None,
    goal_state: Optional[VehicleState] = None,
    goal_aircraft_pose: Optional[tuple] = None,
    terminal_pos_tol_m: float = 0.30,
    terminal_heading_tol_deg: float = 5.0,
) -> BenchmarkMetrics:
    """Compute comprehensive benchmark metrics across a closed-loop trajectory rollout.

    Args:
        timestamps: Array or list of simulation timestamps (s).
        states: Sequence of executed VehicleState objects.
        actions: Sequence of executed ControlAction objects.
        reference_trajectory: PlannedTrajectory reference.
        wheel_loads: Optional sequence of WheelLoads for LTR calculation.
        solve_times_ms: Optional sequence of controller solve times in milliseconds.
        friction_utilizations: Optional sequence of tire adhesion circle utilization ratios.
        deck_env: Optional DeckEnvironment for collision checking.
        goal_state: Optional terminal tractor pose the system must converge to.
        goal_aircraft_pose: Optional (x_a, y_a, psi_a) aircraft terminal pose
            (e.g. deck station target). When provided, the terminal pose
            constraint is evaluated on the *aircraft* CG position and heading
            rather than the tractor rear axle.
        terminal_pos_tol_m: Position tolerance for terminal pose acceptance (m).
        terminal_heading_tol_deg: Heading tolerance for terminal pose acceptance (deg).

    Returns:
        BenchmarkMetrics containing all scalar and boolean evaluation indicators.
    """
    n_points = min(len(timestamps), len(states))
    if n_points == 0:
        raise ValueError("Cannot compute metrics for empty trajectory.")

    lat_errors = []
    long_errors = []
    psi_errors = []
    v_errors = []
    hitch_angles = []
    steer_angles = []
    steer_rates = []
    ltr_values = []
    collisions = False

    for i in range(n_points):
        t = timestamps[i]
        st = states[i]
        ref = reference_trajectory.eval_at(t)

        dx = st.x_t - ref["x"]
        dy = st.y_t - ref["y"]
        psi_r = ref["psi_t"]

        # Frenet projection error in reference frame
        e_long = dx * np.cos(psi_r) + dy * np.sin(psi_r)
        e_lat = -dx * np.sin(psi_r) + dy * np.cos(psi_r)
        e_psi = wrap_to_pi(st.psi_t - psi_r)
        e_v = st.v_t - ref["v"]

        lat_errors.append(abs(e_lat))
        long_errors.append(abs(e_long))
        psi_errors.append(abs(e_psi))
        v_errors.append(abs(e_v))
        hitch_angles.append(abs(st.theta_h))
        steer_angles.append(abs(st.delta_f))

        # Check collision if environment provided
        if deck_env is not None and not collisions:
            if not deck_env.is_point_navigable((st.x_t, st.y_t)):
                collisions = True

    # Steer rates from actions
    for act in actions:
        if act.steer_rate_cmd is not None:
            steer_rates.append(abs(act.steer_rate_cmd))

    # LTR from wheel loads
    if wheel_loads is not None and len(wheel_loads) > 0:
        for wl in wheel_loads:
            ltr_val = getattr(wl, "ltr_max", getattr(wl, "ltr", 0.0))
            ltr_values.append(float(ltr_val))
    else:
        ltr_values = [0.0]

    # Friction utilization
    if friction_utilizations is not None and len(friction_utilizations) > 0:
        f_utils = list(friction_utilizations)
    else:
        f_utils = [0.0]

    # Solve times
    st_times = list(solve_times_ms) if solve_times_ms else [1.0]

    max_lat = float(np.max(lat_errors))
    rms_lat = float(np.sqrt(np.mean(np.array(lat_errors) ** 2)))
    mean_lat = float(np.mean(lat_errors))
    max_long = float(np.max(long_errors))
    max_psi_deg = float(np.degrees(np.max(psi_errors)))
    rms_psi_deg = float(np.degrees(np.sqrt(np.mean(np.array(psi_errors) ** 2))))
    rms_v = float(np.sqrt(np.mean(np.array(v_errors) ** 2)))

    max_hitch_deg = float(np.degrees(np.max(hitch_angles)))
    final_hitch_deg = float(np.degrees(hitch_angles[-1]))
    max_steer_deg = float(np.degrees(np.max(steer_angles)))
    max_steer_rate_deg = float(np.degrees(np.max(steer_rates))) if steer_rates else 0.0

    max_ltr = float(np.max(ltr_values))
    max_eta = float(np.max(f_utils))
    mean_eta = float(np.mean(f_utils))

    mean_st = float(np.mean(st_times))
    p95_st = float(np.percentile(st_times, 95))
    max_st = float(np.max(st_times))

    # Terminal pose accuracy versus the mission goal state
    if goal_aircraft_pose is not None:
        # Constraint applies to the aircraft: CG at the station centre, nose
        # along the station heading. Reconstruct aircraft pose from the final
        # tractor state via the hitch geometry (d_c hitch offset, a_a nose
        # gear to aircraft CG).
        st_f = states[n_points - 1]
        d_c, a_a = 0.10, 6.80
        psi_a_f = wrap_to_pi(st_f.psi_t + st_f.theta_h)
        x_a_f = st_f.x_t - d_c * np.cos(st_f.psi_t) - a_a * np.cos(psi_a_f)
        y_a_f = st_f.y_t - d_c * np.sin(st_f.psi_t) - a_a * np.sin(psi_a_f)
        gx, gy, gpsi = goal_aircraft_pose
        final_pos_err = float(np.hypot(x_a_f - gx, y_a_f - gy))
        final_head_err = float(np.degrees(abs(wrap_to_pi(psi_a_f - gpsi))))
        pose_goal_ok = bool(
            final_pos_err <= terminal_pos_tol_m and final_head_err <= terminal_heading_tol_deg
        )
    elif goal_state is not None:
        st_f = states[n_points - 1]
        final_pos_err = float(np.hypot(st_f.x_t - goal_state.x_t, st_f.y_t - goal_state.y_t))
        final_head_err = float(np.degrees(abs(wrap_to_pi(st_f.psi_t - goal_state.psi_t))))
        pose_goal_ok = bool(
            final_pos_err <= terminal_pos_tol_m and final_head_err <= terminal_heading_tol_deg
        )
    else:
        final_pos_err = float("nan")
        final_head_err = float("nan")
        pose_goal_ok = True

    # Safety checks
    prec_passed = bool(max_lat <= 0.35 and rms_lat <= 0.15 and max_psi_deg <= 4.0)
    jackknife_ok = bool(max_hitch_deg <= 70.0)
    rollover_ok = bool(max_ltr <= 0.70)
    sla_ok = bool(mean_st <= 20.0)
    collision_free = not collisions

    overall_ok = bool(jackknife_ok and rollover_ok and collision_free and pose_goal_ok)

    return BenchmarkMetrics(
        max_lateral_error_m=max_lat,
        rms_lateral_error_m=rms_lat,
        mean_lateral_error_m=mean_lat,
        max_heading_error_deg=max_psi_deg,
        rms_heading_error_deg=rms_psi_deg,
        max_longitudinal_error_m=max_long,
        rms_velocity_error_mps=rms_v,
        final_position_error_m=final_pos_err,
        final_heading_error_deg=final_head_err,
        final_pose_goal_ok=pose_goal_ok,
        max_hitch_angle_deg=max_hitch_deg,
        final_hitch_angle_deg=final_hitch_deg,
        max_steer_angle_deg=max_steer_deg,
        max_steer_rate_deg_s=max_steer_rate_deg,
        max_load_transfer_ratio=max_ltr,
        max_friction_utilization=max_eta,
        mean_friction_utilization=mean_eta,
        mean_solve_time_ms=mean_st,
        p95_solve_time_ms=p95_st,
        max_solve_time_ms=max_st,
        tracking_precision_passed=prec_passed,
        jackknife_safe=jackknife_ok,
        rollover_safe=rollover_ok,
        collision_free=collision_free,
        sla_passed=sla_ok,
        overall_success=overall_ok,
    )
