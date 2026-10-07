"""cts.control.baselines

Geometric and classical feedback baseline controllers:
- Stanley Steering Controller (with hitch articulation damping)
- Pure Pursuit Path Tracking Controller
- Classical Cascaded PID Controller

Used for baseline benchmark comparisons against NMPC and LTV-MPC under dynamic deck motion.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np

from cts.common.config import CTSConfig, TractorConfig
from cts.common.math_utils import wrap_to_pi
from cts.dynamics.multibody_dynamics import VehicleState, ControlAction
from cts.planning.minco_optimizer import PlannedTrajectory


@dataclass
class StanleyConfig:
    """Hyperparameters for Stanley tracking controller."""
    k_lat: float = 1.2             # Cross-track gain
    k_soft: float = 0.5            # Softening speed parameter (m/s)
    k_hitch: float = 0.4           # Trailer articulation damping gain
    k_pv: float = 1.0              # Longitudinal speed proportional gain
    delta_max_rad: float = 0.610865  # Maximum steering angle (35 deg)
    a_max: float = 1.0             # Maximum acceleration (m/s^2)
    slew_rate_max: float = 0.523599  # 30 deg/s


class StanleyController:
    """Stanley Steering Controller adapted for Towbarless Tractor-Aircraft System."""

    def __init__(
        self,
        config: Optional[StanleyConfig] = None,
        cts_config: Optional[CTSConfig] = None,
    ) -> None:
        self.config = config or StanleyConfig()
        self.cts_config = cts_config or CTSConfig()
        self.tractor = self.cts_config.tractor
        self.prev_steer: float = 0.0

    def compute_action(
        self,
        current_state: VehicleState,
        target_traj: PlannedTrajectory,
        t: float = 0.0,
        dt: float = 0.02,
    ) -> ControlAction:
        sample = target_traj.eval_at(t)
        x_r, y_r = sample["x"], sample["y"]
        psi_r = sample["psi_t"]
        v_r = sample["v"]
        theta_r = sample["theta_h"]

        # Cross-track error at front axle
        x_front = current_state.x_t + self.tractor.wheelbase * np.cos(current_state.psi_t)
        y_front = current_state.y_t + self.tractor.wheelbase * np.sin(current_state.psi_t)
        
        dx = x_front - x_r
        dy = y_front - y_r
        e_lat = -np.sin(psi_r) * dx + np.cos(psi_r) * dy
        e_psi = wrap_to_pi(current_state.psi_t - psi_r)

        # Stanley steering law
        v_eff = max(current_state.v_t, 0.0)
        steer_lat = np.arctan2(self.config.k_lat * e_lat, v_eff + self.config.k_soft)
        e_theta = wrap_to_pi(current_state.theta_h - theta_r)
        
        delta_raw = - e_psi - steer_lat - self.config.k_hitch * e_theta
        
        # Slew rate limit
        max_delta_change = self.config.slew_rate_max * dt
        delta_cmd = float(np.clip(delta_raw, self.prev_steer - max_delta_change, self.prev_steer + max_delta_change))
        delta_cmd = float(np.clip(delta_cmd, -self.config.delta_max_rad, self.config.delta_max_rad))
        steer_rate = float((delta_cmd - self.prev_steer) / max(dt, 1e-4))
        self.prev_steer = delta_cmd

        # Longitudinal acceleration
        a_cmd = float(np.clip(self.config.k_pv * (v_r - current_state.v_t), -self.config.a_max, self.config.a_max))

        return ControlAction(a_cmd=a_cmd, steer_angle_cmd=delta_cmd, steer_rate_cmd=steer_rate)


@dataclass
class PurePursuitConfig:
    """Hyperparameters for Pure Pursuit tracking controller."""
    k_dd: float = 1.0              # Speed-proportional lookahead gain (s)
    lookahead_min: float = 1.0     # Minimum lookahead distance (m)
    lookahead_max: float = 4.0     # Maximum lookahead distance (m)
    k_pv: float = 1.0              # Longitudinal speed proportional gain
    delta_max_rad: float = 0.610865  # Maximum steering angle (35 deg)
    a_max: float = 1.0             # Maximum acceleration (m/s^2)
    slew_rate_max: float = 0.523599  # 30 deg/s


class PurePursuitController:
    """Pure Pursuit path tracking controller."""

    def __init__(
        self,
        config: Optional[PurePursuitConfig] = None,
        cts_config: Optional[CTSConfig] = None,
    ) -> None:
        self.config = config or PurePursuitConfig()
        self.cts_config = cts_config or CTSConfig()
        self.tractor = self.cts_config.tractor
        self.prev_steer: float = 0.0

    def compute_action(
        self,
        current_state: VehicleState,
        target_traj: PlannedTrajectory,
        t: float = 0.0,
        dt: float = 0.02,
    ) -> ControlAction:
        sample_now = target_traj.eval_at(t)
        v_r = sample_now["v"]
        
        # Adaptive lookahead
        L_d = float(np.clip(self.config.k_dd * max(current_state.v_t, 0.5), self.config.lookahead_min, self.config.lookahead_max))
        
        # Target lookahead point along trajectory
        dt_look = L_d / max(v_r, 0.5)
        sample_target = target_traj.eval_at(min(t + dt_look, target_traj.timestamps[-1]))
        
        gx, gy = sample_target["x"], sample_target["y"]
        dx = gx - current_state.x_t
        dy = gy - current_state.y_t
        
        alpha = wrap_to_pi(np.arctan2(dy, dx) - current_state.psi_t)
        actual_L = np.hypot(dx, dy)
        if actual_L > 1e-3:
            kappa = 2.0 * np.sin(alpha) / actual_L
            delta_raw = np.arctan(kappa * self.tractor.wheelbase)
        else:
            delta_raw = 0.0

        max_delta_change = self.config.slew_rate_max * dt
        delta_cmd = float(np.clip(delta_raw, self.prev_steer - max_delta_change, self.prev_steer + max_delta_change))
        delta_cmd = float(np.clip(delta_cmd, -self.config.delta_max_rad, self.config.delta_max_rad))
        steer_rate = float((delta_cmd - self.prev_steer) / max(dt, 1e-4))
        self.prev_steer = delta_cmd

        a_cmd = float(np.clip(self.config.k_pv * (v_r - current_state.v_t), -self.config.a_max, self.config.a_max))
        return ControlAction(a_cmd=a_cmd, steer_angle_cmd=delta_cmd, steer_rate_cmd=steer_rate)


@dataclass
class PIDConfig:
    """Hyperparameters for Classical PID tracking controller."""
    kp_lat: float = 1.0
    ki_lat: float = 0.05
    kd_lat: float = 0.4
    kp_psi: float = 1.2
    kp_v: float = 1.5
    ki_v: float = 0.1
    delta_max_rad: float = 0.610865
    a_max: float = 1.0
    slew_rate_max: float = 0.523599


class PIDController:
    """PID Cross-Track and Heading Tracking Controller."""

    def __init__(
        self,
        config: Optional[PIDConfig] = None,
        cts_config: Optional[CTSConfig] = None,
    ) -> None:
        self.config = config or PIDConfig()
        self.cts_config = cts_config or CTSConfig()
        self.tractor = self.cts_config.tractor
        self.e_lat_int: float = 0.0
        self.e_lat_prev: float = 0.0
        self.e_v_int: float = 0.0
        self.prev_steer: float = 0.0

    def compute_action(
        self,
        current_state: VehicleState,
        target_traj: PlannedTrajectory,
        t: float = 0.0,
        dt: float = 0.02,
    ) -> ControlAction:
        sample = target_traj.eval_at(t)
        x_r, y_r = sample["x"], sample["y"]
        psi_r = sample["psi_t"]
        v_r = sample["v"]

        dx = current_state.x_t - x_r
        dy = current_state.y_t - y_r
        e_lat = -np.sin(psi_r) * dx + np.cos(psi_r) * dy
        e_psi = wrap_to_pi(current_state.psi_t - psi_r)

        self.e_lat_int += e_lat * dt
        self.e_lat_int = float(np.clip(self.e_lat_int, -2.0, 2.0))
        de_lat = (e_lat - self.e_lat_prev) / max(dt, 1e-4)
        self.e_lat_prev = e_lat

        u_lat = self.config.kp_lat * e_lat + self.config.ki_lat * self.e_lat_int + self.config.kd_lat * de_lat
        delta_raw = - u_lat - self.config.kp_psi * e_psi

        max_delta_change = self.config.slew_rate_max * dt
        delta_cmd = float(np.clip(delta_raw, self.prev_steer - max_delta_change, self.prev_steer + max_delta_change))
        delta_cmd = float(np.clip(delta_cmd, -self.config.delta_max_rad, self.config.delta_max_rad))
        steer_rate = float((delta_cmd - self.prev_steer) / max(dt, 1e-4))
        self.prev_steer = delta_cmd

        ev = v_r - current_state.v_t
        self.e_v_int += ev * dt
        self.e_v_int = float(np.clip(self.e_v_int, -1.0, 1.0))
        a_cmd = float(np.clip(self.config.kp_v * ev + self.config.ki_v * self.e_v_int, -self.config.a_max, self.config.a_max))

        return ControlAction(a_cmd=a_cmd, steer_angle_cmd=delta_cmd, steer_rate_cmd=steer_rate)
