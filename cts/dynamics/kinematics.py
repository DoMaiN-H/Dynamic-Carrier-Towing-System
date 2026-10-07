"""cts.dynamics.kinematics

Feature F6: Towbarless Tractor-Aircraft Kinematics.
Nonholonomic kinematic equations for Ackermann front steering, turntable clamp hitch articulation,
and analytical dual-body pose and landing gear derivation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Union
import numpy as np

from cts.common.config import TractorConfig, AircraftConfig, CTSConfig
from cts.common.math_utils import wrap_to_pi


@dataclass(frozen=True)
class VehiclePose:
    """Complete 2D poses of both tractor, aircraft, hitch, and landing gear in Deck Frame {D}."""
    x_t: float          # Tractor rear axle center X (m)
    y_t: float          # Tractor rear axle center Y (m)
    psi_t: float        # Tractor heading angle (rad)
    x_a: float          # Aircraft CG X (m)
    y_a: float          # Aircraft CG Y (m)
    psi_a: float        # Aircraft heading angle (rad)
    theta_h: float      # Hitch articulation angle theta_h = psi_a - psi_t (rad)
    x_h: float          # Turntable cradle hitch point X (m)
    y_h: float          # Turntable cradle hitch point Y (m)
    x_m: float          # Aircraft main gear axle center X (m)
    y_m: float          # Aircraft main gear axle center Y (m)
    x_ml: float         # Aircraft main gear left tire contact X (m)
    y_ml: float         # Aircraft main gear left tire contact Y (m)
    x_mr: float         # Aircraft main gear right tire contact X (m)
    y_mr: float         # Aircraft main gear right tire contact Y (m)


class AckermannTurntableKinematics:
    """Nonholonomic kinematic model for towbarless tractor and clamped aircraft."""

    def __init__(
        self,
        tractor_cfg: Optional[TractorConfig] = None,
        aircraft_cfg: Optional[AircraftConfig] = None,
        hitch_offset_dc: Optional[float] = None,
    ) -> None:
        self.tractor = tractor_cfg or TractorConfig()
        self.aircraft = aircraft_cfg or AircraftConfig()
        self.l_t = float(self.tractor.wheelbase)          # 1.80 m
        self.l_a = float(self.aircraft.wheelbase)         # 8.00 m
        self.a_a = float(self.aircraft.cg_to_nose)        # 6.80 m
        self.b_a = float(self.aircraft.cg_to_main)        # 1.20 m
        self.b_t = float(self.tractor.cg_to_rear)         # 0.85 m
        self.a_t = float(self.tractor.cg_to_front)        # 0.95 m
        self.W_r = float(self.aircraft.main_track)        # 3.10 m
        self.d_c = float(self.tractor.hitch_offset if hitch_offset_dc is None else hitch_offset_dc)  # 0.10 m

    def compute_hitch_position(self, x_t: float, y_t: float, psi_t: float) -> Tuple[float, float]:
        """Compute the position of the turntable hitch point H in Deck Frame {D}."""
        x_h = x_t - self.d_c * np.cos(psi_t)
        y_h = y_t - self.d_c * np.sin(psi_t)
        return float(x_h), float(y_h)

    def compute_aircraft_pose(
        self,
        x_t: float,
        y_t: float,
        psi_t: float,
        theta_h: float,
    ) -> Tuple[float, float, float]:
        """Derive aircraft CG position [x_a, y_a] and heading psi_a from tractor pose and hitch angle.

        Returns:
            (x_a, y_a, psi_a) in meters and radians.
        """
        psi_a = wrap_to_pi(psi_t + theta_h)
        x_h, y_h = self.compute_hitch_position(x_t, y_t, psi_t)
        x_a = x_h - self.a_a * np.cos(psi_a)
        y_a = y_h - self.a_a * np.sin(psi_a)
        return float(x_a), float(y_a), float(psi_a)

    def compute_main_gear_positions(
        self,
        x_t: float,
        y_t: float,
        psi_t: float,
        theta_h: float,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Compute positions of main gear axle center, left wheel, and right wheel.

        Returns:
            ((x_m, y_m), (x_ml, y_ml), (x_mr, y_mr))
        """
        psi_a = wrap_to_pi(psi_t + theta_h)
        x_h, y_h = self.compute_hitch_position(x_t, y_t, psi_t)
        x_m = x_h - self.l_a * np.cos(psi_a)
        y_m = y_h - self.l_a * np.sin(psi_a)

        # Lateral unit normal n_a = [-sin(psi_a), cos(psi_a)] (leftward in body frame)
        half_track = 0.5 * self.W_r
        x_ml = x_m - half_track * np.sin(psi_a)
        y_ml = y_m + half_track * np.cos(psi_a)
        x_mr = x_m + half_track * np.sin(psi_a)
        y_mr = y_m - half_track * np.cos(psi_a)

        return (float(x_m), float(y_m)), (float(x_ml), float(y_ml)), (float(x_mr), float(y_mr))

    def compute_all_poses(
        self,
        x_t: float,
        y_t: float,
        psi_t: float,
        theta_h: float,
    ) -> VehiclePose:
        """Compute full geometric pose container including hitch point and landing gear."""
        psi_a = wrap_to_pi(psi_t + theta_h)
        x_h, y_h = self.compute_hitch_position(x_t, y_t, psi_t)
        x_a = x_h - self.a_a * np.cos(psi_a)
        y_a = y_h - self.a_a * np.sin(psi_a)
        (x_m, y_m), (x_ml, y_ml), (x_mr, y_mr) = self.compute_main_gear_positions(x_t, y_t, psi_t, theta_h)

        return VehiclePose(
            x_t=float(x_t), y_t=float(y_t), psi_t=float(psi_t),
            x_a=float(x_a), y_a=float(y_a), psi_a=float(psi_a),
            theta_h=float(theta_h), x_h=float(x_h), y_h=float(y_h),
            x_m=x_m, y_m=y_m, x_ml=x_ml, y_ml=y_ml, x_mr=x_mr, y_mr=y_mr,
        )

    def hitch_articulation_rate(
        self,
        v_t: float,
        delta_f: float,
        theta_h: float,
    ) -> float:
        """Evaluate d(theta_h)/dt under nonholonomic rolling constraints.

        theta_h = psi_a - psi_t.
        dot{theta}_h = - (v_t / l_a) * sin(theta_h) - (v_t / l_t) * tan(delta_f) * (1 + (d_c / l_a) * cos(theta_h))
        """
        steer_rate_t = (v_t / self.l_t) * np.tan(delta_f)
        steer_effect = steer_rate_t * (1.0 + (self.d_c / self.l_a) * np.cos(theta_h))
        trailer_effect = - (v_t / self.l_a) * np.sin(theta_h)
        return float(trailer_effect - steer_effect)

    def tractor_yaw_rate(self, v_t: float, delta_f: float) -> float:
        """Compute kinematic yaw rate of tractor omega_t = dot{psi}_t."""
        return float((v_t / self.l_t) * np.tan(delta_f))

    def aircraft_yaw_rate(self, v_t: float, delta_f: float, theta_h: float) -> float:
        """Compute kinematic yaw rate of aircraft omega_a = dot{psi}_a."""
        omega_t = self.tractor_yaw_rate(v_t, delta_f)
        d_theta_h = self.hitch_articulation_rate(v_t, delta_f, theta_h)
        return float(omega_t + d_theta_h)

    def derivatives(
        self,
        t: float,
        state: np.ndarray,
        a_cmd: float,
        steer_rate_cmd: float,
    ) -> np.ndarray:
        """Kinematic state derivatives [x_t_dot, y_t_dot, psi_t_dot, theta_h_dot, v_t_dot, delta_f_dot, beta_slip_dot, omega_t_dot]."""
        x_t = float(state[0])
        y_t = float(state[1])
        psi_t = float(state[2])
        theta_h = float(state[3])
        v_t = float(state[4])
        delta_f = float(state[5])

        x_t_dot = v_t * np.cos(psi_t)
        y_t_dot = v_t * np.sin(psi_t)
        psi_t_dot = self.tractor_yaw_rate(v_t, delta_f)
        theta_h_dot = self.hitch_articulation_rate(v_t, delta_f, theta_h)
        v_t_dot = float(np.clip(a_cmd, -self.tractor.a_max, self.tractor.a_max))
        delta_f_dot = float(np.clip(steer_rate_cmd, -self.tractor.omega_steer_max_rad, self.tractor.omega_steer_max_rad))

        # In pure kinematics, sideslip is zero and yaw rate equals kinematic yaw rate
        beta_slip_dot = 0.0
        # omega_t derivative under constant acceleration/steer rate:
        # d/dt [ (v_t / l_t) tan(delta_f) ] = (v_t_dot / l_t) tan(delta_f) + (v_t / l_t) (1 + tan^2(delta_f)) delta_f_dot
        omega_t_dot = (v_t_dot / self.l_t) * np.tan(delta_f) + (v_t / self.l_t) * (1.0 + np.tan(delta_f)**2) * delta_f_dot

        res = np.zeros(max(len(state), 6), dtype=np.float64)
        res[0] = x_t_dot
        res[1] = y_t_dot
        res[2] = psi_t_dot
        res[3] = theta_h_dot
        res[4] = v_t_dot
        res[5] = delta_f_dot
        if len(res) > 6:
            res[6] = beta_slip_dot
        if len(res) > 7:
            res[7] = omega_t_dot
        return res

    def step(
        self,
        state: np.ndarray,
        a_cmd: float,
        steer_rate_cmd: float,
        dt: float,
    ) -> np.ndarray:
        """RK4 integration step for nonholonomic kinematic state."""
        h = float(dt)
        s = np.asarray(state, dtype=np.float64)

        k1 = self.derivatives(0.0, s, a_cmd, steer_rate_cmd)
        k2 = self.derivatives(0.0, s + 0.5 * h * k1, a_cmd, steer_rate_cmd)
        k3 = self.derivatives(0.0, s + 0.5 * h * k2, a_cmd, steer_rate_cmd)
        k4 = self.derivatives(0.0, s + h * k3, a_cmd, steer_rate_cmd)

        next_state = s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_state[2] = wrap_to_pi(next_state[2])  # psi_t
        next_state[3] = wrap_to_pi(next_state[3])  # theta_h
        next_state[5] = float(np.clip(next_state[5], -self.tractor.delta_max_rad, self.tractor.delta_max_rad))
        if len(next_state) > 6:
            next_state[6] = 0.0  # beta_slip = 0 in kinematics
        if len(next_state) > 7:
            next_state[7] = self.tractor_yaw_rate(next_state[4], next_state[5])
        return next_state
