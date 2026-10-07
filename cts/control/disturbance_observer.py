"""cts.control.disturbance_observer (Proposed Production Blueprint)

Feature F15: 100 Hz Extended State Observer (ESO) & Disturbance Observer (DOB).
Estimates lumped disturbances d_y(t) and d_x(t) comprising unmodeled tire lateral slip,
trailer side thrust, and deck friction fluctuations, mapping estimates into equivalent
steering compensation delta_dob(t) and traction force compensation a_dob(t).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, Tuple
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.linalg import expm

from cts.common.config import CTSConfig, TractorConfig, AircraftConfig, TireConfig
from cts.dynamics.multibody_dynamics import VehicleState, ControlAction


class ESOConfig(BaseModel):
    """Configuration parameters for 100 Hz Extended State Observer."""
    model_config = ConfigDict(frozen=True)

    order: int = Field(default=2, description="ESO model order: 2 (velocity-level [vy, dy]) or 3 (position-level [ey, vy, dy])")
    omega_obs: float = Field(default=45.0, description="Observer bandwidth omega_obs in [20, 50] rad/s")
    damping_ratio: float = Field(default=0.70710678, description="Observer pole damping ratio zeta (0.707 for Butterworth)")
    b0: Optional[float] = Field(default=None, description="Control effectiveness parameter b0 = C_alpha_f / m_t (m/(s^2*rad))")
    Ts: float = Field(default=0.01, description="Nominal sampling period (100 Hz -> 0.01 s)")
    
    delta_dob_max_deg: float = Field(default=15.0, description="Maximum steering disturbance compensation (deg)")
    delta_dob_max_rad: float = Field(default=0.2617993877991494, description="Maximum steering disturbance compensation (rad)")
    omega_steer_max_deg: float = Field(default=30.0, description="Maximum steering compensation rate (deg/s)")
    omega_steer_max_rad: float = Field(default=0.5235987755982988, description="Maximum steering compensation rate (rad/s)")
    a_dob_max: float = Field(default=0.5, description="Maximum longitudinal acceleration disturbance compensation (m/s^2)")
    
    deadzone: float = Field(default=0.01, description="Disturbance acceleration deadzone threshold (m/s^2)")
    enable_rate_limit: bool = Field(default=True, description="Enable steering compensation slew rate limit")
    enable_output_filter: bool = Field(default=True, description="Enable 1st-order low-pass filter on disturbance output")
    filter_cutoff_rad: float = Field(default=60.0, description="Disturbance post-filter cutoff frequency (rad/s)")
    enable_longitudinal: bool = Field(default=True, description="Enable longitudinal disturbance estimation")


@dataclass(frozen=True)
class DisturbanceEstimate:
    """Output container for disturbance observer telemetry."""
    d_hat_y: float              # Estimated lateral disturbance acceleration (m/s^2)
    F_dist_y: float             # Estimated lateral disturbance force on tractor (N)
    delta_dob: float            # Equivalent steering compensation angle (rad)
    d_hat_x: float              # Estimated longitudinal disturbance acceleration (m/s^2)
    a_dob: float                # Equivalent longitudinal acceleration compensation (m/s^2)
    innovation: float           # Observer measurement innovation / residual (m or m/s)
    state_estimate: np.ndarray  # Internal observer state vector estimate x_hat
    is_saturated: bool          # True if steering compensation clamped at limit


class ExtendedStateObserver:
    """Discrete-time 100 Hz Linear Extended State Observer (LESO).

    State-space model:
        Order 2 (relative degree 1, velocity):
            dot(v_y) = b0 * u + d_y
            dot(d_y) = 0
            y = v_y
        Order 3 (relative degree 2, position):
            dot(e_y) = v_y
            dot(v_y) = b0 * u + d_y
            dot(d_y) = 0
            y = e_y

    Bandwidth parameterization:
        Poles placed with Butterworth damping zeta in [0.707, 1.0]:
        Order 2: s^2 + 2*zeta*omega*s + omega^2 => L = [2*zeta*omega, omega^2]^T
        Order 3: (s + omega)(s^2 + 2*zeta*omega*s + omega^2)

    Exact matrix exponential discretization:
        Phi = expm((A - L C) * Ts)
        Gamma = integral expm((A - L C) * tau) dtau * [B, L]
    """

    def __init__(
        self,
        config: Optional[ESOConfig] = None,
        cts_config: Optional[CTSConfig] = None,
    ) -> None:
        self.cfg = config or ESOConfig()
        self.cts_cfg = cts_config or CTSConfig()
        self.tractor = self.cts_cfg.tractor
        self.tire = self.cts_cfg.tire

        self.m_t = float(self.tractor.mass)
        self.C_alpha_f = 2.0 * float(self.tire.cornering_stiffness_tractor)

        # Control effectiveness b0 = C_alpha_f / m_t
        if self.cfg.b0 is not None:
            self.b0 = float(self.cfg.b0)
        else:
            self.b0 = self.C_alpha_f / self.m_t  # ~ 12.94 m/(s^2 * rad)

        self.order = self.cfg.order
        self.omega_obs = float(self.cfg.omega_obs)
        self.zeta = float(self.cfg.damping_ratio)
        self.Ts = float(self.cfg.Ts)

        # Build continuous state matrices
        if self.order == 3:
            # States: [e_y, v_y, d_y]
            self.A = np.array([
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ], dtype=np.float64)
            self.B = np.array([0.0, self.b0, 0.0], dtype=np.float64)
            self.C = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            p1 = (1.0 + 2.0 * self.zeta) * self.omega_obs
            p2 = (1.0 + 2.0 * self.zeta) * (self.omega_obs ** 2)
            p3 = self.omega_obs ** 3
            self.L = np.array([p1, p2, p3], dtype=np.float64)
        else:
            # Order 2: States [v_y, d_y]
            self.A = np.array([
                [0.0, 1.0],
                [0.0, 0.0],
            ], dtype=np.float64)
            self.B = np.array([self.b0, 0.0], dtype=np.float64)
            self.C = np.array([1.0, 0.0], dtype=np.float64)
            self.L = np.array([
                2.0 * self.zeta * self.omega_obs,
                self.omega_obs ** 2,
            ], dtype=np.float64)

        # Precompute discrete-time update matrices via exact matrix exponential
        self._precompute_discrete_matrices(self.Ts)

        # Longitudinal 2nd-order ESO: states [v_x, d_x]
        self.b0_x = 1.0 / (float(self.tractor.mass) + float(self.cts_cfg.aircraft.mass))
        self.A_x = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
        self.B_x = np.array([1.0, 0.0], dtype=np.float64)
        self.C_x = np.array([1.0, 0.0], dtype=np.float64)
        self.L_x = np.array([2.0 * self.zeta * self.omega_obs, self.omega_obs ** 2], dtype=np.float64)
        self._precompute_discrete_matrices_x(self.Ts)

        # Internal state estimates
        self.x_hat = np.zeros(self.order, dtype=np.float64)
        self.x_hat_long = np.zeros(2, dtype=np.float64)
        self.prev_delta_dob: float = 0.0
        self.prev_d_filtered: float = 0.0
        self.is_initialized: bool = False

    def _precompute_discrete_matrices(self, dt: float) -> None:
        """Precompute discrete state transition matrices using exact matrix exponential."""
        A_cl = self.A - np.outer(self.L, self.C)
        n = self.order
        M = np.zeros((n + 2, n + 2), dtype=np.float64)
        M[:n, :n] = A_cl
        M[:n, n] = self.B
        M[:n, n + 1] = self.L

        expM = expm(M * dt)
        self.Phi = expM[:n, :n]
        self.Gamma_u = expM[:n, n]
        self.Gamma_y = expM[:n, n + 1]
        self._cached_dt = dt

    def _precompute_discrete_matrices_x(self, dt: float) -> None:
        """Precompute discrete matrices for longitudinal observer."""
        A_cl_x = self.A_x - np.outer(self.L_x, self.C_x)
        M_x = np.zeros((4, 4), dtype=np.float64)
        M_x[:2, :2] = A_cl_x
        M_x[:2, 2] = self.B_x
        M_x[:2, 3] = self.L_x

        expM_x = expm(M_x * dt)
        self.Phi_x = expM_x[:2, :2]
        self.Gamma_u_x = expM_x[:2, 2]
        self.Gamma_y_x = expM_x[:2, 3]

    def reset(self) -> None:
        """Reset internal observer states and filters."""
        self.x_hat.fill(0.0)
        self.x_hat_long.fill(0.0)
        self.prev_delta_dob = 0.0
        self.prev_d_filtered = 0.0
        self.is_initialized = False

    def update(
        self,
        y_meas: float,
        u_steer: float,
        v_long_meas: Optional[float] = None,
        u_acc: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> DisturbanceEstimate:
        """Execute one 100 Hz discrete update step.

        Args:
            y_meas: Lateral measurement:
                    Order 2: lateral velocity v_y (m/s)
                    Order 3: lateral cross-track error e_y (m)
            u_steer: Steering control input applied during last interval (rad)
            v_long_meas: Longitudinal velocity measurement v_x (m/s, optional)
            u_acc: Longitudinal acceleration command (m/s^2, optional)
            dt: Time step (s, default config.Ts)

        Returns:
            DisturbanceEstimate with estimated disturbance and compensation commands.
        """
        step_dt = float(dt) if dt is not None else self.Ts
        if abs(step_dt - getattr(self, "_cached_dt", self.Ts)) > 1e-5:
            self._precompute_discrete_matrices(step_dt)
            self._precompute_discrete_matrices_x(step_dt)

        # Output estimate and innovation
        y_hat = float(np.dot(self.C, self.x_hat))
        innovation = float(y_meas - y_hat)

        # Discrete state transition
        self.x_hat = self.Phi @ self.x_hat + self.Gamma_u * float(u_steer) + self.Gamma_y * float(y_meas)

        # Extended state estimate
        d_hat_raw = float(self.x_hat[-1])

        # Deadzone filter (zero small jitter, preserve signal above threshold)
        if abs(d_hat_raw) < self.cfg.deadzone:
            d_hat_dead = 0.0
        else:
            d_hat_dead = d_hat_raw

        # Post low-pass filter
        if self.cfg.enable_output_filter:
            alpha_f = 1.0 - np.exp(-self.cfg.filter_cutoff_rad * step_dt)
            d_hat_y = self.prev_d_filtered + alpha_f * (d_hat_dead - self.prev_d_filtered)
            self.prev_d_filtered = d_hat_y
        else:
            d_hat_y = d_hat_dead

        # Map disturbance acceleration into equivalent steering angle:
        # delta_dob = - d_hat_y / b0
        delta_dob_raw = - d_hat_y / self.b0

        # Slew rate limiter
        if self.cfg.enable_rate_limit:
            max_rate_step = self.cfg.omega_steer_max_rad * step_dt
            delta_change = np.clip(delta_dob_raw - self.prev_delta_dob, -max_rate_step, max_rate_step)
            delta_dob_limited = self.prev_delta_dob + delta_change
        else:
            delta_dob_limited = delta_dob_raw

        # Saturation clamping (|delta_dob| <= 15 deg)
        delta_max = self.cfg.delta_dob_max_rad
        delta_dob = float(np.clip(delta_dob_limited, -delta_max, delta_max))
        is_sat = bool(abs(delta_dob_limited) >= delta_max)
        self.prev_delta_dob = delta_dob

        # Disturbance force on tractor
        F_dist_y = self.m_t * d_hat_y

        # Longitudinal disturbance observer update
        d_hat_x = 0.0
        a_dob = 0.0
        if self.cfg.enable_longitudinal and v_long_meas is not None and u_acc is not None:
            self.x_hat_long = (
                self.Phi_x @ self.x_hat_long
                + self.Gamma_u_x * float(u_acc)
                + self.Gamma_y_x * float(v_long_meas)
            )
            d_hat_x = float(self.x_hat_long[1])
            a_dob = float(np.clip(-d_hat_x, -self.cfg.a_dob_max, self.cfg.a_dob_max))

        return DisturbanceEstimate(
            d_hat_y=float(d_hat_y),
            F_dist_y=float(F_dist_y),
            delta_dob=float(delta_dob),
            d_hat_x=float(d_hat_x),
            a_dob=float(a_dob),
            innovation=float(innovation),
            state_estimate=self.x_hat.copy(),
            is_saturated=is_sat,
        )


class DisturbanceObserver:
    """Production wrapper integrating Extended State Observer (ESO) with CTS plant interface."""

    def __init__(
        self,
        config: Optional[ESOConfig] = None,
        cts_config: Optional[CTSConfig] = None,
    ) -> None:
        self.cfg = config or ESOConfig()
        self.cts_cfg = cts_config or CTSConfig()
        self.eso = ExtendedStateObserver(self.cfg, self.cts_cfg)

    def reset(self) -> None:
        """Reset internal observer state."""
        self.eso.reset()

    def update(
        self,
        y: Union[float, np.ndarray],
        u: Union[float, ControlAction],
        dt: float = 0.01,
        v_long: Optional[float] = None,
        u_acc: Optional[float] = None,
    ) -> DisturbanceEstimate:
        """Matches PROJECT.md interface contract line 56:
        DisturbanceObserver.update(y, u, dt) -> DisturbanceEstimate
        """
        if isinstance(u, ControlAction):
            steer_input = u.steer_angle_cmd if u.steer_angle_cmd is not None else 0.0
            acc_input = u.a_cmd
        else:
            steer_input = float(u)
            acc_input = u_acc

        meas_val = float(y[0]) if isinstance(y, (np.ndarray, list, tuple)) else float(y)

        return self.eso.update(
            y_meas=meas_val,
            u_steer=steer_input,
            v_long_meas=v_long,
            u_acc=acc_input,
            dt=dt,
        )

    def estimate_from_state(
        self,
        current_state: VehicleState,
        control_action: ControlAction,
        ref_y: float = 0.0,
        dt: float = 0.01,
    ) -> DisturbanceEstimate:
        """Convenience method accepting VehicleState and ControlAction directly."""
        if self.cfg.order == 3:
            meas = current_state.y_t - ref_y
        else:
            meas = current_state.v_t * np.sin(current_state.beta_slip)

        return self.update(
            y=meas,
            u=control_action,
            dt=dt,
            v_long=current_state.v_t,
            u_acc=control_action.a_cmd,
        )
