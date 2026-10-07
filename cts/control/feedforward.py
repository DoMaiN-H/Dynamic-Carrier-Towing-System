"""cts.control.feedforward (Proposed Production Blueprint)

Feature F14: Flight Deck Inertial Force Feedforward Compensation.
Calculates analytical steering feedforward compensation delta_ff(t) and
longitudinal acceleration compensation a_ff(t) to counteract dynamic deck
roll/pitch tilt and 6-DOF non-inertial apparent acceleration fields.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, Tuple
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from cts.common.config import CTSConfig, TractorConfig, AircraftConfig, TireConfig
from cts.common.math_utils import wrap_to_pi
from cts.environment.non_inertial_field import DeckState, ApparentAcceleration, GRAVITY
from cts.dynamics.multibody_dynamics import VehicleState


class FeedforwardConfig(BaseModel):
    """Configuration parameters for flight deck inertial force feedforward."""
    model_config = ConfigDict(frozen=True)

    delta_ff_max_deg: float = Field(default=20.0, description="Maximum steering feedforward angle (deg)")
    delta_ff_max_rad: float = Field(default=0.3490658503988659, description="Maximum feedforward angle (rad)")
    omega_steer_max_deg: float = Field(default=30.0, description="Maximum steering slew rate (deg/s)")
    omega_steer_max_rad: float = Field(default=0.5235987755982988, description="Maximum steering slew rate (rad/s)")
    a_ff_max: float = Field(default=0.5, description="Maximum longitudinal acceleration feedforward (m/s^2)")
    
    # Roll angular acceleration derivative damping gain
    kd_roll_acc: float = Field(default=0.03, description="Derivative damping gain on roll angular acceleration (s^2)")
    
    # Low-pass filter cutoff frequency for apparent acceleration input
    cutoff_freq_hz: float = Field(default=5.0, description="Low-pass filter cutoff frequency (Hz)")
    
    # Low-speed soft gating velocity threshold
    v_gate: float = Field(default=0.10, description="Velocity threshold below which feedforward is faded (m/s)")
    
    # Modeling mode: 'direct' (front tire load cancellation) or 'dual_axle' (moment-equilibrium)
    mode: str = Field(default="direct", description="Feedforward formulation mode ('direct' or 'dual_axle')")
    
    # Aircraft nose load fraction override (if None, derived from config b_a / l_a)
    nose_load_fraction: Optional[float] = Field(default=None, description="Aircraft nose gear load fraction")
    
    # Enable switches
    enable_roll_damping: bool = Field(default=True, description="Enable roll angular acceleration damping")
    enable_longitudinal: bool = Field(default=True, description="Enable longitudinal acceleration feedforward")
    enable_slew_rate_limit: bool = Field(default=True, description="Enable steering slew rate limiting")


@dataclass(frozen=True)
class FeedforwardOutput:
    """Output telemetry container for inertial force feedforward."""
    delta_ff: float             # Effective steering compensation after saturation & rate limiting (rad)
    delta_ff_raw: float         # Raw steady-state steering compensation before dynamic limiting (rad)
    delta_ff_damping: float     # Roll acceleration derivative damping contribution (rad)
    a_ff: float                 # Longitudinal acceleration feedforward compensation (m/s^2)
    F_y_inertial: float         # Total lateral inertial disturbance force on tractor (N)
    F_t_inertial: float         # Direct tractor lateral inertial force (N)
    F_hitch_inertial: float     # Aircraft hitch lateral constraint force (N)
    a_app_t_y: float            # Apparent lateral acceleration in tractor body frame (m/s^2)
    d_delta_ff_dt: float        # Instantaneous rate of change of steering feedforward (rad/s)
    is_saturated: bool          # True if steering compensation hit saturation limit


class InertialFeedforward:
    """Analytical feedforward steering and drive compensation for flight deck tilt forces.

    Formulates:
        delta_ff(t) = - (m_t * a_t,y(t) + F_hitch,y(t)) / C_alpha,f + delta_damping(t)
    where:
        F_hitch,y(t) = eta_nose * m_a * (a_a,y(t) * cos(theta_h) + a_a,x(t) * sin(theta_h))
        delta_damping(t) = - kd_roll_acc * alpha_deck,x(t)
    with strict saturation (|delta_ff| <= 20 deg), slew rate filtering (<= 30 deg/s),
    and low-speed smooth gating.
    """

    def __init__(
        self,
        config: Optional[FeedforwardConfig] = None,
        cts_config: Optional[CTSConfig] = None,
    ) -> None:
        self.cfg = config or FeedforwardConfig()
        self.cts_cfg = cts_config or CTSConfig()
        self.tractor = self.cts_cfg.tractor
        self.aircraft = self.cts_cfg.aircraft
        self.tire = self.cts_cfg.tire

        # Inertial parameters
        self.m_t = float(self.tractor.mass)
        self.m_a = float(self.aircraft.mass)
        self.m_total = self.m_t + self.m_a

        # Geometry
        self.l_t = float(self.tractor.wheelbase)
        self.a_t = float(self.tractor.cg_to_front)
        self.b_t = float(self.tractor.cg_to_rear)
        self.d_c = float(self.tractor.hitch_offset)
        self.l_a = float(self.aircraft.wheelbase)
        self.a_a = float(self.aircraft.cg_to_nose)
        self.b_a = float(self.aircraft.cg_to_main)

        # Aircraft nose load fraction: b_a / l_a = 1.2 / 8.0 = 0.15
        if self.cfg.nose_load_fraction is not None:
            self.eta_nose = float(self.cfg.nose_load_fraction)
        else:
            self.eta_nose = self.b_a / self.l_a

        # Cornering stiffness: 2 tires per axle
        # Tractor front axle total cornering stiffness
        self.C_alpha_f = 2.0 * float(self.tire.cornering_stiffness_tractor)
        self.C_alpha_r = 2.0 * float(self.tire.cornering_stiffness_tractor)

        # State storage for slew rate filter and low-pass filter
        self.prev_delta_ff: float = 0.0
        self.prev_t: float = 0.0
        self.is_initialized: bool = False
        self.filtered_a_app_y: float = 0.0

    def reset(self) -> None:
        """Reset internal filter states and rate limits."""
        self.prev_delta_ff = 0.0
        self.prev_t = 0.0
        self.is_initialized = False
        self.filtered_a_app_y = 0.0

    def compute_steering_compensation(
        self,
        state: VehicleState,
        deck_state: DeckState,
        apparent_acc: ApparentAcceleration,
        dt: float = 0.02,
    ) -> FeedforwardOutput:
        """Compute analytical steering feedforward compensation matching PROJECT.md contract.

        Args:
            state: Current vehicle state [x_t, y_t, psi_t, theta_h, v_t, delta_f, ...].
            deck_state: Current 6-DOF deck motion state (DeckState).
            apparent_acc: Decomposed apparent acceleration field at tractor location.
            dt: Time step since last update (s).

        Returns:
            FeedforwardOutput containing delta_ff, a_ff, and disturbance force decomposition.
        """
        # 1. Resolve apparent acceleration in tractor body coordinates
        psi_t = state.psi_t
        theta_h = state.theta_h
        v_t = state.v_t

        a_app_total = apparent_acc.a_apparent_total  # shape (3,) in Deck frame
        ax_d, ay_d = float(a_app_total[0]), float(a_app_total[1])

        # Tractor body frame: x_body forward, y_body left
        c_psi, s_psi = np.cos(psi_t), np.sin(psi_t)
        a_t_x = ax_d * c_psi + ay_d * s_psi
        a_t_y = -ax_d * s_psi + ay_d * c_psi

        # 2. Compute aircraft body apparent acceleration and hitch reaction force
        psi_a = wrap_to_pi(psi_t + theta_h)
        c_psia, s_psia = np.cos(psi_a), np.sin(psi_a)
        a_a_x = ax_d * c_psia + ay_d * s_psia
        a_a_y = -ax_d * s_psia + ay_d * c_psia

        # Aircraft nose gear static load fraction carries eta_nose * m_a
        m_a_nose = self.eta_nose * self.m_a
        # Lateral and longitudinal inertial forces at aircraft hitch in aircraft frame
        F_hitch_ya = m_a_nose * a_a_y
        F_hitch_xa = m_a_nose * a_a_x

        # Transform hitch reaction force into tractor body axes
        c_h, s_h = np.cos(theta_h), np.sin(theta_h)
        F_hitch_y = F_hitch_ya * c_h + F_hitch_xa * s_h
        F_hitch_x = F_hitch_xa * c_h - F_hitch_ya * s_h

        # 3. Direct tractor lateral inertial force
        F_t_inertial = self.m_t * a_t_y

        # Total lateral disturbance force on tractor
        F_y_inertial = F_t_inertial + F_hitch_y

        # 4. Steady-State Steering Angle Calculation
        if self.cfg.mode == "direct":
            # Direct front axle force balancing:
            # Tire lateral force generated by steering is Fy_f = C_alpha_f * delta.
            # To cancel lateral inertial force F_y_inertial, front tires must produce -F_y_inertial:
            # delta_ff_static = - F_y_inertial / C_alpha_f
            delta_ff_static = - F_y_inertial / self.C_alpha_f
        elif self.cfg.mode == "dual_axle":
            # Moment-equilibrium bicycle model:
            # Taking moment balance about rear axle:
            F_yf_req = (self.b_t / self.l_t) * self.m_t * a_t_y - (self.d_c / self.l_t) * F_hitch_y
            # Taking moment balance about front axle:
            F_yr_req = (self.a_t / self.l_t) * self.m_t * a_t_y + ((self.l_t + self.d_c) / self.l_t) * F_hitch_y
            delta_ff_static = - (F_yf_req / self.C_alpha_f - F_yr_req / self.C_alpha_r)
        else:
            delta_ff_static = - F_y_inertial / self.C_alpha_f

        # 5. Roll Angular Acceleration Derivative Damping (Lead Compensation)
        delta_damping = 0.0
        if self.cfg.enable_roll_damping:
            # Deck roll angular acceleration alpha_d[0] (rad/s^2)
            alpha_roll = float(deck_state.alpha_d[0]) if len(deck_state.alpha_d) > 0 else 0.0
            # Anticipatory steering lead: opposes angular acceleration into the roll
            delta_damping = - self.cfg.kd_roll_acc * alpha_roll

        delta_ff_raw = delta_ff_static + delta_damping

        # 6. Low-speed soft gating: fade out feedforward at near zero speed
        v_abs = abs(v_t)
        if v_abs < self.cfg.v_gate:
            w_v = v_abs / max(self.cfg.v_gate, 1e-4)
            delta_ff_target = delta_ff_raw * w_v
        else:
            delta_ff_target = delta_ff_raw

        # 7. Slew Rate Limiter & Saturation
        dt_clamped = max(float(dt), 1e-4)
        if not self.is_initialized:
            self.prev_delta_ff = delta_ff_target
            self.is_initialized = True

        if self.cfg.enable_slew_rate_limit:
            max_delta_change = self.cfg.omega_steer_max_rad * dt_clamped
            delta_change = np.clip(delta_ff_target - self.prev_delta_ff, -max_delta_change, max_delta_change)
            delta_ff_rate_limited = self.prev_delta_ff + delta_change
            d_delta_dt = float(delta_change / dt_clamped)
        else:
            delta_ff_rate_limited = delta_ff_target
            d_delta_dt = float((delta_ff_target - self.prev_delta_ff) / dt_clamped)

        # Saturation clamping (|delta_ff| <= 20 deg)
        delta_max = self.cfg.delta_ff_max_rad
        delta_ff = float(np.clip(delta_ff_rate_limited, -delta_max, delta_max))
        is_sat = bool(abs(delta_ff_rate_limited) >= delta_max)

        self.prev_delta_ff = delta_ff

        # 8. Longitudinal Acceleration Feedforward
        a_ff = 0.0
        if self.cfg.enable_longitudinal:
            # Longitudinal apparent force on tractor + aircraft nose
            F_x_inertial = self.m_t * a_t_x + F_hitch_x
            a_ff_raw = - F_x_inertial / self.m_total
            a_ff = float(np.clip(a_ff_raw, -self.cfg.a_ff_max, self.cfg.a_ff_max))

        return FeedforwardOutput(
            delta_ff=delta_ff,
            delta_ff_raw=float(delta_ff_raw),
            delta_ff_damping=float(delta_damping),
            a_ff=a_ff,
            F_y_inertial=float(F_y_inertial),
            F_t_inertial=float(F_t_inertial),
            F_hitch_inertial=float(F_hitch_y),
            a_app_t_y=float(a_t_y),
            d_delta_ff_dt=d_delta_dt,
            is_saturated=is_sat,
        )
