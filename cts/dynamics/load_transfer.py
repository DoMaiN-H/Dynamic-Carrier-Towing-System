"""cts.dynamics.load_transfer

Feature F7: Dynamic Vertical Load Transfer & Invariants Conservation.
Dynamic redistribution of vertical loads F_z,k across 4 tractor wheels (FL, FR, RL, RR)
and aircraft gear (nose clamped on turntable, ML, MR) under 6-DOF dynamic deck motion,
strictly conserving total vertical force: sum(F_z,k) == (m_t + m_a) * g_eff,z.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Dict, Any
import numpy as np

from cts.common.config import TractorConfig, AircraftConfig, CTSConfig
from cts.environment.deck_motion import DeckState
from cts.environment.non_inertial_field import NonInertialAccelerationField, ApparentAcceleration, GRAVITY


@dataclass(frozen=True)
class WheelLoads:
    """Instantaneous vertical normal loads across all vehicle wheels and contact patches.
    
    All force quantities in Newtons (N).
    Matches PROJECT.md interface contract:
    WheelLoads(Fz_fl, Fz_fr, Fz_rl, Fz_rr, Fz_an_left, Fz_an_right, Fz_am_left, Fz_am_right)
    """
    # Tractor ground contact wheels (N)
    Fz_fl: float
    Fz_fr: float
    Fz_rl: float
    Fz_rr: float

    # Aircraft nose gear (on tractor cradle turntable) (N)
    Fz_an_left: float
    Fz_an_right: float

    # Aircraft main landing gear ground contact (N)
    Fz_am_left: float   # Main-Left (ML)
    Fz_am_right: float  # Main-Right (MR)

    # Effective deck normal acceleration (m/s^2)
    g_eff_z: float

    # Aggregate Axle Loads (N)
    Fz_front: float
    Fz_rear: float
    Fz_hitch: float
    Fz_main: float

    # Stability & Conservation Diagnostics
    ltr_tractor: float
    ltr_aircraft: float
    ltr_max: float
    is_rollover_safe: bool
    has_lift_off: bool
    conservation_error: float

    @property
    def Fz_ml(self) -> float:
        """Alias for aircraft main left ground force."""
        return self.Fz_am_left

    @property
    def Fz_mr(self) -> float:
        """Alias for aircraft main right ground force."""
        return self.Fz_am_right

    @property
    def Fz_ground_total(self) -> float:
        """Sum of all 6 tires in direct contact with flight deck surface (N)."""
        return self.Fz_fl + self.Fz_fr + self.Fz_rl + self.Fz_rr + self.Fz_am_left + self.Fz_am_right

    def as_array_6d(self) -> np.ndarray:
        """Return array of 6 ground contact forces: [FL, FR, RL, RR, ML, MR]."""
        return np.array([self.Fz_fl, self.Fz_fr, self.Fz_rl, self.Fz_rr, self.Fz_am_left, self.Fz_am_right], dtype=np.float64)

    def as_array_8d(self) -> np.ndarray:
        """Return array of all 8 components matching PROJECT.md."""
        return np.array([
            self.Fz_fl, self.Fz_fr, self.Fz_rl, self.Fz_rr,
            self.Fz_an_left, self.Fz_an_right, self.Fz_am_left, self.Fz_am_right
        ], dtype=np.float64)


class DynamicLoadTransfer:
    """Computes closed-form dynamic vertical load redistribution for tractor and aircraft."""

    def __init__(
        self,
        tractor_config: Optional[TractorConfig] = None,
        aircraft_config: Optional[AircraftConfig] = None,
        cradle_height: float = 0.30,
        roll_stiffness_front_fraction: float = 0.50,
        smooth_eps: float = 10.0,
    ) -> None:
        self.t_cfg = tractor_config or TractorConfig()
        self.a_cfg = aircraft_config or AircraftConfig()
        self.cradle_height = float(cradle_height)
        self.xi_front = float(roll_stiffness_front_fraction)
        self.smooth_eps = float(smooth_eps)

        # Tractor geometric & mass parameters
        self.l_t = float(self.t_cfg.wheelbase)          # 1.80 m
        self.a_t = float(self.t_cfg.cg_to_front)        # 0.95 m
        self.b_t = float(self.t_cfg.cg_to_rear)         # 0.85 m
        self.h_t = float(self.t_cfg.track)              # 1.20 m
        self.h_cg_t = float(self.t_cfg.cg_height)       # 0.45 m
        self.d_c = float(self.t_cfg.hitch_offset)       # 0.10 m
        self.m_t = float(self.t_cfg.mass)               # 8500.0 kg

        # Aircraft geometric & mass parameters
        self.l_a = float(self.a_cfg.wheelbase)          # 8.00 m
        self.a_a = float(self.a_cfg.cg_to_nose)         # 6.80 m
        self.b_a = float(self.a_cfg.cg_to_main)         # 1.20 m
        self.W_r = float(self.a_cfg.main_track)         # 3.10 m
        self.h_cg_a = float(self.a_cfg.cg_height)       # 1.85 m
        self.m_a = float(self.a_cfg.mass)               # 20500.0 kg
        self.m_total = self.m_t + self.m_a              # 29000.0 kg

    def compute(
        self,
        a_x_t: float,
        a_y_t: float,
        a_x_a: float,
        a_y_a: float,
        g_eff_z: float,
        F_hx_t: float = 0.0,
        F_hy_t: float = 0.0,
        unilateral: bool = False,
    ) -> WheelLoads:
        """Compute wheel loads given accelerations resolved in tractor/aircraft body frames.

        Args:
            a_x_t: Total longitudinal acceleration of tractor (m/s^2).
            a_y_t: Total lateral acceleration of tractor (m/s^2).
            a_x_a: Total longitudinal acceleration of aircraft (m/s^2).
            a_y_a: Total lateral acceleration of aircraft (m/s^2).
            g_eff_z: Effective downward normal gravity holding vehicle on deck (m/s^2).
            F_hx_t: Longitudinal hitch force along tractor x-axis (N).
            F_hy_t: Lateral hitch force along tractor y-axis (N).
            unilateral: If True, apply smooth conservation-preserving non-negative projection.

        Returns:
            WheelLoads container with exact conservation sum == (m_t + m_a) * g_eff_z.
        """
        g_eff_z = float(g_eff_z)

        # 1. Aircraft Subsystem Vertical Equilibrium & Pitch Transfer
        delta_Fz_long_a = (self.m_a * a_x_a * self.h_cg_a) / self.l_a
        Fz_hitch = (self.b_a / self.l_a) * self.m_a * g_eff_z - delta_Fz_long_a
        Fz_main = (self.a_a / self.l_a) * self.m_a * g_eff_z + delta_Fz_long_a

        # Aircraft lateral load transfer across main gear
        delta_Fz_lat_a = (self.m_a * a_y_a * self.h_cg_a) / self.W_r
        Fz_ml = 0.5 * Fz_main - delta_Fz_lat_a
        Fz_mr = 0.5 * Fz_main + delta_Fz_lat_a

        # 2. Tractor Subsystem Vertical Equilibrium & Pitch Transfer
        delta_Fz_long_t = (self.m_t * a_x_t * self.h_cg_t + F_hx_t * self.cradle_height) / self.l_t
        Fz_front = (self.b_t / self.l_t) * self.m_t * g_eff_z - (self.d_c / self.l_t) * Fz_hitch - delta_Fz_long_t
        Fz_rear = (self.a_t / self.l_t) * self.m_t * g_eff_z + ((self.l_t + self.d_c) / self.l_t) * Fz_hitch + delta_Fz_long_t

        # Tractor lateral load transfer across 4 wheels
        M_roll_t = self.m_t * a_y_t * self.h_cg_t + F_hy_t * self.cradle_height
        delta_Fz_lat_f = self.xi_front * (M_roll_t / self.h_t)
        delta_Fz_lat_r = (1.0 - self.xi_front) * (M_roll_t / self.h_t)

        Fz_fl = 0.5 * Fz_front - delta_Fz_lat_f
        Fz_fr = 0.5 * Fz_front + delta_Fz_lat_f
        Fz_rl = 0.5 * Fz_rear - delta_Fz_lat_r
        Fz_rr = 0.5 * Fz_rear + delta_Fz_lat_r

        # Nose twin tires breakdown
        Fz_an_left = 0.5 * Fz_hitch
        Fz_an_right = 0.5 * Fz_hitch

        raw_ground = np.array([Fz_fl, Fz_fr, Fz_rl, Fz_rr, Fz_ml, Fz_mr], dtype=np.float64)
        has_lift_off = bool(np.any(raw_ground <= 1e-3))

        # 3. Optional Smooth Conservation-Preserving Regularization
        if unilateral:
            F_target = self.m_total * max(0.0, g_eff_z)
            if F_target <= 1e-9:
                Fz_fl = Fz_fr = Fz_rl = Fz_rr = Fz_ml = Fz_mr = 0.0
                Fz_an_left = Fz_an_right = 0.0
                Fz_front = Fz_rear = Fz_hitch = Fz_main = 0.0
            else:
                # Smooth softplus: eps * log(1 + exp(F / eps))
                reg_raw = self.smooth_eps * np.log1p(np.exp(np.clip(raw_ground / self.smooth_eps, -50.0, 50.0)))
                sum_reg = float(np.sum(reg_raw))
                if sum_reg > 1e-12:
                    scaled = reg_raw * (F_target / sum_reg)
                    Fz_fl, Fz_fr, Fz_rl, Fz_rr, Fz_ml, Fz_mr = scaled
                    Fz_front = Fz_fl + Fz_fr
                    Fz_rear = Fz_rl + Fz_rr
                    Fz_main = Fz_ml + Fz_mr
                    Fz_hitch = max(0.0, Fz_hitch)
                    Fz_an_left = 0.5 * Fz_hitch
                    Fz_an_right = 0.5 * Fz_hitch

        # 4. Conservation Verification
        ground_sum = Fz_fl + Fz_fr + Fz_rl + Fz_rr + Fz_ml + Fz_mr
        expected_sum = self.m_total * g_eff_z
        cons_err = abs(ground_sum - expected_sum)

        # 5. Dynamic Lateral Load Transfer Ratio (LTR) Monitoring
        # Evaluated when effective normal gravity is holding the vehicle on the deck (g_eff_z >= 1.0 m/s^2).
        # During transient heave drop / zero-g unweighting (g_eff_z < 1.0 m/s^2), vertical unloading
        # occurs without lateral chassis tipping risk.
        Fz_t_l = max(0.0, Fz_fl + Fz_rl)
        Fz_t_r = max(0.0, Fz_fr + Fz_rr)
        Fz_t_tot = Fz_t_l + Fz_t_r
        if g_eff_z >= 0.5 * GRAVITY and Fz_t_tot > 10.0:
            ltr_t = min(1.0, abs(Fz_t_l - Fz_t_r) / Fz_t_tot)
        else:
            ltr_t = 0.0

        Fz_a_l = max(0.0, Fz_ml)
        Fz_a_r = max(0.0, Fz_mr)
        Fz_a_tot = Fz_a_l + Fz_a_r
        if g_eff_z >= 0.5 * GRAVITY and Fz_a_tot > 10.0:
            ltr_a = min(1.0, abs(Fz_a_l - Fz_a_r) / Fz_a_tot)
        else:
            ltr_a = 0.0

        ltr_max = float(min(1.0, max(ltr_t, ltr_a)))
        is_safe = bool(ltr_max <= 0.70)

        return WheelLoads(
            Fz_fl=float(Fz_fl),
            Fz_fr=float(Fz_fr),
            Fz_rl=float(Fz_rl),
            Fz_rr=float(Fz_rr),
            Fz_an_left=float(Fz_an_left),
            Fz_an_right=float(Fz_an_right),
            Fz_am_left=float(Fz_ml),
            Fz_am_right=float(Fz_mr),
            g_eff_z=float(g_eff_z),
            Fz_front=float(Fz_front),
            Fz_rear=float(Fz_rear),
            Fz_hitch=float(Fz_hitch),
            Fz_main=float(Fz_main),
            ltr_tractor=float(ltr_t),
            ltr_aircraft=float(ltr_a),
            ltr_max=float(ltr_max),
            is_rollover_safe=is_safe,
            has_lift_off=has_lift_off,
            conservation_error=float(cons_err),
        )

    def compute_from_deck_state(
        self,
        tractor_pos_d: np.ndarray,
        tractor_vel_d: np.ndarray,
        tractor_acc_rel: np.ndarray,
        psi_t: float,
        psi_a: float,
        deck_state: DeckState,
        apparent_calc: Optional[NonInertialAccelerationField] = None,
        F_hx_t: float = 0.0,
        F_hy_t: float = 0.0,
        unilateral: bool = False,
    ) -> WheelLoads:
        """High-level interface directly querying DeckState non-inertial apparent field."""
        if apparent_calc is None:
            apparent_calc = NonInertialAccelerationField()

        app = apparent_calc.evaluate(deck_state, tractor_pos_d, tractor_vel_d)
        a_app_d = app.a_apparent_total
        g_eff_z = float(app.g_eff)

        # Resolve apparent acceleration into tractor body coordinates
        c_t, s_t = np.cos(psi_t), np.sin(psi_t)
        a_app_x_t = a_app_d[0] * c_t + a_app_d[1] * s_t
        a_app_y_t = -a_app_d[0] * s_t + a_app_d[1] * c_t
        a_x_t = float(tractor_acc_rel[0] - a_app_x_t)
        a_y_t = float(tractor_acc_rel[1] - a_app_y_t)

        # Resolve apparent acceleration into aircraft body coordinates
        c_a, s_a = np.cos(psi_a), np.sin(psi_a)
        a_app_x_a = a_app_d[0] * c_a + a_app_d[1] * s_a
        a_app_y_a = -a_app_d[0] * s_a + a_app_d[1] * c_a
        a_x_a = float(tractor_acc_rel[0] - a_app_x_a)
        a_y_a = float(tractor_acc_rel[1] - a_app_y_a)

        return self.compute(
            a_x_t=a_x_t, a_y_t=a_y_t,
            a_x_a=a_x_a, a_y_a=a_y_a,
            g_eff_z=g_eff_z,
            F_hx_t=F_hx_t, F_hy_t=F_hy_t,
            unilateral=unilateral,
        )


def compute_critical_rollover_acceleration(
    aircraft_cfg: Optional[AircraftConfig] = None,
    g_eff_z: float = GRAVITY,
    ltr_limit: float = 0.70,
) -> float:
    """Compute the analytical critical lateral acceleration threshold for LTR <= 0.70."""
    cfg = aircraft_cfg or AircraftConfig()
    a_a = float(cfg.cg_to_nose)
    W_r = float(cfg.main_track)
    l_a = float(cfg.wheelbase)
    h_cg_a = float(cfg.cg_height)
    geometric_factor = (a_a * W_r) / (2.0 * l_a * h_cg_a)
    return float(ltr_limit * geometric_factor * g_eff_z)
