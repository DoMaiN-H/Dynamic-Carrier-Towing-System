"""cts.planning.kinodynamic_limits

Feature F10: Time-Varying Kinodynamic Boundaries under 6-DOF Deck Motion.
Calculates dynamic lateral acceleration limits, analytical rollover threshold (LTR <= 0.70),
dynamic curvature bounds, and nonholonomic actuation limits.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Union, List
import numpy as np

from cts.common.config import TractorConfig, AircraftConfig, TireConfig, CarrierConfig
from cts.environment.deck_motion import DeckMotionSimulator, DeckState
from cts.environment.non_inertial_field import ApparentAcceleration, NonInertialAccelerationField
from cts.dynamics.multibody_dynamics import VehicleState


@dataclass(frozen=True)
class KinodynamicBound:
    """Instantaneous time-varying kinodynamic limits at timestamp t and speed v_t.
    
    Attributes:
        t: Simulation timestamp (s).
        v_t: Forward velocity evaluation point (m/s).
        g_eff_z: Effective normal gravity (m/s^2).
        a_deck_y: Apparent lateral deck acceleration (m/s^2).
        a_y_crit_slip: Dynamic lateral tire slip limit (m/s^2).
        a_y_crit_rollover: Dynamic rollover limit corresponding to LTR <= 0.70 (m/s^2).
        a_y_max: Combined governing lateral acceleration cap (m/s^2).
        kappa_max: Maximum allowable path curvature at speed v_t (1/m).
        delta_max: Maximum front wheel steering angle (rad).
        dot_delta_max: Maximum steering slew rate (rad/s).
        theta_h_max: Maximum articulation hitch angle before jackknife (rad).
        v_bounds: (v_min, v_max) allowable velocity range (m/s).
        a_bounds: (a_min, a_max) allowable longitudinal acceleration range (m/s^2).
        governing_constraint: Name of active limit ("tire_slip", "rollover", or "steering_kinematics").
        is_liftoff_risk: True if normal acceleration drops below safe threshold (g_eff_z < 1.0 m/s^2).
    """
    t: float
    v_t: float
    g_eff_z: float
    a_deck_y: float
    a_y_crit_slip: float
    a_y_crit_rollover: float
    a_y_max: float
    kappa_max: float
    delta_max: float
    dot_delta_max: float
    theta_h_max: float
    v_bounds: Tuple[float, float]
    a_bounds: Tuple[float, float]
    governing_constraint: str
    is_liftoff_risk: bool


class KinodynamicLimits:
    """Master evaluator for dynamic feasibility envelopes under 6-DOF deck motion."""

    def __init__(
        self,
        tractor_cfg: Optional[TractorConfig] = None,
        aircraft_cfg: Optional[AircraftConfig] = None,
        tire_cfg: Optional[TireConfig] = None,
        carrier_cfg: Optional[CarrierConfig] = None,
        ltr_limit: float = 0.70,
        v_min_floor: float = 0.10,
        a_y_min_floor: float = 0.0,  # Zero by default: strictly no artificial flooring!
    ) -> None:
        self.t_cfg = tractor_cfg or TractorConfig()
        self.a_cfg = aircraft_cfg or AircraftConfig()
        self.tire_cfg = tire_cfg or TireConfig()
        self.c_cfg = carrier_cfg or CarrierConfig()
        self.ltr_limit = float(ltr_limit)
        self.v_min_floor = float(v_min_floor)
        self.a_y_min_floor = float(a_y_min_floor)

        # Precompute aircraft geometric rollover factor
        # LTR_a = (2 * l_a * h_cg_a) / (a_a * W_r) * (a_y / g_eff_z)
        # Therefore: a_y_crit_rollover = LTR_limit * (a_a * W_r) / (2 * l_a * h_cg_a) * g_eff_z
        self.a_a = float(self.a_cfg.cg_to_nose)       # 6.80 m
        self.W_r = float(self.a_cfg.main_track)       # 3.10 m
        self.l_a = float(self.a_cfg.wheelbase)        # 8.00 m
        self.h_cg_a = float(self.a_cfg.cg_height)     # 1.85 m
        self.k_rollover = (self.a_a * self.W_r) / (2.0 * self.l_a * self.h_cg_a)  # ~0.712162

        # Kinematic turning limit from tractor steering geometry: kappa_kin = tan(delta_max) / l_t
        self.l_t = float(self.t_cfg.wheelbase)        # 1.80 m
        self.delta_max = float(self.t_cfg.delta_max_rad)        # 35 deg (0.610865 rad)
        self.dot_delta_max = float(self.t_cfg.omega_steer_max_rad)  # 30 deg/s (0.523599 rad/s)
        self.theta_h_max = float(self.t_cfg.beta_max_rad)       # 70 deg (1.221730 rad)
        self.kappa_kin_max = float(np.tan(self.delta_max) / self.l_t)  # ~0.389004 m^-1

        self.v_max = float(self.t_cfg.v_max)          # 2.0 m/s
        self.a_max = float(self.t_cfg.a_max)          # 1.0 m/s^2

    def compute_critical_rollover_acceleration(self, g_eff_z: float) -> float:
        """Analytical rollover limit: a_y_crit,rollover = max(floor, LTR_limit * K_roll * g_eff_z)."""
        g_safe = max(float(g_eff_z), 0.0)
        return float(max(self.a_y_min_floor, self.ltr_limit * self.k_rollover * g_safe))

    def compute_critical_slip_acceleration(
        self,
        g_eff_z: float,
        a_deck_y: float,
        mu: Optional[float] = None,
    ) -> float:
        """Dynamic sliding limit: a_y_crit = max(floor, mu * g_eff_z - |a_deck_y|).
        
        When floor is 0.0, this returns negative values when |a_deck_y| > mu * g_eff_z,
        faithfully representing uncontrollable dynamic deck sliding.
        """
        mu_val = float(mu if mu is not None else self.tire_cfg.mu_dry)
        g_safe = max(float(g_eff_z), 0.0)
        raw_slip = mu_val * g_safe - abs(float(a_deck_y))
        if self.a_y_min_floor > 0.0:
            return float(max(self.a_y_min_floor, raw_slip))
        return float(raw_slip)

    def compute_dynamic_curvature_limit(self, v_t: float, a_y_max: float) -> Tuple[float, str]:
        """Compute dynamic curvature limit kappa_max = min(kappa_kin, a_y_max / max(v^2, v_min^2))."""
        v_eff_sq = max(float(v_t)**2, self.v_min_floor**2)
        kappa_dyn = float(max(0.0, a_y_max)) / v_eff_sq
        if kappa_dyn < self.kappa_kin_max:
            return float(kappa_dyn), "dynamic_accel"
        return float(self.kappa_kin_max), "steering_kinematics"

    def compute_admissible_velocity_ceiling(
        self,
        kappa: float,
        t: float,
        deck_motion: DeckMotionSimulator,
        pos_3d: Optional[np.ndarray] = None,
        mu: Optional[float] = None,
    ) -> float:
        """Compute instantaneous admissible velocity ceiling v_ceiling(t) = sqrt(a_y_max / max(|kappa|, 1e-4))."""
        pos = np.asarray(pos_3d, dtype=np.float64) if pos_3d is not None else np.array([160.0, 45.0, 0.0])
        vel = np.zeros(3, dtype=np.float64)
        apparent = deck_motion.compute_non_inertial_acceleration(pos, vel, float(t))
        g_eff_z = float(-apparent.a_apparent_total[2])
        a_deck_y = float(apparent.a_apparent_total[1])

        a_y_slip = self.compute_critical_slip_acceleration(g_eff_z, a_deck_y, mu=mu)
        a_y_rollover = self.compute_critical_rollover_acceleration(g_eff_z)
        a_y_max = max(0.0, min(a_y_slip, a_y_rollover))

        kap = abs(float(kappa))
        if a_y_max <= 1e-4:
            if kap > 1e-3:
                return 0.0
            return self.v_max if g_eff_z > 1.0 else 0.10

        v_lat = float(np.sqrt(a_y_max / max(kap, 1e-4)))
        return float(min(self.v_max, v_lat))

    def compute_limits(
        self,
        t: float,
        deck_motion: DeckMotionSimulator,
        v_t: float = 1.0,
        tractor_pos: Optional[np.ndarray] = None,
        psi_t: float = 0.0,
        mu: Optional[float] = None,
    ) -> KinodynamicBound:
        """Compute instantaneous kinodynamic safety envelope at (t, v_t).
        
        Args:
            t: Query simulation timestamp (s).
            deck_motion: 6-DOF DeckMotionSimulator instance.
            v_t: Tractor forward velocity (m/s).
            tractor_pos: Optional 3D position [x, y, z] on deck (defaults to mid-deck [160, 45, 0]).
            psi_t: Tractor heading angle (rad).
            mu: Optional tire-deck friction coefficient (defaults to dry mu=0.80).
            
        Returns:
            KinodynamicBound instance.
        """
        pos = np.asarray(tractor_pos, dtype=np.float64) if tractor_pos is not None else np.array([160.0, 45.0, 0.0], dtype=np.float64)
        vel = np.array([float(v_t) * np.cos(psi_t), float(v_t) * np.sin(psi_t), 0.0], dtype=np.float64)

        # Apparent acceleration field at position
        apparent = deck_motion.compute_non_inertial_acceleration(pos, vel, float(t))
        g_eff_z = float(-apparent.a_apparent_total[2])
        a_deck_y = float(apparent.a_apparent_total[1])

        # Evaluate critical accelerations
        a_y_slip = self.compute_critical_slip_acceleration(g_eff_z, a_deck_y, mu=mu)
        a_y_rollover = self.compute_critical_rollover_acceleration(g_eff_z)

        # Governing lateral acceleration available for turning
        a_y_max = max(self.a_y_min_floor, min(a_y_slip, a_y_rollover))
        gov_cause = "tire_slip" if a_y_slip <= a_y_rollover else "rollover"

        kappa_max, kappa_cause = self.compute_dynamic_curvature_limit(v_t, a_y_max)
        if kappa_cause == "steering_kinematics":
            governing_constraint = "steering_kinematics"
        else:
            governing_constraint = gov_cause

        is_liftoff = bool(g_eff_z < 1.0)

        return KinodynamicBound(
            t=float(t),
            v_t=float(v_t),
            g_eff_z=g_eff_z,
            a_deck_y=a_deck_y,
            a_y_crit_slip=a_y_slip,
            a_y_crit_rollover=a_y_rollover,
            a_y_max=a_y_max,
            kappa_max=kappa_max,
            delta_max=self.delta_max,
            dot_delta_max=self.dot_delta_max,
            theta_h_max=self.theta_h_max,
            v_bounds=(0.0, self.v_max),
            a_bounds=(-self.a_max, self.a_max),
            governing_constraint=governing_constraint,
            is_liftoff_risk=is_liftoff,
        )

    def check_state_admissibility(
        self,
        state: VehicleState,
        t: float,
        deck_motion: DeckMotionSimulator,
        mu: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Check if vehicle state passes all 4-tier kinodynamic admissibility gates.
        
        Gates:
        1. Anti-jackknife travel limit: |theta_h| <= theta_h_max (70 deg)
        2. Steering angle limit: |delta_f| <= delta_max (35 deg)
        3. Dynamic lateral acceleration: a_y <= a_y_crit_slip
        4. Dynamic rollover threshold: LTR <= 0.70 (a_y <= a_y_crit_rollover)
        """
        # Gate 1: Anti-jackknife
        if abs(state.theta_h) > self.theta_h_max + 1e-4:
            return False, f"Jackknife violation: |theta_h|={abs(state.theta_h):.3f} > {self.theta_h_max:.3f}"

        # Gate 2: Steering angle limit
        if abs(state.delta_f) > self.delta_max + 1e-4:
            return False, f"Steering violation: |delta_f|={abs(state.delta_f):.3f} > {self.delta_max:.3f}"

        # Query dynamic limits
        pos = np.array([state.x_t, state.y_t, 0.0], dtype=np.float64)
        bounds = self.compute_limits(t, deck_motion, v_t=state.v_t, tractor_pos=pos, psi_t=state.psi_t, mu=mu)

        # Planar lateral acceleration: a_y = v_t^2 * kappa_t
        kappa_t = np.tan(state.delta_f) / self.l_t
        a_y_tractor = (state.v_t ** 2) * abs(kappa_t)

        # Gate 3: Dynamic lateral slip
        if a_y_tractor > bounds.a_y_crit_slip + 1e-3:
            return False, f"Lateral slip violation: a_y={a_y_tractor:.3f} > a_crit_slip={bounds.a_y_crit_slip:.3f}"

        # Gate 4: Dynamic rollover
        if a_y_tractor > bounds.a_y_crit_rollover + 1e-3:
            return False, f"Rollover violation: a_y={a_y_tractor:.3f} > a_crit_rollover={bounds.a_y_crit_rollover:.3f}"

        return True, "admissible"
