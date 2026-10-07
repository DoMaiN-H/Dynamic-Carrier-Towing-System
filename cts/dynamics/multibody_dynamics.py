"""cts.dynamics.multibody_dynamics

Feature F9: Coupled 2-Body Multibody Dynamics Plant.
Coupled planar Newton-Euler equations under dynamic deck apparent acceleration fields,
closed-form nonsingular 2x2 hitch constraint force solution, and unified TractorAircraftPlant.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Union, Dict, Any
import numpy as np

from cts.common.config import CTSConfig, TractorConfig, AircraftConfig, TireConfig
from cts.common.math_utils import wrap_to_pi
from cts.environment.deck_motion import DeckMotionSimulator, DeckState
from cts.environment.non_inertial_field import NonInertialAccelerationField, ApparentAcceleration, GRAVITY
from cts.dynamics.kinematics import AckermannTurntableKinematics, VehiclePose
from cts.dynamics.load_transfer import DynamicLoadTransfer, WheelLoads
from cts.dynamics.tire_models import (
    TireModelInterface,
    DugoffTireModel,
    PacejkaTireModel,
    LinearTireModel,
    create_tire_model,
    TireForces,
)
from cts.dynamics.integrator import RK4Integrator


@dataclass
class VehicleState:
    """Standardized vehicle state vector matching PROJECT.md interface contract.

    [x_t, y_t, psi_t, theta_h, v_t, delta_f, beta_slip, omega_t]
    """
    x_t: float
    y_t: float
    psi_t: float
    theta_h: float
    v_t: float
    delta_f: float
    beta_slip: float = 0.0
    omega_t: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.x_t, self.y_t, self.psi_t, self.theta_h,
            self.v_t, self.delta_f, self.beta_slip, self.omega_t
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "VehicleState":
        return cls(
            x_t=float(arr[0]),
            y_t=float(arr[1]),
            psi_t=float(arr[2]),
            theta_h=float(arr[3]),
            v_t=float(arr[4]),
            delta_f=float(arr[5]),
            beta_slip=float(arr[6]) if len(arr) > 6 else 0.0,
            omega_t=float(arr[7]) if len(arr) > 7 else 0.0,
        )


@dataclass
class ControlAction:
    """Control inputs to the tractor."""
    a_cmd: float
    steer_rate_cmd: Optional[float] = None
    steer_angle_cmd: Optional[float] = None


class TractorAircraftPlant:
    """Coupled 2-body planar multibody dynamics plant under dynamic deck motion."""

    def __init__(
        self,
        config: Optional[CTSConfig] = None,
        tire_model: Optional[TireModelInterface] = None,
        load_transfer_model: Optional[DynamicLoadTransfer] = None,
        friction_coeff: Optional[float] = None,
    ) -> None:
        self.config = config or CTSConfig()
        self.tractor = self.config.tractor
        self.aircraft = self.config.aircraft
        self.tire_cfg = self.config.tire

        # Inertial parameters
        self.m_t = float(self.tractor.mass)
        self.I_t = float(self.tractor.izz)
        self.m_a = float(self.aircraft.mass)
        self.I_a = float(self.aircraft.izz)
        self.m_total = self.m_t + self.m_a

        # Geometry
        self.a_t = float(self.tractor.cg_to_front)
        self.b_t = float(self.tractor.cg_to_rear)
        self.l_t = float(self.tractor.wheelbase)
        self.h_t = float(self.tractor.track)
        self.d_c = float(self.tractor.hitch_offset)
        self.c_t = self.b_t + self.d_c  # 0.95 m

        self.a_a = float(self.aircraft.cg_to_nose)   # 6.80 m
        self.b_a = float(self.aircraft.cg_to_main)   # 1.20 m
        self.l_a = float(self.aircraft.wheelbase)    # 8.00 m
        self.W_r = float(self.aircraft.main_track)   # 3.10 m

        # Submodules
        self.kinematics = AckermannTurntableKinematics(self.tractor, self.aircraft, self.d_c)
        self.load_transfer = load_transfer_model or DynamicLoadTransfer(self.tractor, self.aircraft)
        self.tire_model = tire_model or DugoffTireModel(
            cornering_stiffness=self.tire_cfg.cornering_stiffness_tractor,
            longitudinal_stiffness=self.tire_cfg.longitudinal_stiffness_tractor,
            relaxation_length=self.tire_cfg.relaxation_length_tractor,
        )
        self.mu_default = float(friction_coeff if friction_coeff is not None else self.tire_cfg.mu_dry)
        self.apparent_field = NonInertialAccelerationField()

        # --- Fidelity remediation state -----------------------------------
        # Last commanded longitudinal acceleration (feeds pitch load transfer).
        self._prev_v_dot = 0.0
        # Integration horizon used by the operator-split tire relaxation.
        self._step_dt = 0.02
        # Enable first-order relaxation-length tire dynamics (Remediation A).
        self.use_tire_relaxation = True
        self._tire_sigma = float(getattr(self.tire_model, "sigma", 0.2))
        # Frozen per-axle dynamic tire forces; promoted once per step() call so
        # that compute_derivatives stays a pure function inside RK4 sub-steps.
        self._tire_prev = {"front": (0.0, 0.0), "rear": (0.0, 0.0), "main": (0.0, 0.0)}
        self._tire_next = dict(self._tire_prev)

    def _relax_tire_forces(
        self,
        key: str,
        Fx_ss: float,
        Fy_ss: float,
        Fz: float,
        mu: float,
        vx: float,
    ) -> Tuple[float, float]:
        """First-order relaxation-length filter on steady-state tire forces.

        F_dyn = F_ss + (F_prev - F_ss) * exp(-|vx|*dt/sigma), followed by an
        adhesion-circle projection. The stored state is frozen for the whole
        step (operator splitting) and promoted in step() after integration.
        """
        if not self.use_tire_relaxation:
            return Fx_ss, Fy_ss
        px, py = self._tire_prev[key]
        # Effective rolling speed floored at the creep boundary: without it
        # decay -> 1 as v -> 0 and the tire force freezes (unphysical residual
        # force that destabilizes pose-hold / parking maneuvers).
        vx_eff = min(max(abs(vx), 0.30), 50.0)
        decay = float(np.exp(-vx_eff * self._step_dt / max(self._tire_sigma, 1e-3)))
        Fx = Fx_ss + (px - Fx_ss) * decay
        Fy = Fy_ss + (py - Fy_ss) * decay
        f_max = mu * max(Fz, 0.0)
        f_res = float(np.hypot(Fx, Fy))
        if f_res > f_max > 1e-9:
            Fx *= f_max / f_res
            Fy *= f_max / f_res
        self._tire_next[key] = (float(Fx), float(Fy))
        return float(Fx), float(Fy)

    def compute_hitch_force(
        self,
        psi_t: float,
        psi_a: float,
        dpsi_t: float,
        dpsi_a: float,
        F_t_total: np.ndarray,
        M_t_ext: float,
        F_a_total: np.ndarray,
        M_a_ext: float,
    ) -> Tuple[np.ndarray, float, float]:
        """Compute closed-form hitch constraint force F_h = [F_hx, F_hy]^T and yaw accelerations.

        F_h is the constraint force exerted by aircraft on tractor at hitch H.
        -F_h is the constraint force exerted by tractor on aircraft.

        Returns:
            (F_h, d2psi_t, d2psi_a)
        """
        ut = np.array([np.cos(psi_t), np.sin(psi_t)], dtype=np.float64)
        nt = np.array([-np.sin(psi_t), np.cos(psi_t)], dtype=np.float64)
        ua = np.array([np.cos(psi_a), np.sin(psi_a)], dtype=np.float64)
        na = np.array([-np.sin(psi_a), np.cos(psi_a)], dtype=np.float64)

        # 2x2 Effective hitch compliance/inertia matrix M_h
        M_h = (
            (1.0 / self.m_t + 1.0 / self.m_a) * np.eye(2, dtype=np.float64)
            + (self.c_t**2 / self.I_t) * np.outer(nt, nt)
            + (self.a_a**2 / self.I_a) * np.outer(na, na)
        )

        # Unconstrained relative acceleration vector b_h
        b_h = (
            F_a_total / self.m_a - F_t_total / self.m_t
            + (self.a_a * M_a_ext / self.I_a) * na
            + (self.c_t * M_t_ext / self.I_t) * nt
            - (self.a_a * (dpsi_a**2)) * ua
            - (self.c_t * (dpsi_t**2)) * ut
        )

        # Closed-form solve
        F_h = np.linalg.solve(M_h, b_h)

        # Resulting angular accelerations
        d2psi_t = (M_t_ext - self.c_t * float(np.dot(nt, F_h))) / self.I_t
        d2psi_a = (M_a_ext - self.a_a * float(np.dot(na, F_h))) / self.I_a

        return F_h, float(d2psi_t), float(d2psi_a)

    def compute_dynamic_load_transfer(
        self,
        state: VehicleState,
        a_eff: np.ndarray,
        t: float = 0.0,
        F_hx_t: float = 0.0,
        F_hy_t: float = 0.0,
        unilateral: bool = True,
    ) -> WheelLoads:
        """Compute dynamic vertical load transfer strictly conserving (m_t + m_a) * g_eff,z.

        Matches PROJECT.md interface contract line 95:
        compute_dynamic_load_transfer(state: VehicleState, a_eff: ndarray[3], t: float) -> WheelLoads
        """
        a_eff = np.asarray(a_eff, dtype=np.float64)
        # a_eff is the effective apparent acceleration in deck coordinates
        # g_eff,z is -a_eff[2] (downward positive)
        g_eff_z = -float(a_eff[2]) if len(a_eff) > 2 else GRAVITY
        if g_eff_z <= 0.0:
            g_eff_z = 1e-6

        psi_t = state.psi_t
        psi_a = wrap_to_pi(state.psi_t + state.theta_h)

        # Decompose apparent horizontal acceleration into body frames
        c_t, s_t = np.cos(psi_t), np.sin(psi_t)
        a_app_x_t = a_eff[0] * c_t + a_eff[1] * s_t
        a_app_y_t = -a_eff[0] * s_t + a_eff[1] * c_t

        c_a, s_a = np.cos(psi_a), np.sin(psi_a)
        a_app_x_a = a_eff[0] * c_a + a_eff[1] * s_a
        a_app_y_a = -a_eff[0] * s_a + a_eff[1] * c_a

        # Kinematic relative acceleration
        # v_t * omega_t is lateral kinematic centripetal acceleration
        # Remediation C: the vehicle's own longitudinal acceleration (last
        # computed v_dot, one RK4 stage stale) contributes to pitch transfer.
        a_x_t = self._prev_v_dot - a_app_x_t
        a_y_t = float(state.v_t * state.omega_t) - a_app_y_t

        omega_a = self.kinematics.aircraft_yaw_rate(state.v_t, state.delta_f, state.theta_h)
        a_x_a = self._prev_v_dot - a_app_x_a
        a_y_a = float(state.v_t * omega_a) - a_app_y_a

        return self.load_transfer.compute(
            a_x_t=a_x_t, a_y_t=a_y_t,
            a_x_a=a_x_a, a_y_a=a_y_a,
            g_eff_z=g_eff_z,
            F_hx_t=F_hx_t, F_hy_t=F_hy_t,
            unilateral=unilateral,
        )

    def kinematics_step(
        self,
        state: VehicleState,
        action: ControlAction,
        dt: float,
    ) -> VehicleState:
        """Direct kinematic stepping without tire slip (for nominal low speed / baseline)."""
        steer_rate = action.steer_rate_cmd
        if steer_rate is None:
            if action.steer_angle_cmd is not None:
                # Proportional steering rate to reach angle
                steer_rate = (action.steer_angle_cmd - state.delta_f) / max(dt, 1e-4)
            else:
                steer_rate = 0.0

        kin_state = np.array([
            state.x_t, state.y_t, state.psi_t, state.theta_h, state.v_t, state.delta_f
        ], dtype=np.float64)

        next_kin = self.kinematics.step(kin_state, action.a_cmd, steer_rate, dt)
        omega_t = self.kinematics.tractor_yaw_rate(next_kin[4], next_kin[5])

        return VehicleState(
            x_t=float(next_kin[0]),
            y_t=float(next_kin[1]),
            psi_t=float(next_kin[2]),
            theta_h=float(next_kin[3]),
            v_t=float(next_kin[4]),
            delta_f=float(next_kin[5]),
            beta_slip=0.0,
            omega_t=float(omega_t),
        )

    def compute_derivatives(
        self,
        t: float,
        full_state: np.ndarray,
        action: ControlAction,
        deck_motion: Optional[DeckMotionSimulator] = None,
        mu: Optional[float] = None,
    ) -> np.ndarray:
        """Evaluate continuous state derivatives for ODE integration.

        State vector y: [x_t, y_t, psi_t, theta_h, v_t, delta_f, beta_slip, omega_t]
        """
        x_t, y_t, psi_t, theta_h, v_t, delta_f, beta_slip, omega_t = full_state[:8]
        psi_a = wrap_to_pi(psi_t + theta_h)
        friction = self.mu_default if mu is None else float(mu)

        # 1. Deck Motion & Apparent Acceleration Field
        r_t_2d = np.array([x_t, y_t], dtype=np.float64)
        v_t_2d = np.array([v_t * np.cos(psi_t + beta_slip), v_t * np.sin(psi_t + beta_slip)], dtype=np.float64)

        if deck_motion is not None:
            app_acc_t = deck_motion.compute_non_inertial_acceleration(
                pos_d=np.array([x_t, y_t, 0.0]),
                vel_rel_d=np.array([v_t_2d[0], v_t_2d[1], 0.0]),
                t=t,
            )
            a_app_t = app_acc_t.a_apparent_total
            g_eff_z = float(app_acc_t.g_eff)
        else:
            a_app_t = np.array([0.0, 0.0, -GRAVITY], dtype=np.float64)
            g_eff_z = GRAVITY

        # 2. Dynamic Vertical Load Transfer
        veh_state = VehicleState(x_t, y_t, psi_t, theta_h, v_t, delta_f, beta_slip, omega_t)
        loads = self.compute_dynamic_load_transfer(veh_state, a_app_t, t=t, unilateral=True)

        # 3. Wheel Kinematics & Tire Contact Forces
        # Tractor orientation vectors
        ut = np.array([np.cos(psi_t), np.sin(psi_t)], dtype=np.float64)
        nt = np.array([-np.sin(psi_t), np.cos(psi_t)], dtype=np.float64)

        # Front axle center velocity: v_t u_t + v_t_lat n_t + l_t omega_t n_t
        v_tx = v_t * np.cos(beta_slip)
        v_ty = v_t * np.sin(beta_slip)
        v_fx = v_tx
        v_fy = v_ty + self.l_t * omega_t

        # Front wheel slip angles (steered by delta_f)
        # In wheel frame:
        # vx_w = v_fx * cos(delta_f) + v_fy * sin(delta_f)
        # vy_w = -v_fx * sin(delta_f) + v_fy * cos(delta_f)
        cos_d, sin_d = np.cos(delta_f), np.sin(delta_f)
        v_fw_x = v_fx * cos_d + v_fy * sin_d
        v_fw_y = -v_fx * sin_d + v_fy * cos_d

        # Tire forces front (sum of FL and FR)
        Fz_front = loads.Fz_front
        Fz_rear = loads.Fz_rear
        Fz_main = loads.Fz_main

        # Evaluate front tires (Remediation 1: Positive restoring formulation)
        # Remediation A: steady-state targets pass through relaxation-length
        # dynamics instead of instantaneously producing full force.
        alpha_f = np.arctan2(v_fw_y, np.sqrt(v_fw_x**2 + 0.05**2))
        tf_f = self.tire_model.compute_steady_state_forces(alpha=alpha_f, kappa=0.0, Fz=Fz_front, mu=friction)
        fx_f_w, fy_f_w = self._relax_tire_forces("front", tf_f.Fx, tf_f.Fy, Fz_front, friction, v_fw_x)

        # Resolve front tire force to tractor body axes:
        Fx_f_body = fx_f_w * cos_d - fy_f_w * sin_d
        Fy_f_body = fx_f_w * sin_d + fy_f_w * cos_d

        # Rear tires (Remediation 1: Positive restoring formulation)
        alpha_r = np.arctan2(v_ty, np.sqrt(v_tx**2 + 0.05**2))
        tf_r = self.tire_model.compute_steady_state_forces(alpha=alpha_r, kappa=0.0, Fz=Fz_rear, mu=friction)
        Fx_r_body, Fy_r_body = self._relax_tire_forces("rear", tf_r.Fx, tf_r.Fy, Fz_rear, friction, v_tx)

        # Drive tractive force command (applied to rear wheels)
        # Remediation B: drive/brake demand is allocated through the rear-axle
        # adhesion circle. The deliverable longitudinal force is capped by the
        # residual grip sqrt((mu*Fz_rear)^2 - Fy_rear^2), so an over-demanding
        # a_cmd on low-mu decks saturates physically instead of applying
        # unattainable force.
        F_drive_req = float(np.clip(self.m_total * action.a_cmd, -self.tractor.tractive_force_max, self.tractor.tractive_force_max))
        fx_cap = float(np.sqrt(max((friction * Fz_rear) ** 2 - Fy_r_body ** 2, 0.0)))
        F_drive = float(np.clip(F_drive_req, -fx_cap, fx_cap))
        Fx_r_body += F_drive

        # Remediation 5: Standstill creep damping / parking brake
        # Activated when unpowered (|a_cmd| < 0.01) AND near standstill (|v_t| < 0.30 m/s)
        a_tol = 0.01
        v_creep = 0.30
        v_reg = 0.06

        abs_a = abs(action.a_cmd)
        abs_v = abs(v_t)
        if abs_a < a_tol and abs_v < v_creep:
            w_cmd = 1.0 - abs_a / a_tol
            w_spd = 1.0 - abs_v / v_creep
            w_park = w_cmd * w_spd

            Fz_total_t = Fz_front + Fz_rear
            F_brake_t = friction * Fz_total_t * w_park
            Fx_r_body -= F_brake_t * 0.5 * np.tanh(v_tx / v_reg)
            Fy_r_body -= F_brake_t * 0.5 * np.tanh(v_ty / v_reg)
            Fx_f_body -= F_brake_t * 0.5 * np.tanh(v_fw_x / v_reg)
            Fy_f_body -= F_brake_t * 0.5 * np.tanh(v_fw_y / v_reg)

        # Total tire force on tractor in Deck frame
        F_t_tire_body = np.array([Fx_f_body + Fx_r_body, Fy_f_body + Fy_r_body], dtype=np.float64)
        F_t_tire_d = F_t_tire_body[0] * ut + F_t_tire_body[1] * nt

        # Yaw moment about tractor CG
        M_t_tire = self.a_t * Fy_f_body - self.b_t * Fy_r_body

        # Deck apparent body force on tractor
        F_t_app_d = self.m_t * a_app_t[:2]
        F_t_total_d = F_t_tire_d + F_t_app_d
        M_t_ext = M_t_tire

        # Aircraft tires
        ua = np.array([np.cos(psi_a), np.sin(psi_a)], dtype=np.float64)
        na = np.array([-np.sin(psi_a), np.cos(psi_a)], dtype=np.float64)

        # Aircraft CG velocity
        v_H = (v_tx * ut + (v_ty + self.b_t * omega_t) * nt) - self.c_t * omega_t * nt
        omega_a = self.kinematics.aircraft_yaw_rate(v_t, delta_f, theta_h)
        v_m = v_H - self.l_a * omega_a * na

        v_mx_a = float(np.dot(v_m, ua))
        v_my_a = float(np.dot(v_m, na))

        # Remediation 1: Aircraft tire slip angle (positive restoring)
        # Remediation A: relaxation-length dynamics on the main gear too.
        alpha_a = np.arctan2(v_my_a, np.sqrt(v_mx_a**2 + 0.05**2))
        tf_a = self.tire_model.compute_steady_state_forces(alpha=alpha_a, kappa=0.0, Fz=Fz_main, mu=friction)
        Fx_a_body, Fy_a_body = self._relax_tire_forces("main", tf_a.Fx, tf_a.Fy, Fz_main, friction, v_mx_a)

        # Aircraft standstill brake when parked
        if abs_a < a_tol and abs_v < v_creep:
            w_cmd = 1.0 - abs_a / a_tol
            w_spd = 1.0 - abs_v / v_creep
            w_park = w_cmd * w_spd
            F_brake_a = friction * Fz_main * w_park
            Fx_a_body -= F_brake_a * np.tanh(v_mx_a / v_reg)
            Fy_a_body -= F_brake_a * np.tanh(v_my_a / v_reg)

        F_a_tire_d = Fx_a_body * ua + Fy_a_body * na
        M_a_tire = - self.b_a * Fy_a_body

        # Deck apparent body force on aircraft
        if deck_motion is not None:
            pos_a_d = np.array([x_t - self.d_c * np.cos(psi_t) - self.a_a * np.cos(psi_a),
                                y_t - self.d_c * np.sin(psi_t) - self.a_a * np.sin(psi_a), 0.0])
            app_acc_a = deck_motion.compute_non_inertial_acceleration(pos_d=pos_a_d, vel_rel_d=np.array([v_m[0], v_m[1], 0.0]), t=t)
            a_app_a = app_acc_a.a_apparent_total
        else:
            a_app_a = np.array([0.0, 0.0, -GRAVITY], dtype=np.float64)

        F_a_app_d = self.m_a * a_app_a[:2]
        F_a_total_d = F_a_tire_d + F_a_app_d
        M_a_ext = M_a_tire

        # 4. Solve Hitch Constraint Force
        F_h, d2psi_t, d2psi_a = self.compute_hitch_force(
            psi_t=psi_t, psi_a=psi_a,
            dpsi_t=omega_t, dpsi_a=omega_a,
            F_t_total=F_t_total_d, M_t_ext=M_t_ext,
            F_a_total=F_a_total_d, M_a_ext=M_a_ext,
        )

        # 5. State Derivatives
        a_cg_t_d = (F_t_total_d + F_h) / self.m_t
        a_cg_tx = float(np.dot(a_cg_t_d, ut))
        a_cg_ty = float(np.dot(a_cg_t_d, nt))

        # Rear axle inertial acceleration components (used for sideslip rate)
        a_R_x = a_cg_tx + self.b_t * (omega_t**2)
        a_R_y = a_cg_ty - self.b_t * d2psi_t

        # Remediation 2: Exact kinematic acceleration of rear axle longitudinal speed v_t
        # Projected along the trajectory tangent without fictitious centripetal pumping
        v_dot = a_cg_tx * np.cos(beta_slip) + a_cg_ty * np.sin(beta_slip)
        self._prev_v_dot = v_dot  # feeds next-step pitch load transfer (Remediation C)

        x_t_dot = v_t * np.cos(psi_t + beta_slip)
        y_t_dot = v_t * np.sin(psi_t + beta_slip)
        psi_t_dot = omega_t
        # Remediation 3: Clean articulation derivative without (t - t) shortcut
        theta_h_dot = omega_a - omega_t
        # Remediation D: bounded dynamic articulation feedback. The kinematic
        # no-slip identity above ignores the residual yaw-acceleration mismatch
        # between the two bodies (main-gear saturation, hitch moment imbalance).
        # The deficit is converted into a bounded drift rate over the
        # convective timescale b_a/|v_t|, vanishing at standstill.
        if abs(v_t) > 0.05:
            yaw_resid = d2psi_a - d2psi_t
            theta_h_dot += float(np.clip(yaw_resid * self.b_a / abs(v_t), -0.5, 0.5))

        # Steer rate handling
        if action.steer_rate_cmd is not None:
            steer_dot = float(np.clip(action.steer_rate_cmd, -self.tractor.omega_steer_max_rad, self.tractor.omega_steer_max_rad))
        elif action.steer_angle_cmd is not None:
            steer_dot = float(np.clip((action.steer_angle_cmd - delta_f) / 0.02, -self.tractor.omega_steer_max_rad, self.tractor.omega_steer_max_rad))
        else:
            steer_dot = 0.0

        # Remediation 6: Exact, uncoupled sideslip angle derivative
        beta_slip_dot = (a_R_y * np.cos(beta_slip) - a_R_x * np.sin(beta_slip)) / max(abs(v_t) + 0.05, 0.1) - omega_t
        omega_t_dot = d2psi_t

        return np.array([
            x_t_dot, y_t_dot, psi_t_dot, theta_h_dot,
            v_dot, steer_dot, beta_slip_dot, omega_t_dot
        ], dtype=np.float64)

    def step(
        self,
        state: VehicleState,
        action: ControlAction,
        deck_motion: Optional[DeckMotionSimulator] = None,
        t: float = 0.0,
        dt: float = 0.02,
        mu: Optional[float] = None,
    ) -> VehicleState:
        """Full coupled dynamic step integrated with non-inertial deck motion and tires.

        Matches PROJECT.md interface contract line 92:
        step(state: VehicleState, action: ControlAction, deck_motion: DeckMotionSimulator, t: float, dt: float) -> VehicleState
        """
        y0 = state.to_array()
        self._step_dt = dt

        def ode_func(time_curr: float, y_curr: np.ndarray) -> np.ndarray:
            return self.compute_derivatives(time_curr, y_curr, action, deck_motion, mu)

        integrator = RK4Integrator(ode_func, dt=dt)
        y_next = integrator.step(t, y0, dt)

        # Promote tire relaxation state computed at the final RK4 sub-stage.
        self._tire_prev = dict(self._tire_next)

        # Wrap angles
        y_next[2] = wrap_to_pi(y_next[2])  # psi_t
        y_next[3] = wrap_to_pi(y_next[3])  # theta_h
        y_next[5] = float(np.clip(y_next[5], -self.tractor.delta_max_rad, self.tractor.delta_max_rad))
        # Remediation 4: Wrap sideslip angle
        y_next[6] = wrap_to_pi(y_next[6])  # beta_slip

        # Check anti-jackknife travel limit
        if abs(y_next[3]) > self.tractor.beta_max_rad:
            y_next[3] = np.sign(y_next[3]) * self.tractor.beta_max_rad

        return VehicleState.from_array(y_next)
