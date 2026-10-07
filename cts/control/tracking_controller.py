"""cts.control.tracking_controller

Master Closed-Loop Trajectory Tracking Controller (Feature F18).
Synthesizes:
- Flight deck inertial tilt feedforward (Feature F14)
- 100 Hz Extended State Observer (Feature F15)
- CasADi NMPC primary controller with dynamic tire adhesion constraints (Feature F16)
- Portable LTV-MPC fallback controller (Feature F17)
- Anti-jackknife supervisory recovery logic
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

from cts.common.config import CTSConfig, TractorConfig, AircraftConfig, TireConfig
from cts.common.math_utils import wrap_to_pi
from cts.dynamics.multibody_dynamics import VehicleState, ControlAction
from cts.environment.deck_motion import DeckMotionSimulator, DeckState
from cts.environment.non_inertial_field import ApparentAcceleration, compute_apparent_acceleration
from cts.planning.minco_optimizer import PlannedTrajectory

from cts.control.feedforward import InertialFeedforward, FeedforwardConfig, FeedforwardOutput
from cts.control.disturbance_observer import DisturbanceObserver, ESOConfig, DisturbanceEstimate
from cts.control.nmpc_casadi import CasADiNMPCController, NMPCConfig, NMPCResult
from cts.control.ltv_mpc import LTVMPCController, LTVMPCConfig, LTVMPCResult


@dataclass
class MasterControllerConfig:
    """Master controller configuration parameters."""
    control_rate_hz: float = 50.0          # 50 Hz control loop (dt = 0.02 s)
    enable_feedforward: bool = True
    enable_observer: bool = True
    enable_anti_jackknife_recovery: bool = True
    anti_jackknife_trigger_rad: float = 0.698132   # 40 deg
    jackknife_gain: float = 1.8            # Counter-steering gain
    sla_timeout_ms: float = 20.0           # Hard SLA limit <= 20 ms
    slew_rate_limit: float = 0.523599      # 30 deg/s
    preferred_controller: str = "nmpc"     # "nmpc" or "ltv_mpc"


@dataclass
class MasterControlOutput:
    """Comprehensive control telemetry container."""
    action: ControlAction
    a_cmd: float
    delta_cmd: float
    steer_rate_cmd: float
    delta_ff: float
    a_ff: float
    delta_dob: float
    a_dob: float
    delta_fb: float
    a_fb: float
    mode: str
    solve_time_ms: float
    is_jackknife_guard_active: bool


class TrajectoryTrackingController:
    """Master Closed-Loop Trajectory Tracking Controller (Feature F18)."""

    def __init__(
        self,
        config: Optional[MasterControllerConfig] = None,
        cts_config: Optional[CTSConfig] = None,
        primary_controller: Optional[CasADiNMPCController] = None,
        fallback_controller: Optional[LTVMPCController] = None,
    ) -> None:
        self.config = config or MasterControllerConfig()
        self.cts_config = cts_config or CTSConfig()
        self.tractor = self.cts_config.tractor
        self.dt = 1.0 / self.config.control_rate_hz

        # Submodules
        self.feedforward = InertialFeedforward(cts_config=self.cts_config)
        # Configure ESO for position-level cross-track error observation (order=3)
        eso_cfg = ESOConfig(order=3, omega_obs=25.0, Ts=self.dt)
        self.observer = DisturbanceObserver(config=eso_cfg, cts_config=self.cts_config)
        
        self.primary_controller = primary_controller
        if self.primary_controller is None and self.config.preferred_controller == "nmpc":
            self.primary_controller = CasADiNMPCController(
                tractor_cfg=self.tractor,
                aircraft_cfg=self.cts_config.aircraft,
                tire_cfg=self.cts_config.tire,
            )

        self.fallback_controller = fallback_controller or LTVMPCController(cts_config=self.cts_config)
        self.last_action = ControlAction(a_cmd=0.0, steer_angle_cmd=0.0, steer_rate_cmd=0.0)

    def compute_control(
        self,
        current_state: VehicleState,
        reference_trajectory: PlannedTrajectory,
        deck_env: Optional[Any] = None,
        sea_state: Optional[DeckMotionSimulator] = None,
        t: float = 0.0,
    ) -> MasterControlOutput:
        """Main compute step matching PROJECT.md interface contract."""
        t_start = time.perf_counter()

        # 1. Deck Motion and Apparent Acceleration
        deck_state = None
        app_acc = None
        if sea_state is not None:
            deck_state = sea_state.get_deck_state(t)
            pos_d = np.array([current_state.x_t, current_state.y_t, 0.0], dtype=np.float64)
            vel_rel_d = np.array([
                current_state.v_t * np.cos(current_state.psi_t),
                current_state.v_t * np.sin(current_state.psi_t),
                0.0
            ], dtype=np.float64)
            app_acc = compute_apparent_acceleration(deck_state, pos_d, vel_rel_d)

        # 2. Feedforward Channel (F14)
        delta_ff = 0.0
        a_ff = 0.0
        if self.config.enable_feedforward and app_acc is not None and deck_state is not None:
            ff_out = self.feedforward.compute_steering_compensation(
                state=current_state,
                deck_state=deck_state,
                apparent_acc=app_acc,
                dt=self.dt,
            )
            delta_ff = ff_out.delta_ff
            a_ff = ff_out.a_ff

        # 3. Disturbance Observer Channel (F15)
        # The ESO lumped residual lateral acceleration estimate is passed
        # INTO the MPC prediction models (in-model disturbance preview), not
        # added as an out-of-optimization steering bump. The previous
        # additive delta_dob channel double-counted the feedforward
        # compensation and destabilized the loop (ablation: C2 worse than C1
        # on both MPCs). delta_dob/a_dob are kept in telemetry only.
        delta_dob = 0.0
        a_dob = 0.0
        d_hat_y = 0.0
        if self.config.enable_observer:
            sample0 = reference_trajectory.eval_at(t)
            dx = current_state.x_t - sample0["x"]
            dy = current_state.y_t - sample0["y"]
            psi_r = sample0["psi_t"]
            e_lat = -np.sin(psi_r) * dx + np.cos(psi_r) * dy

            obs_est = self.observer.update(
                y=e_lat,
                u=self.last_action,
                dt=self.dt,
                v_long=current_state.v_t,
                u_acc=self.last_action.a_cmd,
            )
            delta_dob = obs_est.delta_dob
            a_dob = obs_est.a_dob
            d_hat_y = obs_est.d_hat_y
            # Reliability gate: e_lat mixes true disturbance response with
            # controller tracking lag, which is NOT a forward-predictable
            # disturbance. Injecting a lag-contaminated d_hat creates a
            # positive observer-controller feedback (each over-correction
            # produces a bigger estimate). Gate by tracking-error magnitude
            # and clamp to the physical residual scale (~<=0.8 m/s^2 once
            # the modeled deck accel is already compensated).
            e_mag = abs(float(e_lat))
            d_hat_y *= max(0.0, 1.0 - e_mag / 1.0)
            d_hat_y = float(np.clip(d_hat_y, -0.8, 0.8))

        # 4. Feedback Optimization Channel (F16 / F17)
        mode = "NMPC" if self.primary_controller is not None else "LTV_MPC"
        delta_fb = 0.0
        a_fb = 0.0
        use_primary = self.primary_controller is not None and self.config.preferred_controller == "nmpc"

        if use_primary:
            try:
                t_prim_start = time.perf_counter()
                prim_res = self.primary_controller.compute_control(
                    state=current_state,
                    target_traj=reference_trajectory,
                    t=t,
                    apparent_acc=app_acc,
                    disturbance_acc_y=d_hat_y,
                )
                t_prim = (time.perf_counter() - t_prim_start) * 1000.0
                if prim_res.success and t_prim <= self.config.sla_timeout_ms:
                    delta_fb = prim_res.delta_cmd
                    a_fb = prim_res.a_cmd
                    mode = "NMPC"
                else:
                    use_primary = False
            except Exception:
                use_primary = False

        if not use_primary:
            # Fallback to high-speed LTV-MPC
            mpc_res = self.fallback_controller.compute_action(
                current_state=current_state,
                reference_trajectory=reference_trajectory,
                t=t,
                disturbance_acc_y=d_hat_y,
                apparent_tilt_acc_y=(
                    app_acc.a_apparent_total[1]
                    if (app_acc is not None and self.config.enable_feedforward)
                    else 0.0
                ),
            )
            delta_fb = mpc_res.delta_cmd
            a_fb = mpc_res.a_cmd
            mode = "LTV_MPC"

        # 5. Composite Control Blending
        # delta_dob is intentionally NOT blended here: the ESO residual now
        # enters the MPC prediction models as an in-loop disturbance
        # parameter (see step 3). It remains available in telemetry.
        delta_total = delta_fb + delta_ff
        # Longitudinal speed regulation:
        # a_fb provides the primary trajectory tracking acceleration from NMPC / LTV-MPC.
        # Add proportional speed error damping against reference velocity to eliminate
        # steady-state speed drift induced by flight deck pitch and wave heave.
        ref_sample = reference_trajectory.eval_at(t)
        v_ref = float(ref_sample.get("v", 1.0))
        a_total = a_fb + 1.2 * (v_ref - current_state.v_t)

        # 6. Anti-Jackknife Recovery Supervisor
        is_jackknife_guard = False
        if self.config.enable_anti_jackknife_recovery:
            theta_h = current_state.theta_h
            abs_theta = abs(theta_h)
            if abs_theta > self.config.anti_jackknife_trigger_rad:
                is_jackknife_guard = True
                mode = f"{mode}+JACKKNIFE_GUARD"
            
            # Anti-whipping steering envelope: when hitch articulation is large,
            # aggressively restrict tractor steering so cross-track feedback cannot
            # excite lateral pendulum resonance of the 20-ton aircraft.
            if abs_theta > np.radians(18.0):
                # Smooth progressive steering limit from 35 deg down to 6 deg
                steer_cap = max(np.radians(6.0), self.tractor.delta_max_rad - 0.7 * abs_theta)
                delta_total = float(np.clip(delta_total, -steer_cap, steer_cap))
                # Maintain positive towing speed (0.7 - 1.1 m/s) to pull trailer straight
                if current_state.v_t < 0.7:
                    a_total = max(a_total, 0.2)
                elif current_state.v_t > 1.1:
                    a_total = min(a_total, -0.2)

        # 7. Slew Rate and Physical Actuation Clamping
        delta_prev = self.last_action.steer_angle_cmd if self.last_action.steer_angle_cmd is not None else 0.0
        max_delta_change = self.config.slew_rate_limit * self.dt
        delta_cmd = float(np.clip(delta_total, delta_prev - max_delta_change, delta_prev + max_delta_change))
        delta_cmd = float(np.clip(delta_cmd, -self.tractor.delta_max_rad, self.tractor.delta_max_rad))
        steer_rate_cmd = float((delta_cmd - delta_prev) / self.dt)
        a_cmd = float(np.clip(a_total, -self.tractor.a_max, self.tractor.a_max))

        final_action = ControlAction(
            a_cmd=a_cmd,
            steer_angle_cmd=delta_cmd,
            steer_rate_cmd=steer_rate_cmd,
        )
        self.last_action = final_action
        t_total_ms = (time.perf_counter() - t_start) * 1000.0

        return MasterControlOutput(
            action=final_action,
            a_cmd=a_cmd,
            delta_cmd=delta_cmd,
            steer_rate_cmd=steer_rate_cmd,
            delta_ff=delta_ff,
            a_ff=a_ff,
            delta_dob=delta_dob,
            a_dob=a_dob,
            delta_fb=delta_fb,
            a_fb=a_fb,
            mode=mode,
            solve_time_ms=t_total_ms,
            is_jackknife_guard_active=is_jackknife_guard,
        )

    def compute_action(
        self,
        current_state: VehicleState,
        target_traj: PlannedTrajectory,
        deck_state: Optional[DeckState] = None,
        apparent_acc: Optional[ApparentAcceleration] = None,
        t: float = 0.0,
    ) -> ControlAction:
        """Alias matching PROJECT.md interface contract line 107."""
        out = self.compute_control(
            current_state=current_state,
            reference_trajectory=target_traj,
            deck_env=None,
            sea_state=None,
            t=t,
        )
        return out.action
