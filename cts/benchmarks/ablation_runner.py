"""cts.benchmarks.ablation_runner

Factorial Ablation Experiment Suite (Feature F20).
Executes systematic 3x3 factorial comparative studies:
1. Dynamic Base Compensation:
   - C0: No compensation (nominal feedback only)
   - C1: Dynamic deck tilt feedforward only (F14)
   - C2: Full feedforward + 100 Hz Extended State Observer (F14 + F15)
2. Tire Contact Models:
   - T0: Linear cornering stiffness
   - T1: Dugoff combined-slip nonlinear model
   - T2: Pacejka Magic Formula MF-5.2
3. Trajectory Tracking Controllers:
   - CTRL0: Classical Stanley / Pure Pursuit + PID Baseline
   - CTRL1: Linear Time-Varying MPC (LTV-MPC)
   - CTRL2: Full Nonlinear Model Predictive Control (CasADi NMPC)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import time
import json
from pathlib import Path
import numpy as np

from cts.common.config import CTSConfig
from cts.dynamics.multibody_dynamics import VehicleState, TractorAircraftPlant, ControlAction
from cts.dynamics.tire_models import LinearTireModel, DugoffTireModel, PacejkaTireModel
from cts.environment.deck_env import DeckEnvironment
from cts.environment.deck_motion import DeckMotionSimulator
from cts.planning.corridor_planner import SpatiotemporalCorridorPlanner
from cts.control.tracking_controller import TrajectoryTrackingController, MasterControllerConfig
from cts.control.baselines import StanleyController, PurePursuitController, PIDController
from cts.control.ltv_mpc import LTVMPCController
from cts.control.nmpc_casadi import CasADiNMPCController
from cts.benchmarks.scenarios import BenchmarkScenario, load_canonical_scenario, run_scenario
from cts.benchmarks.metrics import BenchmarkMetrics, compute_trajectory_metrics


@dataclass
class AblationConfiguration:
    """Specification of an ablation experiment condition."""
    compensation_level: str   # "C0" (None), "C1" (Feedforward), "C2" (Full FF+ESO)
    tire_model: str           # "T0" (Linear), "T1" (Dugoff), "T2" (Pacejka)
    controller_type: str      # "CTRL0" (Stanley/PID), "CTRL1" (LTV_MPC), "CTRL2" (CasADi NMPC)
    scenario_id: str = "SC-1"
    # Unmodeled lateral acceleration bias (m/s^2) applied ONLY to the plant's
    # apparent field -- a constant disturbance the feedforward model cannot
    # see (e.g. steady wind loading / payload CG offset). Used for the
    # model-mismatch ablation: FF degrades while the ESO can still observe
    # and cancel the residual through the tracking error.
    unmodeled_bias_y: float = 0.0
    # Optional sinusoidal component (wind gusts / shifting loads):
    # a_bias(t) = bias_y + amp * sin(2 pi freq t). Time-varying unmodeled
    # terms are the regime where an observer should beat pure feedback.
    unmodeled_bias_amp: float = 0.0
    unmodeled_bias_hz: float = 0.5

    @property
    def label(self) -> str:
        base = f"{self.compensation_level}_{self.tire_model}_{self.controller_type}"
        if abs(self.unmodeled_bias_y) > 1e-9:
            base += f"_BIAS{self.unmodeled_bias_y:+.2f}"
        if abs(self.unmodeled_bias_amp) > 1e-9:
            base += f"_GUST{self.unmodeled_bias_amp:.2f}@{self.unmodeled_bias_hz:g}Hz"
        return base


class _BiasedDeckMotion:
    """Plant-side deck view carrying an unmodeled lateral acceleration bias.

    Wraps a DeckMotionSimulator: the plant integrates the biased apparent
    field (physical truth) while controllers keep the clean simulator, so
    the bias is a genuine model mismatch invisible to feedforward.
    """

    def __init__(
        self,
        base: DeckMotionSimulator,
        bias_y: float,
        bias_amp: float = 0.0,
        bias_hz: float = 0.5,
    ) -> None:
        self._base = base
        self.bias_y = float(bias_y)
        self.bias_amp = float(bias_amp)
        self.bias_hz = float(bias_hz)

    def _bias(self, t: float) -> float:
        return self.bias_y + self.bias_amp * float(
            np.sin(2.0 * np.pi * self.bias_hz * float(t))
        )

    def compute_non_inertial_acceleration(self, pos_d, vel_rel_d, t):
        from cts.environment.non_inertial_field import ApparentAcceleration
        out = self._base.compute_non_inertial_acceleration(
            pos_d=pos_d, vel_rel_d=vel_rel_d, t=t,
        )
        a_tot = np.array(out.a_apparent_total, dtype=np.float64, copy=True)
        a_tot[..., 1] += self._bias(t)
        return ApparentAcceleration(
            a_apparent_total=a_tot,
            g_tilt=out.g_tilt,
            a_trans=out.a_trans,
            a_euler=out.a_euler,
            a_centripetal=out.a_centripetal,
            a_coriolis=out.a_coriolis,
        )

    def __getattr__(self, name):
        return getattr(self._base, name)


@dataclass
class AblationResult:
    """Telemetry and metrics result from a single ablation run."""
    config: AblationConfiguration
    metrics: BenchmarkMetrics
    execution_time_s: float
    status: str = "COMPLETED"


class AblationSuiteRunner:
    """Automated runner for multi-dimensional ablation matrices."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or Path("outputs/ablations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_single_ablation(
        self,
        config: AblationConfiguration,
        scenario: Optional[BenchmarkScenario] = None,
        sim_duration_s: float = 15.0,
        sim_dt: float = 0.02,
    ) -> AblationResult:
        """Run an individual ablation trial under specified configuration."""
        t_start = time.perf_counter()
        sc = scenario or load_canonical_scenario(config.scenario_id)
        sc.duration_s = sim_duration_s

        cts_cfg = sc.config or CTSConfig()
        deck_env = DeckEnvironment.from_default()
        sea_sim = DeckMotionSimulator(sea_state=sc.sea_state_level, dt=sim_dt, seed=42)

        # 1. Setup Tire Model in Plant
        if config.tire_model == "T0":
            tire_model = LinearTireModel(
                cornering_stiffness=cts_cfg.tire.cornering_stiffness_tractor,
                longitudinal_stiffness=cts_cfg.tire.longitudinal_stiffness_tractor,
            )
        elif config.tire_model == "T1":
            tire_model = DugoffTireModel(
                cornering_stiffness=cts_cfg.tire.cornering_stiffness_tractor,
                longitudinal_stiffness=cts_cfg.tire.longitudinal_stiffness_tractor,
            )
        else: # T2: Pacejka MF-5.2
            tire_model = PacejkaTireModel()

        plant = TractorAircraftPlant(
            config=cts_cfg,
            tire_model=tire_model,
            friction_coeff=sc.friction_coeff,
        )

        # 2. Plan Reference Trajectory
        planner = SpatiotemporalCorridorPlanner(config=cts_cfg)
        traj = planner.plan(
            start_state=sc.start_state,
            goal_state=sc.goal_state,
            deck_env=deck_env,
            sea_state=sea_sim,
            t0=0.0,
        )

        # 3. Setup Controller and Compensation
        enable_ff = config.compensation_level in ("C1", "C2")
        enable_obs = config.compensation_level == "C2"

        if config.controller_type == "CTRL0":
            # Classical Baseline: Stanley Controller
            stanley = StanleyController(cts_config=cts_cfg)
            master_ctrl = None
        elif config.controller_type == "CTRL1":
            # LTV-MPC
            ctrl_cfg = MasterControllerConfig(
                control_rate_hz=1.0 / sim_dt,
                preferred_controller="ltv_mpc",
                enable_feedforward=enable_ff,
                enable_observer=enable_obs,
            )
            master_ctrl = TrajectoryTrackingController(config=ctrl_cfg, cts_config=cts_cfg)
        else:
            # CasADi NMPC
            ctrl_cfg = MasterControllerConfig(
                control_rate_hz=1.0 / sim_dt,
                preferred_controller="nmpc",
                enable_feedforward=enable_ff,
                enable_observer=enable_obs,
            )
            master_ctrl = TrajectoryTrackingController(config=ctrl_cfg, cts_config=cts_cfg)

        # 4. Closed-Loop Simulation
        # Model-mismatch channel: the plant integrates a biased apparent
        # field; controllers keep the clean sea_sim so the bias is
        # invisible to feedforward/planning.
        plant_motion = (
            _BiasedDeckMotion(
                sea_sim, config.unmodeled_bias_y,
                config.unmodeled_bias_amp, config.unmodeled_bias_hz,
            )
            if abs(config.unmodeled_bias_y) + abs(config.unmodeled_bias_amp) > 1e-9
            else sea_sim
        )
        times = np.arange(0.0, min(sc.duration_s, float(traj.duration)) + sim_dt, sim_dt)
        init_ref = traj.eval_at(0.0)
        curr_state = VehicleState(
            x_t=float(init_ref["x"]),
            y_t=float(init_ref["y"]),
            psi_t=float(init_ref["psi_t"]),
            theta_h=float(init_ref["theta_h"]),
            v_t=float(init_ref["v"]),
            delta_f=0.0,
        )

        states = [curr_state]
        actions = []
        solve_times = []
        wheel_loads = []
        f_utils = []

        for k, t_curr in enumerate(times[:-1]):
            t0_step = time.perf_counter()
            if config.controller_type == "CTRL0":
                act = stanley.compute_action(curr_state, traj, t=t_curr, dt=sim_dt)
                st_ms = (time.perf_counter() - t0_step) * 1000.0
            else:
                out = master_ctrl.compute_control(
                    current_state=curr_state,
                    reference_trajectory=traj,
                    sea_state=sea_sim,
                    t=t_curr,
                )
                act = out.action
                st_ms = out.solve_time_ms

            actions.append(act)
            solve_times.append(st_ms)

            # Loads and tire metrics
            app_acc = plant_motion.compute_non_inertial_acceleration(
                pos_d=np.array([curr_state.x_t, curr_state.y_t, 0.0]),
                vel_rel_d=np.array([curr_state.v_t * np.cos(curr_state.psi_t),
                                    curr_state.v_t * np.sin(curr_state.psi_t), 0.0]),
                t=t_curr,
            )
            loads = plant.compute_dynamic_load_transfer(curr_state, app_acc.a_apparent_total, t=t_curr)
            wheel_loads.append(loads)
            
            fz_total = loads.Fz_front + loads.Fz_rear
            f_norm = abs(act.a_cmd) * cts_cfg.tractor.mass
            eta = min(1.0, f_norm / max(sc.friction_coeff * fz_total, 1.0))
            f_utils.append(eta)

            curr_state = plant.step(
                state=curr_state,
                action=act,
                deck_motion=plant_motion,
                t=t_curr,
                dt=sim_dt,
                mu=sc.friction_coeff,
            )
            states.append(curr_state)

        metrics = compute_trajectory_metrics(
            timestamps=times,
            states=states,
            actions=actions,
            reference_trajectory=traj,
            wheel_loads=wheel_loads,
            solve_times_ms=solve_times,
            friction_utilizations=f_utils,
            deck_env=deck_env,
        )

        exec_time = time.perf_counter() - t_start
        return AblationResult(
            config=config,
            metrics=metrics,
            execution_time_s=exec_time,
            status="COMPLETED",
        )

    def run_compensation_ablation(self, scenario_id: str = "SC-1", sim_duration_s: float = 12.0) -> List[AblationResult]:
        """Run 3-level compensation ablation (C0 vs C1 vs C2) with NMPC and Pacejka tires."""
        results = []
        for c_mode in ["C0", "C1", "C2"]:
            cfg = AblationConfiguration(
                compensation_level=c_mode,
                tire_model="T2",
                controller_type="CTRL2",
                scenario_id=scenario_id,
            )
            res = self.run_single_ablation(cfg, sim_duration_s=sim_duration_s)
            results.append(res)
        return results

    def run_controller_ablation(self, scenario_id: str = "SC-1", sim_duration_s: float = 12.0) -> List[AblationResult]:
        """Run 3-level controller ablation (CTRL0 vs CTRL1 vs CTRL2) with full compensation and Pacejka."""
        results = []
        for ctrl_type in ["CTRL0", "CTRL1", "CTRL2"]:
            cfg = AblationConfiguration(
                compensation_level="C2",
                tire_model="T2",
                controller_type=ctrl_type,
                scenario_id=scenario_id,
            )
            res = self.run_single_ablation(cfg, sim_duration_s=sim_duration_s)
            results.append(res)
        return results

    def run_tire_ablation(self, scenario_id: str = "SC-1", sim_duration_s: float = 12.0) -> List[AblationResult]:
        """Run 3-level tire model ablation (T0 vs T1 vs T2) with NMPC and full compensation."""
        results = []
        for tire_mode in ["T0", "T1", "T2"]:
            cfg = AblationConfiguration(
                compensation_level="C2",
                tire_model=tire_mode,
                controller_type="CTRL2",
                scenario_id=scenario_id,
            )
            res = self.run_single_ablation(cfg, sim_duration_s=sim_duration_s)
            results.append(res)
        return results

    def export_results_summary(self, results: List[AblationResult], filename: str = "ablation_summary.json") -> Path:
        """Export list of ablation results to structured JSON and CSV."""
        out_path = self.output_dir / filename
        data = []
        for r in results:
            item = {
                "config": r.config.label,
                "compensation": r.config.compensation_level,
                "tire_model": r.config.tire_model,
                "controller": r.config.controller_type,
                "scenario": r.config.scenario_id,
                "execution_time_s": r.execution_time_s,
                **r.metrics.to_dict(),
            }
            data.append(item)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Also export CSV
        csv_path = self.output_dir / (out_path.stem + ".csv")
        if data:
            headers = list(data[0].keys())
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(headers) + "\n")
                for d in data:
                    f.write(",".join(str(d[h]) for h in headers) + "\n")

        return out_path
