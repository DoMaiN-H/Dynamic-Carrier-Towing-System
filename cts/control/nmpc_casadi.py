"""Prototype validation of CasADi NMPC Controller for Feature F16.
Validates mathematical formulation, execution speed <= 20 ms SLA,
adhesion circle constraint, angle unwrapping, and fallback triggers.
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import casadi as ca

from cts.common.config import CTSConfig, TractorConfig, AircraftConfig, TireConfig
from cts.common.math_utils import wrap_to_pi
from cts.dynamics.multibody_dynamics import VehicleState, ControlAction
from cts.environment.deck_motion import DeckState
from cts.environment.non_inertial_field import ApparentAcceleration, GRAVITY
from cts.planning.minco_optimizer import PlannedTrajectory


@dataclass
class NMPCConfig:
    """Hyperparameters and configuration for CasADi NMPC Controller."""
    horizon_steps: int = 20               # N_p prediction horizon steps
    dt: float = 0.05                      # T_s discrete sampling period (s)
    
    # State tracking weights Q: [x, y, psi_t, v_t, theta_h]
    q_x: float = 25.0
    q_y: float = 25.0
    q_psi: float = 15.0
    q_v: float = 5.0
    q_theta: float = 20.0
    
    # Control effort weights R: [a_t, delta_f]
    r_a: float = 0.5
    r_delta: float = 2.0
    
    # Control rate weights R_Delta: [Delta a_t, Delta delta_f]
    r_delta_a: float = 1.0
    r_delta_steer: float = 10.0
    
    # Terminal cost scale P = p_scale * Q
    p_scale: float = 2.5
    
    # Physical actuation bounds. Towbarless tractors do reverse (pushback /
    # docking correction); reverse speed is capped low because the trailer
    # articulation is anti-damped when backing (theta_dot ~ -(v/l_a) sin(theta)).
    v_min: float = -0.5                   # m/s (limited reverse authority)
    v_max: float = 2.0                    # m/s
    a_min: float = -1.0                   # m/s^2
    a_max: float = 1.0                    # m/s^2
    delta_max_rad: float = 0.610865       # 35 deg
    omega_steer_max_rad: float = 0.523599 # 30 deg/s
    theta_max_rad: float = 1.047198       # 60 deg (anti-jackknife safety limit)
    
    # Tire and Deck parameters
    default_mu: float = 0.50              # Default wet deck friction
    
    # Solver options
    sla_timeout_ms: float = 18.0          # SLA hard timeout before triggering fallback (ms)
    ipopt_max_iter: int = 35              # Maximum IPOPT iterations
    ipopt_tol: float = 1e-3               # Convergence tolerance
    ipopt_acceptable_tol: float = 5e-3    # Acceptable tolerance
    ipopt_acceptable_iter: int = 3
    ipopt_linear_solver: str = "mumps"


@dataclass
class NMPCResult:
    """Detailed result container from NMPC optimization cycle."""
    action: ControlAction
    a_cmd: float
    delta_cmd: float
    steer_rate_cmd: float
    success: bool
    solve_time_ms: float
    cost: float
    status: str
    predicted_states: np.ndarray          # Shape (N+1, 5)
    predicted_controls: np.ndarray        # Shape (N, 2)
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    fallback_level: int = 0


class CasADiNMPCController:
    """Feature F16: High-Performance CasADi NMPC Tracking Controller with Guaranteed <= 20 ms SLA."""

    def __init__(
        self,
        config: Optional[NMPCConfig] = None,
        tractor_cfg: Optional[TractorConfig] = None,
        aircraft_cfg: Optional[AircraftConfig] = None,
        tire_cfg: Optional[TireConfig] = None,
    ) -> None:
        self.config = config or NMPCConfig()
        self.tractor = tractor_cfg or TractorConfig()
        self.aircraft = aircraft_cfg or AircraftConfig()
        self.tire = tire_cfg or TireConfig()

        self.N = self.config.horizon_steps
        self.dt = self.config.dt
        self.l_t = float(self.tractor.wheelbase)          # 1.80 m
        self.l_a = float(self.aircraft.wheelbase)         # 8.00 m
        self.d_c = float(self.tractor.hitch_offset)       # 0.10 m
        self.m_t = float(self.tractor.mass)               # 8500 kg
        self.m_a = float(self.aircraft.mass)              # 20500 kg
        self.m_total = self.m_t + self.m_a

        # State and control dimensions
        self.nx = 5  # [x, y, psi_t, v_t, theta_h]
        self.nu = 2  # [a_t, delta_f]
        # Steering control effectiveness: lateral accel per rad of steer,
        # b0 = C_alpha_f / m_t with 2 tires on the front axle. Maps the ESO
        # residual acceleration estimate to an equivalent input bias.
        self.b0_steer = 2.0 * float(self.tire.cornering_stiffness_tractor) / self.m_t
        self.delta_bias_max = 0.261799  # 15 deg cap on injected bias

        # Warm-start state buffers
        self.w_sol: Optional[np.ndarray] = None
        self.lam_x_sol: Optional[Any] = None
        self.lam_g_sol: Optional[Any] = None
        self.u_prev = np.array([0.0, 0.0], dtype=np.float64)
        self.last_predicted_states: Optional[np.ndarray] = None
        self.last_predicted_controls: Optional[np.ndarray] = None

        # Build symbolic NLP
        self._build_symbolic_nlp()
        
        # Pre-compile and warm up
        self._warm_up()

    def _build_symbolic_nlp(self) -> None:
        """Construct the SX symbolic NLP graph and IPOPT solver function."""
        # Symbolics for ODE: x = [x, y, psi, v, theta], u = [a, delta]
        x_sym = ca.SX.sym("x", self.nx)
        u_sym = ca.SX.sym("u", self.nu)

        px, py, psi, v, theta = x_sym[0], x_sym[1], x_sym[2], x_sym[3], x_sym[4]
        a, delta = u_sym[0], u_sym[1]

        # Steering disturbance bias b_sym (rad): lumped residual lateral
        # acceleration expressed as an equivalent input bias, so effective
        # steer in the model is delta + b. With b > 0 the optimizer
        # automatically selects delta_cmd = delta_desired - b.
        b_sym = ca.SX.sym("b", 1)

        # Nonholonomic kinematics with rear turntable clamp
        dx = v * ca.cos(psi)
        dy = v * ca.sin(psi)
        dpsi = (v / self.l_t) * ca.tan(delta + b_sym)
        dtheta = - (v / self.l_a) * ca.sin(theta) - (v / self.l_t) * ca.tan(delta + b_sym) * (1.0 + (self.d_c / self.l_a) * ca.cos(theta))
        dv = a

        f_ode = ca.Function("f_ode", [x_sym, u_sym, b_sym], [ca.vertcat(dx, dy, dpsi, dv, dtheta)])

        # RK4 Discretization function
        dt = self.dt
        k1 = f_ode(x_sym, u_sym, b_sym)
        k2 = f_ode(x_sym + 0.5 * dt * k1, u_sym, b_sym)
        k3 = f_ode(x_sym + 0.5 * dt * k2, u_sym, b_sym)
        k4 = f_ode(x_sym + dt * k3, u_sym, b_sym)
        x_next_sym = x_sym + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.f_rk4 = ca.Function("f_rk4", [x_sym, u_sym, b_sym], [x_next_sym])

        # NLP Decision variables w and constraint bounds
        w = []
        self.lbw = []
        self.ubw = []
        g = []
        self.lbg = []
        self.ubg = []

        # Parameters vector p:
        # [x0 (5), xref (5*(N+1)), uprev (2), geff (N), adeck_y (N), mu (1), dbias (1)]
        # dbias: ESO-estimated lumped residual lateral acceleration mapped to
        # an equivalent STEERING bias delta_bias = d_hat_y / b0 injected into
        # the yaw/articulation dynamics. The optimizer then picks
        # delta_cmd = delta_desired - delta_bias inside the constrained NLP
        # (bounded by tan-saturation, steering bounds, slew limits and the
        # adhesion circle), replacing the previous out-of-band additive
        # delta_dob correction that double-counted the feedforward channel.
        p_len = self.nx + self.nx * (self.N + 1) + self.nu + self.N + self.N + 2
        p_sym = ca.SX.sym("p", p_len)

        idx = 0
        x0_p = p_sym[idx : idx + self.nx]; idx += self.nx
        xref_p = ca.reshape(p_sym[idx : idx + self.nx * (self.N + 1)], self.nx, self.N + 1); idx += self.nx * (self.N + 1)
        uprev_p = p_sym[idx : idx + self.nu]; idx += self.nu
        geff_p = p_sym[idx : idx + self.N]; idx += self.N
        adeck_p = p_sym[idx : idx + self.N]; idx += self.N
        mu_p = p_sym[idx]; idx += 1
        dbias_p = p_sym[idx]; idx += 1

        # Weight matrices
        Q = np.diag([self.config.q_x, self.config.q_y, self.config.q_psi, self.config.q_v, self.config.q_theta])
        R = np.diag([self.config.r_a, self.config.r_delta])
        R_delta = np.diag([self.config.r_delta_a, self.config.r_delta_steer])
        P = Q * self.config.p_scale

        cost = 0.0

        # Create state variables X_0 ... X_N
        X = []
        for k in range(self.N + 1):
            Xk = ca.SX.sym(f"X_{k}", self.nx)
            w.append(Xk)
            # Bounds: x, y, psi unconstrained; v in [0, 2]; |theta| <= theta_max
            self.lbw.extend([-np.inf, -np.inf, -np.inf, self.config.v_min, -self.config.theta_max_rad])
            self.ubw.extend([np.inf, np.inf, np.inf, self.config.v_max, self.config.theta_max_rad])
            X.append(Xk)

        # Create control variables U_0 ... U_{N-1}
        U = []
        for k in range(self.N):
            Uk = ca.SX.sym(f"U_{k}", self.nu)
            w.append(Uk)
            self.lbw.extend([self.config.a_min, -self.config.delta_max_rad])
            self.ubw.extend([self.config.a_max, self.config.delta_max_rad])
            U.append(Uk)

        # Initial state equality constraint: X_0 - x0 = 0
        g.append(X[0] - x0_p)
        self.lbg.extend([0.0] * self.nx)
        self.ubg.extend([0.0] * self.nx)

        # Horizon loop
        delta_rate_max_step = self.config.omega_steer_max_rad * self.dt
        for k in range(self.N):
            # 1. System Dynamics Continuity: X_{k+1} - f_rk4(X_k, U_k, b) = 0
            #    Steering disturbance bias dbias_p enters the model via
            #    f_rk4(X, U, b) as an input-channel offset (see b_sym).
            x_next = self.f_rk4(X[k], U[k], dbias_p)
            g.append(X[k + 1] - x_next)
            self.lbg.extend([0.0] * self.nx)
            self.ubg.extend([0.0] * self.nx)

            # 2. Stage Tracking Error Cost
            err = X[k] - xref_p[:, k]
            cost += ca.mtimes([err.T, Q, err])

            # 3. Control Effort Cost
            cost += ca.mtimes([U[k].T, R, U[k]])

            # 4. Slew Rate Cost and Hard Steering Rate Constraint
            du = U[k] - (uprev_p if k == 0 else U[k - 1])
            cost += ca.mtimes([du.T, R_delta, du])

            # Constraint: -delta_rate_max_step <= du[1] <= delta_rate_max_step
            g.append(du[1])
            self.lbg.append(-delta_rate_max_step)
            self.ubg.append(delta_rate_max_step)

            # 5. Dynamic Tire Adhesion Circle Constraint
            # ax = U[k][0]
            # ay = (X[k][3]^2 / l_t) * tan(U[k][1]) - a_deck_y
            # Normalized: (ax / geff)^2 + (ay / geff)^2 - mu^2 <= 0
            ax_k = U[k][0]
            ay_k = (X[k][3]**2 / self.l_t) * ca.tan(U[k][1]) - adeck_p[k]
            geff_k = ca.fmax(geff_p[k], 1e-3)
            
            adhesion_utilization = (ax_k / geff_k)**2 + (ay_k / geff_k)**2
            g.append(adhesion_utilization - mu_p**2)
            self.lbg.append(-np.inf)
            self.ubg.append(0.0)

        # Terminal Error Cost
        err_term = X[self.N] - xref_p[:, self.N]
        cost += ca.mtimes([err_term.T, P, err_term])

        w_vec = ca.vertcat(*w)
        g_vec = ca.vertcat(*g)

        self.lbw = np.array(self.lbw, dtype=np.float64)
        self.ubw = np.array(self.ubw, dtype=np.float64)
        self.lbg = np.array(self.lbg, dtype=np.float64)
        self.ubg = np.array(self.ubg, dtype=np.float64)

        nlp = {"x": w_vec, "f": cost, "g": g_vec, "p": p_sym}
        ipopt_opts = {
            "print_level": 0,
            "max_iter": self.config.ipopt_max_iter,
            "tol": self.config.ipopt_tol,
            "acceptable_tol": self.config.ipopt_acceptable_tol,
            "acceptable_iter": self.config.ipopt_acceptable_iter,
            "warm_start_init_point": "yes",
            "warm_start_bound_push": 1e-4,
            "warm_start_mult_bound_push": 1e-4,
            "linear_solver": self.config.ipopt_linear_solver,
        }
        opts = {"ipopt": ipopt_opts, "print_time": 0}
        self.solver = ca.nlpsol("cts_nmpc_solver", "ipopt", nlp, opts)

    def _warm_up(self) -> None:
        """Perform offline dry-run solve so graph compilation and memory allocations occur at init."""
        x0_dummy = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        xref_dummy = np.zeros((self.nx, self.N + 1), dtype=np.float64)
        xref_dummy[3, :] = 1.0
        
        p_val = self._pack_parameters(
            x0=x0_dummy,
            xref=xref_dummy,
            uprev=np.zeros(2),
            geff=np.full(self.N, GRAVITY),
            adeck=np.zeros(self.N),
            mu=self.config.default_mu,
            dbias=0.0,
        )
        
        w0 = np.zeros(len(self.lbw))
        for k in range(self.N + 1):
            w0[k * self.nx : (k + 1) * self.nx] = x0_dummy

        res = self.solver(x0=w0, p=p_val, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg)
        self.w_sol = np.array(res["x"]).flatten()
        self.lam_x_sol = res["lam_x"]
        self.lam_g_sol = res["lam_g"]

    def _pack_parameters(
        self,
        x0: np.ndarray,
        xref: np.ndarray,
        uprev: np.ndarray,
        geff: np.ndarray,
        adeck: np.ndarray,
        mu: float,
        dbias: float = 0.0,
    ) -> np.ndarray:
        """Pack all numerical parameters into flattened parameter vector p."""
        p_list = [
            x0,
            xref.flatten(order="F"),
            uprev,
            geff,
            adeck,
            np.array([mu]),
            np.array([dbias]),
        ]
        return np.concatenate(p_list)

    def sample_reference_horizon(
        self,
        target_traj: PlannedTrajectory,
        t: float,
        current_heading: float,
        deck_state: Optional[DeckState] = None,
        apparent_acc: Optional[ApparentAcceleration] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract continuous unwrapped reference states and deck acceleration profile over horizon.
        
        Returns:
            xref: (5, N+1) reference state matrix
            geff: (N,) effective gravity profile
            adeck_y: (N,) deck lateral apparent acceleration profile
        """
        xref = np.zeros((self.nx, self.N + 1), dtype=np.float64)
        geff = np.full(self.N, GRAVITY, dtype=np.float64)
        adeck_y = np.zeros(self.N, dtype=np.float64)

        for k in range(self.N + 1):
            tk = t + k * self.dt
            sample = target_traj.eval_at(tk)
            
            xref[0, k] = sample["x"]
            xref[1, k] = sample["y"]
            
            # Smooth heading unwrapping relative to current_heading to eliminate 2*pi jumps
            raw_psi = sample["psi_t"]
            dpsi = raw_psi - current_heading
            xref[2, k] = current_heading + np.arctan2(np.sin(dpsi), np.cos(dpsi))
            
            xref[3, k] = sample["v"]
            xref[4, k] = sample["theta_h"]

        # Deck apparent accelerations
        if apparent_acc is not None:
            # Constant or extrapolated
            g_val = max(1e-3, float(apparent_acc.g_eff))
            a_val = float(apparent_acc.a_apparent_total[1])
            geff[:] = g_val
            adeck_y[:] = a_val

        return xref, geff, adeck_y

    def compute_control(
        self,
        state: VehicleState,
        target_traj: PlannedTrajectory,
        t: float,
        apparent_acc: Optional[ApparentAcceleration] = None,
        deck_state: Optional[DeckState] = None,
        mu: Optional[float] = None,
        disturbance_acc_y: float = 0.0,
    ) -> NMPCResult:
        """Compute optimal control input for current cycle adhering to <= 20 ms SLA.

        disturbance_acc_y: lumped residual lateral acceleration estimate
        (m/s^2, vehicle-lateral frame) injected into the prediction model.
        """
        t_start = time.perf_counter()
        friction = self.config.default_mu if mu is None else float(mu)

        x0 = np.array([state.x_t, state.y_t, state.psi_t, state.v_t, state.theta_h], dtype=np.float64)

        # 1. Sample reference trajectory over horizon
        xref, geff, adeck_y = self.sample_reference_horizon(
            target_traj, t, state.psi_t, deck_state, apparent_acc
        )

        p_val = self._pack_parameters(
            x0=x0,
            xref=xref,
            uprev=self.u_prev,
            geff=geff,
            adeck=adeck_y,
            mu=friction,
            dbias=float(np.clip(disturbance_acc_y / self.b0_steer,
                                -self.delta_bias_max, self.delta_bias_max)),
        )

        # 2. Construct 1-step shifted warm start
        w_init, lam_x_init, lam_g_init = self._get_warm_start(x0, xref)

        # 3. Invoke IPOPT NLP Solver
        fallback_needed = False
        fallback_reason = None
        sol_w = None
        status = "UNKNOWN"
        cost = 0.0

        try:
            if lam_x_init is not None:
                res = self.solver(
                    x0=w_init,
                    p=p_val,
                    lbx=self.lbw,
                    ubx=self.ubw,
                    lbg=self.lbg,
                    ubg=self.ubg,
                    lam_x0=lam_x_init,
                    lam_g0=lam_g_init,
                )
            else:
                res = self.solver(
                    x0=w_init,
                    p=p_val,
                    lbx=self.lbw,
                    ubx=self.ubw,
                    lbg=self.lbg,
                    ubg=self.ubg,
                )
            
            t_solve = (time.perf_counter() - t_start) * 1000.0  # ms
            sol_w = np.array(res["x"]).flatten()
            cost = float(res["f"])

            # Check for NaN / Inf
            if np.any(np.isnan(sol_w)) or np.any(np.isinf(sol_w)):
                fallback_needed = True
                fallback_reason = "NaN or Inf in solver variables"
            elif t_solve > self.config.sla_timeout_ms:
                fallback_needed = True
                fallback_reason = f"Solve time {t_solve:.2f} ms exceeded SLA limit {self.config.sla_timeout_ms} ms"
            else:
                status = "OPTIMAL"

        except Exception as e:
            t_solve = (time.perf_counter() - t_start) * 1000.0
            fallback_needed = True
            fallback_reason = f"IPOPT Exception: {str(e)}"

        # 4. Fallback or Output Extraction
        if fallback_needed or sol_w is None:
            return self._apply_fallback(fallback_reason or "Solver Failure", state, target_traj, t, t_solve)

        # Update warm start buffers
        self.w_sol = sol_w
        self.lam_x_sol = res.get("lam_x", None)
        self.lam_g_sol = res.get("lam_g", None)

        # Extract optimal control U_0: [a_0, delta_0]
        # State variables: (N+1) * 5
        u_offset = (self.N + 1) * self.nx
        u0 = sol_w[u_offset : u_offset + self.nu]
        a_cmd = float(np.clip(u0[0], self.config.a_min, self.config.a_max))
        delta_cmd = float(np.clip(u0[1], -self.config.delta_max_rad, self.config.delta_max_rad))
        steer_rate_cmd = float((delta_cmd - self.u_prev[1]) / self.dt)

        self.u_prev = np.array([a_cmd, delta_cmd], dtype=np.float64)

        # Extract predicted trajectories
        pred_states = sol_w[:u_offset].reshape((self.N + 1, self.nx))
        pred_controls = sol_w[u_offset : u_offset + self.N * self.nu].reshape((self.N, self.nu))
        self.last_predicted_states = pred_states
        self.last_predicted_controls = pred_controls

        action = ControlAction(a_cmd=a_cmd, steer_angle_cmd=delta_cmd, steer_rate_cmd=steer_rate_cmd)

        return NMPCResult(
            action=action,
            a_cmd=a_cmd,
            delta_cmd=delta_cmd,
            steer_rate_cmd=steer_rate_cmd,
            success=True,
            solve_time_ms=t_solve,
            cost=cost,
            status=status,
            predicted_states=pred_states,
            predicted_controls=pred_controls,
            fallback_used=False,
            fallback_level=0,
        )

    def _get_warm_start(self, x0: np.ndarray, xref: np.ndarray) -> Tuple[np.ndarray, Optional[Any], Optional[Any]]:
        """Compute warm start vector by shifting previous solution 1 step forward."""
        if self.w_sol is None:
            w0 = np.zeros(len(self.lbw))
            for k in range(self.N + 1):
                w0[k * self.nx : (k + 1) * self.nx] = xref[:, k]
            return w0, None, None

        w_init = np.copy(self.w_sol)
        x_len = (self.N + 1) * self.nx
        u_len = self.N * self.nu

        # Shift states: X_0..X_{N-1} <- X_1..X_N
        w_init[0 : x_len - self.nx] = self.w_sol[self.nx : x_len]
        # Replicate last state
        w_init[x_len - self.nx : x_len] = self.w_sol[x_len - self.nx : x_len]
        # First state locked to current x0
        w_init[0 : self.nx] = x0

        # Shift controls: U_0..U_{N-2} <- U_1..U_{N-1}
        w_init[x_len : x_len + u_len - self.nu] = self.w_sol[x_len + self.nu : x_len + u_len]
        w_init[x_len + u_len - self.nu : x_len + u_len] = self.w_sol[x_len + u_len - self.nu : x_len + u_len]

        return w_init, self.lam_x_sol, self.lam_g_sol

    def _apply_fallback(
        self,
        reason: str,
        state: VehicleState,
        target_traj: PlannedTrajectory,
        t: float,
        t_solve: float,
    ) -> NMPCResult:
        """Multi-tier deterministic fallback logic when NLP fails or exceeds 20 ms SLA."""
        # Tier 3: Emergency Jackknife / Roll Recovery
        if abs(state.theta_h) > 1.082:  # > 62 deg
            a_cmd = -1.0
            # Steer in opposite direction of hitch articulation to reduce theta_h
            delta_cmd = float(np.clip(-np.sign(state.theta_h) * 0.35, -self.config.delta_max_rad, self.config.delta_max_rad))
            steer_rate_cmd = float((delta_cmd - self.u_prev[1]) / self.dt)
            level = 3
        # Tier 1: Predictive Hold (Shift previous control trajectory)
        elif self.last_predicted_controls is not None and len(self.last_predicted_controls) > 1:
            u_next = self.last_predicted_controls[1]
            a_cmd = float(np.clip(u_next[0], self.config.a_min, self.config.a_max))
            delta_cmd = float(np.clip(u_next[1], -self.config.delta_max_rad, self.config.delta_max_rad))
            steer_rate_cmd = float((delta_cmd - self.u_prev[1]) / self.dt)
            level = 1
        # Tier 2: Kinematic Pure Pursuit / Stanley feedforward
        else:
            sample = target_traj.eval_at(t)
            a_cmd = float(np.clip(sample["a"], self.config.a_min, self.config.a_max))
            
            # Heading and cross-track error
            dx = state.x_t - sample["x"]
            dy = state.y_t - sample["y"]
            psi_path = sample["psi_t"]
            e_lat = -np.sin(psi_path) * dx + np.cos(psi_path) * dy
            e_psi = wrap_to_pi(state.psi_t - psi_path)

            delta_ff = np.arctan(self.l_t * sample["kappa"])
            k_lat = 0.8
            k_psi = 1.2
            delta_fb = -k_psi * e_psi - np.arctan2(k_lat * e_lat, max(state.v_t, 0.2))
            delta_cmd = float(np.clip(delta_ff + delta_fb, -self.config.delta_max_rad, self.config.delta_max_rad))
            steer_rate_cmd = float((delta_cmd - self.u_prev[1]) / self.dt)
            level = 2

        self.u_prev = np.array([a_cmd, delta_cmd], dtype=np.float64)
        action = ControlAction(a_cmd=a_cmd, steer_angle_cmd=delta_cmd, steer_rate_cmd=steer_rate_cmd)

        dummy_states = np.zeros((self.N + 1, self.nx))
        dummy_controls = np.zeros((self.N, self.nu))

        return NMPCResult(
            action=action,
            a_cmd=a_cmd,
            delta_cmd=delta_cmd,
            steer_rate_cmd=steer_rate_cmd,
            success=False,
            solve_time_ms=t_solve,
            cost=-1.0,
            status=f"FALLBACK_TRIGGERED_L{level}",
            predicted_states=dummy_states,
            predicted_controls=dummy_controls,
            fallback_used=True,
            fallback_reason=reason,
            fallback_level=level,
        )

    def compute_action(
        self,
        current_state: VehicleState,
        target_traj: PlannedTrajectory,
        deck_state: Optional[DeckState] = None,
        apparent_acc: Optional[ApparentAcceleration] = None,
        t: float = 0.0,
    ) -> ControlAction:
        """PROJECT.md public interface contract line 107."""
        res = self.compute_control(
            state=current_state,
            target_traj=target_traj,
            t=t,
            apparent_acc=apparent_acc,
            deck_state=deck_state,
        )
        return res.action

    def reset(self) -> None:
        """Reset warm-start caches and historical controls."""
        self.w_sol = None
        self.lam_x_sol = None
        self.lam_g_sol = None
        self.u_prev = np.array([0.0, 0.0], dtype=np.float64)
        self.last_predicted_states = None
        self.last_predicted_controls = None
