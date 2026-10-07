"""cts.environment.deck_motion

6-DOF Carrier Deck Dynamic Motion Generation.
Supports:
1. Fossen 2nd-order wave-shaping filters in continuous-to-discrete state-space with covariance matching.
2. Multi-harmonic RAO spectrum superposition with wave encounter heading.
3. Full 3D finite rotation kinematics, DeckState generation, and zero-drift numerical integration.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Union, Sequence
import numpy as np
from scipy.linalg import expm

from cts.environment.sea_state import (
    SeaStateLevel,
    SeaStateGenerator,
    WaveSpectrumConfig,
    SEASTATE_4_CONFIG,
    SEASTATE_5_CONFIG,
)
from cts.environment.non_inertial_field import (
    DeckState,
    ApparentAcceleration,
    compute_apparent_acceleration,
)

GRAVITY = 9.80665


@dataclass(frozen=True)
class FossenDOFParam:
    """Parameters for a single degree of freedom Fossen 2nd-order filter."""
    name: str
    zeta: float      # Damping ratio [-]
    omega_n: float   # Natural circular frequency [rad/s]
    sigma: float     # Target standard deviation [m or rad]

    @property
    def gain_kw(self) -> float:
        """Filter gain Kw = 2 * sigma * sqrt(zeta * omega_n)."""
        return 2.0 * self.sigma * np.sqrt(self.zeta * self.omega_n)


@dataclass(frozen=True)
class FossenSeaStateParams:
    """6-DOF Fossen filter parameters for a specific sea state."""
    level: SeaStateLevel
    surge: FossenDOFParam
    sway: FossenDOFParam
    heave: FossenDOFParam
    roll: FossenDOFParam
    pitch: FossenDOFParam
    yaw: FossenDOFParam


# Calibrated Presets for Aircraft Carrier
FOSSEN_SS4_PARAMS = FossenSeaStateParams(
    level=SeaStateLevel.SS4,
    surge=FossenDOFParam("surge", zeta=0.35, omega_n=0.55, sigma=0.40),
    sway=FossenDOFParam("sway",   zeta=0.30, omega_n=0.45, sigma=0.55),
    heave=FossenDOFParam("heave", zeta=0.25, omega_n=0.60, sigma=0.95),
    roll=FossenDOFParam("roll",   zeta=0.08, omega_n=0.40, sigma=float(np.deg2rad(3.0))),
    pitch=FossenDOFParam("pitch", zeta=0.20, omega_n=0.55, sigma=float(np.deg2rad(1.25))),
    yaw=FossenDOFParam("yaw",     zeta=0.30, omega_n=0.35, sigma=float(np.deg2rad(0.60))),
)

FOSSEN_SS5_PARAMS = FossenSeaStateParams(
    level=SeaStateLevel.SS5,
    surge=FossenDOFParam("surge", zeta=0.35, omega_n=0.48, sigma=0.85),
    sway=FossenDOFParam("sway",   zeta=0.30, omega_n=0.40, sigma=1.10),
    heave=FossenDOFParam("heave", zeta=0.25, omega_n=0.52, sigma=1.65),
    roll=FossenDOFParam("roll",   zeta=0.08, omega_n=0.36, sigma=float(np.deg2rad(5.5))),
    pitch=FossenDOFParam("pitch", zeta=0.20, omega_n=0.48, sigma=float(np.deg2rad(2.4))),
    yaw=FossenDOFParam("yaw",     zeta=0.30, omega_n=0.30, sigma=float(np.deg2rad(1.20))),
)


class DiscreteFossenFilterChannel:
    """Single-DOF Fossen 2nd-Order Wave-Shaping Filter in Discrete State-Space.

    Exact continuous-to-discrete covariance matching guarantees E[y^2] == sigma^2 for any dt.
    """

    def __init__(self, param: FossenDOFParam, dt: float):
        self.param = param
        self.dt = float(dt)

        # Continuous state-space matrices
        w0 = param.omega_n
        z0 = param.zeta
        kw = param.gain_kw

        self.Ac = np.array([
            [0.0, 1.0],
            [-w0**2, -2.0 * z0 * w0]
        ], dtype=np.float64)
        self.Bc = np.array([[0.0], [1.0]], dtype=np.float64)
        self.Cc = np.array([0.0, kw], dtype=np.float64)

        # Continuous stationary covariance Pc: A*Pc + Pc*A' + B*B' = 0
        p11 = 1.0 / (4.0 * z0 * (w0 ** 3))
        p22 = 1.0 / (4.0 * z0 * w0)
        self.Pc = np.array([
            [p11, 0.0],
            [0.0, p22]
        ], dtype=np.float64)

        # Discrete state transition matrix Phi = expm(Ac * dt)
        self.Phi = expm(self.Ac * self.dt)

        # Exact discrete noise covariance matching: Qd = Pc - Phi * Pc * Phi'
        Qd = self.Pc - self.Phi @ self.Pc @ self.Phi.T
        Qd = 0.5 * (Qd + Qd.T)
        eigvals, eigvecs = np.linalg.eigh(Qd)
        eigvals = np.maximum(eigvals, 1e-15)
        # Sort descending so column 0 corresponds to dominant mode
        idx_desc = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx_desc]
        eigvecs = eigvecs[:, idx_desc]
        self.Gamma = eigvecs @ np.diag(np.sqrt(eigvals))

        # Internal state vector x = [x1, x2]^T
        self.x = np.zeros(2, dtype=np.float64)
        self.y_prev = 0.0
        self.v_filt = 0.0
        self.a_filt = 0.0
        self.tau_diff = 0.05  # Dirty derivative time constant [s]

    def reset(self, rng: Optional[np.random.Generator] = None, warmup_steps: int = 2600):
        """Warm up filter from initial state to stationary distribution."""
        self.x = np.zeros(2, dtype=np.float64)
        if rng is not None and warmup_steps > 0:
            for _ in range(warmup_steps):
                w = rng.standard_normal(2)
                self.x = self.Phi @ self.x + self.Gamma @ w
        self.y_prev = float(self.Cc @ self.x)
        self.v_filt = 0.0
        self.a_filt = 0.0

    def step(self, w_noise: Union[float, np.ndarray, Sequence[float]]) -> Tuple[float, float, float]:
        """Advance filter by one step dt.

        Args:
            w_noise: 2D Gaussian vector or scalar.
        Returns:
            (displacement y, velocity dy, acceleration ddy).
        """
        w_arr = np.asarray(w_noise, dtype=np.float64)
        if w_arr.ndim == 0 or len(w_arr) == 1:
            noise_vec = self.Gamma @ np.array([float(w_arr), 0.0])
        else:
            noise_vec = self.Gamma @ w_arr[:2]

        self.x = self.Phi @ self.x + noise_vec
        y = float(self.Cc @ self.x)

        # Band-limited derivative filter: H(s) = s / (tau*s + 1)
        alpha = self.dt / (self.tau_diff + self.dt)
        raw_v = (y - self.y_prev) / self.dt
        v_next = self.v_filt + alpha * (raw_v - self.v_filt)
        raw_a = (v_next - self.v_filt) / self.dt
        a_next = self.a_filt + alpha * (raw_a - self.a_filt)

        self.y_prev = y
        self.v_filt = v_next
        self.a_filt = a_next

        return y, v_next, a_next


class DeckMotionSimulator:
    """Unified 6-DOF Deck Motion Simulator.

    Supports:
    - Mode 'fossen': Real-time step-by-step state-space simulation.
    - Mode 'rao': Multi-harmonic superposition for arbitrary time queries get_deck_state(t).
    """

    def __init__(
        self,
        sea_state: Union[SeaStateLevel, str, int] = SeaStateLevel.SS4,
        mode: str = "fossen",
        dt: float = 0.02,
        seed: Optional[int] = 2026,
    ):
        if isinstance(sea_state, (int, str)):
            if str(sea_state) in ("5", "SS5"):
                self.sea_state_level = SeaStateLevel.SS5
            else:
                self.sea_state_level = SeaStateLevel.SS4
        else:
            self.sea_state_level = sea_state

        self.mode = mode.lower()
        self.dt = float(dt)
        self.t = 0.0
        self.rng = np.random.default_rng(seed)

        if self.sea_state_level == SeaStateLevel.SS4:
            self.params = FOSSEN_SS4_PARAMS
            self.wave_config = SEASTATE_4_CONFIG
        else:
            self.params = FOSSEN_SS5_PARAMS
            self.wave_config = SEASTATE_5_CONFIG

        # Initialize Fossen channels
        self.channels: Dict[str, DiscreteFossenFilterChannel] = {
            "surge": DiscreteFossenFilterChannel(self.params.surge, self.dt),
            "sway":  DiscreteFossenFilterChannel(self.params.sway,  self.dt),
            "heave": DiscreteFossenFilterChannel(self.params.heave, self.dt),
            "roll":  DiscreteFossenFilterChannel(self.params.roll,  self.dt),
            "pitch": DiscreteFossenFilterChannel(self.params.pitch, self.dt),
            "yaw":   DiscreteFossenFilterChannel(self.params.yaw,   self.dt),
        }

        # Initialize RAO multi-harmonic components for get_deck_state(t) queries
        self.sea_generator = SeaStateGenerator(self.wave_config, n_frequencies=120, seed=seed)
        self._init_rao_amplitudes(seed)
        self.reset(seed)

    def _init_rao_amplitudes(self, seed: Optional[int] = None):
        """Construct RAO transfer function gains for 6 DOFs."""
        c = self.sea_generator.components
        w = c.omega
        rng_rao = np.random.default_rng(seed if seed is not None else 2026)

        def get_rao(param: FossenDOFParam):
            num = param.gain_kw * w
            den = np.sqrt((param.omega_n**2 - w**2)**2 + (2.0 * param.zeta * param.omega_n * w)**2)
            return num / np.maximum(den, 1e-6)

        self.rao_amplitudes = {
            "surge": c.amplitudes * get_rao(self.params.surge),
            "sway":  c.amplitudes * get_rao(self.params.sway),
            "heave": c.amplitudes * get_rao(self.params.heave),
            "roll":  c.amplitudes * get_rao(self.params.roll),
            "pitch": c.amplitudes * get_rao(self.params.pitch),
            "yaw":   c.amplitudes * get_rao(self.params.yaw),
        }
        self.rao_phases = {
            dof: rng_rao.uniform(0.0, 2.0 * np.pi, size=len(w))
            for dof in self.rao_amplitudes
        }

    def reset(self, seed: Optional[int] = None):
        """Reset simulation states and optionally reseed RNG."""
        self.t = 0.0
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for ch in self.channels.values():
            ch.reset(self.rng, warmup_steps=2600)

    def step(self) -> DeckState:
        """Advance simulation by dt and return updated DeckState."""
        self.t += self.dt
        if self.mode == "fossen":
            surge, v_surge, a_surge = self.channels["surge"].step(self.rng.standard_normal(2))
            sway,  v_sway,  a_sway  = self.channels["sway"].step(self.rng.standard_normal(2))
            heave, v_heave, a_heave = self.channels["heave"].step(self.rng.standard_normal(2))
            roll,  p_rate,  p_acc   = self.channels["roll"].step(self.rng.standard_normal(2))
            pitch, q_rate,  q_acc   = self.channels["pitch"].step(self.rng.standard_normal(2))
            yaw,   r_rate,  r_acc   = self.channels["yaw"].step(self.rng.standard_normal(2))

            pos_w = np.array([surge, sway, heave], dtype=np.float64)
            rpy = np.array([roll, pitch, yaw], dtype=np.float64)
            vel_w = np.array([v_surge, v_sway, v_heave], dtype=np.float64)
            acc_w = np.array([a_surge, a_sway, a_heave], dtype=np.float64)
            rpy_dot = np.array([p_rate, q_rate, r_rate], dtype=np.float64)
            rpy_ddot = np.array([p_acc, q_acc, r_acc], dtype=np.float64)
            return self._build_deck_state(self.t, pos_w, rpy, vel_w, acc_w, rpy_dot, rpy_ddot)
        else:
            return self.get_deck_state(self.t)

    def get_deck_state(self, t: float) -> DeckState:
        """Analytical query at arbitrary time t via multi-harmonic superposition."""
        t_val = float(t)
        if np.isnan(t_val) or np.isinf(t_val):
            raise ValueError(f"Invalid timestamp query: {t_val}")

        c = self.sea_generator.components
        w = c.omega
        pos_list, vel_list, acc_list = [], [], []

        for dof in ["surge", "sway", "heave"]:
            amp = self.rao_amplitudes[dof]
            ph = self.rao_phases[dof]
            arg = w * t_val + ph
            pos_list.append(np.sum(amp * np.cos(arg)))
            vel_list.append(-np.sum(amp * w * np.sin(arg)))
            acc_list.append(-np.sum(amp * (w**2) * np.cos(arg)))

        rpy_list, rpy_dot_list, rpy_ddot_list = [], [], []
        for dof in ["roll", "pitch", "yaw"]:
            amp = self.rao_amplitudes[dof]
            ph = self.rao_phases[dof]
            arg = w * t_val + ph
            rpy_list.append(np.sum(amp * np.cos(arg)))
            rpy_dot_list.append(-np.sum(amp * w * np.sin(arg)))
            rpy_ddot_list.append(-np.sum(amp * (w**2) * np.cos(arg)))

        pos_w = np.array(pos_list, dtype=np.float64)
        rpy = np.array(rpy_list, dtype=np.float64)
        vel_w = np.array(vel_list, dtype=np.float64)
        acc_w = np.array(acc_list, dtype=np.float64)
        rpy_dot = np.array(rpy_dot_list, dtype=np.float64)
        rpy_ddot = np.array(rpy_ddot_list, dtype=np.float64)

        return self._build_deck_state(t_val, pos_w, rpy, vel_w, acc_w, rpy_dot, rpy_ddot)

    def _build_deck_state(
        self,
        t: float,
        pos_w: np.ndarray,
        rpy: np.ndarray,
        vel_w: np.ndarray,
        acc_w: np.ndarray,
        rpy_dot: np.ndarray,
        rpy_ddot: np.ndarray,
    ) -> DeckState:
        """Construct full nonlinear 3D rotation matrices, angular rates, and accelerations."""
        phi, theta, psi = float(rpy[0]), float(rpy[1]), float(rpy[2])
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)

        # Z-Y-X rotation matrix R_D^W (Deck to World)
        R_d_to_w = np.array([
            [cpsi * cth, cpsi * sth * sphi - spsi * cphi, cpsi * sth * cphi + spsi * sphi],
            [spsi * cth, spsi * sth * sphi + cpsi * cphi, spsi * sth * cphi - cpsi * sphi],
            [-sth,       cth * sphi,                      cth * cphi]
        ], dtype=np.float64)
        R_w_to_d = R_d_to_w.T.copy()

        # Angular velocity in deck frame: omega_d = T_euler * rpy_dot
        T_euler = np.array([
            [1.0, 0.0,   -sth],
            [0.0, cphi,  sphi * cth],
            [0.0, -sphi, cphi * cth]
        ], dtype=np.float64)
        omega_d = T_euler @ rpy_dot

        # Angular acceleration in deck frame: alpha_d = T_euler * rpy_ddot + T_dot * rpy_dot
        phi_dot, th_dot = float(rpy_dot[0]), float(rpy_dot[1])
        T_dot = np.array([
            [0.0, 0.0,                 -th_dot * cth],
            [0.0, -phi_dot * sphi,      phi_dot * cphi * cth - th_dot * sphi * sth],
            [0.0, -phi_dot * cphi,     -phi_dot * sphi * cth - th_dot * cphi * sth]
        ], dtype=np.float64)
        alpha_d = T_euler @ rpy_ddot + T_dot @ rpy_dot

        # Return copies of arrays for immutability
        return DeckState(
            pos_w=pos_w.copy(),
            rpy_rad=rpy.copy(),
            vel_w=vel_w.copy(),
            omega_d=omega_d.copy(),
            acc_w=acc_w.copy(),
            alpha_d=alpha_d.copy(),
            R_d_to_w=R_d_to_w.copy(),
            R_w_to_d=R_w_to_d.copy(),
            t=float(t),
        )

    def compute_non_inertial_acceleration(
        self,
        pos_d: np.ndarray,
        vel_rel_d: np.ndarray,
        t: Optional[float] = None,
    ) -> ApparentAcceleration:
        """Compute non-inertial apparent acceleration field at given position and velocity."""
        if t is not None:
            state = self.get_deck_state(float(t))
        else:
            state = self.get_deck_state(self.t)
        return compute_apparent_acceleration(state, pos_d, vel_rel_d)
