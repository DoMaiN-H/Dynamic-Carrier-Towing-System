"""cts.dynamics.tire_models

Feature F8: Nonlinear Tire Contact & Friction Models for Carrier Towing Operations.
Unified tire models providing:
  - LinearTireModel (Ablation T0)
  - DugoffTireModel (Ablation T1, C^1 continuous combined slip)
  - PacejkaTireModel (Ablation T2, MF-5.2 Magic Formula with adhesion ellipse)
  - First-order tire relaxation length dynamics via unconditional exponential integration
  - Three-tier low-speed singularity regularization
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np

from cts.common.config import TireConfig


@dataclass(frozen=True)
class TireForces:
    """Standardized output container for tire contact patch forces and telemetry."""
    Fx: float               # Longitudinal tractive/braking force (N, + forward)
    Fy: float               # Lateral restoring force (N, opposes lateral slip)
    Fz: float               # Normal vertical contact load (N)
    adhesion_ratio: float   # Normalized adhesion utilization eta in [0, 1]
    is_saturated: bool      # True if force is clamped on adhesion ellipse
    alpha: float            # Lateral slip angle (rad)
    kappa: float            # Longitudinal slip ratio s_x


@dataclass
class DynamicTireState:
    """State storage for first-order relaxation length ODE."""
    Fx_dyn: float = 0.0     # Dynamically filtered longitudinal force (N)
    Fy_dyn: float = 0.0     # Dynamically filtered lateral force (N)
    t_prev: float = 0.0     # Timestamp of last update (s)


def compute_slip_quantities(
    vx_wheel: float,
    vy_wheel: float,
    omega_wheel: float,
    Re: float,
    v_eps: float = 0.05,
) -> Tuple[float, float]:
    """Compute regularized lateral slip angle alpha and longitudinal slip ratio kappa.

    Args:
        vx_wheel: Longitudinal wheel velocity along wheel forward heading (m/s).
        vy_wheel: Lateral wheel velocity perpendicular to wheel heading (m/s).
        omega_wheel: Wheel rotational angular velocity (rad/s).
        Re: Effective tire rolling radius (m).
        v_eps: Regularization parameter preventing zero denominator (m/s).

    Returns:
        (alpha, kappa): Slip angle in radians and longitudinal slip ratio.
    """
    v_denom = np.sqrt(float(vx_wheel)**2 + float(v_eps)**2)
    alpha = np.arctan2(float(vy_wheel), v_denom)
    kappa = (float(omega_wheel) * float(Re) - float(vx_wheel)) / v_denom
    return float(alpha), float(kappa)


class TireModelInterface(ABC):
    """Abstract interface defining the standardized public contract for all CTS tire models."""

    @abstractmethod
    def compute_steady_state_forces(
        self,
        alpha: float,
        kappa: float,
        Fz: float,
        mu: float,
    ) -> TireForces:
        """Compute instantaneous steady-state contact forces without relaxation lag."""
        pass

    @abstractmethod
    def step_dynamic(
        self,
        alpha: float,
        kappa: float,
        Fz: float,
        mu: float,
        vx: float,
        dt: float,
        state: DynamicTireState,
    ) -> Tuple[TireForces, DynamicTireState]:
        """Compute dynamic contact forces integrated via first-order relaxation ODE."""
        pass

    @abstractmethod
    def compute_from_wheel_velocity(
        self,
        vx_wheel: float,
        vy_wheel: float,
        omega_wheel: float,
        Fz: float,
        mu: float,
        Re: float,
        dt: Optional[float] = None,
        state: Optional[DynamicTireState] = None,
    ) -> Tuple[TireForces, Optional[DynamicTireState]]:
        """Convenience evaluation directly from contact patch linear and angular velocities."""
        pass


class LinearTireModel(TireModelInterface):
    """Linear tire model with adhesion circle saturation (Ablation T0)."""

    def __init__(
        self,
        cornering_stiffness: float = 55000.0,
        longitudinal_stiffness: float = 75000.0,
        relaxation_length: float = 0.15,
    ) -> None:
        self.C_alpha = float(cornering_stiffness)
        self.C_s = float(longitudinal_stiffness)
        self.sigma = float(relaxation_length)

    def compute_steady_state_forces(self, alpha: float, kappa: float, Fz: float, mu: float) -> TireForces:
        Fz_eff = max(float(Fz), 0.0)
        if Fz_eff <= 1e-6 or mu <= 1e-6:
            return TireForces(0.0, 0.0, Fz_eff, 0.0, False, float(alpha), float(kappa))

        Fx_lin = self.C_s * float(kappa)
        Fy_lin = - self.C_alpha * float(alpha)

        # Adhesion circle saturation
        F_max = float(mu) * Fz_eff
        F_res = np.hypot(Fx_lin, Fy_lin)
        eta = F_res / F_max if F_max > 1e-9 else 0.0

        if eta > 1.0:
            Fx = Fx_lin / eta
            Fy = Fy_lin / eta
            return TireForces(float(Fx), float(Fy), Fz_eff, 1.0, True, float(alpha), float(kappa))
        return TireForces(float(Fx_lin), float(Fy_lin), Fz_eff, float(eta), False, float(alpha), float(kappa))

    def step_dynamic(
        self,
        alpha: float,
        kappa: float,
        Fz: float,
        mu: float,
        vx: float,
        dt: float,
        state: DynamicTireState,
    ) -> Tuple[TireForces, DynamicTireState]:
        Fz_eff = max(float(Fz), 0.0)
        mu_val = max(float(mu), 0.0)

        # 1. Unilateral lift-off and zero friction reset: eliminate mid-air ghost forces
        if Fz_eff <= 1e-6 or mu_val <= 1e-6:
            state.Fx_dyn = 0.0
            state.Fy_dyn = 0.0
            return TireForces(
                Fx=0.0,
                Fy=0.0,
                Fz=float(Fz_eff),
                adhesion_ratio=0.0,
                is_saturated=False,
                alpha=float(alpha),
                kappa=float(kappa),
            ), state

        # 2. Steady-state target evaluation & exponential relaxation ODE integration
        steady = self.compute_steady_state_forces(alpha, kappa, Fz, mu)
        vx_abs = max(abs(float(vx)), 0.01)
        decay = np.exp(- (vx_abs * float(dt)) / self.sigma)

        Fx_dyn = steady.Fx + (state.Fx_dyn - steady.Fx) * decay
        Fy_dyn = steady.Fy + (state.Fy_dyn - steady.Fy) * decay

        # 3. Dynamic adhesion circle projection operator
        F_max = mu_val * Fz_eff
        F_res = np.hypot(Fx_dyn, Fy_dyn)

        if F_res > F_max and F_max > 1e-9:
            scale = F_max / F_res
            Fx_dyn *= scale
            Fy_dyn *= scale
            eta_dyn = 1.0
            is_sat = True
        else:
            eta_dyn = min(float(F_res / F_max), 1.0) if F_max > 1e-9 else 0.0
            is_sat = bool(eta_dyn >= 0.9999 or steady.is_saturated)

        state.Fx_dyn = float(Fx_dyn)
        state.Fy_dyn = float(Fy_dyn)

        dyn_forces = TireForces(
            Fx=float(Fx_dyn),
            Fy=float(Fy_dyn),
            Fz=float(Fz_eff),
            adhesion_ratio=float(eta_dyn),
            is_saturated=bool(is_sat),
            alpha=float(alpha),
            kappa=float(kappa),
        )
        return dyn_forces, state

    def compute_from_wheel_velocity(
        self,
        vx_wheel: float,
        vy_wheel: float,
        omega_wheel: float,
        Fz: float,
        mu: float,
        Re: float,
        dt: Optional[float] = None,
        state: Optional[DynamicTireState] = None,
    ) -> Tuple[TireForces, Optional[DynamicTireState]]:
        alpha, kappa = compute_slip_quantities(vx_wheel, vy_wheel, omega_wheel, Re)
        if dt is not None and state is not None:
            return self.step_dynamic(alpha, kappa, Fz, mu, vx_wheel, dt, state)
        return self.compute_steady_state_forces(alpha, kappa, Fz, mu), None


class DugoffTireModel(TireModelInterface):
    """Dugoff Combined Slip Tire Model with C^1 continuous f(lambda) (Ablation T1)."""

    def __init__(
        self,
        cornering_stiffness: float = 55000.0,
        longitudinal_stiffness: float = 75000.0,
        relaxation_length: float = 0.15,
        lambda_thresh: float = 1.0,
    ) -> None:
        self.C_alpha = float(cornering_stiffness)
        self.C_s = float(longitudinal_stiffness)
        self.sigma = float(relaxation_length)
        self.lambda_thresh = float(lambda_thresh)

    def compute_steady_state_forces(self, alpha: float, kappa: float, Fz: float, mu: float) -> TireForces:
        Fz_eff = max(float(Fz), 0.0)
        if Fz_eff <= 1e-6 or mu <= 1e-6:
            return TireForces(0.0, 0.0, Fz_eff, 0.0, False, float(alpha), float(kappa))

        s_x = float(kappa)
        tan_alpha = np.tan(float(alpha))
        denom_slip = 1.0 + abs(s_x)

        sigma_x = s_x / denom_slip
        sigma_y = tan_alpha / denom_slip

        Fx0 = self.C_s * sigma_x
        Fy0 = - self.C_alpha * sigma_y

        F_linear_res = np.hypot(Fx0, Fy0)
        mu_Fz = float(mu) * Fz_eff

        if F_linear_res < 1e-9:
            lam = 100.0
            f_lam = 1.0
        else:
            lam = mu_Fz / (2.0 * F_linear_res)
            if lam < 1.0:
                f_lam = (2.0 - lam) * lam
            else:
                f_lam = 1.0

        Fx = Fx0 * f_lam
        Fy = Fy0 * f_lam

        # Friction circle safety clamp
        res_force = np.hypot(Fx, Fy)
        eta = res_force / mu_Fz if mu_Fz > 1e-9 else 0.0
        if eta > 1.0:
            Fx /= eta
            Fy /= eta
            eta = 1.0
            is_sat = True
        else:
            is_sat = (lam < 1.0)

        return TireForces(float(Fx), float(Fy), Fz_eff, float(eta), bool(is_sat), float(alpha), float(kappa))

    def step_dynamic(
        self,
        alpha: float,
        kappa: float,
        Fz: float,
        mu: float,
        vx: float,
        dt: float,
        state: DynamicTireState,
    ) -> Tuple[TireForces, DynamicTireState]:
        Fz_eff = max(float(Fz), 0.0)
        mu_val = max(float(mu), 0.0)

        # 1. Unilateral lift-off and zero friction reset: eliminate mid-air ghost forces
        if Fz_eff <= 1e-6 or mu_val <= 1e-6:
            state.Fx_dyn = 0.0
            state.Fy_dyn = 0.0
            return TireForces(
                Fx=0.0,
                Fy=0.0,
                Fz=float(Fz_eff),
                adhesion_ratio=0.0,
                is_saturated=False,
                alpha=float(alpha),
                kappa=float(kappa),
            ), state

        # 2. Steady-state target evaluation & exponential relaxation ODE integration
        steady = self.compute_steady_state_forces(alpha, kappa, Fz, mu)
        vx_abs = max(abs(float(vx)), 0.01)
        decay = np.exp(- (vx_abs * float(dt)) / self.sigma)

        Fx_dyn = steady.Fx + (state.Fx_dyn - steady.Fx) * decay
        Fy_dyn = steady.Fy + (state.Fy_dyn - steady.Fy) * decay

        # 3. Dynamic adhesion circle projection operator
        F_max = mu_val * Fz_eff
        F_res = np.hypot(Fx_dyn, Fy_dyn)

        if F_res > F_max and F_max > 1e-9:
            scale = F_max / F_res
            Fx_dyn *= scale
            Fy_dyn *= scale
            eta_dyn = 1.0
            is_sat = True
        else:
            eta_dyn = min(float(F_res / F_max), 1.0) if F_max > 1e-9 else 0.0
            is_sat = bool(eta_dyn >= 0.9999 or steady.is_saturated)

        state.Fx_dyn = float(Fx_dyn)
        state.Fy_dyn = float(Fy_dyn)

        dyn_forces = TireForces(
            Fx=float(Fx_dyn),
            Fy=float(Fy_dyn),
            Fz=float(Fz_eff),
            adhesion_ratio=float(eta_dyn),
            is_saturated=bool(is_sat),
            alpha=float(alpha),
            kappa=float(kappa),
        )
        return dyn_forces, state

    def compute_from_wheel_velocity(
        self,
        vx_wheel: float,
        vy_wheel: float,
        omega_wheel: float,
        Fz: float,
        mu: float,
        Re: float,
        dt: Optional[float] = None,
        state: Optional[DynamicTireState] = None,
    ) -> Tuple[TireForces, Optional[DynamicTireState]]:
        alpha, kappa = compute_slip_quantities(vx_wheel, vy_wheel, omega_wheel, Re)
        if dt is not None and state is not None:
            return self.step_dynamic(alpha, kappa, Fz, mu, vx_wheel, dt, state)
        return self.compute_steady_state_forces(alpha, kappa, Fz, mu), None


class PacejkaTireModel(TireModelInterface):
    """Pacejka Magic Formula MF-5.2 with Combined Slip & Friction Ellipse (Ablation T2)."""

    def __init__(
        self,
        cornering_stiffness: float = 55000.0,
        longitudinal_stiffness: float = 75000.0,
        relaxation_length: float = 0.15,
        B: float = 10.0,
        C_lat: float = 1.30,
        C_long: float = 1.65,
        E: float = -0.10,
    ) -> None:
        self.C_alpha = float(cornering_stiffness)
        self.C_s = float(longitudinal_stiffness)
        self.sigma = float(relaxation_length)
        self.C_lat = float(C_lat)
        self.C_long = float(C_long)
        self.E = float(E)

    def compute_steady_state_forces(self, alpha: float, kappa: float, Fz: float, mu: float) -> TireForces:
        Fz_eff = max(float(Fz), 0.0)
        if Fz_eff <= 1e-6 or mu <= 1e-6:
            return TireForces(0.0, 0.0, Fz_eff, 0.0, False, float(alpha), float(kappa))

        mu_Fz = float(mu) * Fz_eff

        # Pure Longitudinal Slip
        Dx = mu_Fz
        Bx = self.C_s / max(self.C_long * Dx, 1e-6)
        sx = float(kappa)
        arg_x = Bx * sx - self.E * (Bx * sx - np.arctan(Bx * sx))
        Fx0 = Dx * np.sin(self.C_long * np.arctan(arg_x))

        # Pure Lateral Slip
        Dy = mu_Fz
        By = self.C_alpha / max(self.C_lat * Dy, 1e-6)
        al = float(alpha)
        arg_y = By * al - self.E * (By * al - np.arctan(By * al))
        Fy0 = - Dy * np.sin(self.C_lat * np.arctan(arg_y))

        # PAC2002 Weighting & Adhesion Ellipse Projection
        G_xa = np.cos(1.15 * np.arctan(8.0 * al))
        G_yk = np.cos(1.10 * np.arctan(10.0 * sx))

        Fx_cand = Fx0 * G_xa
        Fy_cand = Fy0 * G_yk

        # Adhesion Ellipse Check
        eta = np.hypot(Fx_cand, Fy_cand) / mu_Fz if mu_Fz > 1e-9 else 0.0
        peak_y = np.pi / (2.0 * self.C_lat)
        peak_x = np.pi / (2.0 * self.C_long)
        is_sat = bool(eta >= 0.95 or abs(arg_y) >= peak_y or abs(arg_x) >= peak_x)
        if eta > 1.0:
            Fx = Fx_cand / eta
            Fy = Fy_cand / eta
            return TireForces(float(Fx), float(Fy), Fz_eff, 1.0, True, float(alpha), float(kappa))
        return TireForces(float(Fx_cand), float(Fy_cand), Fz_eff, float(eta), is_sat, float(alpha), float(kappa))

    def step_dynamic(
        self,
        alpha: float,
        kappa: float,
        Fz: float,
        mu: float,
        vx: float,
        dt: float,
        state: DynamicTireState,
    ) -> Tuple[TireForces, DynamicTireState]:
        Fz_eff = max(float(Fz), 0.0)
        mu_val = max(float(mu), 0.0)

        # 1. Unilateral lift-off and zero friction reset: eliminate mid-air ghost forces
        if Fz_eff <= 1e-6 or mu_val <= 1e-6:
            state.Fx_dyn = 0.0
            state.Fy_dyn = 0.0
            return TireForces(
                Fx=0.0,
                Fy=0.0,
                Fz=float(Fz_eff),
                adhesion_ratio=0.0,
                is_saturated=False,
                alpha=float(alpha),
                kappa=float(kappa),
            ), state

        # 2. Steady-state target evaluation & exponential relaxation ODE integration
        steady = self.compute_steady_state_forces(alpha, kappa, Fz, mu)
        vx_abs = max(abs(float(vx)), 0.01)
        decay = np.exp(- (vx_abs * float(dt)) / self.sigma)

        Fx_dyn = steady.Fx + (state.Fx_dyn - steady.Fx) * decay
        Fy_dyn = steady.Fy + (state.Fy_dyn - steady.Fy) * decay

        # 3. Dynamic adhesion circle projection operator
        F_max = mu_val * Fz_eff
        F_res = np.hypot(Fx_dyn, Fy_dyn)

        if F_res > F_max and F_max > 1e-9:
            scale = F_max / F_res
            Fx_dyn *= scale
            Fy_dyn *= scale
            eta_dyn = 1.0
            is_sat = True
        else:
            eta_dyn = min(float(F_res / F_max), 1.0) if F_max > 1e-9 else 0.0
            is_sat = bool(eta_dyn >= 0.9999 or steady.is_saturated)

        state.Fx_dyn = float(Fx_dyn)
        state.Fy_dyn = float(Fy_dyn)

        dyn_forces = TireForces(
            Fx=float(Fx_dyn),
            Fy=float(Fy_dyn),
            Fz=float(Fz_eff),
            adhesion_ratio=float(eta_dyn),
            is_saturated=bool(is_sat),
            alpha=float(alpha),
            kappa=float(kappa),
        )
        return dyn_forces, state

    def compute_from_wheel_velocity(
        self,
        vx_wheel: float,
        vy_wheel: float,
        omega_wheel: float,
        Fz: float,
        mu: float,
        Re: float,
        dt: Optional[float] = None,
        state: Optional[DynamicTireState] = None,
    ) -> Tuple[TireForces, Optional[DynamicTireState]]:
        alpha, kappa = compute_slip_quantities(vx_wheel, vy_wheel, omega_wheel, Re)
        if dt is not None and state is not None:
            return self.step_dynamic(alpha, kappa, Fz, mu, vx_wheel, dt, state)
        return self.compute_steady_state_forces(alpha, kappa, Fz, mu), None


def create_tire_model(
    model_type: str = "dugoff",
    config: Optional[TireConfig] = None,
    is_aircraft: bool = False,
) -> TireModelInterface:
    """Factory creating tire models initialized from TireConfig."""
    cfg = config or TireConfig()
    if is_aircraft:
        c_alpha = cfg.cornering_stiffness_aircraft_main
        c_s = cfg.longitudinal_stiffness_aircraft_main
        sigma = cfg.relaxation_length_aircraft
    else:
        c_alpha = cfg.cornering_stiffness_tractor
        c_s = cfg.longitudinal_stiffness_tractor
        sigma = cfg.relaxation_length_tractor

    m_lower = model_type.lower()
    if m_lower in ("linear", "t0"):
        return LinearTireModel(c_alpha, c_s, sigma)
    elif m_lower in ("dugoff", "t1"):
        return DugoffTireModel(c_alpha, c_s, sigma, cfg.dugoff_lambda_thresh)
    elif m_lower in ("pacejka", "t2", "mf52", "magic_formula"):
        return PacejkaTireModel(c_alpha, c_s, sigma, cfg.pacejka_B, cfg.pacejka_C, 1.65, cfg.pacejka_E)
    else:
        raise ValueError(f"Unknown tire model type: {model_type}. Expected 'linear', 'dugoff', or 'pacejka'.")
