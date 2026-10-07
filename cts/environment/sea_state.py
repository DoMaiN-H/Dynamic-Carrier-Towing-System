"""cts.environment.sea_state

Ocean Wave Spectrum Modeling and Random Surface Generation.
Supports Pierson-Moskowitz (PM) and JONSWAP spectra with analytical moments,
frequency discretization (Uniform, Jittered, Equal-Energy), and 1D/2D spatial wave fields.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union
import numpy as np

GRAVITY = 9.80665

# Cross-version numpy trapezoid integration helper
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


class SeaStateLevel(Enum):
    SS4 = "SS4"  # Moderate: Hs = 2.5m, Tp = 8.8s
    SS5 = "SS5"  # Rough:    Hs = 4.0m, Tp = 10.2s


class WaveSpectrumType(Enum):
    PIERSON_MOSKOWITZ = "PM"
    JONSWAP = "JONSWAP"


class DiscretizationMethod(Enum):
    UNIFORM = "uniform"
    JITTERED = "jittered"
    EQUAL_ENERGY = "equal_energy"


@dataclass(frozen=True)
class WaveSpectrumConfig:
    """Wave spectrum configuration parameters."""
    hs: float                # Significant wave height [m]
    tp: float                # Peak wave period [s]
    spectrum_type: WaveSpectrumType = WaveSpectrumType.JONSWAP
    gamma: float = 3.3       # Peak enhancement factor (JONSWAP)
    sigma_a: float = 0.07    # Spectral width factor (omega <= omega_p)
    sigma_b: float = 0.09    # Spectral width factor (omega > omega_p)
    principal_dir_rad: float = 0.0  # Primary wave heading [rad]
    spreading_s: int = 4     # Cosine-2s directional spreading parameter

    @property
    def omega_p(self) -> float:
        return 2.0 * np.pi / self.tp

    @property
    def theoretical_m0(self) -> float:
        return (self.hs ** 2) / 16.0

    @property
    def theoretical_rms(self) -> float:
        return self.hs / 4.0


# Predefined Sea State Presets
SEASTATE_4_CONFIG = WaveSpectrumConfig(
    hs=2.5, tp=8.8, spectrum_type=WaveSpectrumType.JONSWAP, gamma=3.3
)
SEASTATE_5_CONFIG = WaveSpectrumConfig(
    hs=4.0, tp=10.2, spectrum_type=WaveSpectrumType.JONSWAP, gamma=3.3
)


@dataclass
class WaveComponents:
    """Harmonic wave components resulting from spectral discretization."""
    omega: np.ndarray       # Frequencies [rad/s] (N_w,)
    k_num: np.ndarray       # Wave numbers [rad/m] (N_w,)
    amplitudes: np.ndarray  # Wave amplitudes [m] (N_w,)
    phases: np.ndarray      # Random phases [rad] (N_w,)
    directions: np.ndarray  # Wave directions [rad] (N_w,)


class WaveSpectrum:
    """Base class for ocean wave energy spectrum."""

    def __init__(self, config: WaveSpectrumConfig):
        self.config = config

    def evaluate(self, omega: Union[float, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def compute_moments(
        self, omega_max: float = 6.0, n_pts: int = 5000
    ) -> Tuple[float, float, float, float]:
        """Compute spectral moments m_0, m_1, m_2, m_4 numerically."""
        w = np.linspace(0.01, omega_max, n_pts)
        S = self.evaluate(w)
        m0 = float(_trapz(S, w))
        m1 = float(_trapz(S * w, w))
        m2 = float(_trapz(S * (w ** 2), w))
        m4 = float(_trapz(S * (w ** 4), w))
        return m0, m1, m2, m4


class PiersonMoskowitzSpectrum(WaveSpectrum):
    """Pierson-Moskowitz (PM) Wave Spectrum.

    S_PM(w) = (5/16) * (Hs^2 * wp^4 / w^5) * exp(-1.25 * (wp / w)^4)
    """

    def evaluate(self, omega: Union[float, np.ndarray]) -> np.ndarray:
        w = np.maximum(np.asarray(omega, dtype=np.float64), 1e-6)
        hs = self.config.hs
        wp = self.config.omega_p
        A = (5.0 / 16.0) * (hs ** 2) * (wp ** 4)
        B = 1.25 * (wp ** 4)
        s_pm = (A / (w ** 5)) * np.exp(-B / (w ** 4))
        if np.ndim(omega) == 0:
            return float(s_pm)
        return s_pm


class JONSWAPSpectrum(WaveSpectrum):
    """JONSWAP Wave Spectrum with Goda exact energy normalization.

    S_J(w) = A_gamma * S_PM(w) * gamma^exp(-(w - wp)^2 / (2 * sigma^2 * wp^2))
    """

    def __init__(self, config: WaveSpectrumConfig):
        super().__init__(config)
        self.pm = PiersonMoskowitzSpectrum(config)
        # Compute exact normalization constant A_gamma to enforce m0 == Hs^2 / 16
        wp = self.config.omega_p
        w_norm = np.linspace(0.01, 6.0 * wp, 5000)
        raw_jonswap = self._raw_evaluate(w_norm)
        raw_m0 = float(_trapz(raw_jonswap, w_norm))
        target_m0 = self.config.theoretical_m0
        self.a_gamma = target_m0 / raw_m0 if raw_m0 > 0 else 1.0

    def _raw_evaluate(self, omega: np.ndarray) -> np.ndarray:
        w = np.maximum(np.asarray(omega, dtype=np.float64), 1e-6)
        s_pm = self.pm.evaluate(w)
        wp = self.config.omega_p
        sigma = np.where(w <= wp, self.config.sigma_a, self.config.sigma_b)
        exponent = -((w - wp) ** 2) / (2.0 * (sigma ** 2) * (wp ** 2))
        peak_enhancement = self.config.gamma ** np.exp(exponent)
        return s_pm * peak_enhancement

    def evaluate(self, omega: Union[float, np.ndarray]) -> np.ndarray:
        w = np.asarray(omega, dtype=np.float64)
        res = self.a_gamma * self._raw_evaluate(w)
        if np.ndim(omega) == 0:
            return float(res)
        return res


class SeaStateGenerator:
    """High-level Ocean Sea State and Wave Field Generator.

    Supports random wave elevation calculation and deterministic seed control.
    """

    def __init__(
        self,
        config: WaveSpectrumConfig,
        n_frequencies: int = 150,
        omega_min: float = 0.2,
        omega_max: float = 2.5,
        discretization: DiscretizationMethod = DiscretizationMethod.JITTERED,
        seed: Optional[int] = 2026,
    ):
        self.config = config
        self.n_frequencies = int(n_frequencies)
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.discretization = discretization
        self.rng = np.random.default_rng(seed)

        if config.spectrum_type == WaveSpectrumType.PIERSON_MOSKOWITZ:
            self.spectrum = PiersonMoskowitzSpectrum(config)
        else:
            self.spectrum = JONSWAPSpectrum(config)

        self.components = self._discretize()

    def _discretize(self) -> WaveComponents:
        N = self.n_frequencies
        if self.discretization == DiscretizationMethod.UNIFORM:
            delta_w = (self.omega_max - self.omega_min) / N
            w = self.omega_min + (np.arange(N) + 0.5) * delta_w
            delta_w_vec = np.full(N, delta_w)
        elif self.discretization == DiscretizationMethod.JITTERED:
            delta_w = (self.omega_max - self.omega_min) / N
            jitter = self.rng.uniform(-0.4, 0.4, size=N) * delta_w
            w = self.omega_min + (np.arange(N) + 0.5) * delta_w + jitter
            delta_w_vec = np.full(N, delta_w)
        elif self.discretization == DiscretizationMethod.EQUAL_ENERGY:
            # Constant area bins
            w_dense = np.linspace(self.omega_min, self.omega_max, 5000)
            S_dense = self.spectrum.evaluate(w_dense)
            cum_m0 = np.concatenate([[0.0], np.cumsum((S_dense[:-1] + S_dense[1:]) * 0.5 * np.diff(w_dense))])
            total_m0 = cum_m0[-1]
            target_cum = (np.arange(N) + 0.5) * (total_m0 / N)
            w = np.interp(target_cum, cum_m0, w_dense)
            delta_w_vec = np.gradient(w)
        else:
            raise ValueError(f"Unknown discretization method: {self.discretization}")

        S_w = self.spectrum.evaluate(w)
        if self.discretization == DiscretizationMethod.EQUAL_ENERGY:
            delta_m = self.config.theoretical_m0 / N
            amp = np.sqrt(2.0 * delta_m) * np.ones(N)
        else:
            amp = np.sqrt(2.0 * S_w * delta_w_vec)

        k = (w ** 2) / GRAVITY
        phases = self.rng.uniform(0.0, 2.0 * np.pi, size=N)
        dirs = np.full(N, self.config.principal_dir_rad)

        return WaveComponents(omega=w, k_num=k, amplitudes=amp, phases=phases, directions=dirs)

    def elevation_1d(self, x: float, t: float) -> float:
        """Evaluate 1D long-crested wave elevation at position x and time t."""
        c = self.components
        arg = c.k_num * float(x) - c.omega * float(t) + c.phases
        return float(np.sum(c.amplitudes * np.cos(arg)))

    def elevation_mesh(self, x_grid: np.ndarray, y_grid: np.ndarray, t: float) -> np.ndarray:
        """Vectorized 2D wave surface elevation field over coordinate grid."""
        c = self.components
        kx = c.k_num * np.cos(c.directions)
        ky = c.k_num * np.sin(c.directions)
        arg = (
            kx[:, None, None] * x_grid[None, :, :] +
            ky[:, None, None] * y_grid[None, :, :] -
            c.omega[:, None, None] * float(t) +
            c.phases[:, None, None]
        )
        return np.sum(c.amplitudes[:, None, None] * np.cos(arg), axis=0)

    def reset_seed(self, seed: Optional[int] = None):
        """Reset RNG and re-sample random phases."""
        self.rng = np.random.default_rng(seed)
        self.components = self._discretize()
