"""cts.common.math_utils

SE(2) and SO(3) rotations, Euler angles (RPY), rotation matrix conversions R_D^W and R_W^D,
angular velocity kinematics, skew-symmetric operators, and Tustin low-pass differentiator.
"""

from __future__ import annotations
from typing import Tuple, Union, Optional
import numpy as np


def wrap_to_pi(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wrap angle to [-pi, pi)."""
    return ((angle + np.pi) % (2.0 * np.pi)) - np.pi


def wrap_to_2pi(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wrap angle to [0, 2*pi)."""
    return angle % (2.0 * np.pi)


def se2_matrix(x: float, y: float, theta: float) -> np.ndarray:
    """Construct 3x3 homogeneous transformation matrix in SE(2)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, float(x)],
        [s,  c, float(y)],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def se2_inv(T: np.ndarray) -> np.ndarray:
    """Compute exact inverse of SE(2) homogeneous transformation matrix."""
    R_t = T[:2, :2].T
    t = T[:2, 2]
    T_inv = np.eye(3, dtype=np.float64)
    T_inv[:2, :2] = R_t
    T_inv[:2, 2] = -R_t @ t
    return T_inv


def se2_apply(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply SE(2) transform to points of shape (N, 2) or (2,)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        return (T[:2, :2] @ pts) + T[:2, 2]
    return (pts @ T[:2, :2].T) + T[:2, 2]


def rpy_to_R_d_to_w(phi: float, theta: float, psi: float) -> np.ndarray:
    """Compute rotation matrix R_D^W (Deck frame to World frame).

    Tait-Bryan ZYX convention: R_D^W = Rz(psi) * Ry(theta) * Rx(phi).
    Converts vectors from deck coordinates to world coordinates: v_W = R_D^W * v_D.
    """
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    return np.array([
        [cps * cth, cps * sth * sph - sps * cph, cps * sth * cph + sps * sph],
        [sps * cth, sps * sth * sph + cps * cph, sps * sth * cph - cps * sph],
        [-sth,      cth * sph,                  cth * cph]
    ], dtype=np.float64)


def rpy_to_R_w_to_d(phi: float, theta: float, psi: float) -> np.ndarray:
    """Compute rotation matrix R_W^D (World frame to Deck frame).

    Inverse/transpose of R_D^W: v_D = R_W^D * v_W.
    """
    return rpy_to_R_d_to_w(phi, theta, psi).T


def R_d_to_w_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    """Extract Euler angles (phi, theta, psi) from rotation matrix R_D^W.

    Returns:
        (phi, theta, psi) in radians: Roll, Pitch, Yaw.
    """
    r31 = float(np.clip(R[2, 0], -1.0, 1.0))
    theta = -np.arcsin(r31)
    cos_th = np.cos(theta)

    if np.abs(cos_th) > 1e-6:
        phi = float(np.arctan2(R[2, 1], R[2, 2]))
        psi = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        # Gimbal lock singularity handling
        psi = 0.0
        if r31 < 0.0:  # theta = +pi/2
            phi = float(np.arctan2(R[0, 1], R[1, 1]))
        else:          # theta = -pi/2
            phi = float(np.arctan2(-R[0, 1], R[1, 1]))

    return phi, theta, psi


def euler_rates_to_body_rates(phi: float, theta: float, dphi: float, dtheta: float, dpsi: float) -> np.ndarray:
    """Convert Euler angle rates [dphi, dtheta, dpsi] to body angular velocity omega_D = [p, q, r]^T."""
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    J = np.array([
        [1.0, 0.0, -sth],
        [0.0, cph,  sph * cth],
        [0.0, -sph, cph * cth]
    ], dtype=np.float64)
    rates = np.array([dphi, dtheta, dpsi], dtype=np.float64)
    return J @ rates


def body_rates_to_euler_rates(phi: float, theta: float, p: float, q: float, r: float) -> np.ndarray:
    """Convert body angular velocity omega_D = [p, q, r]^T to Euler angle rates [dphi, dtheta, dpsi]^T."""
    cph, sph = np.cos(phi), np.sin(phi)
    cth = np.cos(theta)
    if np.abs(cth) < 1e-6:
        cth = 1e-6 if cth >= 0 else -1e-6
    tan_th = np.tan(theta)

    J_inv = np.array([
        [1.0, sph * tan_th, cph * tan_th],
        [0.0, cph,          -sph],
        [0.0, sph / cth,    cph / cth]
    ], dtype=np.float64)
    omega = np.array([p, q, r], dtype=np.float64)
    return J_inv @ omega


def skew_symmetric_3x3(vector: np.ndarray) -> np.ndarray:
    """Compute 3x3 skew-symmetric cross-product matrix [v]_x."""
    v = np.asarray(vector, dtype=np.float64)
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0]
    ], dtype=np.float64)


class LowPassDifferentiator:
    """Tustin bilinear numerical differentiator with 1st-order low-pass filter H(s) = s / (tau * s + 1)."""

    def __init__(self, tau: float = 0.05, dt: float = 0.02):
        self.tau = float(tau)
        self.dt = float(dt)
        self.reset()

    def reset(self, initial_x: float = 0.0, initial_y: float = 0.0):
        self._x_prev = float(initial_x)
        self._y_prev = float(initial_y)
        self._initialized = False

    def step(self, x: float, dt: Optional[float] = None) -> float:
        """Process one time step and return filtered derivative dx/dt."""
        dt_val = float(dt) if dt is not None else self.dt
        if not self._initialized:
            self._x_prev = float(x)
            self._y_prev = 0.0
            self._initialized = True
            return 0.0

        a0 = 2.0 * self.tau + dt_val
        a1 = dt_val - 2.0 * self.tau
        b0 = 2.0
        b1 = -2.0

        y = -(a1 / a0) * self._y_prev + (b0 / a0) * x + (b1 / a0) * self._x_prev
        self._x_prev = float(x)
        self._y_prev = float(y)
        return float(y)

    def filter_series(self, x_series: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        """Batch filter an entire 1D array of positions."""
        dt_val = float(dt) if dt is not None else self.dt
        x_arr = np.asarray(x_series, dtype=np.float64)
        y_arr = np.zeros_like(x_arr)
        self.reset(initial_x=float(x_arr[0]), initial_y=0.0)
        for i, val in enumerate(x_arr):
            y_arr[i] = self.step(float(val), dt=dt_val)
        return y_arr
