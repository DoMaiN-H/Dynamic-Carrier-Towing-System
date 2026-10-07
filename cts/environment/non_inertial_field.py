"""cts.environment.non_inertial_field

Closed-form vectorized computation of the 5-component non-inertial apparent
acceleration field and effective gravity on an aircraft carrier flight deck.

References:
    - ORIGINAL_REQUEST.md: Requirements R1 and Acceptance Criteria AC 1.
    - PROJECT.md: Feature F4 (Non-Inertial Acceleration Field) & Feature F5.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Union, Tuple, Optional
import numpy as np

GRAVITY = 9.80665


@dataclass
class DeckState:
    """Instantaneous 6-DOF kinematic state of the carrier flight deck.

    All coordinates follow right-handed Cartesian conventions:
      - World Frame {W}: +X_W = Forward/North, +Y_W = Port/West, +Z_W = Up
      - Deck Frame {D}:  +X_D = Bow (forward),  +Y_D = Port (left),  +Z_D = Up (deck normal)

    Attributes:
        pos_w: (3,) Ship CG position in World frame [surge, sway, heave] (m).
        rpy_rad: (3,) Deck attitude Euler angles [roll phi, pitch theta, yaw psi] (rad).
        vel_w: (3,) Ship CG translational velocity in World frame (m/s).
        omega_d: (3,) Deck angular velocity resolved in Deck frame [p, q, r] (rad/s).
        acc_w: (3,) Ship CG translational acceleration in World frame (m/s^2).
        alpha_d: (3,) Deck angular acceleration resolved in Deck frame [dot_p, dot_q, dot_r] (rad/s^2).
        R_d_to_w: (3, 3) SO(3) rotation matrix transforming vectors from {D} to {W}.
        R_w_to_d: (3, 3) SO(3) rotation matrix transforming vectors from {W} to {D}.
        t: Simulation timestamp in seconds (default: 0.0).
    """
    pos_w: np.ndarray
    rpy_rad: np.ndarray
    vel_w: np.ndarray
    omega_d: np.ndarray
    acc_w: np.ndarray
    alpha_d: np.ndarray
    R_d_to_w: np.ndarray
    R_w_to_d: np.ndarray
    t: float = 0.0

    def __post_init__(self) -> None:
        self.pos_w = np.asarray(self.pos_w, dtype=np.float64)
        self.rpy_rad = np.asarray(self.rpy_rad, dtype=np.float64)
        self.vel_w = np.asarray(self.vel_w, dtype=np.float64)
        self.omega_d = np.asarray(self.omega_d, dtype=np.float64)
        self.acc_w = np.asarray(self.acc_w, dtype=np.float64)
        self.alpha_d = np.asarray(self.alpha_d, dtype=np.float64)
        self.R_d_to_w = np.asarray(self.R_d_to_w, dtype=np.float64)
        self.R_w_to_d = np.asarray(self.R_w_to_d, dtype=np.float64)
        self.t = float(self.t)


@dataclass(frozen=True)
class ApparentAcceleration:
    """Container for the decomposed non-inertial apparent acceleration components.

    All vectors are resolved in the moving Deck Frame {D}.
    The relative acceleration of a mass element on the deck satisfies:
        a_rel = F_contact / m + a_apparent_total
    where:
        a_apparent_total = g_tilt + a_trans + a_euler + a_centripetal + a_coriolis

    Attributes:
        a_apparent_total: (..., 3) Total apparent acceleration field vector (m/s^2).
        g_tilt: (..., 3) Dynamic tilting gravity component: R_w_to_d @ [0, 0, -g]^T (m/s^2).
        a_trans: (..., 3) Translational apparent acceleration: -R_w_to_d @ acc_w (m/s^2).
        a_euler: (..., 3) Euler angular acceleration inertial component: -alpha_d x r_d (m/s^2).
        a_centripetal: (..., 3) Centripetal apparent acceleration: -omega_d x (omega_d x r_d) (m/s^2).
        a_coriolis: (..., 3) Coriolis apparent acceleration: -2 * omega_d x vel_rel_d (m/s^2).
    """
    a_apparent_total: np.ndarray
    g_tilt: np.ndarray
    a_trans: np.ndarray
    a_euler: np.ndarray
    a_centripetal: np.ndarray
    a_coriolis: np.ndarray

    @property
    def g_eff(self) -> Union[float, np.ndarray]:
        """Effective normal gravity holding the vehicle against the deck surface (m/s^2).

        Defined as the downward normal component along -Z_D:
            g_eff = - a_apparent_total[..., 2]
        """
        val = -self.a_apparent_total[..., 2]
        if np.ndim(val) == 0:
            return float(val)
        return val


def compute_dynamic_tilting_gravity(
    R_w_to_d: np.ndarray,
    g: float = GRAVITY,
) -> np.ndarray:
    """Compute dynamic tilting gravity vector resolved in Deck Frame {D}.

    g^(D) = R_w_to_d @ [0, 0, -g]^T = -g * R_w_to_d[:, 2]
    """
    return -float(g) * np.asarray(R_w_to_d, dtype=np.float64)[:, 2]


def compute_translational_apparent_acceleration(
    R_w_to_d: np.ndarray,
    acc_w: np.ndarray,
) -> np.ndarray:
    """Compute linear translational apparent acceleration in Deck Frame {D}.

    a_trans = - R_w_to_d @ acc_w
    """
    return -(np.asarray(R_w_to_d, dtype=np.float64) @ np.asarray(acc_w, dtype=np.float64))


def compute_euler_acceleration(
    alpha_d: np.ndarray,
    pos_d: np.ndarray,
) -> np.ndarray:
    """Compute Euler angular acceleration inertial force in Deck Frame {D}.

    a_euler = - alpha_d x pos_d
    """
    pos_d = np.asarray(pos_d, dtype=np.float64)
    alpha_d = np.asarray(alpha_d, dtype=np.float64)

    px = pos_d[..., 0]
    py = pos_d[..., 1]
    pz = pos_d[..., 2]

    ax, ay, az = alpha_d[0], alpha_d[1], alpha_d[2]

    out_x = az * py - ay * pz
    out_y = ax * pz - az * px
    out_z = ay * px - ax * py

    return np.stack([out_x, out_y, out_z], axis=-1)


def compute_centripetal_acceleration(
    omega_d: np.ndarray,
    pos_d: np.ndarray,
) -> np.ndarray:
    """Compute centripetal apparent acceleration in Deck Frame {D}.

    a_centripetal = - omega_d x (omega_d x pos_d)
                  = (omega_d . omega_d) * pos_d - (omega_d . pos_d) * omega_d
    """
    pos_d = np.asarray(pos_d, dtype=np.float64)
    omega_d = np.asarray(omega_d, dtype=np.float64)

    omega_sq = float(np.sum(omega_d ** 2))
    dot = np.sum(pos_d * omega_d, axis=-1, keepdims=True)
    return omega_sq * pos_d - dot * omega_d


def compute_coriolis_acceleration(
    omega_d: np.ndarray,
    vel_rel_d: np.ndarray,
) -> np.ndarray:
    """Compute Coriolis apparent acceleration in Deck Frame {D}.

    a_coriolis = -2 * (omega_d x vel_rel_d)
    """
    vel_rel_d = np.asarray(vel_rel_d, dtype=np.float64)
    omega_d = np.asarray(omega_d, dtype=np.float64)

    vx = vel_rel_d[..., 0]
    vy = vel_rel_d[..., 1]
    vz = vel_rel_d[..., 2]

    p, q, r = omega_d[0], omega_d[1], omega_d[2]

    out_x = 2.0 * (r * vy - q * vz)
    out_y = 2.0 * (p * vz - r * vx)
    out_z = 2.0 * (q * vx - p * vy)

    return np.stack([out_x, out_y, out_z], axis=-1)


def compute_apparent_acceleration(
    deck_state: DeckState,
    pos_d: np.ndarray,
    vel_rel_d: np.ndarray,
    g: float = GRAVITY,
) -> ApparentAcceleration:
    """Compute the 5-component apparent acceleration field at given deck position(s) and velocity.

    Supports arbitrary leading dimensions for pos_d and vel_rel_d ((3,), (N, 3), (T, N, 3)).
    """
    pos_arr = np.asarray(pos_d, dtype=np.float64)
    vel_arr = np.asarray(vel_rel_d, dtype=np.float64)

    pos_b, vel_b = np.broadcast_arrays(pos_arr, vel_arr)
    target_shape = pos_b.shape

    # 1. Dynamic Tilting Gravity (broadcast to target_shape)
    g_tilt_1d = compute_dynamic_tilting_gravity(deck_state.R_w_to_d, g=g)
    g_tilt = np.broadcast_to(g_tilt_1d, target_shape).copy()

    # 2. Translational apparent acceleration (broadcast to target_shape)
    a_trans_1d = compute_translational_apparent_acceleration(deck_state.R_w_to_d, deck_state.acc_w)
    a_trans = np.broadcast_to(a_trans_1d, target_shape).copy()

    # 3. Euler angular acceleration
    a_euler = compute_euler_acceleration(deck_state.alpha_d, pos_b)

    # 4. Centripetal acceleration
    a_centripetal = compute_centripetal_acceleration(deck_state.omega_d, pos_b)

    # 5. Coriolis acceleration
    a_coriolis = compute_coriolis_acceleration(deck_state.omega_d, vel_b)

    # Total vector sum
    a_apparent_total = g_tilt + a_trans + a_euler + a_centripetal + a_coriolis

    return ApparentAcceleration(
        a_apparent_total=a_apparent_total,
        g_tilt=g_tilt,
        a_trans=a_trans,
        a_euler=a_euler,
        a_centripetal=a_centripetal,
        a_coriolis=a_coriolis,
    )


# Alias for interface compatibility with PROJECT.md and conftest.py
compute_non_inertial_acceleration = compute_apparent_acceleration


def compute_effective_gravity(
    deck_state: DeckState,
    pos_d: np.ndarray,
    vel_rel_d: np.ndarray,
    g: float = GRAVITY,
) -> Union[float, np.ndarray]:
    """Convenience evaluator returning effective normal gravity g_eff = -a_apparent_total[..., 2]."""
    app = compute_apparent_acceleration(deck_state, pos_d, vel_rel_d, g=g)
    return app.g_eff


class NonInertialAccelerationField:
    """Stateful calculator and batch evaluator for flight deck non-inertial acceleration fields."""

    def __init__(self, g: float = GRAVITY) -> None:
        self.g = float(g)

    def evaluate(
        self,
        deck_state: DeckState,
        pos_d: np.ndarray,
        vel_rel_d: np.ndarray,
    ) -> ApparentAcceleration:
        """Compute apparent acceleration at specified position(s) and velocity."""
        return compute_apparent_acceleration(deck_state, pos_d, vel_rel_d, g=self.g)

    def evaluate_effective_gravity(
        self,
        deck_state: DeckState,
        pos_d: np.ndarray,
        vel_rel_d: np.ndarray,
    ) -> Union[float, np.ndarray]:
        """Compute effective normal gravity g_eff = -a_apparent_z."""
        return compute_effective_gravity(deck_state, pos_d, vel_rel_d, g=self.g)

    def apparent_force(
        self,
        mass: float,
        deck_state: DeckState,
        pos_d: np.ndarray,
        vel_rel_d: np.ndarray,
    ) -> np.ndarray:
        """Compute the fictitious inertial body force F_apparent = m * a_apparent."""
        app = self.evaluate(deck_state, pos_d, vel_rel_d)
        return float(mass) * app.a_apparent_total

    def evaluate_wheel_normal_loads_scaling(
        self,
        deck_state: DeckState,
        wheel_positions_d: np.ndarray,
        vehicle_velocity_d: np.ndarray,
    ) -> np.ndarray:
        """Compute local effective normal gravity scaling factors g_eff_i at multiple wheel locations."""
        app = self.evaluate(deck_state, wheel_positions_d, vehicle_velocity_d)
        return np.asarray(app.g_eff)
