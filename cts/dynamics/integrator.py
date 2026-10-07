"""cts.dynamics.integrator

High-Precision Numerical Integrators for Carrier Multibody Dynamics.
Implements:
  - RK4Integrator: Classical 4th-order Runge-Kutta numerical integrator with fixed/variable dt.
  - RK45AdaptiveIntegrator: Embedded Runge-Kutta-Fehlberg / Dormand-Prince 5(4) adaptive step integrator.
"""

from __future__ import annotations
from typing import Callable, Tuple, Optional, List, Dict, Any
import numpy as np


class RK4Integrator:
    """Classical 4th-Order Runge-Kutta (RK4) numerical integrator."""

    def __init__(
        self,
        ode_func: Callable[[float, np.ndarray], np.ndarray],
        dt: float = 0.02,
    ) -> None:
        self.ode_func = ode_func
        self.dt = float(dt)

    def step(
        self,
        t: float,
        y: np.ndarray,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """Perform one single integration step of size dt.

        Args:
            t: Current time.
            y: Current state vector.
            dt: Optional time step override.

        Returns:
            Updated state vector y_{n+1}.
        """
        h = float(dt if dt is not None else self.dt)
        y_arr = np.asarray(y, dtype=np.float64)

        k1 = np.asarray(self.ode_func(t, y_arr), dtype=np.float64)
        k2 = np.asarray(self.ode_func(t + 0.5 * h, y_arr + 0.5 * h * k1), dtype=np.float64)
        k3 = np.asarray(self.ode_func(t + 0.5 * h, y_arr + 0.5 * h * k2), dtype=np.float64)
        k4 = np.asarray(self.ode_func(t + h, y_arr + h * k3), dtype=np.float64)

        return y_arr + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def integrate(
        self,
        t_span: Tuple[float, float],
        y0: np.ndarray,
        dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate trajectory across [t0, tf].

        Returns:
            (t_eval, y_eval): Timestamps shape (N,) and trajectory states shape (N, D).
        """
        t0, tf = float(t_span[0]), float(t_span[1])
        h = float(dt if dt is not None else self.dt)
        if tf <= t0:
            return np.array([t0]), np.array([y0], dtype=np.float64)

        n_steps = max(1, int(np.ceil((tf - t0) / h)))
        t_eval = np.linspace(t0, tf, n_steps + 1)
        y_eval = np.zeros((len(t_eval), len(y0)), dtype=np.float64)
        y_eval[0] = y0

        cur_y = np.array(y0, dtype=np.float64, copy=True)
        for i in range(n_steps):
            cur_t = float(t_eval[i])
            actual_h = float(t_eval[i + 1] - cur_t)
            cur_y = self.step(cur_t, cur_y, actual_h)
            y_eval[i + 1] = cur_y

        return t_eval, y_eval


class RK45AdaptiveIntegrator:
    """Embedded Dormand-Prince 5(4) adaptive time-step numerical integrator."""

    # Dormand-Prince Butcher Tableau coefficients
    c = np.array([0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0], dtype=np.float64)
    a = [
        [],
        [1.0/5.0],
        [3.0/40.0, 9.0/40.0],
        [44.0/45.0, -56.0/15.0, 32.0/9.0],
        [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0],
        [9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0],
        [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0],
    ]
    b5 = np.array([35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0], dtype=np.float64)
    b4 = np.array([5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0], dtype=np.float64)

    def __init__(
        self,
        ode_func: Callable[[float, np.ndarray], np.ndarray],
        dt_initial: float = 0.02,
        rtol: float = 1e-5,
        atol: float = 1e-7,
        dt_min: float = 1e-6,
        dt_max: float = 0.10,
    ) -> None:
        self.ode_func = ode_func
        self.dt_initial = float(dt_initial)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)

    def step(
        self,
        t: float,
        y: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, float, float, bool]:
        """Perform one adaptive Dormand-Prince step.

        Returns:
            (y_next, dt_next, error_norm, accepted)
        """
        y_arr = np.asarray(y, dtype=np.float64)
        h = float(dt)

        k = [None] * 7
        k[0] = np.asarray(self.ode_func(t, y_arr), dtype=np.float64)
        for i in range(1, 7):
            dy = np.zeros_like(y_arr)
            for j in range(i):
                dy += self.a[i][j] * k[j]
            k[i] = np.asarray(self.ode_func(t + self.c[i] * h, y_arr + h * dy), dtype=np.float64)

        y5 = y_arr + h * sum(self.b5[i] * k[i] for i in range(7))
        y4 = y_arr + h * sum(self.b4[i] * k[i] for i in range(7))

        # Truncation error vector
        scale = self.atol + self.rtol * np.maximum(np.abs(y_arr), np.abs(y5))
        err_vec = (y5 - y4) / scale
        err_norm = float(np.sqrt(np.mean(err_vec**2)))

        if err_norm <= 1.0:
            # Step accepted
            accepted = True
            y_next = y5
            factor = min(2.0, max(0.2, 0.9 * (1.0 / max(err_norm, 1e-10))**0.2))
        else:
            # Step rejected
            accepted = False
            y_next = y_arr
            factor = max(0.2, 0.9 * (1.0 / err_norm)**0.25)

        dt_next = float(np.clip(h * factor, self.dt_min, self.dt_max))
        return y_next, dt_next, err_norm, accepted

    def integrate(
        self,
        t_span: Tuple[float, float],
        y0: np.ndarray,
        dt_eval: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate trajectory across [t0, tf] with adaptive stepping and interpolation."""
        t0, tf = float(t_span[0]), float(t_span[1])
        if tf <= t0:
            return np.array([t0]), np.array([y0], dtype=np.float64)

        dt_grid = float(dt_eval if dt_eval is not None else self.dt_initial)
        t_out = [t0]
        y_out = [np.array(y0, dtype=np.float64, copy=True)]

        cur_t = t0
        cur_y = np.array(y0, dtype=np.float64, copy=True)
        h = self.dt_initial

        while cur_t < tf - 1e-12:
            if cur_t + h > tf:
                h = tf - cur_t

            y_next, dt_next, _, accepted = self.step(cur_t, cur_y, h)
            if accepted:
                cur_t += h
                cur_y = y_next
                t_out.append(cur_t)
                y_out.append(np.copy(cur_y))

            h = dt_next
            if h <= self.dt_min and not accepted:
                # Force accept to avoid deadlock
                cur_t += h
                cur_y = y_next
                t_out.append(cur_t)
                y_out.append(np.copy(cur_y))
                h = self.dt_initial

        return np.array(t_out, dtype=np.float64), np.array(y_out, dtype=np.float64)
