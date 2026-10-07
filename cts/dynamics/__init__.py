"""cts.dynamics

Milestone M2: High-Fidelity Multibody Carrier Dynamics, Kinematics,
Tire Models, Vertical Load Transfer, and Numerical Integrators.
"""

from __future__ import annotations

from cts.dynamics.kinematics import (
    VehiclePose,
    AckermannTurntableKinematics,
)
from cts.dynamics.load_transfer import (
    WheelLoads,
    DynamicLoadTransfer,
    compute_critical_rollover_acceleration,
)
from cts.dynamics.tire_models import (
    TireForces,
    DynamicTireState,
    TireModelInterface,
    LinearTireModel,
    DugoffTireModel,
    PacejkaTireModel,
    compute_slip_quantities,
    create_tire_model,
)
from cts.dynamics.integrator import (
    RK4Integrator,
    RK45AdaptiveIntegrator,
)
from cts.dynamics.multibody_dynamics import (
    VehicleState,
    ControlAction,
    TractorAircraftPlant,
)

__all__ = [
    # Kinematics
    "VehiclePose",
    "AckermannTurntableKinematics",
    # Load Transfer
    "WheelLoads",
    "DynamicLoadTransfer",
    "compute_critical_rollover_acceleration",
    # Tire Models
    "TireForces",
    "DynamicTireState",
    "TireModelInterface",
    "LinearTireModel",
    "DugoffTireModel",
    "PacejkaTireModel",
    "compute_slip_quantities",
    "create_tire_model",
    # Integrators
    "RK4Integrator",
    "RK45AdaptiveIntegrator",
    # Multibody Dynamics
    "VehicleState",
    "ControlAction",
    "TractorAircraftPlant",
]
