"""cts.control

Package providing high-performance trajectory tracking control, disturbance observers,
inertial feedforward compensation, CasADi NMPC, and geometric baselines for carrier
towbarless tractor operations under dynamic deck motion.
"""

from cts.control.feedforward import (
    InertialFeedforward,
    FeedforwardConfig,
    FeedforwardOutput,
)
from cts.control.disturbance_observer import (
    ExtendedStateObserver,
    DisturbanceObserver,
    ESOConfig,
    DisturbanceEstimate,
)
from cts.control.nmpc_casadi import (
    CasADiNMPCController,
    NMPCConfig,
    NMPCResult,
)
from cts.control.ltv_mpc import (
    LTVMPCController,
    LTVMPCConfig,
    LTVMPCResult,
)
from cts.control.baselines import (
    StanleyController,
    PurePursuitController,
    PIDController,
)
from cts.control.tracking_controller import (
    TrajectoryTrackingController,
    MasterControllerConfig,
    MasterControlOutput,
)

__all__ = [
    "InertialFeedforward",
    "FeedforwardConfig",
    "FeedforwardOutput",
    "ExtendedStateObserver",
    "DisturbanceObserver",
    "ESOConfig",
    "DisturbanceEstimate",
    "CasADiNMPCController",
    "NMPCConfig",
    "NMPCResult",
    "LTVMPCController",
    "LTVMPCConfig",
    "LTVMPCResult",
    "StanleyController",
    "PurePursuitController",
    "PIDController",
    "TrajectoryTrackingController",
    "MasterControllerConfig",
    "MasterControlOutput",
]
