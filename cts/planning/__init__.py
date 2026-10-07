"""cts.planning

Spatiotemporal Safe Corridor Kinodynamic Planner suite (Milestone M3, Features F10-F13):
- deck_map: Carrier deck spatial representation, runway centerline, dual-body footprint & EDT.
- kinodynamic_limits: Time-varying dynamic acceleration & rollover limits under deck motion.
- hybrid_a_star: 5D spatiotemporal kinodynamic search with 4-tier gating.
- safe_corridor: Convex polyhedral safe flight corridor (SFC) generation.
- minco_optimizer: Minimum-jerk C^2 polynomial spline trajectory optimizer.
- corridor_planner: Master SpatiotemporalCorridorPlanner matching PROJECT.md interface contract.
"""

from cts.planning.deck_map import (
    RunwayCenterline,
    DualBodyFootprint,
    DualBodyModel,
    DeckGridMap,
    CarrierDeckMap,
    default_runway_centerline,
)
from cts.planning.kinodynamic_limits import (
    KinodynamicBound,
    KinodynamicLimits,
)
from cts.planning.hybrid_a_star import (
    HybridAStarConfig,
    HybridAStarNode,
    HybridAStarResult,
    ObstacleDistanceHeuristic,
    DubinsReedsSheppHeuristic,
    SpatiotemporalHybridAStar,
)
from cts.planning.safe_corridor import (
    Polytope,
    SafeFlightCorridorConfig,
    SafeFlightCorridorGenerator,
)
from cts.planning.minco_optimizer import (
    PlannedTrajectory,
    MincoTrajectoryConfig,
    PolynomialPiece,
    MincoSpline,
    MincoTrajectoryOptimizer,
    compute_minco_q_matrix,
)
from cts.planning.corridor_planner import (
    SpatiotemporalCorridorPlanner,
)

__all__ = [
    # F10: Deck Map & Kinodynamic Boundaries
    "RunwayCenterline",
    "DualBodyFootprint",
    "DualBodyModel",
    "DeckGridMap",
    "CarrierDeckMap",
    "default_runway_centerline",
    "KinodynamicBound",
    "KinodynamicLimits",
    # F11: 5D Spatiotemporal Hybrid A*
    "HybridAStarConfig",
    "HybridAStarNode",
    "HybridAStarResult",
    "ObstacleDistanceHeuristic",
    "DubinsReedsSheppHeuristic",
    "SpatiotemporalHybridAStar",
    # F12: Safe Flight Corridors
    "Polytope",
    "SafeFlightCorridorConfig",
    "SafeFlightCorridorGenerator",
    # F13: MINCO Trajectory Optimizer
    "PlannedTrajectory",
    "MincoTrajectoryConfig",
    "PolynomialPiece",
    "MincoSpline",
    "MincoTrajectoryOptimizer",
    "compute_minco_q_matrix",
    # Master Planner Integration
    "SpatiotemporalCorridorPlanner",
]
