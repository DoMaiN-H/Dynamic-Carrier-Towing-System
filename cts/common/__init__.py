"""cts.common

Common configuration models, 2D computational geometry, and mathematical utilities.
"""

from cts.common.config import (
    CarrierConfig,
    TractorConfig,
    AircraftConfig,
    TireConfig,
    OceanConfig,
    CTSConfig,
)
from cts.common.geometry import (
    Polygon2D,
    sat_convex_collision,
    decompose_island_to_convex,
    check_polygon_collision,
    is_polygon_contained,
)
from cts.common.math_utils import (
    wrap_to_pi,
    wrap_to_2pi,
    se2_matrix,
    se2_inv,
    se2_apply,
    rpy_to_R_d_to_w,
    rpy_to_R_w_to_d,
    R_d_to_w_to_rpy,
    euler_rates_to_body_rates,
    body_rates_to_euler_rates,
    skew_symmetric_3x3,
    LowPassDifferentiator,
)

__all__ = [
    # Config
    "CarrierConfig",
    "TractorConfig",
    "AircraftConfig",
    "TireConfig",
    "OceanConfig",
    "CTSConfig",
    # Geometry
    "Polygon2D",
    "sat_convex_collision",
    "decompose_island_to_convex",
    "check_polygon_collision",
    "is_polygon_contained",
    # Math Utils
    "wrap_to_pi",
    "wrap_to_2pi",
    "se2_matrix",
    "se2_inv",
    "se2_apply",
    "rpy_to_R_d_to_w",
    "rpy_to_R_w_to_d",
    "R_d_to_w_to_rpy",
    "euler_rates_to_body_rates",
    "body_rates_to_euler_rates",
    "skew_symmetric_3x3",
    "LowPassDifferentiator",
]
