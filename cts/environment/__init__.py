"""cts.environment

Standardized Environment Interface for the Carrier Towing System (CTS).

This module exports:
    - DeckState: Dataclass for 6-DOF instantaneous ship motion and kinematic state.
    - ApparentAcceleration: Dataclass for decomposed non-inertial apparent accelerations.
    - NonInertialAccelerationField: Vectorized calculator for deck apparent accelerations.
    - DeckMotionSimulator: 6-DOF dynamic ship motion generator under wave spectra.
    - DeckEnvironment: Flight deck boundary parser, station regions, and obstacles.
    - SeaStateGenerator, WaveSpectrum, PiersonMoskowitzSpectrum, JONSWAPSpectrum.
"""

from __future__ import annotations

from cts.environment.non_inertial_field import (
    DeckState,
    ApparentAcceleration,
    NonInertialAccelerationField,
    compute_apparent_acceleration,
    compute_non_inertial_acceleration,
    compute_dynamic_tilting_gravity,
    compute_translational_apparent_acceleration,
    compute_euler_acceleration,
    compute_centripetal_acceleration,
    compute_coriolis_acceleration,
    compute_effective_gravity,
)

from cts.environment.sea_state import (
    SeaStateLevel,
    WaveSpectrumType,
    DiscretizationMethod,
    WaveSpectrumConfig,
    WaveComponents,
    WaveSpectrum,
    PiersonMoskowitzSpectrum,
    JONSWAPSpectrum,
    SeaStateGenerator,
    SEASTATE_4_CONFIG,
    SEASTATE_5_CONFIG,
)

from cts.environment.deck_motion import (
    FossenDOFParam,
    FossenSeaStateParams,
    FOSSEN_SS4_PARAMS,
    FOSSEN_SS5_PARAMS,
    DiscreteFossenFilterChannel,
    DeckMotionSimulator,
)

from cts.environment.deck_env import (
    DeckRegion,
    DeckObstacle,
    TouchdownPoint,
    DeckEnvironment,
)

__all__ = [
    # Data contracts
    "DeckState",
    "ApparentAcceleration",
    # Calculators & simulators
    "NonInertialAccelerationField",
    "DeckMotionSimulator",
    "DeckEnvironment",
    "DeckRegion",
    "DeckObstacle",
    "TouchdownPoint",
    "SeaStateGenerator",
    # Wave Spectra
    "SeaStateLevel",
    "WaveSpectrumType",
    "DiscretizationMethod",
    "WaveSpectrumConfig",
    "WaveComponents",
    "WaveSpectrum",
    "PiersonMoskowitzSpectrum",
    "JONSWAPSpectrum",
    "SEASTATE_4_CONFIG",
    "SEASTATE_5_CONFIG",
    # Fossen Filter
    "FossenDOFParam",
    "FossenSeaStateParams",
    "FOSSEN_SS4_PARAMS",
    "FOSSEN_SS5_PARAMS",
    "DiscreteFossenFilterChannel",
    # Calculation functions
    "compute_apparent_acceleration",
    "compute_non_inertial_acceleration",
    "compute_dynamic_tilting_gravity",
    "compute_translational_apparent_acceleration",
    "compute_euler_acceleration",
    "compute_centripetal_acceleration",
    "compute_coriolis_acceleration",
    "compute_effective_gravity",
]
