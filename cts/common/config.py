"""cts.common.config

Typed configuration models for Carrier Flight Deck, Tractor, Aircraft,
Tire-Deck friction, and 6-DOF Ocean wave simulation.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class CarrierConfig(BaseModel):
    """Aircraft carrier geometric, map, and reference frame parameters."""
    model_config = ConfigDict(frozen=True)

    map_length: float = Field(default=350.0, description="Map bounding box length along X (m)")
    map_width: float = Field(default=120.0, description="Map bounding box width along Y (m)")
    grid_resolution: float = Field(default=1.0, description="Planning grid cell resolution (m)")
    deck_length: float = Field(default=304.0, description="Flight deck longitudinal span (m)")
    deck_width: float = Field(default=77.0, description="Flight deck lateral span (m)")
    angled_runway_angle_deg: float = Field(default=14.0, description="Angled runway axis heading (deg)")
    angled_runway_angle_rad: float = Field(default=0.24434609527920614, description="Angled runway axis (rad)")
    ship_cg_deck_frame: Tuple[float, float, float] = Field(
        default=(160.0, 45.5, -6.2), description="Ship CG position in deck coordinates (m)"
    )
    touchdown_cg_offset: Tuple[float, float, float] = Field(
        default=(-105.0, 4.5, 6.2), description="3rd wire touchdown point offset relative to CG (m)"
    )
    touchdown_deck_frame: Tuple[float, float, float] = Field(
        default=(55.0, 50.0, 0.0), description="3rd wire touchdown point on deck surface (m)"
    )
    safety_margin: float = Field(default=1.5, description="Default safety margin to deck edge/obstacles (m)")


class TractorConfig(BaseModel):
    """Towbarless tractor geometric, kinematic, and dynamic limits."""
    model_config = ConfigDict(frozen=True)

    wheelbase: float = Field(default=1.8, description="Tractor wheelbase l_t (m)")
    width: float = Field(default=1.4, description="Tractor overall chassis width W_t (m)")
    track: float = Field(default=1.2, description="Tractor track width h_t (m)")
    v_max: float = Field(default=2.0, description="Maximum operational forward velocity (m/s)")
    a_max: float = Field(default=1.0, description="Maximum acceleration / deceleration (m/s^2)")
    delta_max_deg: float = Field(default=35.0, description="Maximum front wheel steering angle (deg)")
    delta_max_rad: float = Field(default=0.6108652381980153, description="Maximum front wheel steering angle (rad)")
    omega_steer_max_deg: float = Field(default=30.0, description="Maximum front wheel steering rate (deg/s)")
    omega_steer_max_rad: float = Field(default=0.5235987755982988, description="Maximum steering rate (rad/s)")
    beta_max_deg: float = Field(default=70.0, description="Maximum hitch articulation angle beta (deg)")
    beta_max_rad: float = Field(default=1.2217304763960306, description="Maximum articulation angle beta (rad)")
    mass: float = Field(default=8500.0, description="Tractor curb mass m_t (kg)")
    izz: float = Field(default=9800.0, description="Tractor yaw moment of inertia I_zz,t (kg*m^2)")
    cg_to_front: float = Field(default=0.95, description="Distance from tractor CG to front axle a_t (m)")
    cg_to_rear: float = Field(default=0.85, description="Distance from tractor CG to rear axle b_t (m)")
    cg_height: float = Field(default=0.45, description="Tractor CG height above deck h_cg,t (m)")
    hitch_offset: float = Field(default=0.10, description="Cradle clamp hitch offset behind rear axle d_c (m)")
    tractive_force_max: float = Field(default=35000.0, description="Maximum tractor total tractive force (N)")
    wheel_drive_torque_max: float = Field(default=4200.0, description="Maximum single wheel drive torque (N*m)")


class AircraftConfig(BaseModel):
    """Carrier fighter aircraft geometric, inertial, and landing gear parameters (J-15 class)."""
    model_config = ConfigDict(frozen=True)

    length: float = Field(default=18.3, description="Overall aircraft fuselage length L_a (m)")
    wheelbase: float = Field(default=8.0, description="Wheelbase from nose gear to main gear l_a (m)")
    folded_width: float = Field(default=8.6, description="Wings folded overall width W_a (m)")
    main_track: float = Field(default=3.1, description="Main landing gear track width W_r (m)")
    rear_overhang: float = Field(default=6.4, description="Main gear axle to tail overhang l_r (m)")
    nose_overhang: float = Field(default=3.9, description="Nose tip to nose landing gear overhang l_f (m)")
    mass: float = Field(default=20500.0, description="Towing gross mass m_a (kg)")
    izz: float = Field(default=285000.0, description="Aircraft yaw moment of inertia I_zz,a (kg*m^2)")
    cg_to_nose: float = Field(default=6.8, description="Longitudinal distance from aircraft CG to nose gear a_a (m)")
    cg_to_main: float = Field(default=1.2, description="Longitudinal distance from aircraft CG to main gear b_a (m)")
    cg_height: float = Field(default=1.85, description="Aircraft CG height above deck h_cg,a (m)")
    static_nose_fraction: float = Field(default=0.15, description="Static vertical load fraction on nose gear (15%)")
    static_main_fraction: float = Field(default=0.85, description="Static vertical load fraction on main gear (85%)")
    static_nose_load: float = Field(default=30155.44875, description="Static vertical load on nose gear F_z,n0 (N)")
    static_main_load_single: float = Field(default=85440.438125, description="Static load on single main gear tire F_z,m0 (N)")
    main_tire_diameter: float = Field(default=1.05, description="Main gear tire outer diameter (m)")
    nose_tire_diameter: float = Field(default=0.68, description="Nose gear tire outer diameter (m)")


class TireConfig(BaseModel):
    """Tire friction coefficients, cornering stiffness, and relaxation parameters."""
    model_config = ConfigDict(frozen=True)

    mu_dry: float = Field(default=0.80, description="Dry flight deck friction coefficient")
    mu_wet: float = Field(default=0.50, description="Wet flight deck friction coefficient")
    mu_spray: float = Field(default=0.30, description="Saline sea spray deck friction coefficient (SS5)")
    cornering_stiffness_tractor: float = Field(default=55000.0, description="Per-tire cornering stiffness C_alpha,t (N/rad)")
    cornering_stiffness_aircraft_nose: float = Field(default=42000.0, description="Per-tire nose cornering stiffness C_alpha,an (N/rad)")
    cornering_stiffness_aircraft_main: float = Field(default=155000.0, description="Per-tire main gear cornering stiffness C_alpha,am (N/rad)")
    longitudinal_stiffness_tractor: float = Field(default=75000.0, description="Tractor longitudinal slip stiffness C_x,t (N/slip)")
    longitudinal_stiffness_aircraft_main: float = Field(default=210000.0, description="Main gear longitudinal slip stiffness C_x,am (N/slip)")
    vertical_stiffness_tractor: float = Field(default=6.5e5, description="Tractor tire vertical stiffness K_z,t (N/m)")
    vertical_stiffness_aircraft: float = Field(default=1.1e6, description="Aircraft tire vertical stiffness K_z,a (N/m)")
    relaxation_length_tractor: float = Field(default=0.15, description="Tractor tire relaxation length sigma_t (m)")
    relaxation_length_aircraft: float = Field(default=0.25, description="Aircraft tire relaxation length sigma_a (m)")
    dugoff_lambda_thresh: float = Field(default=1.0, description="Dugoff critical adhesion-sliding threshold")
    pacejka_B: float = Field(default=10.0, description="Pacejka stiffness factor B")
    pacejka_C: float = Field(default=1.30, description="Pacejka shape factor C")
    pacejka_D: float = Field(default=1.00, description="Pacejka peak value factor D")
    pacejka_E: float = Field(default=-0.10, description="Pacejka curvature factor E")


class OceanConfig(BaseModel):
    """6-DOF Ocean wave spectrum, Fossen transfer functions, and sea state parameters."""
    model_config = ConfigDict(frozen=True)

    sea_state: int = Field(default=4, description="Sea state integer code (4 or 5)")
    dt: float = Field(default=0.02, description="Simulation discrete time step (s)")
    sim_time: float = Field(default=150.0, description="Default simulation duration (s)")
    g: float = Field(default=9.80665, description="Gravitational acceleration (m/s^2)")
    hs: float = Field(default=2.5, description="Significant wave height H_1/3 (m)")
    wind_speed_knots: float = Field(default=20.0, description="Mean wind speed (knots)")
    omega_p: float = Field(default=0.71, description="Wave spectrum peak frequency (rad/s)")
    tp: float = Field(default=8.85, description="Wave peak period T_p (s)")
    heave_sigma: float = Field(default=0.95, description="Heave motion standard deviation (m)")
    heave_omega: float = Field(default=0.60, description="Heave peak circular frequency (rad/s)")
    heave_zeta: float = Field(default=0.25, description="Heave relative damping ratio")
    heave_gain: float = Field(default=0.73587, description="Heave wave filter gain K_z (m*s^-0.5)")
    pitch_sigma_deg: float = Field(default=1.25, description="Pitch motion standard deviation (deg)")
    pitch_sigma_rad: float = Field(default=0.02181661564992912, description="Pitch motion standard deviation (rad)")
    pitch_omega: float = Field(default=0.55, description="Pitch peak circular frequency (rad/s)")
    pitch_zeta: float = Field(default=0.20, description="Pitch relative damping ratio")
    pitch_gain: float = Field(default=0.01447, description="Pitch wave filter gain K_theta (rad*s^-0.5)")
    roll_sigma_deg: float = Field(default=3.00, description="Roll motion standard deviation (deg)")
    roll_sigma_rad: float = Field(default=0.05235987755982989, description="Roll motion standard deviation (rad)")
    roll_omega: float = Field(default=0.40, description="Roll peak circular frequency (rad/s)")
    roll_zeta: float = Field(default=0.08, description="Roll relative damping ratio")
    roll_gain: float = Field(default=0.01874, description="Roll wave filter gain K_phi (rad*s^-0.5)")
    diff_tau: float = Field(default=0.05, description="Velocity differentiator low-pass filter time constant (s)")
    z_bound: float = Field(default=3.0, description="Safe touchdown vertical position bound +/- (m)")
    dz_bound: float = Field(default=1.5, description="Safe touchdown vertical velocity bound +/- (m/s)")


class CTSConfig(BaseModel):
    """Master configuration encapsulating all domain models."""
    model_config = ConfigDict(frozen=True)

    carrier: CarrierConfig = Field(default_factory=CarrierConfig)
    tractor: TractorConfig = Field(default_factory=TractorConfig)
    aircraft: AircraftConfig = Field(default_factory=AircraftConfig)
    tire: TireConfig = Field(default_factory=TireConfig)
    ocean: OceanConfig = Field(default_factory=OceanConfig)

    @classmethod
    def default_ss4(cls) -> "CTSConfig":
        """Factory for standard Sea State 4 configuration."""
        return cls(ocean=OceanConfig(sea_state=4))

    @classmethod
    def default_ss5(cls) -> "CTSConfig":
        """Factory for Sea State 5 configuration with calibrated wave scaling."""
        ocean_ss5 = OceanConfig(
            sea_state=5,
            hs=4.0,
            wind_speed_knots=28.0,
            omega_p=0.58,
            tp=10.83,
            heave_sigma=1.52,
            heave_omega=0.52,
            heave_zeta=0.22,
            heave_gain=1.0289,
            pitch_sigma_deg=2.00,
            pitch_sigma_rad=0.03490658503988659,
            pitch_omega=0.48,
            pitch_zeta=0.18,
            pitch_gain=0.02052,
            roll_sigma_deg=4.80,
            roll_sigma_rad=0.08377580409572782,
            roll_omega=0.36,
            roll_zeta=0.07,
            roll_gain=0.02661,
            z_bound=4.5,
            dz_bound=2.2,
        )
        return cls(ocean=ocean_ss5)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    def to_json(self, path: Optional[Union[Path, str]] = None, indent: int = 2) -> str:
        """Serialize configuration to JSON string or write to file."""
        json_str = self.model_dump_json(indent=indent)
        if path:
            Path(path).write_text(json_str, encoding="utf-8")
        return json_str

    @classmethod
    def from_json(cls, json_path: Union[Path, str]) -> "CTSConfig":
        """Load configuration from JSON file."""
        content = Path(json_path).read_text(encoding="utf-8")
        return cls.model_validate_json(content)
