"""cts.environment.deck_env

Loader and spatial validation engine for carrier flight deck environment,
including 18-vertex deck polygon, decomposed island obstacle, 16 parking spots,
8 support spots, 4 catapult launch spots, and angled runway touchdown center.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Sequence
import json
import numpy as np

from cts.common.config import CarrierConfig
from cts.common.geometry import (
    Polygon2D,
    sat_convex_collision,
    decompose_island_to_convex,
    is_polygon_contained,
)


class DeckRegion:
    """Represents a functional deck station (parking, support, takeoff)."""

    def __init__(
        self,
        region_id: str,
        region_type: str,
        polygon: Polygon2D,
        center: Union[Sequence[float], np.ndarray],
        heading_deg: float,
        heading_rad: float,
        preferred_heading_deg: Optional[float] = None,
        source_theta_deck: Optional[float] = None,
        description: str = "",
    ):
        self.id = region_id
        self.region_type = region_type
        self.polygon = polygon
        self.center = np.array(center, dtype=np.float64)
        self.heading_deg = float(heading_deg)
        self.heading_rad = float(heading_rad)
        self.preferred_heading_deg = float(preferred_heading_deg if preferred_heading_deg is not None else heading_deg)
        self.source_theta_deck = source_theta_deck
        self.description = description

    def to_se2_pose(self) -> Tuple[float, float, float]:
        """Return station target pose (x, y, heading_rad)."""
        return (float(self.center[0]), float(self.center[1]), self.heading_rad)


class DeckObstacle:
    """Represents a static superstructure obstacle (e.g. carrier island)."""

    def __init__(
        self,
        obstacle_id: str,
        polygon: Polygon2D,
        center: Union[Sequence[float], np.ndarray],
        description: str = "",
    ):
        self.id = obstacle_id
        self.polygon = polygon
        self.center = np.array(center, dtype=np.float64)
        self.description = description
        # Precompute convex decomposition for fast SAT collision checks
        if not polygon.is_convex and polygon.num_vertices == 6:
            self.convex_pieces = decompose_island_to_convex(polygon)
        else:
            self.convex_pieces = [polygon]

    def collides_with(self, vehicle_poly: Polygon2D) -> bool:
        """Check collision using cached convex pieces with SAT."""
        for piece in self.convex_pieces:
            collides, _, _ = sat_convex_collision(piece, vehicle_poly)
            if collides:
                return True
        return False


class TouchdownPoint:
    """3rd arresting wire touchdown point and angled runway axis."""

    def __init__(
        self,
        deck_pos: Tuple[float, float, float] = (55.0, 50.0, 0.0),
        cg_offset: Tuple[float, float, float] = (-105.0, 4.5, 6.2),
        runway_heading_deg: float = 14.0,
    ):
        self.deck_pos = np.array(deck_pos, dtype=np.float64)
        self.cg_offset = np.array(cg_offset, dtype=np.float64)
        self.runway_heading_deg = float(runway_heading_deg)
        self.runway_heading_rad = float(np.deg2rad(runway_heading_deg))


class DeckEnvironment:
    """Complete carrier flight deck spatial environment model."""

    def __init__(
        self,
        outer_boundary: Polygon2D,
        island: DeckObstacle,
        parking_regions: Dict[str, DeckRegion],
        support_regions: Dict[str, DeckRegion],
        takeoff_regions: Dict[str, DeckRegion],
        touchdown: TouchdownPoint,
        config: Optional[CarrierConfig] = None,
    ):
        self.outer_boundary = outer_boundary
        self.island = island
        self.parking_regions = parking_regions
        self.support_regions = support_regions
        self.takeoff_regions = takeoff_regions
        self.touchdown = touchdown
        self.config = config or CarrierConfig()

        # Merged dictionary of all 28 spots
        self.all_regions: Dict[str, DeckRegion] = {
            **parking_regions,
            **support_regions,
            **takeoff_regions,
        }

    @classmethod
    def from_json(
        cls,
        json_path: Union[Path, str],
        config: Optional[CarrierConfig] = None,
    ) -> "DeckEnvironment":
        """Load deck environment from carrier_deck_environment.json."""
        path = Path(json_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        outer_boundary = Polygon2D(data["outer_boundary"], ensure_ccw=True)
        island_poly = Polygon2D(data["island"]["polygon"], ensure_ccw=True)
        island = DeckObstacle(
            obstacle_id="island",
            polygon=island_poly,
            center=tuple(data["island"]["center"]),
            description="Carrier island superstructure",
        )

        def parse_region(item: dict) -> DeckRegion:
            poly = Polygon2D(item["polygon"], ensure_ccw=True)
            return DeckRegion(
                region_id=item["id"],
                region_type=item["type"],
                polygon=poly,
                center=tuple(item["center"]),
                heading_deg=item["heading_deg"],
                heading_rad=item["heading_rad"],
                preferred_heading_deg=item.get("preferred_heading_deg", item["heading_deg"]),
                source_theta_deck=item.get("source_theta_deck"),
                description=item.get("description", ""),
            )

        parking = {r["id"]: parse_region(r) for r in data["parking_regions"]}
        support = {r["id"]: parse_region(r) for r in data["support_regions"]}
        takeoff = {r["id"]: parse_region(r) for r in data["takeoff_regions"]}
        touchdown = TouchdownPoint()

        return cls(
            outer_boundary=outer_boundary,
            island=island,
            parking_regions=parking,
            support_regions=support,
            takeoff_regions=takeoff,
            touchdown=touchdown,
            config=config,
        )

    @classmethod
    def from_default(cls) -> "DeckEnvironment":
        """Load from default project data path `data/carrier_deck_environment.json`."""
        default_path = Path(__file__).resolve().parents[2] / "data" / "carrier_deck_environment.json"
        if not default_path.exists():
            default_path = Path("data/carrier_deck_environment.json")
        return cls.from_json(default_path)

    def is_point_on_deck(
        self,
        point: Union[np.ndarray, Tuple[float, float], Sequence[float]],
        safety_margin: float = 0.0,
    ) -> bool:
        """Check if point is strictly inside the flight deck outer boundary."""
        if not self.outer_boundary.contains_point(point):
            return False
        if safety_margin > 0.0:
            return self.outer_boundary.distance_to_point(point) >= safety_margin
        return True

    def is_point_in_island(
        self,
        point: Union[np.ndarray, Tuple[float, float], Sequence[float]],
        safety_margin: float = 0.0,
    ) -> bool:
        """Check if point penetrates the island obstacle or its safety buffer."""
        if self.island.polygon.contains_point(point):
            return True
        if safety_margin > 0.0:
            return self.island.polygon.distance_to_point(point) < safety_margin
        return False

    def is_point_navigable(
        self,
        point: Union[np.ndarray, Tuple[float, float], Sequence[float]],
        safety_margin: float = 0.0,
    ) -> bool:
        """Check if point is navigable (on deck and outside island obstacle)."""
        return self.is_point_on_deck(point, safety_margin) and not self.is_point_in_island(point, safety_margin)

    def is_pose_valid(
        self,
        vehicle_polygon: Polygon2D,
        safety_margin: float = 0.0,
    ) -> bool:
        """Check if vehicle footprint is completely inside deck and collision-free with island."""
        if not is_polygon_contained(vehicle_polygon, self.outer_boundary, margin=safety_margin):
            return False
        if self.island.collides_with(vehicle_polygon):
            return False
        return True

    def distance_to_boundary(self, point: Union[np.ndarray, Tuple[float, float], Sequence[float]]) -> float:
        """Compute Euclidean distance to outer deck boundary."""
        return self.outer_boundary.distance_to_point(point)

    def distance_to_island(self, point: Union[np.ndarray, Tuple[float, float], Sequence[float]]) -> float:
        """Compute Euclidean distance to island obstacle boundary."""
        return self.island.polygon.distance_to_point(point)

    def get_region_by_id(self, region_id: str) -> Optional[DeckRegion]:
        """Look up station by ID (e.g. 'P1', 'S5', 'L2')."""
        return self.all_regions.get(region_id)

    def get_parking_spots(self) -> List[DeckRegion]:
        """Return list of all 16 parking spots."""
        return list(self.parking_regions.values())

    def get_support_spots(self) -> List[DeckRegion]:
        """Return list of all 8 support spots."""
        return list(self.support_regions.values())

    def get_catapult_spots(self) -> List[DeckRegion]:
        """Return list of all 4 catapult spots."""
        return list(self.takeoff_regions.values())
