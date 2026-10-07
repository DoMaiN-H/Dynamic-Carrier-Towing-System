"""cts.visualization.aircraft_shape

Loads the J-15-class aircraft planform silhouette from data/飞机模型.svg and
returns it as a polygon in the aircraft body frame (origin at the nose-gear
hitch point, +x along aircraft heading). Used by the deck figures to render
the real airframe outline instead of coarse OBB approximations.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Optional

import numpy as np

_SVG_PATH = Path(__file__).resolve().parents[2] / "data" / "飞机模型.svg"


@functools.lru_cache(maxsize=4)
def load_aircraft_outline(
    fuselage_length: float = 18.3,
    nose_gear_offset: float = 3.9,
    svg_path: Optional[str] = None,
) -> np.ndarray:
    """Parse the aircraft silhouette path and return a (N, 2) outline.

    The SVG path is a closed polyline in pixel units (+x forward, +y down).
    It is scaled so that its longitudinal extent equals ``fuselage_length``
    and anchored so that the nose gear sits at the origin: the gear is
    ``nose_gear_offset`` metres aft of the radome tip (matches AircraftConfig
    nose_overhang = 3.9 m over length = 18.3 m).

    Args:
        fuselage_length: Real aircraft length the silhouette is scaled to (m).
        nose_gear_offset: Distance from nose tip to the towbarless clamp (m).
        svg_path: Optional override SVG file path.

    Returns:
        (N, 2) array of outline vertices in the aircraft body frame:
        origin = nose gear / hitch, +x = aircraft heading, +y = port side.
    """
    path = Path(svg_path) if svg_path else _SVG_PATH
    text = path.read_text(encoding="utf-8")
    m = re.search(r'd="([^"]+)"', text)
    if m is None:
        raise ValueError(f"No path 'd' attribute found in {path}")
    nums = [float(v) for v in re.findall(r"-?\d+\.?\d*", m.group(1))]
    pts = np.asarray(nums, dtype=np.float64).reshape(-1, 2)

    x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
    extent = x_max - x_min
    if extent <= 0:
        raise ValueError("Degenerate silhouette x-extent")
    scale = fuselage_length / extent

    # Nose gear location inside the silhouette (fraction of x-extent aft of nose)
    x_ng = x_min + (nose_gear_offset / fuselage_length) * extent
    # Fuselage centreline: median y of the nose-region vertices
    nose_pts = pts[pts[:, 0] < x_min + 0.15 * extent]
    y_c = float(np.median(nose_pts[:, 1])) if len(nose_pts) else 0.5 * float(pts[:, 1].min() + pts[:, 1].max())

    # Aircraft frame: +x forward toward the nose, +y toward port.
    # In the SVG the nose tip is the leftmost vertex and +y points down,
    # so both axes are mirrored into the aircraft convention.
    return np.stack([-(pts[:, 0] - x_ng) * scale, -(pts[:, 1] - y_c) * scale], axis=1)


def silhouette_world(
    x_t: float,
    y_t: float,
    psi_t: float,
    theta_h: float,
    hitch_offset: float = 0.10,
    **kwargs,
) -> np.ndarray:
    """Place the scaled aircraft silhouette into deck-world coordinates.

    Args:
        x_t, y_t: Tractor rear-axle position (m).
        psi_t: Tractor heading (rad).
        theta_h: Hitch articulation angle (rad); aircraft heading psi_a = psi_t + theta_h.
        hitch_offset: Hitch offset behind tractor rear axle d_c (m).

    Returns:
        (N, 2) world-frame outline vertices for the aircraft silhouette.
    """
    pts = load_aircraft_outline(**kwargs)
    psi_a = psi_t + theta_h
    hx = x_t - hitch_offset * np.cos(psi_t)
    hy = y_t - hitch_offset * np.sin(psi_t)
    c, s = np.cos(psi_a), np.sin(psi_a)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return pts @ R.T + np.array([hx, hy], dtype=np.float64)


def silhouette_at_aircraft_pose(
    x_a: float,
    y_a: float,
    psi_a: float,
    cg_to_nose_gear: float = 6.80,
    **kwargs,
) -> np.ndarray:
    """Place the aircraft silhouette at an aircraft pose (CG + heading).

    The station pose convention is aircraft CG at the region centre with the
    nose along the region heading; the silhouette is anchored at the nose
    gear, which sits ``cg_to_nose_gear`` metres ahead of the CG.
    """
    pts = load_aircraft_outline(**kwargs)
    hx = x_a + cg_to_nose_gear * np.cos(psi_a)
    hy = y_a + cg_to_nose_gear * np.sin(psi_a)
    c, s = np.cos(psi_a), np.sin(psi_a)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return pts @ R.T + np.array([hx, hy], dtype=np.float64)
