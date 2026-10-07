"""cts.visualization.publication_plots

Publication-Ready Vector Visualization Suite (Feature F21).
Generates IEEE / AIAA formatted academic figures:
- Birds-Eye Flight Deck Trajectory & Swept Footprints
- Multi-Panel Closed-Loop Tracking Error & Actuation Profiles
- State Phase Portraits (theta_h, dtheta_h) & (beta, dbeta)
- Normalized Tire Adhesion Circle Saturation Envelopes
- Factorial Ablation Comparison Bar Charts
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Sequence, Dict, Any, Union
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-interactive headless backend
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle as MplCircle
import matplotlib.gridspec as gridspec

from cts.common.math_utils import wrap_to_pi
from cts.dynamics.multibody_dynamics import VehicleState, ControlAction, WheelLoads
from cts.planning.minco_optimizer import PlannedTrajectory
from cts.planning.deck_map import DualBodyModel
from cts.planning.safe_corridor import Polytope
from cts.visualization.aircraft_shape import silhouette_world, silhouette_at_aircraft_pose
from cts.environment.deck_env import DeckEnvironment
from cts.benchmarks.metrics import BenchmarkMetrics
from cts.benchmarks.ablation_runner import AblationResult


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Save figure with retry against transient Windows file locks.

    External processes (image viewers, antivirus, file indexing) can hold a
    short-lived lock on freshly written PNGs, which surfaces as a spurious
    OSError(EINVAL) from PIL. Retry with a brief backoff instead of aborting
    the whole benchmark run.
    """
    import time
    last_err: Optional[Exception] = None
    for _ in range(6):
        try:
            fig.savefig(str(path), dpi=300, bbox_inches="tight")
            return
        except OSError as exc:
            last_err = exc
            time.sleep(0.4)
    raise last_err  # type: ignore[misc]


def set_academic_style() -> None:
    """Configure matplotlib with clean publication-ready styling."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "grid.color": "#D3D3D3",
        "grid.linestyle": ":",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.8,
    })


def plot_carrier_deck_trajectory(
    reference_trajectory: PlannedTrajectory,
    states: Sequence[VehicleState],
    deck_env: Optional[DeckEnvironment] = None,
    goal_aircraft_pose: Optional[tuple] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot birds-eye view of flight deck with planned reference and executed paths."""
    set_academic_style()
    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=300)

    # 1. Draw Carrier Deck Geometry
    if deck_env is None:
        deck_env = DeckEnvironment.from_default()

    # Deck Outer Perimeter
    deck_poly = MplPolygon(deck_env.outer_boundary.vertices, closed=True,
                           facecolor="#E8ECEF", edgecolor="#2C3E50", linewidth=1.8, label="Flight Deck Boundary")
    ax.add_patch(deck_poly)

    # Functional Deck Stations: Parking P1-P16, Support S1-S8, Catapult L1-L4
    region_groups = (
        (deck_env.get_parking_spots(), "#5D6D7E", "#D6E4F0", "Parking Spot"),
        (deck_env.get_support_spots(), "#B9770E", "#FDEBD0", "Support Station"),
        (deck_env.get_catapult_spots(), "#1E8449", "#D5F5E3", "Catapult Position"),
    )
    for regions, edge_c, face_c, group_label in region_groups:
        for i, region in enumerate(regions):
            ax.add_patch(MplPolygon(
                region.polygon.vertices, closed=True,
                facecolor=face_c, edgecolor=edge_c, linewidth=0.7,
                linestyle="--", alpha=0.75, zorder=1.3,
                label=group_label if i == 0 else None,
            ))
            ax.annotate(region.id, xy=(float(region.center[0]), float(region.center[1])),
                        fontsize=4.5, color=edge_c, ha="center", va="center", zorder=2)

    # Island Obstacle
    island_poly = MplPolygon(deck_env.island.polygon.vertices, closed=True,
                            facecolor="#7F8C8D", edgecolor="#2C3E50", linewidth=1.5, zorder=1.6, label="Island Superstructure")
    ax.add_patch(island_poly)

    # 2. Draw Trajectories
    ref_x = reference_trajectory.x
    ref_y = reference_trajectory.y
    ax.plot(ref_x, ref_y, "r--", linewidth=1.8, label="Planned Reference Path", zorder=3)

    act_x = [st.x_t for st in states]
    act_y = [st.y_t for st in states]
    ax.plot(act_x, act_y, "b-", linewidth=1.5, label="Actual Tractor Path", zorder=4)

    # Aircraft Main Gear Path (through hitch point, incl. hitch offset)
    d_c = 0.10
    l_a = 8.0 # wheelbase
    air_x = []
    air_y = []
    for st in states:
        psi_a = wrap_to_pi(st.psi_t + st.theta_h)
        hx = st.x_t - d_c * np.cos(st.psi_t)
        hy = st.y_t - d_c * np.sin(st.psi_t)
        air_x.append(hx - l_a * np.cos(psi_a))
        air_y.append(hy - l_a * np.sin(psi_a))
    ax.plot(air_x, air_y, color="#27AE60", linestyle="-.", linewidth=1.2, label="Aircraft Main Gear", zorder=4)

    # 3. Draw the complete towbarless tractor-aircraft system footprint
    #    (rectangle tractor hitched to the real airframe silhouette),
    #    sparsely sampled by arc length so silhouettes stay readable
    body_model = DualBodyModel()
    act_x_arr = np.asarray(act_x, dtype=np.float64)
    act_y_arr = np.asarray(act_y, dtype=np.float64)
    for k, idx in enumerate(_arc_length_sample(act_x_arr, act_y_arr, max_poses=6)):
        st = states[int(idx)]
        _draw_towbarless_system(ax, body_model, st.x_t, st.y_t, st.psi_t, st.theta_h,
                                with_label=(k == 0))

    # Target aircraft pose outline (station missions: CG on station, nose on heading)
    if goal_aircraft_pose is not None:
        gx, gy, gpsi = goal_aircraft_pose
        goal_outline = silhouette_at_aircraft_pose(gx, gy, gpsi)
        ax.add_patch(MplPolygon(
            goal_outline, closed=True, facecolor="none",
            edgecolor="#1E8449", linewidth=1.4, linestyle="--",
            zorder=5.0, label="Goal Aircraft Pose",
        ))

    ax.set_xlabel("Flight Deck Longitudinal Coordinate X (m)")
    ax.set_ylabel("Flight Deck Lateral Coordinate Y (m)")
    ax.set_title("Carrier Flight Deck Towing Maneuver: Reference vs. Tracked Trajectory")

    # Adaptive view limits: full deck bounding box + trajectory extents + padding
    xs_all = np.concatenate([
        deck_env.outer_boundary.vertices[:, 0],
        np.asarray(ref_x, dtype=np.float64),
        np.asarray(act_x, dtype=np.float64),
        np.asarray(air_x, dtype=np.float64),
    ])
    ys_all = np.concatenate([
        deck_env.outer_boundary.vertices[:, 1],
        np.asarray(ref_y, dtype=np.float64),
        np.asarray(act_y, dtype=np.float64),
        np.asarray(air_y, dtype=np.float64),
    ])
    pad = 6.0
    ax.set_xlim(float(xs_all.min()) - pad, float(xs_all.max()) + pad)
    ax.set_ylim(float(ys_all.min()) - pad, float(ys_all.max()) + pad)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        _save_figure(fig, p)

    return fig


def _arc_length_sample(xs: np.ndarray, ys: np.ndarray, max_poses: int) -> np.ndarray:
    """Return path indices evenly spaced by arc length, incl. both endpoints."""
    n = len(xs)
    if n <= 1:
        return np.arange(n)
    seg = np.hypot(np.diff(xs), np.diff(ys))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 1e-9:
        return np.array([0, n - 1])
    targets = np.linspace(0.0, total, max_poses)
    idx = np.unique(np.searchsorted(s, targets).clip(0, n - 1))
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def _draw_towbarless_system(
    ax: plt.Axes,
    body_model: DualBodyModel,
    x_t: float,
    y_t: float,
    psi_t: float,
    theta_h: float,
    with_label: bool = False,
) -> None:
    """Draw the full towbarless towing system at one pose.

    Aircraft: real J-15 planform silhouette (from data SVG), anchored at the
    nose-gear hitch point. Tractor: oriented chassis rectangle connected to
    the nose gear by a visible hitch link.
    """
    d_c = 0.10
    fp = body_model.get_footprint(x_t, y_t, psi_t, theta_h)
    airframe = silhouette_world(x_t, y_t, psi_t, theta_h, hitch_offset=d_c)

    ax.add_patch(MplPolygon(
        airframe, closed=True,
        facecolor="#E67E22", edgecolor="#6E2C00", linewidth=0.9,
        alpha=0.50, zorder=5.1,
        label="Towbarless Towing System" if with_label else None,
    ))
    ax.add_patch(MplPolygon(
        fp.tractor.vertices, closed=True,
        facecolor="#3498DB", edgecolor="#1B4F72", linewidth=0.9,
        alpha=0.65, zorder=5.2,
    ))
    # Hitch link: tractor rear axle -> nose gear clamp point
    hx = x_t - d_c * np.cos(psi_t)
    hy = y_t - d_c * np.sin(psi_t)
    ax.plot([x_t, hx], [y_t, hy], color="#1B4F72", linewidth=1.4, zorder=5.3)
    ax.scatter([hx], [hy], s=10, color="#1B4F72", zorder=5.4)


def _polytope_vertices(poly: Polytope) -> Optional[np.ndarray]:
    """Enumerate vertices of a bounded 2D H-polytope {p : A p <= b}.

    Returns vertices ordered counter-clockwise, or None if the polytope is
    empty or degenerate.
    """
    A, b = np.asarray(poly.A, dtype=np.float64), np.asarray(poly.b, dtype=np.float64)
    verts: List[np.ndarray] = []
    m = len(b)
    for i in range(m):
        for j in range(i + 1, m):
            A2 = A[[i, j]]
            if abs(np.linalg.det(A2)) < 1e-9:
                continue
            p = np.linalg.solve(A2, b[[i, j]])
            if np.all(A @ p <= b + 1e-6):
                verts.append(p)
    if len(verts) < 3:
        return None
    pts = np.array(verts, dtype=np.float64)
    centroid = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0]))
    return pts[order]


def plot_planning_overview(
    reference_trajectory: PlannedTrajectory,
    deck_env: Optional[DeckEnvironment] = None,
    goal_aircraft_pose: Optional[tuple] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Planning map: deck geometry, safe flight corridors, planned path, and
    the complete towbarless tractor-aircraft footprint along the plan."""
    set_academic_style()
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=300)

    if deck_env is None:
        deck_env = DeckEnvironment.from_default()

    # Deck boundary and island
    ax.add_patch(MplPolygon(deck_env.outer_boundary.vertices, closed=True,
                            facecolor="#E8ECEF", edgecolor="#2C3E50",
                            linewidth=1.8, label="Flight Deck Boundary"))
    ax.add_patch(MplPolygon(deck_env.island.polygon.vertices, closed=True,
                            facecolor="#7F8C8D", edgecolor="#2C3E50",
                            linewidth=1.5, zorder=1.6, label="Island Superstructure"))

    # Functional stations (subdued)
    region_groups = (
        (deck_env.get_parking_spots(), "#5D6D7E", "#D6E4F0", "Parking Spot"),
        (deck_env.get_support_spots(), "#B9770E", "#FDEBD0", "Support Station"),
        (deck_env.get_catapult_spots(), "#1E8449", "#D5F5E3", "Catapult Position"),
    )
    for regions, edge_c, face_c, group_label in region_groups:
        for i, region in enumerate(regions):
            ax.add_patch(MplPolygon(
                region.polygon.vertices, closed=True,
                facecolor=face_c, edgecolor=edge_c, linewidth=0.7,
                linestyle="--", alpha=0.55, zorder=1.3,
                label=group_label if i == 0 else None,
            ))
            ax.annotate(region.id, xy=(float(region.center[0]), float(region.center[1])),
                        fontsize=4.5, color=edge_c, ha="center", va="center", zorder=2)

    # Safe flight corridor polytopes
    n_corr = len(reference_trajectory.corridors)
    for k, poly in enumerate(reference_trajectory.corridors):
        verts = _polytope_vertices(poly)
        if verts is None:
            continue
        ax.add_patch(MplPolygon(
            verts, closed=True, facecolor="#9B59B6", edgecolor="#6C3483",
            linewidth=0.9, linestyle="-", alpha=0.16, zorder=2.5,
            label="Safe Flight Corridor" if k == 0 else None,
        ))

    # Planned reference path
    ref_x = np.asarray(reference_trajectory.x)
    ref_y = np.asarray(reference_trajectory.y)
    ax.plot(ref_x, ref_y, color="#C0392B", linewidth=2.0,
            label="Planned Reference Path", zorder=3)
    ax.scatter([ref_x[0]], [ref_y[0]], marker="o", s=42, color="#1E8449",
               zorder=4, label="Start")
    ax.scatter([ref_x[-1]], [ref_y[-1]], marker="*", s=120, color="#B03A2E",
               zorder=4, label="Goal")

    # Complete towbarless system footprint along the planned path,
    # sampled by arc length so silhouettes stay sparse and readable
    body_model = DualBodyModel()
    for k, idx in enumerate(_arc_length_sample(ref_x, ref_y, max_poses=4)):
        _draw_towbarless_system(
            ax, body_model,
            float(ref_x[idx]), float(ref_y[idx]),
            float(reference_trajectory.psi_t[idx]),
            float(reference_trajectory.theta_h[idx]),
            with_label=(k == 0),
        )

    # Target aircraft pose outline (station missions)
    if goal_aircraft_pose is not None:
        gx, gy, gpsi = goal_aircraft_pose
        goal_outline = silhouette_at_aircraft_pose(gx, gy, gpsi)
        ax.add_patch(MplPolygon(
            goal_outline, closed=True, facecolor="none",
            edgecolor="#1E8449", linewidth=1.4, linestyle="--",
            zorder=5.0, label="Goal Aircraft Pose",
        ))

    ax.set_xlabel("Flight Deck Longitudinal Coordinate X (m)")
    ax.set_ylabel("Flight Deck Lateral Coordinate Y (m)")
    ax.set_title(f"Kinodynamic Path Planning: {n_corr} Safe Corridor Segments")

    xs_all = np.concatenate([deck_env.outer_boundary.vertices[:, 0], ref_x])
    ys_all = np.concatenate([deck_env.outer_boundary.vertices[:, 1], ref_y])
    pad = 6.0
    ax.set_xlim(float(xs_all.min()) - pad, float(xs_all.max()) + pad)
    ax.set_ylim(float(ys_all.min()) - pad, float(ys_all.max()) + pad)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.9, ncol=2)
    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        _save_figure(fig, p)

    return fig


def _get_box_vertices(x: float, y: float, psi: float, length: float, width: float) -> np.ndarray:
    """Generate 2D corner vertices for an oriented bounding box."""
    hl = length / 2.0
    hw = width / 2.0
    corners = np.array([
        [hl, hw],
        [-hl, hw],
        [-hl, -hw],
        [hl, -hw],
    ])
    c, s = np.cos(psi), np.sin(psi)
    R = np.array([[c, -s], [s, c]])
    rot = (R @ corners.T).T
    rot[:, 0] += x
    rot[:, 1] += y
    return rot


def plot_tracking_error_profiles(
    timestamps: Sequence[float],
    states: Sequence[VehicleState],
    actions: Sequence[ControlAction],
    reference_trajectory: PlannedTrajectory,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Multi-panel publication plot for tracking errors and control inputs."""
    set_academic_style()
    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

    times = np.asarray(timestamps)[:len(states)]
    lat_errors = []
    psi_errors_deg = []
    hitch_deg = []
    v_actual = []
    v_ref = []
    a_cmds = []
    delta_cmds_deg = []

    for idx, t in enumerate(times):
        st = states[idx]
        ref = reference_trajectory.eval_at(float(t))
        dx = st.x_t - ref["x"]
        dy = st.y_t - ref["y"]
        psi_r = ref["psi_t"]

        e_lat = -dx * np.sin(psi_r) + dy * np.cos(psi_r)
        e_psi = wrap_to_pi(st.psi_t - psi_r)

        lat_errors.append(e_lat)
        psi_errors_deg.append(np.degrees(e_psi))
        hitch_deg.append(np.degrees(st.theta_h))
        v_actual.append(st.v_t)
        v_ref.append(ref["v"])

        if idx < len(actions):
            a_cmds.append(actions[idx].a_cmd)
            delta_cmds_deg.append(np.degrees(actions[idx].steer_angle_cmd if actions[idx].steer_angle_cmd is not None else 0.0))
        else:
            a_cmds.append(0.0)
            delta_cmds_deg.append(0.0)

    # 1. Lateral Error e_lat(t)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(times, lat_errors, "b-", label="Lateral Error $e_{lat}$")
    ax1.axhline(0.35, color="r", linestyle="--", linewidth=1.0, label="AC 3 Threshold ($+0.35$ m)")
    ax1.axhline(-0.35, color="r", linestyle="--", linewidth=1.0, label="AC 3 Threshold ($-0.35$ m)")
    ax1.set_ylabel("$e_{lat}$ (m)")
    ax1.set_title("(a) Cross-Track Lateral Error")
    ax1.grid(True)
    ax1.legend(loc="upper right", fontsize=8)

    # 2. Heading Error e_psi(t)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(times, psi_errors_deg, color="#8E44AD", label=r"Heading Error $e_{\psi}$")
    ax2.axhline(3.0, color="r", linestyle="--", linewidth=1.0, label=r"$\pm 3^\circ$ Bound")
    ax2.axhline(-3.0, color="r", linestyle="--", linewidth=1.0)
    ax2.set_ylabel(r"$e_{\psi}$ (deg)")
    ax2.set_title(r"(b) Orientation Heading Error")
    ax2.grid(True)
    ax2.legend(loc="upper right", fontsize=8)

    # 3. Hitch Articulation Angle theta_h(t)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(times, hitch_deg, color="#D35400", label=r"Hitch Angle $\theta_h$")
    ax3.axhline(70.0, color="r", linestyle=":", linewidth=1.2, label=r"Jackknife Limit ($\pm 70^\circ$)")
    ax3.axhline(-70.0, color="r", linestyle=":", linewidth=1.2)
    ax3.set_ylabel(r"$\theta_h$ (deg)")
    ax3.set_title(r"(c) Turntable Hitch Articulation Angle")
    ax3.grid(True)
    ax3.legend(loc="upper right", fontsize=8)

    # 4. Longitudinal Speed Tracking v(t)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(times, v_ref, "k--", label=r"Reference $v_{ref}$")
    ax4.plot(times, v_actual, "g-", label=r"Actual $v_t$")
    ax4.set_ylabel("Speed (m/s)")
    ax4.set_title(r"(d) Longitudinal Velocity Tracking")
    ax4.grid(True)
    ax4.legend(loc="upper right", fontsize=8)

    # 5. Steering Angle Command delta_f(t)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(times, delta_cmds_deg, color="#16A085", label=r"Steer $\delta_f$")
    ax5.axhline(35.0, color="k", linestyle=":", linewidth=0.8, label=r"$\pm 35^\circ$ Limit")
    ax5.axhline(-35.0, color="k", linestyle=":", linewidth=0.8)
    ax5.set_xlabel("Time $t$ (s)")
    ax5.set_ylabel(r"$\delta_f$ (deg)")
    ax5.set_title(r"(e) Front Steering Command")
    ax5.grid(True)
    ax5.legend(loc="upper right", fontsize=8)

    # 6. Drive/Braking Acceleration Command a_cmd(t)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(times, a_cmds, color="#C0392B", label=r"Acc $a_{cmd}$")
    ax6.axhline(1.0, color="k", linestyle=":", linewidth=0.8, label=r"$\pm 1.0\,\text{m/s}^2$ Limit")
    ax6.axhline(-1.0, color="k", linestyle=":", linewidth=0.8)
    ax6.set_xlabel("Time $t$ (s)")
    ax6.set_ylabel(r"$a_{cmd}$ ($\text{m/s}^2$)")
    ax6.set_title(r"(f) Longitudinal Acceleration Command")
    ax6.grid(True)
    ax6.legend(loc="upper right", fontsize=8)

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        _save_figure(fig, p)

    return fig


def plot_phase_portraits(
    states: Sequence[VehicleState],
    dt: float = 0.02,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot phase portrait trajectories for articulation (theta_h, dtheta_h) and sideslip (beta, dbeta)."""
    set_academic_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    theta = np.array([st.theta_h for st in states])
    beta = np.array([st.beta_slip for st in states])

    # Derivatives
    dtheta = np.gradient(theta, dt)
    dbeta = np.gradient(beta, dt)

    # 1. Articulation Phase Portrait
    ax1.plot(np.degrees(theta), np.degrees(dtheta), color="#2980B9", linewidth=1.5)
    ax1.scatter([np.degrees(theta[0])], [np.degrees(dtheta[0])], color="green", marker="o", s=40, label="Start", zorder=5)
    ax1.scatter([np.degrees(theta[-1])], [np.degrees(dtheta[-1])], color="red", marker="x", s=50, label="End", zorder=5)
    ax1.axvline(0, color="k", linestyle=":", alpha=0.5)
    ax1.axhline(0, color="k", linestyle=":", alpha=0.5)
    ax1.set_xlabel(r"Articulation Angle $\theta_h$ (deg)")
    ax1.set_ylabel(r"Articulation Rate $\dot{\theta}_h$ (deg/s)")
    ax1.set_title(r"Hitch Articulation Phase Portrait $(\theta_h, \dot{\theta}_h)$")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    # 2. Sideslip Phase Portrait
    ax2.plot(np.degrees(beta), np.degrees(dbeta), color="#8E44AD", linewidth=1.5)
    ax2.scatter([np.degrees(beta[0])], [np.degrees(dbeta[0])], color="green", marker="o", s=40, label="Start", zorder=5)
    ax2.scatter([np.degrees(beta[-1])], [np.degrees(dbeta[-1])], color="red", marker="x", s=50, label="End", zorder=5)
    ax2.axvline(0, color="k", linestyle=":", alpha=0.5)
    ax2.axhline(0, color="k", linestyle=":", alpha=0.5)
    ax2.set_xlabel(r"Sideslip Angle $\beta$ (deg)")
    ax2.set_ylabel(r"Sideslip Rate $\dot{\beta}$ (deg/s)")
    ax2.set_title(r"Vehicle Sideslip Phase Portrait $(\beta, \dot{\beta})$")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        _save_figure(fig, p)

    return fig


def plot_tire_adhesion_circle(
    actions: Sequence[ControlAction],
    states: Sequence[VehicleState],
    friction_coeff: float = 0.50,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot normalized tire adhesion circle and trajectory operating points."""
    set_academic_style()
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    # Unit Adhesion Circle
    unit_circle = MplCircle((0, 0), 1.0, color="#E74C3C", fill=False, linestyle="--", linewidth=1.8, label=r"Adhesion Limit $\mu F_z$")
    ax.add_patch(unit_circle)

    # Normalized Forces: fx = a_cmd / (mu * g), fy = v * r / (mu * g)
    g = 9.81
    denom = max(friction_coeff * g, 1e-3)
    n = min(len(actions), len(states))
    fx_norm = [actions[i].a_cmd / denom for i in range(n)]
    fy_norm = [(states[i].v_t * states[i].omega_t) / denom for i in range(n)]

    sc = ax.scatter(fx_norm, fy_norm, c=range(n), cmap="viridis", s=15, alpha=0.8, label="Operating Points", zorder=3)
    plt.colorbar(sc, ax=ax, label="Step index", fraction=0.046, pad=0.04)

    ax.set_xlabel(r"Normalized Longitudinal Force $\bar{F}_x = F_x / (\mu F_z)$")
    ax.set_ylabel(r"Normalized Lateral Force $\bar{F}_y = F_y / (\mu F_z)$")
    ax.set_title("Tire Adhesion Utilization & Saturation Envelope")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axhline(0, color="k", linestyle=":", alpha=0.4)
    ax.axvline(0, color="k", linestyle=":", alpha=0.4)
    ax.grid(True)
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        _save_figure(fig, p)

    return fig


def plot_ablation_comparison(
    ablation_results: Sequence[AblationResult],
    metric_keys: Tuple[str, str, str] = ("max_lateral_error_m", "rms_lateral_error_m", "mean_solve_time_ms"),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Grouped bar chart comparing ablation conditions."""
    set_academic_style()
    fig, axes = plt.subplots(1, len(metric_keys), figsize=(4 * len(metric_keys), 4), dpi=300)

    labels = [r.config.label for r in ablation_results]
    n_res = len(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, n_res))

    titles = {
        "max_lateral_error_m": "Max Lateral Error (m)",
        "rms_lateral_error_m": "RMS Lateral Error (m)",
        "max_heading_error_deg": "Max Heading Error (deg)",
        "mean_solve_time_ms": "Mean Solve Latency (ms)",
        "max_load_transfer_ratio": "Max LTR",
    }

    for i, m_key in enumerate(metric_keys):
        ax = axes[i] if len(metric_keys) > 1 else axes
        values = [getattr(r.metrics, m_key) for r in ablation_results]
        bars = ax.bar(range(n_res), values, color=colors, edgecolor="black", alpha=0.85)
        ax.set_xticks(range(n_res))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(titles.get(m_key, m_key))
        ax.grid(True, axis="y")

        # Add text labels on top of bars
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        _save_figure(fig, p)

    return fig
