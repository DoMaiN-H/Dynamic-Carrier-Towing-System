"""cts.visualization.interactive_deck

Interactive Web Visualizer and Animation Generator (Feature F22).
Generates:
1. Self-contained, zero-dependency browser HTML application with interactive timeline,
   trajectory playback, dynamic vehicle postures, and telemetry HUD.
2. Animated GIF rendering of flight deck towing operations.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Sequence, Optional, Union, List
import numpy as np

from cts.common.math_utils import wrap_to_pi
from cts.dynamics.multibody_dynamics import VehicleState
from cts.planning.minco_optimizer import PlannedTrajectory
from cts.planning.deck_map import default_runway_centerline
from cts.environment.deck_env import DeckEnvironment


def generate_interactive_html(
    reference_trajectory: PlannedTrajectory,
    states: Sequence[VehicleState],
    deck_env: Optional[DeckEnvironment] = None,
    save_path: Optional[Union[str, Path]] = None,
    dt: float = 0.02,
) -> str:
    """Generate self-contained interactive HTML flight deck inspection tool."""
    if deck_env is None:
        deck_env = DeckEnvironment.from_default()

    deck_poly = deck_env.outer_boundary.vertices.tolist()
    island_poly = deck_env.island.polygon.vertices.tolist()

    # Functional stations & angled runway geometry
    parking_polys = [r.polygon.vertices.tolist() for r in deck_env.get_parking_spots()]
    support_polys = [r.polygon.vertices.tolist() for r in deck_env.get_support_spots()]
    takeoff_polys = [r.polygon.vertices.tolist() for r in deck_env.get_catapult_spots()]
    region_labels = [
        {"id": r.id, "x": round(float(r.center[0]), 2), "y": round(float(r.center[1]), 2)}
        for r in deck_env.all_regions.values()
    ]
    runway = default_runway_centerline()
    rw_n = runway.normal
    rw_hw = 0.5 * runway.width
    runway_band = [
        (runway.start_point + rw_n * rw_hw).tolist(),
        (runway.end_point + rw_n * rw_hw).tolist(),
        (runway.end_point - rw_n * rw_hw).tolist(),
        (runway.start_point - rw_n * rw_hw).tolist(),
    ]
    runway_axis = [runway.start_point.tolist(), runway.end_point.tolist()]
    touchdown_xy = runway.touchdown_point.tolist()

    # SVG renders +Y downward; FLIP_Y maps deck data coordinates to screen space
    # so that +Y (port side) appears on top, consistent with the data convention.
    flip_y = 115.0

    # Downsample points if too dense (target ~400 frames for smooth web playback)
    n_pts = len(states)
    stride = max(1, n_pts // 500)

    sampled_indices = list(range(0, n_pts, stride))
    if sampled_indices[-1] != n_pts - 1:
        sampled_indices.append(n_pts - 1)

    frames_data = []
    ref_points = []
    act_points = []

    for idx in sampled_indices:
        st = states[idx]
        t = idx * dt
        ref = reference_trajectory.eval_at(t)

        dx = st.x_t - ref["x"]
        dy = st.y_t - ref["y"]
        psi_r = ref["psi_t"]
        e_lat = -dx * np.sin(psi_r) + dy * np.cos(psi_r)
        psi_a = wrap_to_pi(st.psi_t + st.theta_h)

        frames_data.append({
            "t": round(t, 2),
            "x": round(st.x_t, 2),
            "y": round(st.y_t, 2),
            "psi": round(st.psi_t, 4),
            "psi_deg": round(float(np.degrees(st.psi_t)), 1),
            "theta": round(st.theta_h, 4),
            "theta_deg": round(float(np.degrees(st.theta_h)), 1),
            "psi_a": round(psi_a, 4),
            "v": round(st.v_t, 2),
            "delta": round(st.delta_f, 4),
            "delta_deg": round(float(np.degrees(st.delta_f)), 1),
            "e_lat": round(float(e_lat), 3),
            "ref_x": round(float(ref["x"]), 2),
            "ref_y": round(float(ref["y"]), 2),
        })

    for t_step in np.linspace(0.0, float(reference_trajectory.duration), 200):
        r = reference_trajectory.eval_at(t_step)
        ref_points.append([round(float(r["x"]), 2), round(float(r["y"]), 2)])

    for st in states[::max(1, len(states)//250)]:
        act_points.append([round(st.x_t, 2), round(st.y_t, 2)])

    # Self-contained HTML Template
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Carrier Towing System - Interactive Deck Playback</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 20px;
      background: #1E222B;
      color: #ECEFF4;
    }}
    h1 {{
      font-size: 20px;
      margin-bottom: 8px;
      color: #88C0D0;
    }}
    .subtitle {{
      font-size: 13px;
      color: #D8DEE9;
      margin-bottom: 16px;
    }}
    .container {{
      display: flex;
      flex-direction: column;
      gap: 16px;
      max-width: 1200px;
      margin: 0 auto;
    }}
    .canvas-card {{
      background: #2E3440;
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      position: relative;
    }}
    svg {{
      width: 100%;
      height: 400px;
      background: #242933;
      border-radius: 6px;
    }}
    .controls {{
      display: flex;
      align-items: center;
      gap: 16px;
      margin-top: 12px;
    }}
    button {{
      background: #5E81AC;
      color: white;
      border: none;
      padding: 8px 18px;
      border-radius: 4px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }}
    button:hover {{ background: #81A1C1; }}
    input[type=range] {{
      flex: 1;
      accent-color: #88C0D0;
    }}
    .hud {{
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 12px;
      background: #2E3440;
      padding: 14px;
      border-radius: 8px;
    }}
    .hud-item {{
      background: #3B4252;
      padding: 10px;
      border-radius: 6px;
      text-align: center;
    }}
    .hud-label {{
      font-size: 11px;
      color: #9AA7BD;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .hud-value {{
      font-size: 18px;
      font-weight: bold;
      color: #ECEFF4;
      margin-top: 4px;
    }}
    .legend {{
      display: flex;
      gap: 20px;
      font-size: 12px;
      margin-top: 8px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .legend-box {{
      width: 14px;
      height: 14px;
      border-radius: 3px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div>
      <h1>Carrier Flight Deck Towing Simulator & Trajectory Inspection</h1>
      <div class="subtitle">6-DOF Dynamic Deck Motion | MINCO Spatiotemporal Corridor | Non-Linear Tire Slip NMPC Control</div>
    </div>

    <div class="canvas-card">
      <svg id="deckSvg" viewBox="10 -10 330 110">
        <!-- Grid pattern -->
        <defs>
          <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#2E3440" stroke-width="0.5"/>
          </pattern>
        </defs>
        <rect x="0" y="-15" width="360" height="125" fill="url(#grid)" />

        <!-- World group: y-flip so data +Y (port side) renders upward -->
        <g id="world" transform="translate(0, {flip_y}) scale(1,-1)">
          <!-- Deck Polygon -->
          <polygon id="deckPoly" points="" fill="#3B4252" stroke="#4C566A" stroke-width="1.2" />

          <!-- Angled Runway Corridor & Centerline -->
          <polygon id="runwayBand" points="" fill="#D8DEE9" opacity="0.10" stroke="none" />
          <polyline id="runwayAxis" points="" fill="none" stroke="#D8DEE9" stroke-width="0.7" stroke-dasharray="4,2" opacity="0.55" />

          <!-- Functional Station Layers (populated by JS) -->
          <g id="parkingLayer"></g>
          <g id="supportLayer"></g>
          <g id="takeoffLayer"></g>

          <!-- Island Superstructure -->
          <polygon id="islandPoly" points="" fill="#4C566A" stroke="#ECEFF4" stroke-width="0.8" />
          <circle id="tdPoint" cx="0" cy="0" r="1.1" fill="none" stroke="#BF616A" stroke-width="0.5" />

          <!-- Reference Trajectory -->
          <polyline id="refPath" points="" fill="none" stroke="#EBCB8B" stroke-width="0.8" stroke-dasharray="2,2" />

          <!-- Actual Trajectory -->
          <polyline id="actPath" points="" fill="none" stroke="#88C0D0" stroke-width="0.9" />

          <!-- Aircraft Body (Nose at hitch, Main gear behind) -->
          <g id="aircraftGroup">
            <line id="airFuselage" x1="0" y1="0" x2="0" y2="0" stroke="#D08770" stroke-width="2.2" stroke-linecap="round" />
            <line id="airWings" x1="0" y1="0" x2="0" y2="0" stroke="#BF616A" stroke-width="1.4" stroke-linecap="round" />
            <circle id="noseGear" cx="0" cy="0" r="1.0" fill="#EBCB8B" />
            <circle id="mainGearL" cx="0" cy="0" r="0.9" fill="#A3BE8C" />
            <circle id="mainGearR" cx="0" cy="0" r="0.9" fill="#A3BE8C" />
          </g>

          <!-- Tractor Vehicle Body -->
          <g id="tractorGroup">
            <polygon id="tractorBox" points="" fill="#81A1C1" stroke="#ECEFF4" stroke-width="0.6" />
            <line id="tractorHeading" x1="0" y1="0" x2="0" y2="0" stroke="#ECEFF4" stroke-width="0.8" />
          </g>
        </g>

        <!-- Upright text labels (outside the y-flip group) -->
        <g id="labelLayer"></g>
      </svg>

      <div class="legend">
        <div class="legend-item"><div class="legend-box" style="background: #3B4252; border: 1px solid #4C566A;"></div>Flight Deck</div>
        <div class="legend-item"><div class="legend-box" style="background: #4C566A;"></div>Island Superstructure</div>
        <div class="legend-item"><div class="legend-box" style="background: rgba(94,129,172,0.35); border: 1px dashed #5E81AC;"></div>Parking</div>
        <div class="legend-item"><div class="legend-box" style="background: rgba(180,142,173,0.35); border: 1px dashed #B48EAD;"></div>Support</div>
        <div class="legend-item"><div class="legend-box" style="background: rgba(163,190,140,0.35); border: 1px dashed #A3BE8C;"></div>Catapult</div>
        <div class="legend-item"><div class="legend-box" style="background: rgba(216,222,233,0.25);"></div>Angled Runway</div>
        <div class="legend-item"><div class="legend-box" style="background: #EBCB8B;"></div>Reference Path</div>
        <div class="legend-item"><div class="legend-box" style="background: #88C0D0;"></div>Actual Path</div>
        <div class="legend-item"><div class="legend-box" style="background: #81A1C1;"></div>Tractor</div>
        <div class="legend-item"><div class="legend-box" style="background: #D08770;"></div>Carrier Aircraft</div>
      </div>

      <div class="controls">
        <button id="playBtn">Play</button>
        <input type="range" id="timeSlider" min="0" max="{len(frames_data)-1}" value="0">
        <span id="timeLabel" style="font-family: monospace; font-size: 13px; width: 60px;">0.00s</span>
        <button id="speedBtn" style="padding: 6px 12px; font-size: 12px;">1x</button>
      </div>
    </div>

    <div class="hud">
      <div class="hud-item">
        <div class="hud-label">Time</div>
        <div class="hud-value" id="hudTime">0.00 s</div>
      </div>
      <div class="hud-item">
        <div class="hud-label">Position (X, Y)</div>
        <div class="hud-value" id="hudPos">0.0, 0.0</div>
      </div>
      <div class="hud-item">
        <div class="hud-label">Velocity</div>
        <div class="hud-value" id="hudVel">0.00 m/s</div>
      </div>
      <div class="hud-item">
        <div class="hud-label">Lateral Error</div>
        <div class="hud-value" id="hudLatErr">0.000 m</div>
      </div>
      <div class="hud-item">
        <div class="hud-label">Hitch Angle &theta;<sub>h</sub></div>
        <div class="hud-value" id="hudHitch">0.0&deg;</div>
      </div>
      <div class="hud-item">
        <div class="hud-label">Steering &delta;<sub>f</sub></div>
        <div class="hud-value" id="hudSteer">0.0&deg;</div>
      </div>
    </div>
  </div>

  <script>
    const deckPoly = {json.dumps(deck_poly)};
    const islandPoly = {json.dumps(island_poly)};
    const parkingPolys = {json.dumps(parking_polys)};
    const supportPolys = {json.dumps(support_polys)};
    const takeoffPolys = {json.dumps(takeoff_polys)};
    const regionLabels = {json.dumps(region_labels)};
    const runwayBandPts = {json.dumps(runway_band)};
    const runwayAxisPts = {json.dumps(runway_axis)};
    const touchdownPt = {json.dumps(touchdown_xy)};
    const FLIP_Y = {flip_y};
    const refPath = {json.dumps(ref_points)};
    const actPath = {json.dumps(act_points)};
    const frames = {json.dumps(frames_data)};

    // Init static SVG elements
    document.getElementById("deckPoly").setAttribute("points", deckPoly.map(p => `${{p[0]}},${{p[1]}}`).join(" "));
    document.getElementById("islandPoly").setAttribute("points", islandPoly.map(p => `${{p[0]}},${{p[1]}}`).join(" "));
    document.getElementById("refPath").setAttribute("points", refPath.map(p => `${{p[0]}},${{p[1]}}`).join(" "));
    document.getElementById("actPath").setAttribute("points", actPath.map(p => `${{p[0]}},${{p[1]}}`).join(" "));
    document.getElementById("runwayBand").setAttribute("points", runwayBandPts.map(p => `${{p[0]}},${{p[1]}}`).join(" "));
    document.getElementById("runwayAxis").setAttribute("points", runwayAxisPts.map(p => `${{p[0]}},${{p[1]}}`).join(" "));
    document.getElementById("tdPoint").setAttribute("cx", touchdownPt[0]);
    document.getElementById("tdPoint").setAttribute("cy", touchdownPt[1]);

    // Populate functional station polygons (data coordinates; world group applies y-flip)
    const SVGNS = "http://www.w3.org/2000/svg";
    function addRegionPolys(arr, layerId, stroke, fill) {{
      const layer = document.getElementById(layerId);
      arr.forEach(poly => {{
        const el = document.createElementNS(SVGNS, "polygon");
        el.setAttribute("points", poly.map(p => `${{p[0]}},${{p[1]}}`).join(" "));
        el.setAttribute("fill", fill);
        el.setAttribute("stroke", stroke);
        el.setAttribute("stroke-width", "0.5");
        el.setAttribute("stroke-dasharray", "1.6,1.6");
        layer.appendChild(el);
      }});
    }}
    addRegionPolys(parkingPolys, "parkingLayer", "#5E81AC", "rgba(94,129,172,0.10)");
    addRegionPolys(supportPolys, "supportLayer", "#B48EAD", "rgba(180,142,173,0.12)");
    addRegionPolys(takeoffPolys, "takeoffLayer", "#A3BE8C", "rgba(163,190,140,0.15)");

    // Station ID labels in upright overlay layer (manual y-flip)
    const labelLayer = document.getElementById("labelLayer");
    regionLabels.forEach(r => {{
      const t = document.createElementNS(SVGNS, "text");
      t.setAttribute("x", r.x);
      t.setAttribute("y", FLIP_Y - r.y);
      t.setAttribute("font-size", "3.2");
      t.setAttribute("fill", "#9AA7BD");
      t.setAttribute("text-anchor", "middle");
      t.setAttribute("dominant-baseline", "middle");
      t.textContent = r.id;
      labelLayer.appendChild(t);
    }});
    const tdLabel = document.createElementNS(SVGNS, "text");
    tdLabel.setAttribute("x", touchdownPt[0]);
    tdLabel.setAttribute("y", FLIP_Y - touchdownPt[1] + 4.5);
    tdLabel.setAttribute("font-size", "3.4");
    tdLabel.setAttribute("fill", "#BF616A");
    tdLabel.setAttribute("text-anchor", "middle");
    tdLabel.textContent = "TD";
    labelLayer.appendChild(tdLabel);

    let currentIdx = 0;
    let isPlaying = false;
    let playInterval = null;
    let speed = 1.0;

    const slider = document.getElementById("timeSlider");
    const playBtn = document.getElementById("playBtn");
    const speedBtn = document.getElementById("speedBtn");
    const timeLabel = document.getElementById("timeLabel");

    function renderFrame(idx) {{
      const f = frames[idx];
      if (!f) return;

      slider.value = idx;
      timeLabel.innerText = f.t.toFixed(2) + "s";
      document.getElementById("hudTime").innerText = f.t.toFixed(2) + " s";
      document.getElementById("hudPos").innerText = `${{f.x.toFixed(1)}}, ${{f.y.toFixed(1)}}`;
      document.getElementById("hudVel").innerText = f.v.toFixed(2) + " m/s";
      document.getElementById("hudLatErr").innerText = f.e_lat.toFixed(3) + " m";
      document.getElementById("hudHitch").innerText = f.theta_deg.toFixed(1) + "°";
      document.getElementById("hudSteer").innerText = f.delta_deg.toFixed(1) + "°";

      // Tractor box: 1.8m length, 1.4m width centered at tractor CG
      const c_t = Math.cos(f.psi), s_t = Math.sin(f.psi);
      const hl = 0.9, hw = 0.7;
      const tPts = [
        [f.x + c_t*hl - s_t*hw, f.y + s_t*hl + c_t*hw],
        [f.x - c_t*hl - s_t*hw, f.y - s_t*hl + c_t*hw],
        [f.x - c_t*hl + s_t*hw, f.y - s_t*hl - c_t*hw],
        [f.x + c_t*hl + s_t*hw, f.y + s_t*hl - c_t*hw],
      ];
      document.getElementById("tractorBox").setAttribute("points", tPts.map(p => `${{p[0]}},${{p[1]}}`).join(" "));
      document.getElementById("tractorHeading").setAttribute("x1", f.x);
      document.getElementById("tractorHeading").setAttribute("y1", f.y);
      document.getElementById("tractorHeading").setAttribute("x2", f.x + c_t * 1.5);
      document.getElementById("tractorHeading").setAttribute("y2", f.y + s_t * 1.5);

      // Aircraft: Nose gear at hitch point (~0.1m behind tractor rear axle)
      const hx = f.x - c_t * 0.95;
      const hy = f.y - s_t * 0.95;
      const c_a = Math.cos(f.psi_a), s_a = Math.sin(f.psi_a);
      const airLen = 14.0;
      const tailX = hx - c_a * airLen;
      const tailY = hy - s_a * airLen;

      document.getElementById("airFuselage").setAttribute("x1", hx);
      document.getElementById("airFuselage").setAttribute("y1", hy);
      document.getElementById("airFuselage").setAttribute("x2", tailX);
      document.getElementById("airFuselage").setAttribute("y2", tailY);

      // Main landing gear: ~8.0m behind nose gear, track 3.1m
      const mgX = hx - c_a * 8.0;
      const mgY = hy - s_a * 8.0;
      const ht = 1.55;
      const mgLx = mgX - s_a * ht, mgLy = mgY + c_a * ht;
      const mgRx = mgX + s_a * ht, mgRy = mgY - c_a * ht;

      document.getElementById("noseGear").setAttribute("cx", hx);
      document.getElementById("noseGear").setAttribute("cy", hy);
      document.getElementById("mainGearL").setAttribute("cx", mgLx);
      document.getElementById("mainGearL").setAttribute("cy", mgLy);
      document.getElementById("mainGearR").setAttribute("cx", mgRx);
      document.getElementById("mainGearR").setAttribute("cy", mgRy);

      // Wings line at 7.0m behind nose, width 9.0m
      const wX = hx - c_a * 7.0;
      const wY = hy - s_a * 7.0;
      const hw_span = 4.5;
      document.getElementById("airWings").setAttribute("x1", wX - s_a * hw_span);
      document.getElementById("airWings").setAttribute("y1", wY + c_a * hw_span);
      document.getElementById("airWings").setAttribute("x2", wX + s_a * hw_span);
      document.getElementById("airWings").setAttribute("y2", wY - c_a * hw_span);
    }}

    function togglePlay() {{
      if (isPlaying) {{
        clearInterval(playInterval);
        isPlaying = false;
        playBtn.innerText = "Play";
      }} else {{
        isPlaying = true;
        playBtn.innerText = "Pause";
        playInterval = setInterval(() => {{
          currentIdx++;
          if (currentIdx >= frames.length) {{
            currentIdx = 0;
          }}
          renderFrame(currentIdx);
        }}, 30 / speed);
      }}
    }}

    playBtn.addEventListener("click", togglePlay);
    slider.addEventListener("input", (e) => {{
      currentIdx = parseInt(e.target.value);
      renderFrame(currentIdx);
    }});

    speedBtn.addEventListener("click", () => {{
      if (speed === 1.0) speed = 2.0;
      else if (speed === 2.0) speed = 5.0;
      else speed = 1.0;
      speedBtn.innerText = speed + "x";
      if (isPlaying) {{
        clearInterval(playInterval);
        playInterval = setInterval(() => {{
          currentIdx = (currentIdx + 1) % frames.length;
          renderFrame(currentIdx);
        }}, 30 / speed);
      }}
    }});

    // Initial frame
    renderFrame(0);
  </script>
</body>
</html>
"""

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(html_content)

    return html_content
