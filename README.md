# CTS — Carrier Towing System

Simulation, planning and control toolkit for **towbarless carrier aircraft tractor operations**
under 6-DOF sea-state-induced flight deck motion.

The package `cts` covers the full pipeline:

| Layer | Module | Role |
|---|---|---|
| Environment | `cts.environment` | Ocean wave spectra (PM / JONSWAP), 6-DOF deck motion (Fossen filters + RAO), non-inertial apparent acceleration field, deck geometry parsing |
| Dynamics | `cts.dynamics` | Towbarless tractor-aircraft kinematics, dynamic vertical load transfer, Dugoff / Pacejka tire models, coupled 2-body multibody plant, RK4 integrators |
| Planning | `cts.planning` | Time-varying kinodynamic limits, 5D spatiotemporal Hybrid A*, safe flight corridors (SFC), MINCO C² trajectory optimization |
| Control | `cts.control` | Inertial feedforward, 100 Hz ESO/DOB, CasADi NMPC, LTV-MPC fallback, Stanley / Pure Pursuit / PID baselines |
| Benchmarks | `cts.benchmarks` | Canonical scenarios (SC-1/2/3), quantitative metrics, factorial ablation runner |
| Visualization | `cts.visualization` | Publication-ready vector figures, interactive HTML deck playback |

## Repository layout

```
cts/        Python package (source)
data/       Carrier deck geometry and reference assets
requirements.txt
README.md
```

`data/carrier_deck_environment.json` and `data/飞机模型.svg` are loaded at runtime by
`cts.environment` and `cts.visualization`, so keep the `data/` directory next to `cts/`.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
```

Run all commands from the repository root so the `cts` package is importable.

## Quick start

```python
import numpy as np

from cts.environment import DeckMotionSimulator, DeckEnvironment
from cts.dynamics.multibody_dynamics import TractorAircraftPlant, VehicleState, ControlAction

# 1) 6-DOF deck motion and non-inertial acceleration field (Sea State 4)
deck = DeckMotionSimulator(sea_state=4, mode="fossen", dt=0.02, seed=2026)
deck_state = deck.get_deck_state(t=10.0)
apparent = deck.compute_non_inertial_acceleration(np.zeros(3), np.zeros(3), t=10.0)
print("deck rpy [rad]:", np.round(deck_state.rpy_rad, 4))
print("apparent acc [m/s^2]:", np.round(apparent.a_apparent_total, 4))

# 2) One coupled tractor-aircraft dynamics step on the moving deck
plant = TractorAircraftPlant()
x0 = VehicleState(x_t=150.0, y_t=60.0, psi_t=0.0, theta_h=0.0, v_t=1.0, delta_f=0.0)
x1 = plant.step(
    x0,
    ControlAction(a_cmd=0.5, steer_rate_cmd=0.1),
    deck_motion=deck,
    t=0.0,
    dt=0.02,
)
print("next state:", np.round(x1.to_array(), 4))

# 3) Flight deck geometry queries
env = DeckEnvironment.from_default()
print("navigable:", env.is_point_navigable((150.0, 60.0)))
```

## Dependencies

Python 3.10+ with `numpy`, `scipy`, `pydantic`, `casadi` and `matplotlib`
(see `requirements.txt`).
