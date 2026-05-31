# ATC-RL — AI-Powered Air Traffic Control with ADS-B Data

> **100 % free / open-source stack.** No paid tools required at any layer.

---

## Architecture Overview

```
Live ADS-B feed / CSV
        │
        ▼
┌──────────────────┐
│  adsb_processor  │  Parse → clean → discretize → conflict-detect
└────────┬─────────┘
         │  encoded state vector
         ▼
┌──────────────────┐
│    rl_agent      │  DQN (Dueling Double DQN)
│  - encode_state  │    forward pass  → Q(s,a)
│  - select_action │    ε-greedy      → action
│  - learn()       │    backward pass → ∇θ
└────────┬─────────┘
         │  ATC instruction
         ▼
┌──────────────────┐
│   simulation_    │  BlueSky (preferred) or internal physics
│   integration    │  propagator  →  next state
└──────────────────┘
         │
         ▼
   CSV + JSON logs / training curves
```

---

## File Map

| File | Purpose |
|------|---------|
| `adsb_processor.py` | Data ingestion, discretisation, conflict detection |
| `rl_agent.py` | DQNNetwork, ATCDQNAgent, reward function, ε-greedy |
| `training.py` | Training loop, ATCEnvironment |
| `simulation_integration.py` | BlueSky / physics propagator, scenario runner |
| `requirements.txt` | All dependencies (free/OSS) |

---

## State Space

Each aircraft is encoded as an **11-dimensional** normalised vector:

| Feature | Range | Description |
|---------|-------|-------------|
| x_norm | [−1, 1] | East position in sector |
| y_norm | [−1, 1] | North position in sector |
| alt_norm | [−1, 1] | Altitude (normalised around FL280) |
| spd_norm | [−1, 1] | Ground speed |
| sin(hdg) | [−1, 1] | Heading sine component |
| cos(hdg) | [−1, 1] | Heading cosine component |
| vr_norm | [−1, 1] | Vertical rate |
| x_bin | [0, 1] | Discretised x bin / N_X |
| y_bin | [0, 1] | Discretised y bin / N_Y |
| alt_bin | [0, 1] | Altitude band index / N_ALT |
| spd_bin | [0, 1] | Speed band index / N_SPD |

Up to **MAX_AIRCRAFT = 12** aircraft are packed into one flat vector (zero-padded).
Total state dimension: **12 × 11 = 132**.

### Discretisation bins

| Dimension | Bins | Edges |
|-----------|------|-------|
| X (lateral) | 10 | uniform across sector width |
| Y (longitudinal) | 10 | uniform across sector height |
| Altitude | 6 | 0 / FL100 / FL180 / FL240 / FL290 / FL370 / ∞ |
| Speed | 5 | 0 / 150 / 250 / 320 / 400 / ∞ kts |
| Heading | 8 | 45° octants |
| Vertical rate | 3 | descend / level / climb (±200 fpm threshold) |

---

## Action Space

**N_ACTIONS = MAX_AIRCRAFT × 17 = 204**

Each action targets one aircraft slot with one of 17 instructions:

| Type | Options |
|------|---------|
| Heading change | −30°, −15°, −5°, 0°, +5°, +15°, +30° (7) |
| Altitude change | −2000, −1000, 0, +1000, +2000 ft (5) |
| Speed change | −20, −10, 0, +10, +20 kts (5) |

---

## Reward Function

```
R = R_safety + R_efficiency + R_workload + R_exit

R_safety     = −100 × (# active LOS conflicts)
             + −10  × (# pairs in alert zone ≤ 1.5× minima)

R_efficiency = −0.05 × (# aircraft)        ← fuel/time cost per step
R_workload   = −0.5  if non-HOLD instruction issued
R_exit       = +5.0  × (# aircraft exiting sector cleanly)
```

ICAO separation minima enforced:
- Horizontal: **5 NM**
- Vertical:   **1 000 ft**

---

## DQN Architecture (Dueling Double DQN)

```
Input (132)
    │
    ├─ Linear(132, 256) → LayerNorm → ReLU
    ├─ Linear(256, 256) → LayerNorm → ReLU
    └─ Linear(256, 128) → ReLU
              │
    ┌─────────┴─────────┐
    │                   │
Value stream      Advantage stream
Linear(128,64)    Linear(128,128)
→ ReLU            → ReLU
→ Linear(64,1)    → Linear(128, 204)
    │                   │
    └──────── Q = V + (A − mean(A)) ──────→ 204 Q-values
```

**Loss:** Huber (Smooth L1) — robust to outliers in early training  
**Optimiser:** Adam (lr = 1e-4)  
**Target net:** Polyak soft-update (τ = 0.005)  
**Gradient clipping:** max-norm = 10

---

## ε-Greedy Exploration

```
ε(t) = ε_end + (ε_start − ε_end) × exp(−t / decay)

ε_start = 1.0   (fully random)
ε_end   = 0.05  (5 % random floor)
decay   = 10 000 steps
```

At each step:
- With probability ε  → pick **random** valid action (explore)
- With probability 1−ε → pick **argmax Q** over valid actions (exploit)

---

## Simulation Tools

### BlueSky (preferred)
- **License:** GNU GPL v3  
- **Repo:** https://github.com/TUDelft-CNS-ATM/bluesky  
- Full ATC simulator from TU Delft; supports BADA performance models,
  airspace definitions, realistic flight dynamics, 2D/3D visualisation.
- Install: `pip install bluesky-simulator`

### Internal Physics Propagator (built-in fallback)
- Zero additional dependencies
- Kinematic model: straight-and-level with first-order response
- Adequate for RL training loops

### Free ADS-B Data Sources
| Source | URL | Notes |
|--------|-----|-------|
| OpenSky Network | https://opensky-network.org | REST API + history |
| ADS-B Exchange | https://www.adsbexchange.com/data | Community feed |
| dump1090 | https://github.com/antirez/dump1090 | Local SDR receiver |
| pyModeS | https://github.com/junzis/pyModeS | Python Mode-S decoder |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Install BlueSky
pip install bluesky-simulator

# 3. Train on synthetic data
python training.py --episodes 500 --device cpu

# 4. Run scenario simulations
python simulation_integration.py --checkpoint checkpoints/atc_dqn_final.pt --scenario ALL

# 5. Train on your own ADS-B CSV
python training.py --no-synthetic --adsb-csv my_adsb_data.csv

# 6. Download airport coordinates
Place `airports.csv` from https://ourairports.com/data/ in the repo root to enable airport-ground conflict filtering in `adsb_processor.py` and airport-ground annotation in `preprocess_adsb.py`.
`preprocess_adsb.py` will load `airports.csv` automatically by default.
```

## ADS-B Visualization Prep

Use `preprocess_adsb.py` to convert raw CSV to the compact grouped JSON format required by `adsb_viz2.html`.

```bash
# Default: decimate to 1 point every 5 seconds (recommended — fast, ~20-50 MB output)
python preprocess_adsb.py ads-b_extra.csv

# If output is still >100 MB (very long recording), use --decimate 30
python preprocess_adsb.py ads-b_extra.csv --decimate 30
```

Then open `adsb_viz.html`:
1. Click `Import Data`
2. Select `Upload File`
3. Choose `ads-b_extra_viz.json`
4. Click `Load Data`


CSV format expected:
```
timestamp,icao24,lat,lon,altitude_ft,speed_kts,heading_deg,vertical_rate_fpm,callsign
1700000000,A12345,40.5,-74.2,35000,280,90,0,UAL123
```

---

## Component Cost Table

| Component | Tool | Cost |
|-----------|------|------|
| ADS-B decoding | pyModeS / dump1090 | Free |
| Data processing | pandas / numpy | Free |
| RL framework | PyTorch | Free |
| ATC simulation | BlueSky | Free (GPL-3) |
| Visualisation | matplotlib | Free |
| Live data | OpenSky Network API | Free (rate-limited) |

**No paid components are required at any layer.**
