"""
simulation_integration.py
--------------------------
Integrates the trained DQN agent with free / open-source air traffic
simulators.  Two simulators are supported:

  1. BlueSky  (https://github.com/TUDelft-CNS-ATM/bluesky)   ← PRIMARY
       Free, open-source, Python-native ATC simulator developed at
       TU Delft.  Install: pip install bluesky-simulator
       OR clone: git clone https://github.com/TUDelft-CNS-ATM/bluesky

  2. POSIX-socket "headless" mode  ← FALLBACK (no extra deps)
       When BlueSky is not available we run our own physics propagator
       and log results to CSV / JSON — fully self-contained.

NOT required (and not used):
  - NATS / TAAM / RAMS+ (commercial, expensive)
  - X-Plane / Prepar3D (commercial)
  - Any paid data feed

Free datasets that pair well:
  - OpenSky Network REST API  (https://opensky-network.org/apidoc/)
  - ADS-B Exchange  (https://www.adsbexchange.com/data/)

Dependencies (all free):
    pip install numpy pandas torch
    pip install bluesky-simulator          # optional but recommended
"""

import math
import time
import json
import csv
import copy
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np

from adsb_processor import (
    ADSBFrame, AirspaceSector, ADSBLoader,
    detect_conflicts, haversine_nm,
)
from rl_agent import (
    ATCDQNAgent, encode_state,
    MAX_AIRCRAFT, N_INSTR,
)

# ============================================================
# BLUESKY ADAPTER
# ============================================================

class BlueSkyAdapter:
    """
    Thin wrapper around BlueSky's headless / client interface.

    BlueSky exposes a Python API that lets you:
      - Create / delete aircraft (bluesky.traffic.create / delete)
      - Issue commands (HEADING, ALT, SPD, DIRECT, etc.)
      - Advance the simulation clock (bluesky.sim.step)
      - Read traffic state (bluesky.traffic.lat, .lon, .alt, …)

    Installation:
        git clone https://github.com/TUDelft-CNS-ATM/bluesky
        cd bluesky && pip install -e .

    If BlueSky is not installed, this class degrades gracefully to
    our internal physics propagator (PhysicsPropagator below).
    """

    def __init__(self):
        self._bs    = None
        self._avail = False
        self._try_import()

    def _try_import(self):
        try:
            import bluesky as bs
            from bluesky import settings
            settings.init('headless')
            bs.init(mode='sim', configfile=None)
            self._bs    = bs
            self._avail = True
            print("[BlueSky] Simulator initialised in headless mode ✓")
        except ImportError:
            print("[BlueSky] Not installed — falling back to internal "
                  "physics propagator.")
            print("          To install: pip install bluesky-simulator")
        except Exception as e:
            print(f"[BlueSky] Init failed ({e}) — using fallback propagator.")

    @property
    def available(self) -> bool:
        return self._avail

    # ---- populate BlueSky with ADS-B frames --------------------------------
    def load_traffic(self, frames: List[ADSBFrame]):
        if not self._avail:
            return
        traf = self._bs.traffic
        for f in frames:
            # CRE callsign type lat lon alt spd hdg
            traf.create(
                n     = 1,
                actype= "B738",
                aclat = f.lat,
                aclon = f.lon,
                acalt = f.altitude_ft * 0.3048,   # ft→m
                acspd = f.speed_kts * 0.514444,   # kts→m/s
                achdg = f.heading_deg,
                acid  = f.icao24,
            )

    # ---- apply ATC command -------------------------------------------------
    def apply_command(self, icao24: str, instruction: Dict):
        if not self._avail:
            return
        traf = self._bs.traffic
        idx  = traf.id2idx(icao24)
        if idx < 0:
            return
        t = instruction['instruction_type']
        d = instruction['delta_value']
        if t == 'HEADING':
            new_hdg = (traf.hdg[idx] + d) % 360
            traf.ap.selhdg(idx, new_hdg)
        elif t == 'ALTITUDE':
            new_alt = traf.alt[idx] + d * 0.3048
            traf.ap.selalt(idx, new_alt)
        elif t == 'SPEED':
            new_spd = traf.tas[idx] + d * 0.514444
            traf.ap.selspd(idx, new_spd)

    # ---- advance simulation ------------------------------------------------
    def step(self, dt_s: float = 10.0):
        if self._avail:
            self._bs.sim.step()

    # ---- read current state ------------------------------------------------
    def read_state(self) -> List[ADSBFrame]:
        if not self._avail:
            return []
        traf = self._bs.traffic
        frames = []
        t_now  = self._bs.sim.simt
        for i, acid in enumerate(traf.id):
            frames.append(ADSBFrame(
                timestamp        = t_now,
                icao24           = acid,
                callsign         = acid,
                lat              = float(traf.lat[i]),
                lon              = float(traf.lon[i]),
                altitude_ft      = float(traf.alt[i] / 0.3048),
                speed_kts        = float(traf.tas[i] / 0.514444),
                heading_deg      = float(traf.hdg[i]),
                vertical_rate_fpm= float(traf.vs[i] / 0.00508),
            ))
        return frames


# ============================================================
# INTERNAL PHYSICS PROPAGATOR  (fallback — no extra deps)
# ============================================================

class PhysicsPropagator:
    """
    Simplified kinematic aircraft model used when BlueSky is absent.

    State: lat, lon, alt_ft, spd_kts, hdg_deg, vr_fpm
    Model: straight-and-level flight with first-order response to commands.

    This is adequate for RL training / scenario testing when a full
    simulator is unavailable.
    """

    TURN_RATE_DEG_S    = 3.0    # standard rate turn ≈ 3°/s
    CLIMB_ACCEL        = 100    # fpm change per second toward target
    SPEED_ACCEL        = 1.0    # kts change per second toward target

    def __init__(self, frames: List[ADSBFrame]):
        self._aircraft: Dict[str, Dict] = {}
        self._t: float = 0.0
        for f in frames:
            self._aircraft[f.icao24] = dict(
                icao24       = f.icao24,
                callsign     = f.callsign,
                lat          = f.lat,
                lon          = f.lon,
                alt_ft       = f.altitude_ft,
                spd_kts      = f.speed_kts,
                hdg_deg      = f.heading_deg,
                vr_fpm       = f.vertical_rate_fpm,
                target_hdg   = f.heading_deg,
                target_alt   = f.altitude_ft,
                target_spd   = f.speed_kts,
            )

    def apply_command(self, icao24: str, instruction: Dict):
        if icao24 not in self._aircraft:
            return
        ac = self._aircraft[icao24]
        t  = instruction['instruction_type']
        d  = instruction['delta_value']
        if t == 'HEADING':
            ac['target_hdg'] = (ac['hdg_deg'] + d) % 360
        elif t == 'ALTITUDE':
            ac['target_alt'] = float(np.clip(ac['alt_ft'] + d, 1000, 45_000))
        elif t == 'SPEED':
            ac['target_spd'] = float(np.clip(ac['spd_kts'] + d, 100, 500))

    def step(self, dt_s: float = 10.0):
        self._t += dt_s
        for ac in self._aircraft.values():
            # ---- heading -------------------------------------------------
            dh = (ac['target_hdg'] - ac['hdg_deg'] + 180) % 360 - 180
            max_turn = self.TURN_RATE_DEG_S * dt_s
            ac['hdg_deg'] = (ac['hdg_deg'] +
                             np.clip(dh, -max_turn, max_turn)) % 360

            # ---- altitude ------------------------------------------------
            da = ac['target_alt'] - ac['alt_ft']
            vr_fpm_target = np.clip(da * 6, -3000, 3000)   # proportional
            ac['vr_fpm']  = float(np.clip(
                ac['vr_fpm'] + np.clip(
                    vr_fpm_target - ac['vr_fpm'],
                    -self.CLIMB_ACCEL * dt_s,
                     self.CLIMB_ACCEL * dt_s),
                -3000, 3000))
            ac['alt_ft'] = float(np.clip(
                ac['alt_ft'] + ac['vr_fpm'] * dt_s / 60, 0, 60_000))

            # ---- speed ---------------------------------------------------
            ds = ac['target_spd'] - ac['spd_kts']
            ac['spd_kts'] = float(np.clip(
                ac['spd_kts'] + np.clip(
                    ds, -self.SPEED_ACCEL * dt_s,
                         self.SPEED_ACCEL * dt_s),
                80, 600))

            # ---- position ------------------------------------------------
            spd_ms   = ac['spd_kts'] * 0.514444
            hdg_rad  = math.radians(ac['hdg_deg'])
            dlat = (spd_ms * math.cos(hdg_rad) * dt_s / 111_320)
            dlon = (spd_ms * math.sin(hdg_rad) * dt_s /
                    (111_320 * math.cos(math.radians(ac['lat']))))
            ac['lat'] += dlat
            ac['lon'] += dlon

    def read_state(self) -> List[ADSBFrame]:
        return [ADSBFrame(
            timestamp        = self._t,
            icao24           = ac['icao24'],
            callsign         = ac['callsign'],
            lat              = ac['lat'],
            lon              = ac['lon'],
            altitude_ft      = ac['alt_ft'],
            speed_kts        = ac['spd_kts'],
            heading_deg      = ac['hdg_deg'],
            vertical_rate_fpm= ac['vr_fpm'],
        ) for ac in self._aircraft.values()]


# ============================================================
# SCENARIO DEFINITIONS
# ============================================================

@dataclass
class Scenario:
    name: str
    description: str
    aircraft: List[ADSBFrame]   # initial traffic state

def head_on_scenario(sector: AirspaceSector) -> Scenario:
    cx = (sector.lon_min + sector.lon_max) / 2
    cy = (sector.lat_min + sector.lat_max) / 2
    return Scenario(
        name        = "HEAD_ON",
        description = "Two aircraft approaching head-on at same altitude",
        aircraft    = [
            ADSBFrame(0, "A00001", cy, cx - 0.8, 35_000, 280, 90, 0, "UAL1"),
            ADSBFrame(0, "A00002", cy, cx + 0.8, 35_000, 280, 270, 0, "DAL2"),
        ]
    )

def crossing_scenario(sector: AirspaceSector) -> Scenario:
    cx = (sector.lon_min + sector.lon_max) / 2
    cy = (sector.lat_min + sector.lat_max) / 2
    return Scenario(
        name        = "CROSSING",
        description = "Two aircraft crossing at 90° same altitude",
        aircraft    = [
            ADSBFrame(0, "A00001", cy - 0.5, cx, 35_000, 280, 0, 0, "UAL3"),
            ADSBFrame(0, "A00002", cy, cx - 0.5, 35_000, 280, 90, 0, "DAL4"),
        ]
    )

def high_density_scenario(sector: AirspaceSector) -> Scenario:
    rng = np.random.default_rng(7)
    aircraft = []
    for i in range(10):
        aircraft.append(ADSBFrame(
            timestamp        = 0,
            icao24           = f"A{i:05X}",
            lat              = rng.uniform(sector.lat_min, sector.lat_max),
            lon              = rng.uniform(sector.lon_min, sector.lon_max),
            altitude_ft      = rng.choice([29_000, 33_000, 35_000,
                                           37_000, 39_000]).astype(float),
            speed_kts        = rng.uniform(240, 320),
            heading_deg      = rng.uniform(0, 360),
            vertical_rate_fpm= 0.0,
            callsign         = f"SWA{100+i}",
        ))
    return Scenario(
        name        = "HIGH_DENSITY",
        description = "10 aircraft in a congested sector",
        aircraft    = aircraft,
    )

ALL_SCENARIOS = [head_on_scenario, crossing_scenario, high_density_scenario]


# ============================================================
# SIMULATION RUNNER
# ============================================================

class SimulationRunner:
    """
    Ties together:
      agent  → DQN policy (or random baseline)
      sim    → BlueSky or PhysicsPropagator
      logger → CSV + JSON event log
    """

    DT_S        = 10.0    # physics / decision step (seconds)
    MAX_STEPS   = 300

    def __init__(self, agent: ATCDQNAgent, sector: AirspaceSector,
                 output_dir: str = "sim_results"):
        self.agent  = agent
        self.sector = sector
        self.out    = Path(output_dir)
        self.out.mkdir(exist_ok=True)

        self._bs  = BlueSkyAdapter()
        self._use_bs = self._bs.available

    # ---- run one scenario --------------------------------------------------
    def run_scenario(self, scenario: Scenario,
                     use_agent: bool = True,
                     tag: str = "") -> Dict:
        print(f"\n{'='*60}")
        print(f"Scenario : {scenario.name}  [{scenario.description}]")
        print(f"Mode     : {'DQN Agent' if use_agent else 'Baseline (random)'}")
        print(f"Simulator: {'BlueSky' if self._use_bs else 'Physics propagator'}")
        print(f"{'='*60}")

        # ---- init simulator
        if self._use_bs:
            self._bs.load_traffic(scenario.aircraft)
            sim = self._bs
        else:
            sim = PhysicsPropagator(scenario.aircraft)

        from adsb_processor import StateDiscretizer
        disc = StateDiscretizer(self.sector)

        log_rows = []
        events   = []
        total_reward = 0.0
        n_conflicts  = 0
        step = 0

        while step < self.MAX_STEPS:
            frames = sim.read_state()
            if not frames:
                break

            state = encode_state(frames, self.sector, disc)

            # ---- agent decision
            if use_agent:
                action, _ = self.agent.select_action(
                    state, n_aircraft=len(frames))
            else:
                # Baseline: random action
                import random
                action = random.randrange(
                    min(len(frames), MAX_AIRCRAFT) * N_INSTR)

            instruction = ATCDQNAgent.decode_action(action)
            target_slot = instruction['aircraft_slot']
            if target_slot < len(frames):
                target_icao = frames[target_slot].icao24
                sim.apply_command(target_icao, instruction)
                events.append(dict(
                    step=step,
                    icao24=target_icao,
                    instruction=instruction['description'],
                ))

            # ---- advance physics
            sim.step(self.DT_S)

            # ---- read updated state
            frames = sim.read_state()
            if not frames:
                break

            conflicts = detect_conflicts(frames)
            n_conflicts += len(conflicts)

            from rl_agent import compute_reward
            r = compute_reward(frames, frames, self.sector, action)
            total_reward += r

            # ---- log each aircraft
            for f in frames:
                log_rows.append(dict(
                    step             = step,
                    icao24           = f.icao24,
                    lat              = round(f.lat, 5),
                    lon              = round(f.lon, 5),
                    alt_ft           = round(f.altitude_ft),
                    spd_kts          = round(f.speed_kts, 1),
                    hdg_deg          = round(f.heading_deg, 1),
                    vr_fpm           = round(f.vertical_rate_fpm),
                    conflicts        = len(conflicts),
                ))

            if step % 30 == 0:
                print(f"  Step {step:3d}  aircraft={len(frames)}  "
                      f"conflicts={len(conflicts)}  "
                      f"reward={r:.1f}  cumulative={total_reward:.1f}")

            step += 1

        # ---- save outputs
        label = f"{scenario.name}_{tag}_{'agent' if use_agent else 'baseline'}"
        csv_path  = self.out / f"{label}_trajectory.csv"
        json_path = self.out / f"{label}_events.json"

        with open(csv_path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=log_rows[0].keys())
            w.writeheader(); w.writerows(log_rows)

        with open(json_path, 'w') as fh:
            json.dump(dict(scenario=scenario.name,
                           mode='agent' if use_agent else 'baseline',
                           total_reward=total_reward,
                           total_conflict_steps=n_conflicts,
                           steps=step,
                           events=events), fh, indent=2)

        summary = dict(
            scenario=scenario.name, mode='agent' if use_agent else 'baseline',
            total_reward=round(total_reward, 2),
            total_conflict_steps=n_conflicts, steps=step,
        )
        print(f"\n  Summary: {summary}")
        print(f"  Output : {csv_path}")
        return summary

    # ---- run all scenarios (agent vs baseline) ------------------------------
    def benchmark(self) -> List[Dict]:
        sector   = self.sector
        results  = []
        for fn in ALL_SCENARIOS:
            scenario = fn(sector)
            for use_agent in [True, False]:
                r = self.run_scenario(scenario, use_agent=use_agent)
                results.append(r)
        # Save comparison table
        with open(self.out / "benchmark_results.json", 'w') as fh:
            json.dump(results, fh, indent=2)
        print("\n\n===== BENCHMARK SUMMARY =====")
        for r in results:
            print(f"  {r['scenario']:15s} {r['mode']:8s}  "
                  f"reward={r['total_reward']:8.1f}  "
                  f"conflicts={r['total_conflict_steps']}")
        return results


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to trained model .pt file (optional)")
    p.add_argument("--scenario",   type=str,
                   choices=["HEAD_ON","CROSSING","HIGH_DENSITY","ALL"],
                   default="ALL")
    p.add_argument("--output-dir", type=str, default="sim_results")
    args = p.parse_args()

    sector = AirspaceSector(lat_min=40.0, lat_max=41.5,
                            lon_min=-75.0, lon_max=-73.0)
    agent  = ATCDQNAgent(sector)
    if args.checkpoint:
        agent.load(args.checkpoint)
    else:
        print("[Sim] No checkpoint — using untrained agent (random weights)")

    runner = SimulationRunner(agent, sector, output_dir=args.output_dir)

    if args.scenario == "ALL":
        runner.benchmark()
    else:
        fn_map = {
            "HEAD_ON"     : head_on_scenario,
            "CROSSING"    : crossing_scenario,
            "HIGH_DENSITY": high_density_scenario,
        }
        scenario = fn_map[args.scenario](sector)
        runner.run_scenario(scenario, use_agent=True)
        runner.run_scenario(scenario, use_agent=False)
