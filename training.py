"""
training.py
-----------
Training loop that drives the DQN agent through the ATC environment.

Free / open-source only.
Run: python training.py
"""

import time
import math
import copy
import json
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np

from adsb_processor import (
    ADSBFrame, AirspaceSector, ADSBLoader,
    detect_conflicts, haversine_nm,
)
from rl_agent import (
    ATCDQNAgent, encode_state, compute_reward,
    MAX_AIRCRAFT, N_INSTR,
)


# ============================================================
# ATC GYM ENVIRONMENT
# ============================================================

class ATCEnvironment:
    """
    Wraps a sequence of ADS-B frames into a step-based RL environment.

    step() contract:
        state, reward, done, info = env.step(action_idx)
    """

    STEP_INTERVAL_S   = 10      # seconds between ATC decisions
    MAX_STEPS_EP      = 500     # max steps per episode
    CONFLICT_LIMIT    = 5       # episode ends if this many simultaneous conflicts

    def __init__(self, sector: AirspaceSector,
                 frames: Optional[List[ADSBFrame]] = None,
                 synthetic: bool = True):
        self.sector  = sector
        self._raw    = frames or ADSBLoader.synthetic(
            n_aircraft=8, duration_s=3600, sector=sector)
        self._disc   = __import__('adsb_processor').StateDiscretizer(sector)

        # Group frames by timestamp
        self._by_time: Dict[float, List[ADSBFrame]] = {}
        for f in self._raw:
            self._by_time.setdefault(f.timestamp, []).append(f)
        self._times = sorted(self._by_time.keys())

        self._t_idx    = 0
        self._step_cnt = 0
        self._current_frames: List[ADSBFrame] = []
        self._episode_rewards: List[float] = []

    # ---- reset -------------------------------------------------------------
    def reset(self) -> np.ndarray:
        # Start at a random time in the first half of the data
        max_start = max(0, len(self._times) // 2)
        self._t_idx    = np.random.randint(0, max(1, max_start))
        self._step_cnt = 0
        self._current_frames = copy.deepcopy(
            self._by_time[self._times[self._t_idx]])
        self._episode_rewards = []
        return encode_state(self._current_frames, self.sector, self._disc)

    # ---- step --------------------------------------------------------------
    def step(self, action_idx: int):
        prev_frames = copy.deepcopy(self._current_frames)

        # Apply ATC instruction
        modified = ATCDQNAgent.apply_action(self._current_frames, action_idx)

        # Advance time → propagate positions
        self._t_idx   = min(self._t_idx + 1, len(self._times) - 1)
        next_t_frames = copy.deepcopy(
            self._by_time[self._times[self._t_idx]])

        # Merge ATC modifications into the next time step
        # (aircraft keep heading/alt/spd changes)
        mod_map = {f.icao24: f for f in modified}
        for f in next_t_frames:
            if f.icao24 in mod_map:
                src = mod_map[f.icao24]
                f.heading_deg  = src.heading_deg
                f.altitude_ft  = src.altitude_ft
                f.speed_kts    = src.speed_kts

        self._current_frames = next_t_frames
        self._step_cnt      += 1

        # Reward
        reward = compute_reward(prev_frames, self._current_frames,
                                self.sector, action_idx)
        self._episode_rewards.append(reward)

        # Terminal conditions
        conflicts = detect_conflicts(self._current_frames)
        done = (
            self._step_cnt >= self.MAX_STEPS_EP or
            self._t_idx   >= len(self._times) - 1 or
            len(conflicts) >= self.CONFLICT_LIMIT
        )

        next_state = encode_state(self._current_frames, self.sector, self._disc)
        info = dict(
            conflicts       = len(conflicts),
            n_aircraft      = len(self._current_frames),
            step            = self._step_cnt,
            eps_total_reward= sum(self._episode_rewards),
            epsilon         = None,   # filled by training loop
        )
        return next_state, reward, done, info


# ============================================================
# TRAINING LOOP
# ============================================================

def train(
    n_episodes: int         = 500,
    checkpoint_every: int   = 50,
    checkpoint_dir: str     = "checkpoints",
    device: str             = "cpu",
    use_synthetic: bool     = True,
    adsb_csv: Optional[str] = None,
    verbose: bool           = True,
):
    """
    Main training loop.

    Phase 1 – Warm-up  : Fill replay buffer with random actions
    Phase 2 – Learning : ε-greedy + DQN updates each step
    """
    Path(checkpoint_dir).mkdir(exist_ok=True)

    sector = AirspaceSector(
        lat_min=40.0, lat_max=41.5,
        lon_min=-75.0, lon_max=-73.0,
    )

    frames = None
    if not use_synthetic and adsb_csv:
        frames = ADSBLoader.from_csv(adsb_csv)
        print(f"[Train] Loaded {len(frames)} frames from {adsb_csv}")
    else:
        frames = ADSBLoader.synthetic(n_aircraft=8,
                                      duration_s=7200, sector=sector)
        print(f"[Train] Generated {len(frames)} synthetic frames")

    env   = ATCEnvironment(sector, frames)
    agent = ATCDQNAgent(sector, device=device)

    log = dict(
        episode_rewards  = [],
        episode_conflicts= [],
        losses           = [],
        epsilons         = [],
    )

    total_steps = 0
    t0 = time.time()

    for ep in range(1, n_episodes + 1):
        state   = env.reset()
        ep_reward  = 0.0
        ep_conflicts = 0
        done    = False

        while not done:
            n_ac     = len(env._current_frames)
            action, is_random = agent.select_action(state, n_ac)
            next_state, reward, done, info = env.step(action)

            agent.store(state, action, reward, next_state, done)
            loss = agent.learn()

            state       = next_state
            ep_reward  += reward
            ep_conflicts = max(ep_conflicts, info['conflicts'])
            total_steps += 1

            if loss is not None:
                log['losses'].append(loss)

        log['episode_rewards'].append(ep_reward)
        log['episode_conflicts'].append(ep_conflicts)
        log['epsilons'].append(agent.epsilon())

        if verbose and ep % 10 == 0:
            avg_r   = np.mean(log['episode_rewards'][-20:])
            avg_c   = np.mean(log['episode_conflicts'][-20:])
            avg_l   = np.mean(log['losses'][-100:]) if log['losses'] else 0.0
            elapsed = time.time() - t0
            print(f"Ep {ep:4d}/{n_episodes}  "
                  f"AvgR(20)={avg_r:8.1f}  "
                  f"AvgConflict(20)={avg_c:.2f}  "
                  f"Loss={avg_l:.4f}  "
                  f"ε={agent.epsilon():.3f}  "
                  f"Steps={total_steps}  "
                  f"t={elapsed:.0f}s")

        if ep % checkpoint_every == 0:
            ck_path = f"{checkpoint_dir}/atc_dqn_ep{ep}.pt"
            agent.save(ck_path)
            with open(f"{checkpoint_dir}/training_log.json", 'w') as fh:
                json.dump(log, fh, indent=2)

    print("[Train] Done. Saving final model …")
    agent.save(f"{checkpoint_dir}/atc_dqn_final.pt")
    with open(f"{checkpoint_dir}/training_log.json", 'w') as fh:
        json.dump(log, fh, indent=2)
    return agent, log


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train ATC DQN agent")
    p.add_argument("--episodes",    type=int,  default=500)
    p.add_argument("--device",      type=str,  default="cpu")
    p.add_argument("--adsb-csv",    type=str,  default=None)
    p.add_argument("--synthetic",   action="store_true", default=True)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = p.parse_args()

    train(
        n_episodes       = args.episodes,
        device           = args.device,
        use_synthetic    = args.synthetic,
        adsb_csv         = args.adsb_csv,
        checkpoint_dir   = args.checkpoint_dir,
    )
