"""
rl_agent.py
-----------
Deep Q-Network (DQN) agent for Air Traffic Control.

Architecture
────────────
  State  : Concatenated feature vector for ALL aircraft in sector
           (each aircraft → 11-D vector; padded to MAX_AIRCRAFT)
  Action : Per-aircraft ATC instruction (heading change / altitude change /
           speed change / hold)
  Reward : Composite function penalising conflicts and fuel burn,
           rewarding separation and sector throughput

All dependencies are free / open-source:
    pip install torch numpy
"""

import math
import random
import collections
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from adsb_processor import (
    ADSBFrame, AirspaceSector, StateDiscretizer,
    detect_conflicts, SEPARATION_NM, SEPARATION_VERT_FT,
    haversine_nm,
)

# ============================================================
# 1.  ACTION SPACE
# ============================================================

# Each action is an instruction to ONE aircraft.
# We encode it as (aircraft_index, instruction_type, magnitude_index)

HDG_DELTAS  = [-30, -15, -5, 0, +5, +15, +30]   # degrees
ALT_DELTAS  = [-2000, -1000, 0, +1000, +2000]    # feet
SPD_DELTAS  = [-20, -10, 0, +10, +20]            # knots

N_HDG_ACTS  = len(HDG_DELTAS)
N_ALT_ACTS  = len(ALT_DELTAS)
N_SPD_ACTS  = len(SPD_DELTAS)
N_INSTR     = N_HDG_ACTS + N_ALT_ACTS + N_SPD_ACTS  # 17 per aircraft

MAX_AIRCRAFT = 12   # pad state to this size

# Total Q-head output = MAX_AIRCRAFT × N_INSTR
N_ACTIONS = MAX_AIRCRAFT * N_INSTR


# ============================================================
# 2.  STATE ENCODING
# ============================================================

AIRCRAFT_FEATURE_DIM = 11   # see encode_aircraft()

def encode_aircraft(frame: ADSBFrame, sector: AirspaceSector,
                    disc: StateDiscretizer) -> np.ndarray:
    """
    Encode one aircraft into an 11-D normalised feature vector:
        [x_norm, y_norm, alt_norm, spd_norm,
         sin(hdg), cos(hdg),
         vr_norm,
         x_bin/N_X, y_bin/N_Y, alt_bin/N_ALT, spd_bin/N_SPD]
    """
    x, y = sector.to_local_nm(frame.lat, frame.lon)
    w, h = sector.width_nm(), sector.height_nm()

    x_n   = np.clip(x / (w/2), -1, 1)
    y_n   = np.clip(y / (h/2), -1, 1)
    alt_n = np.clip((frame.altitude_ft - 20_000) / 25_000, -1, 1)
    spd_n = np.clip((frame.speed_kts - 275) / 150, -1, 1)
    hdg_r = math.radians(frame.heading_deg)
    vr_n  = np.clip(frame.vertical_rate_fpm / 3000, -1, 1)

    d = disc.discretize(frame)
    return np.array([
        x_n, y_n, alt_n, spd_n,
        math.sin(hdg_r), math.cos(hdg_r),
        vr_n,
        d[0]/disc.N_X, d[1]/disc.N_Y,
        d[2]/disc.N_ALT, d[3]/disc.N_SPD,
    ], dtype=np.float32)


def encode_state(frames: List[ADSBFrame], sector: AirspaceSector,
                 disc: StateDiscretizer) -> np.ndarray:
    """
    Encode all aircraft in sector to a flat vector of length
    MAX_AIRCRAFT × AIRCRAFT_FEATURE_DIM.  Extras are zero-padded.
    """
    vec = np.zeros(MAX_AIRCRAFT * AIRCRAFT_FEATURE_DIM, dtype=np.float32)
    for i, f in enumerate(frames[:MAX_AIRCRAFT]):
        start = i * AIRCRAFT_FEATURE_DIM
        vec[start: start + AIRCRAFT_FEATURE_DIM] = encode_aircraft(
            f, sector, disc)
    return vec


# ============================================================
# 3.  REWARD FUNCTION
# ============================================================

def compute_reward(
    prev_frames: List[ADSBFrame],
    curr_frames: List[ADSBFrame],
    sector: AirspaceSector,
    action_taken: int,
) -> float:
    """
    Composite reward:

    R = R_safety + R_efficiency + R_workload + R_exit

    R_safety    : −100 per active conflict pair (LOS)
                  −10  per pair within 1.5× separation minima (alert)
    R_efficiency: −0.05 per aircraft per step (fuel/time cost)
                  +1.0  per aircraft that exits sector on its planned route
    R_workload  : −0.5  per non-HOLD instruction (ATC cognitive load proxy)
    R_exit      : +5.0  per aircraft that cleanly exits sector this step
    """
    r = 0.0

    # --- safety --------------------------------------------------------------
    conflicts = detect_conflicts(curr_frames)
    r -= 100.0 * len(conflicts)

    # Alert zone (1.5× minima)
    for i, a in enumerate(curr_frames):
        for b in curr_frames[i+1:]:
            h = haversine_nm(a.lat, a.lon, b.lat, b.lon)
            v = abs(a.altitude_ft - b.altitude_ft)
            if (h < SEPARATION_NM * 1.5 and
                    v < SEPARATION_VERT_FT * 1.5 and
                    (a.icao24, b.icao24) not in
                    [(c[0], c[1]) for c in conflicts]):
                r -= 10.0

    # --- efficiency ----------------------------------------------------------
    r -= 0.05 * len(curr_frames)

    # --- workload ------------------------------------------------------------
    # Decode whether action was a HOLD (magnitude index == centre of range)
    instr_idx = action_taken % N_INSTR
    if instr_idx != 3:   # heading hold index (delta=0)
        r -= 0.5

    # --- exit bonus ----------------------------------------------------------
    prev_ids = {f.icao24 for f in prev_frames}
    curr_ids = {f.icao24 for f in curr_frames}
    exited = prev_ids - curr_ids
    r += 5.0 * len(exited)

    return float(r)


# ============================================================
# 4.  REPLAY BUFFER
# ============================================================

Transition = collections.namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)


# ============================================================
# 5.  DQN NETWORK  (forward pass defined here)
# ============================================================

class DQNNetwork(nn.Module):
    """
    Dueling Double DQN with noisy linear layers option.

    Input  : flat state vector (MAX_AIRCRAFT × AIRCRAFT_FEATURE_DIM)
    Output : Q-values for each of N_ACTIONS discrete actions

    Forward pass:
        x → shared_fc → advantage_stream + value_stream
        Q(s,a) = V(s) + A(s,a) − mean(A(s,·))
    """

    def __init__(self, state_dim: int, n_actions: int,
                 hidden: int = 256):
        super().__init__()
        self.state_dim  = state_dim
        self.n_actions  = n_actions

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )

        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(hidden // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden // 2, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, state_dim) float tensor

        Returns:
            q: (batch, n_actions) Q-value estimates
        """
        features  = self.shared(x)                         # (B, hidden/2)
        value     = self.value_stream(features)             # (B, 1)
        advantage = self.advantage_stream(features)         # (B, N_ACTIONS)

        # Dueling aggregation:  Q = V + A - mean(A)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q


# ============================================================
# 6.  DQN AGENT  (epsilon-greedy + backward pass)
# ============================================================

class ATCDQNAgent:
    """
    Double DQN with:
      - ε-greedy exploration
      - Prioritised experience replay (uniform for simplicity)
      - Target network soft-updates (Polyak)
      - Huber loss for stable gradients
    """

    GAMMA          = 0.99
    LR             = 1e-4
    BATCH_SIZE     = 64
    MIN_REPLAY     = 1_000
    TARGET_UPDATE  = 200    # steps between hard target syncs
    EPS_START      = 1.0
    EPS_END        = 0.05
    EPS_DECAY      = 10_000   # steps over which to anneal
    CLIP_GRAD      = 10.0

    def __init__(self, sector: AirspaceSector, device: str = 'cpu'):
        self.sector = sector
        self.disc   = StateDiscretizer(sector)
        self.device = torch.device(device)

        state_dim = MAX_AIRCRAFT * AIRCRAFT_FEATURE_DIM

        self.policy_net = DQNNetwork(state_dim, N_ACTIONS).to(self.device)
        self.target_net = DQNNetwork(state_dim, N_ACTIONS).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LR)
        self.replay    = ReplayBuffer()

        self.steps_done = 0
        self.losses: List[float] = []

    # ------------------------------------------------------------------ #
    # 6a. ε-GREEDY DECISION MAKING
    # ------------------------------------------------------------------ #
    def epsilon(self) -> float:
        """Current exploration rate (exponential decay)."""
        return (self.EPS_END +
                (self.EPS_START - self.EPS_END) *
                math.exp(-self.steps_done / self.EPS_DECAY))

    def select_action(self, state: np.ndarray,
                      n_aircraft: int) -> Tuple[int, bool]:
        """
        ε-greedy action selection.

        Returns:
            action_idx : flat index into (aircraft × instruction)
            is_random  : True if exploration, False if exploitation
        """
        eps = self.epsilon()
        self.steps_done += 1

        # Mask actions for non-existent aircraft slots
        valid_actions = list(range(n_aircraft * N_INSTR))

        if random.random() < eps:
            # EXPLORE: uniform random over valid actions
            return random.choice(valid_actions), True
        else:
            # EXPLOIT: argmax Q over valid actions
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
                q = self.policy_net(s).squeeze(0)     # (N_ACTIONS,)
                # Mask out actions for empty aircraft slots
                mask = torch.full((N_ACTIONS,), float('-inf'),
                                  device=self.device)
                mask[valid_actions] = 0.0
                q = q + mask
                return int(q.argmax().item()), False

    # ------------------------------------------------------------------ #
    # 6b. STORE TRANSITION
    # ------------------------------------------------------------------ #
    def store(self, state, action, reward, next_state, done):
        self.replay.push(
            torch.tensor(state,      dtype=torch.float32),
            torch.tensor([action],   dtype=torch.long),
            torch.tensor([reward],   dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor([done],     dtype=torch.bool),
        )

    # ------------------------------------------------------------------ #
    # 6c. BACKWARD PASS (LEARNING STEP)
    # ------------------------------------------------------------------ #
    def learn(self) -> Optional[float]:
        """
        One gradient-descent step using a sampled mini-batch.

        Algorithm  : Double DQN
            target = r + γ · Q_target(s', argmax_a Q_policy(s', a))

        Loss       : Huber (smooth L1) — less sensitive to outliers
                     than MSE, stabilising early training.

        Returns average loss for logging, or None if buffer too small.
        """
        if len(self.replay) < self.MIN_REPLAY:
            return None

        batch = self.replay.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*batch))

        states      = torch.stack(batch.state).to(self.device)       # (B, S)
        actions     = torch.stack(batch.action).to(self.device)      # (B, 1)
        rewards     = torch.stack(batch.reward).to(self.device)      # (B, 1)
        next_states = torch.stack(batch.next_state).to(self.device)  # (B, S)
        dones       = torch.stack(batch.done).to(self.device)        # (B, 1)

        # ---- FORWARD PASS on policy net ----------------------------------
        q_values = self.policy_net(states)            # (B, N_ACTIONS)
        q_sa     = q_values.gather(1, actions)        # (B, 1)  — chosen Q

        # ---- DOUBLE DQN TARGET ------------------------------------------
        with torch.no_grad():
            # Step 1: policy net selects best next action
            next_q_policy = self.policy_net(next_states)          # (B, N)
            best_next_acts = next_q_policy.argmax(dim=1, keepdim=True)  # (B,1)

            # Step 2: target net evaluates that action
            next_q_target  = self.target_net(next_states)         # (B, N)
            q_next = next_q_target.gather(1, best_next_acts)      # (B, 1)

            # Bellman target
            target = rewards + self.GAMMA * q_next * (~dones)     # (B, 1)

        # ---- LOSS & BACKWARD PASS ----------------------------------------
        loss = F.smooth_l1_loss(q_sa, target)   # Huber loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.CLIP_GRAD)
        self.optimizer.step()

        # ---- SOFT TARGET UPDATE (Polyak τ = 0.005) -----------------------
        tau = 0.005
        for tp, pp in zip(self.target_net.parameters(),
                          self.policy_net.parameters()):
            tp.data.copy_(tau * pp.data + (1.0 - tau) * tp.data)

        loss_val = float(loss.item())
        self.losses.append(loss_val)
        return loss_val

    # ------------------------------------------------------------------ #
    # 6d. ACTION DECODING
    # ------------------------------------------------------------------ #
    @staticmethod
    def decode_action(action_idx: int) -> Dict:
        """
        Convert flat action index → human-readable ATC instruction.

        Returns dict with keys: aircraft_slot, instruction_type,
                                delta_value, description
        """
        aircraft_slot = action_idx // N_INSTR
        instr_idx     = action_idx %  N_INSTR

        if instr_idx < N_HDG_ACTS:
            delta = HDG_DELTAS[instr_idx]
            return dict(aircraft_slot=aircraft_slot,
                        instruction_type='HEADING',
                        delta_value=delta,
                        description=f"Turn {'left' if delta<0 else 'right'} {abs(delta)}°"
                                    if delta != 0 else "Maintain heading")
        instr_idx -= N_HDG_ACTS
        if instr_idx < N_ALT_ACTS:
            delta = ALT_DELTAS[instr_idx]
            return dict(aircraft_slot=aircraft_slot,
                        instruction_type='ALTITUDE',
                        delta_value=delta,
                        description=f"{'Climb' if delta>0 else 'Descend'} {abs(delta)//100} FL"
                                    if delta != 0 else "Maintain altitude")
        instr_idx -= N_ALT_ACTS
        delta = SPD_DELTAS[instr_idx]
        return dict(aircraft_slot=aircraft_slot,
                    instruction_type='SPEED',
                    delta_value=delta,
                    description=f"{'Increase' if delta>0 else 'Decrease'} speed {abs(delta)} kts"
                                if delta != 0 else "Maintain speed")

    # ------------------------------------------------------------------ #
    # 6e. APPLY ACTION TO AIRCRAFT LIST
    # ------------------------------------------------------------------ #
    @staticmethod
    def apply_action(frames: List[ADSBFrame],
                     action_idx: int) -> List[ADSBFrame]:
        """
        Apply decoded ATC instruction to the target aircraft.
        Returns a new list of ADSBFrame (immutable-style).
        """
        import copy
        instruction = ATCDQNAgent.decode_action(action_idx)
        slot        = instruction['aircraft_slot']

        new_frames = copy.deepcopy(frames)
        if slot >= len(new_frames):
            return new_frames

        ac = new_frames[slot]
        t  = instruction['instruction_type']
        d  = instruction['delta_value']

        if t == 'HEADING':
            ac.heading_deg = (ac.heading_deg + d) % 360
        elif t == 'ALTITUDE':
            ac.altitude_ft = float(np.clip(
                ac.altitude_ft + d, 1000, 45_000))
        elif t == 'SPEED':
            ac.speed_kts = float(np.clip(
                ac.speed_kts + d, 100, 500))

        return new_frames

    # ------------------------------------------------------------------ #
    # 6f. SAVE / LOAD
    # ------------------------------------------------------------------ #
    def save(self, path: str):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'steps_done': self.steps_done,
        }, path)
        print(f"[ATCDQNAgent] Saved → {path}")

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ck['policy_net'])
        self.target_net.load_state_dict(ck['target_net'])
        self.optimizer.load_state_dict(ck['optimizer'])
        self.steps_done = ck['steps_done']
        print(f"[ATCDQNAgent] Loaded ← {path}")
