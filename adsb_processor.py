"""
adsb_processor.py
-----------------
Ingests live or recorded ADS-B data, cleans it, and discretizes the
continuous state space into a form suitable for tabular / DQN-style RL.

Dependencies (all free / open-source):
    pip install pandas numpy scipy pyModeS  (optional: opensky-api)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import math

# ---------------------------------------------------------------------------
# 1.  Raw data schema
# ---------------------------------------------------------------------------

@dataclass
class ADSBFrame:
    """One ADS-B message decoded from the wire (or CSV row)."""
    timestamp: float          # Unix epoch seconds
    icao24: str               # Aircraft hex identifier
    lat: float                # degrees  [-90, 90]
    lon: float                # degrees  [-180, 180]
    altitude_ft: float        # feet MSL
    speed_kts: float          # ground speed, knots
    heading_deg: float        # true track, 0–360
    vertical_rate_fpm: float  # feet per minute, + = climbing
    callsign: str = ""


# ---------------------------------------------------------------------------
# 2.  Sector / airspace definition
# ---------------------------------------------------------------------------

@dataclass
class Waypoint:
    name: str
    lat: float
    lon: float

@dataclass
class AirspaceSector:
    """
    Rectangular bounding box that defines the controlled sector.
    All positions are projected to a local Cartesian frame (nm).
    """
    lat_min: float;  lat_max: float
    lon_min: float;  lon_max: float
    alt_min_ft: float = 0.0
    alt_max_ft: float = 45_000.0
    waypoints: List[Waypoint] = field(default_factory=list)

    # ---- coordinate helpers -----------------------------------------------
    @property
    def lat_center(self): return (self.lat_min + self.lat_max) / 2
    @property
    def lon_center(self): return (self.lon_min + self.lon_max) / 2

    def to_local_nm(self, lat: float, lon: float) -> Tuple[float, float]:
        """Convert WGS-84 lat/lon to local nautical miles (x=East, y=North)."""
        dlat = lat - self.lat_center
        dlon = lon - self.lon_center
        x_nm = dlon * math.cos(math.radians(self.lat_center)) * 60.0
        y_nm = dlat * 60.0
        return x_nm, y_nm

    def width_nm(self):
        return (self.lon_max - self.lon_min) * math.cos(
            math.radians(self.lat_center)) * 60.0

    def height_nm(self):
        return (self.lat_max - self.lat_min) * 60.0


# ---------------------------------------------------------------------------
# 3.  Discretization
# ---------------------------------------------------------------------------

class StateDiscretizer:
    """
    Maps continuous aircraft state → discrete integer tuple.

    Dimensions discretized:
        x_bin    : lateral position in sector  (N_X bins)
        y_bin    : longitudinal position        (N_Y bins)
        alt_bin  : altitude band               (N_ALT bins)
        spd_bin  : speed band                  (N_SPD bins)
        hdg_bin  : heading octant              (N_HDG bins = 8)
        vr_bin   : vertical rate category      (3: descend/level/climb)
    """

    N_X   = 10
    N_Y   = 10
    N_ALT = 6     # FL0-100, 100-180, 180-240, 240-290, 290-370, 370+
    N_SPD = 5     # <150, 150-250, 250-320, 320-400, >400 kts
    N_HDG = 8     # 45° bins
    N_VR  = 3     # descend / level / climb

    ALT_EDGES = [0, 10_000, 18_000, 24_000, 29_000, 37_000, 999_999]
    SPD_EDGES = [0, 150, 250, 320, 400, 9999]
    VR_THRESH = 200   # fpm threshold for climb/descend vs level

    def __init__(self, sector: AirspaceSector):
        self.sector = sector
        self.x_edges = np.linspace(-sector.width_nm()/2,
                                    sector.width_nm()/2, self.N_X + 1)
        self.y_edges = np.linspace(-sector.height_nm()/2,
                                    sector.height_nm()/2, self.N_Y + 1)

    # ---- individual bin functions -----------------------------------------
    def _bin(self, value, edges) -> int:
        idx = np.searchsorted(edges, value, side='right') - 1
        return int(np.clip(idx, 0, len(edges) - 2))

    def x_bin(self, x_nm):  return self._bin(x_nm, self.x_edges)
    def y_bin(self, y_nm):  return self._bin(y_nm, self.y_edges)
    def alt_bin(self, ft):  return self._bin(ft, self.ALT_EDGES)
    def spd_bin(self, kts): return self._bin(kts, self.SPD_EDGES)

    def hdg_bin(self, deg: float) -> int:
        return int((deg % 360) / 45)   # 0–7

    def vr_bin(self, fpm: float) -> int:
        if fpm > self.VR_THRESH:  return 2   # climb
        if fpm < -self.VR_THRESH: return 0   # descend
        return 1                              # level

    # ---- full state tuple ---------------------------------------------------
    def discretize(self, frame: ADSBFrame) -> Tuple[int, ...]:
        x, y = self.sector.to_local_nm(frame.lat, frame.lon)
        return (
            self.x_bin(x),
            self.y_bin(y),
            self.alt_bin(frame.altitude_ft),
            self.spd_bin(frame.speed_kts),
            self.hdg_bin(frame.heading_deg),
            self.vr_bin(frame.vertical_rate_fpm),
        )

    @property
    def state_space_size(self):
        return (self.N_X * self.N_Y * self.N_ALT *
                self.N_SPD * self.N_HDG * self.N_VR)


# ---------------------------------------------------------------------------
# 4.  ADS-B data loader (CSV / live OpenSky)
# ---------------------------------------------------------------------------

class ADSBLoader:
    """
    Supports:
      - CSV files with columns: timestamp,icao24,lat,lon,
                                altitude_ft,speed_kts,heading_deg,
                                vertical_rate_fpm[,callsign]
      - Synthetic generator for offline testing
    """

    @staticmethod
    def from_csv(path: str) -> List[ADSBFrame]:
        df = pd.read_csv(path)
        frames = []
        for _, row in df.iterrows():
            frames.append(ADSBFrame(
                timestamp=float(row['ts']),
                icao24=str(row['hex']),
                lat=float(row['lat']),
                lon=float(row['lon']),
                altitude_ft=float(row.get('alt_baro', 35000)),
                speed_kts=float(row.get('gs', 280)),
                heading_deg=float(row.get('track', 0)),
                vertical_rate_fpm=float(row.get('baro_rate', 0)),
                callsign=str(row.get('flight', '')),
            ))
        return frames

    @staticmethod
    def synthetic(n_aircraft: int = 8, duration_s: int = 3600,
                  sector: Optional[AirspaceSector] = None) -> List[ADSBFrame]:
        """
        Generate synthetic ADS-B traffic for offline training / testing.
        Aircraft fly straight tracks with small random perturbations.
        """
        if sector is None:
            sector = AirspaceSector(lat_min=40.0, lat_max=41.5,
                                    lon_min=-75.0, lon_max=-73.0)
        rng = np.random.default_rng(42)
        frames: List[ADSBFrame] = []
        dt = 10   # 10-second update interval

        aircraft = []
        for i in range(n_aircraft):
            aircraft.append(dict(
                icao24=f"A{i:05X}",
                callsign=f"UAL{i+100}",
                lat=rng.uniform(sector.lat_min, sector.lat_max),
                lon=rng.uniform(sector.lon_min, sector.lon_max),
                alt=rng.choice([25000, 29000, 33000, 37000,
                                39000, 41000]).astype(float),
                spd=rng.uniform(240, 320),
                hdg=rng.uniform(0, 360),
                vr=0.0,
            ))

        t = 0.0
        while t < duration_s:
            for ac in aircraft:
                # Propagate position
                spd_ms = ac['spd'] * 0.514444
                hdg_rad = math.radians(ac['hdg'])
                dlat = (spd_ms * math.cos(hdg_rad) * dt /
                        111_320)
                dlon = (spd_ms * math.sin(hdg_rad) * dt /
                        (111_320 * math.cos(math.radians(ac['lat']))))
                ac['lat'] = np.clip(ac['lat'] + dlat,
                                    sector.lat_min, sector.lat_max)
                ac['lon'] = np.clip(ac['lon'] + dlon,
                                    sector.lon_min, sector.lon_max)
                ac['hdg'] = (ac['hdg'] + rng.normal(0, 1)) % 360
                ac['vr'] = float(rng.normal(0, 50))

                frames.append(ADSBFrame(
                    timestamp=t,
                    icao24=ac['icao24'],
                    callsign=ac['callsign'],
                    lat=ac['lat'],
                    lon=ac['lon'],
                    altitude_ft=ac['alt'],
                    speed_kts=ac['spd'],
                    heading_deg=ac['hdg'],
                    vertical_rate_fpm=ac['vr'],
                ))
            t += dt
        return frames


# ---------------------------------------------------------------------------
# 5.  Conflict detection utility
# ---------------------------------------------------------------------------

SEPARATION_NM      = 5.0    # ICAO horizontal minimum
SEPARATION_VERT_FT = 1000   # ICAO vertical minimum

AIRPORTS_CSV = "airports.csv"
GROUND_ALTITUDE_FT = 100
AIRPORT_RADIUS_NM = 2.0


def load_airport_coordinates(path: str = AIRPORTS_CSV) -> Dict[str, Tuple[float, float]]:
    """Load ICAO airport coordinates from OurAirports airports.csv."""
    if not Path(path).is_file():
        return {}
    try:
        df = pd.read_csv(path, usecols=["ident", "latitude_deg", "longitude_deg"])
    except Exception:
        return {}
    df = df.dropna(subset=["ident", "latitude_deg", "longitude_deg"])
    df["ident"] = df["ident"].astype(str).str.strip().str.upper()
    coords: Dict[str, Tuple[float, float]] = {}
    for _, row in df.iterrows():
        ident = row["ident"]
        if not ident:
            continue
        try:
            coords[ident] = (float(row["latitude_deg"]), float(row["longitude_deg"]))
        except (TypeError, ValueError):
            continue
    return coords


AIRPORT_COORDINATES = load_airport_coordinates()


def haversine_nm(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in nautical miles."""
    R = 3440.065   # Earth radius in nm
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi/2)**2 +
         math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2)
    return 2 * R * math.asin(math.sqrt(a))


def is_on_ground_at_airport(frame: ADSBFrame,
                             airports=AIRPORT_COORDINATES,
                             radius_nm=AIRPORT_RADIUS_NM,
                             alt_threshold_ft=GROUND_ALTITUDE_FT) -> bool:
    """Return True if a frame is effectively on the ground at a known airport."""
    if frame.altitude_ft > alt_threshold_ft or not airports:
        return False
    for lat, lon in airports.values():
        if haversine_nm(frame.lat, frame.lon, lat, lon) <= radius_nm:
            return True
    return False


def detect_conflicts(frames_at_t: List[ADSBFrame]) -> List[Tuple[str, str, float, float]]:
    """
    Return list of (icao_a, icao_b, horiz_sep_nm, vert_sep_ft) for conflicts.
    """
    conflicts = []
    n = len(frames_at_t)
    for i in range(n):
        for j in range(i+1, n):
            a, b = frames_at_t[i], frames_at_t[j]
            h = haversine_nm(a.lat, a.lon, b.lat, b.lon)
            v = abs(a.altitude_ft - b.altitude_ft)
            if h < SEPARATION_NM and v < SEPARATION_VERT_FT:
                if is_on_ground_at_airport(a) and is_on_ground_at_airport(b):
                    continue
                conflicts.append((a.icao24, b.icao24, h, v))
    return conflicts
