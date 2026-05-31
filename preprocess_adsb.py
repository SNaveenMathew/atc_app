#!/usr/bin/env python3
"""
preprocess_adsb.py  (v3)
─────────────────────────
Converts raw ADS-B CSV to the compact grouped-per-aircraft JSON format
expected by adsb_viz2.html.

Why grouped?
  Raw CSV:     1,646 MB  (one row per ping, every field repeated)
  Grouped JSON:  ~20 MB  (one object per aircraft, waypoints array)
  Browser can parse 20 MB easily; 1.6 GB will crash it.

Key flags
  --decimate N    Keep one position report per N seconds per aircraft (default 5).
                  At 1-second ADS-B intervals, --decimate 5 gives one point every 5 s.
                  Use --decimate 1 to keep every ping (slower, larger file).
  --min-points N  Drop aircraft with fewer than N position reports after decimation (default 3).
  --sector-*      Circular sector filter (default: 40.64°N 73.78°W r=100 nm).
  --no-filter     Skip sector filter.

Usage
  python preprocess_adsb.py ads-b_extra.csv
  python preprocess_adsb.py ads-b_extra.csv --decimate 10
  python preprocess_adsb.py ads-b_extra.csv -o mydata.json
  python preprocess_adsb.py ads-b_extra.csv --no-filter --decimate 30
"""

import argparse, json, math, sys, time
from pathlib import Path

import pandas as pd
import numpy as np

# ── CLI ───────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("input")
ap.add_argument("-o", "--output", default=None)
ap.add_argument("--decimate",    type=int,   default=5,
                help="Keep one point per N seconds per aircraft (default 5)")
ap.add_argument("--min-points",  type=int,   default=3,
                help="Min waypoints after decimation to keep aircraft (default 3)")
ap.add_argument("--sector-lat",  type=float, default=40.64)
ap.add_argument("--sector-lon",  type=float, default=-73.78)
ap.add_argument("--sector-r",    type=float, default=100.0)
ap.add_argument("--no-filter",   action="store_true")
args = ap.parse_args()

t0 = time.time()

# ── 1. Load ───────────────────────────────────────────────────────────────
print(f"[1/6] Loading {args.input} …")
try:
    df = pd.read_csv(args.input, low_memory=False)
except Exception as e:
    sys.exit(f"ERROR: {e}")

df.columns = df.columns.str.strip().str.lower()
# Preserve 't' (ICAO type code e.g. "B738") BEFORE we create our own 't' time-offset column
if "t" in df.columns:
    df.rename(columns={"t": "actype"}, inplace=True)
n_hex = df["hex"].nunique() if "hex" in df.columns else "?"
print(f"      {len(df):,} rows · {n_hex} aircraft")

# ── 2. Timestamp ──────────────────────────────────────────────────────────
print("[2/6] Timestamps …")
if "ts" in df.columns:
    df["ts_raw"] = pd.to_numeric(df["ts"], errors="coerce")
elif "seen" in df.columns:
    df["ts_raw"] = pd.to_numeric(df["seen"], errors="coerce")
    print("      WARNING: no 'ts' column – using 'seen' as relative time")
else:
    df["ts_raw"] = np.arange(len(df), dtype=float)
    print("      WARNING: using row index as time")

df = df.dropna(subset=["ts_raw"])
df["ts_raw"] = df["ts_raw"].astype(float)
if df["ts_raw"].median() > 1e11:          # milliseconds → seconds
    df["ts_raw"] /= 1000.0
    print("      Millisecond timestamps → divided by 1000")

# Normalise to t=0 at first record
t_min = df["ts_raw"].min()
df["t"] = (df["ts_raw"] - t_min).round(1)
df = df[df["t"]<=3600].reset_index(drop = True)# Keep just 1 hour of data
print(f"      Time range: 0 – {df['t'].max():.0f} s  "
      f"({df['t'].max()/3600:.2f} h)  "
      f"epoch base = {t_min:.0f}")

# ── 3. Positions ──────────────────────────────────────────────────────────
print("[3/6] Positions …")
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
before = len(df)
df = df.dropna(subset=["lat", "lon"])
print(f"      Dropped {before - len(df):,} rows with missing lat/lon")

# ── 4. Sector filter ──────────────────────────────────────────────────────
if not args.no_filter:
    slat, slon, sr = args.sector_lat, args.sector_lon, args.sector_r
    dlat = (df["lat"] - slat) * 60.0
    dlon = (df["lon"] - slon) * 60.0 * math.cos(math.radians(slat))
    before2 = len(df)
    df = df[np.sqrt(dlat**2 + dlon**2) <= sr].copy()
    print(f"[4/6] Sector filter: {len(df):,}/{before2:,} rows within "
          f"{sr} nm of ({slat},{slon})")
else:
    df = df.copy()
    print("[4/6] Sector filter: skipped")

# ── 5. Derive measurement columns ─────────────────────────────────────────
print("[5/6] Deriving columns …")

df["icao24"] = (df["hex"].fillna("UNKN").astype(str).str.strip().str.lower()
                if "hex" in df.columns else "UNKN")

# callsign: first token of 'flight' field ("AAL123 N12345" → "AAL123")
if "flight" in df.columns:
    df["callsign"] = (df["flight"].fillna("").astype(str)
                        .str.strip().str.split().str[0].fillna(""))
else:
    df["callsign"] = ""
blank = df["callsign"] == ""
if blank.any() and "r" in df.columns:
    df.loc[blank, "callsign"] = df.loc[blank, "r"].fillna("").astype(str).str.strip()

# altitude
if "alt_baro" in df.columns:
    alt = pd.to_numeric(df["alt_baro"], errors="coerce")
    alt[df["alt_baro"].astype(str).str.strip().str.lower() == "ground"] = 0.0
elif "alt_geom" in df.columns:
    alt = pd.to_numeric(df["alt_geom"], errors="coerce")
else:
    alt = pd.Series(0.0, index=df.index)
df["alt"] = alt.fillna(0).clip(lower=0).round(0)

# speed
df["spd"] = (pd.to_numeric(df["gs"], errors="coerce").fillna(0).clip(lower=0).round(1)
             if "gs" in df.columns else 0.0)

# heading  (true_heading > mag_heading > track > calc_track)
hdg = pd.Series(np.nan, index=df.index)
for col in ["true_heading", "mag_heading", "track", "calc_track"]:
    if col in df.columns:
        hdg = hdg.where(hdg.notna(), pd.to_numeric(df[col], errors="coerce"))
df["hdg"] = (hdg.fillna(0) % 360).round(1)

# vertical rate
vr = pd.Series(0.0, index=df.index)
for col in ["baro_rate", "geom_rate"]:
    if col in df.columns:
        cand = pd.to_numeric(df[col], errors="coerce")
        vr = vr.where(cand.isna(), cand)
        break
df["vr"] = vr.fillna(0).round(0)

# optional extras kept per-aircraft (first value wins)
META_COLS = {
    "actype": "aircraft_type", "desc": "description",
    "ownop": "operator",       "year": "year",
    "r":     "registration",   "category": "category",
}

# ── 6. Group by aircraft → compact waypoints ──────────────────────────────
print("[6/6] Grouping and decimating …")

WP_COLS = ["t", "lat", "lon", "alt", "spd", "hdg", "vr"]
aircraft_out = []
groups = df.sort_values("t").groupby("icao24", sort=False)

skipped = 0
for icao24, grp in groups:
    grp = grp.sort_values("t").reset_index(drop=True)

    # Decimate: keep one point per --decimate seconds
    if args.decimate > 1:
        grp["bucket"] = (grp["t"] // args.decimate).astype(int)
        grp = grp.groupby("bucket", as_index=False).first()

    if len(grp) < args.min_points:
        skipped += 1
        continue

    # Build waypoints list — only the 7 numeric fields; compact rounding
    waypoints = []
    for _, row in grp[WP_COLS].iterrows():
        wp = {
            "t":   round(float(row["t"]),   1),
            "lat": round(float(row["lat"]), 6),
            "lon": round(float(row["lon"]), 6),
            "alt": int(row["alt"]),
            "spd": round(float(row["spd"]), 1),
            "hdg": round(float(row["hdg"]), 1),
            "vr":  int(row["vr"]),
        }
        waypoints.append(wp)

    # Per-aircraft metadata (first non-null value in each meta column)
    meta = {"id": icao24, "callsign": str(grp["callsign"].iloc[0])}
    for src, dst in META_COLS.items():
        if src in grp.columns:
            val = grp[src].dropna()
            if len(val):
                meta[dst] = val.iloc[0]

    meta["actual"] = waypoints
    aircraft_out.append(meta)

print(f"      {len(aircraft_out)} aircraft kept · {skipped} skipped "
      f"(< {args.min_points} pts after {args.decimate}s decimation)")
total_wps = sum(len(a["actual"]) for a in aircraft_out)
print(f"      Total waypoints: {total_wps:,}")

# ── Write ─────────────────────────────────────────────────────────────────
inp_path = Path(args.input)
out_path  = Path(args.output) if args.output else inp_path.with_name(inp_path.stem + "_viz.json")

def clean_val(v):
    """Replace NaN/inf/numpy scalars with JSON-safe values."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    # numpy int/float → native Python
    if hasattr(v, 'item'):
        return clean_val(v.item())
    return v

def clean_obj(obj):
    if isinstance(obj, dict):
        return {k: clean_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_obj(i) for i in obj]
    return clean_val(obj)

print(f"      Writing {out_path} …")
with open(out_path, "w") as fh:
    json.dump(clean_obj(aircraft_out), fh, allow_nan=False, separators=(",", ":"))

size_mb = out_path.stat().st_size / 1024 / 1024
elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s  →  {out_path}  ({size_mb:.1f} MB)")
print(f"\nLoad into adsb_viz2.html:")
print(f"  1. Open adsb_viz2.html in your browser")
print(f"  2. Click 'Import Data'  →  'Upload File'  →  select  {out_path.name}")
print(f"  3. Click 'Load Data'")
print(f"\nIf the file is still >100 MB, run again with a higher decimate value:")
print(f"  python preprocess_adsb.py {args.input} --decimate 30")
