"""
Spotify Dataset — Cleaning Pipeline
Author: Maisha Khatoon

Purpose:
    Clean both sheets from the raw Excel file, identify and normalise
    the merge key, resolve six data quality issues, and produce a
    single analysis-ready dataset.

Data manipulation challenges resolved:
    1. Merge key (Song_Id) formatted differently across sheets:
       - Song_Deatils: comma-formatted text  e.g. "4,137"
       - Node_Deatils: zero-padded text      e.g. "0000003788"
    2. ~136 duplicate Song_Ids in Songs sheet, ~138 in Nodes sheet
    3. released_month stored as mixed numeric (1-12) AND text ("January"-"December")
    4. Row 570: audio feature dump pasted into the streams column (corrupted)
    5. 88 rows with broken UTF-8 encoding in artist/track names
    6. Missing values: 80 in 'key', 44 in audio features, 1 in streams

Input:  data/spotify_data.xlsx
Output: outputs/spotify_clean.csv
"""

import os
import pandas as pd
import numpy as np

# ── PATHS ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SRC = os.path.join(DATA_DIR, "spotify_data.xlsx")
OUT = os.path.join(OUTPUT_DIR, "spotify_clean.csv")

if not os.path.exists(SRC):
    alt_src = os.environ.get("SPOTIFY_DATA_PATH", "").strip()
    if alt_src and os.path.exists(alt_src):
        SRC = alt_src
    else:
        raise FileNotFoundError(
            f"Missing input workbook: {SRC}\n"
            f"Place spotify_data.xlsx in {DATA_DIR} or set the SPOTIFY_DATA_PATH environment variable."
        )


# ── 1. Load ───────────────────────────────────────────────────────────────────
songs = pd.read_excel(SRC, sheet_name="Song_Deatils", dtype=str)
nodes = pd.read_excel(SRC, sheet_name="Node_Deatils", dtype=str)
print(f"[load] Songs: {songs.shape}, Nodes: {nodes.shape}")


# ── 2. Normalise merge key ────────────────────────────────────────────────────
def normalize_id(x):
    if pd.isna(x):
        return None
    s = str(x).replace(",", "").strip().lstrip("0")
    return s if s else "0"

songs["Song_Id"] = songs["Song_Id"].apply(normalize_id)
nodes["Song_Id"] = nodes["Song_Id"].apply(normalize_id)
print(f"[merge key] Songs unique: {songs['Song_Id'].nunique()}, "
      f"Nodes unique: {nodes['Song_Id'].nunique()}")


# ── 3. De-duplicate ───────────────────────────────────────────────────────────
songs = songs.drop_duplicates(subset="Song_Id", keep="first").reset_index(drop=True)
nodes = nodes.drop_duplicates(subset="Song_Id", keep="first").reset_index(drop=True)
print(f"[dedupe] Songs: {songs.shape}, Nodes: {nodes.shape}")


# ── 4. Standardise released_month ────────────────────────────────────────────
month_map = {
    "January": 1, "February": 2, "March": 3,  "April": 4,
    "May": 5,     "June": 6,     "July": 7,    "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

def norm_month(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s in month_map:
        return month_map[s]
    try:
        v = int(s)
        return v if 1 <= v <= 12 else None
    except ValueError:
        return None

songs["released_month"] = songs["released_month"].apply(norm_month)


# ── 5. Convert numeric columns ────────────────────────────────────────────────
numeric_cols = [
    "artist_count", "released_year", "released_day",
    "in_spotify_playlists", "in_spotify_charts", "streams",
    "in_apple_playlists",   "in_apple_charts",
    "in_deezer_playlists",  "in_deezer_charts", "in_shazam_charts",
    "bpm",
]
for c in numeric_cols:
    songs[c] = pd.to_numeric(
        songs[c].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
# Row 570 streams corrupted — now NaN after coerce. Documented above.

node_numeric = [
    "danceability_%", "valence_%",  "energy_%",
    "acousticness_%", "instrumentalness_%", "liveness_%", "speechiness_%",
]
for c in node_numeric:
    nodes[c] = pd.to_numeric(nodes[c], errors="coerce")


# ── 6. Best-effort fix for encoding errors ────────────────────────────────────
def clean_encoding(s):
    if pd.isna(s):
        return s
    return str(s).replace("ï¿½", "").replace("  ", " ").strip()

songs["track_name"]      = songs["track_name"].apply(clean_encoding)
songs["artist(s)_name"]  = songs["artist(s)_name"].apply(clean_encoding)


# ── 7. Build release_date ─────────────────────────────────────────────────────
def build_date(row):
    y, m, d = row["released_year"], row["released_month"], row["released_day"]
    try:
        return pd.Timestamp(year=int(y), month=int(m), day=int(d))
    except (ValueError, TypeError):
        return pd.NaT

songs["release_date"] = songs.apply(build_date, axis=1)


# ── 8. Merge ──────────────────────────────────────────────────────────────────
merged = songs.merge(nodes, on="Song_Id", how="inner")
print(f"[merge] Final shape: {merged.shape}")


# ── 9. Summary + save ─────────────────────────────────────────────────────────
print("\n=== Missing values after cleaning ===")
print(merged.isna().sum()[merged.isna().sum() > 0])

merged.to_csv(OUT, index=False)
print(f"\n[save] {OUT}")
print(f"[save] Final rows: {len(merged)}, columns: {merged.shape[1]}")
