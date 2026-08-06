"""
Spotify Dataset — Exploratory Data Analysis
Author: Maisha Khatoon

Purpose:
    Profile the cleaned dataset, compute correlations, and surface
    the key signals before modelling. Output is console-only —
    no files are written by this script.

Input:  outputs/spotify_clean.csv
"""

import os
import pandas as pd
import numpy as np

# ── PATHS ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

df = pd.read_csv(os.path.join(OUTPUT_DIR, "spotify_clean.csv"))


# ── Helper: ASCII bar chart ───────────────────────────────────────────────────
def print_bar_chart(values, labels, width=50, title=None):
    if title:
        print(title)
    max_val = max(values) if values else 1
    for label, value in zip(labels, values):
        length = int((value / max_val) * width) if max_val else 0
        print(f"{label:>10s} | {'#' * length} {value:,.0f}")
    print()


# ── 1. Streams distribution ───────────────────────────────────────────────────
print("=== STREAMS DISTRIBUTION ===")
print(df["streams"].describe().apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "NaN"))
print(f"Songs > 1B streams : {(df['streams'] > 1e9).sum()}")
print(f"Songs > 500M streams: {(df['streams'] > 5e8).sum()}")
print(f"Songs > 100M streams: {(df['streams'] > 1e8).sum()}")
print(f"Median streams      : {df['streams'].median():,.0f}\n")


# ── 2. Release year ───────────────────────────────────────────────────────────
print("=== RELEASE YEAR ===")
print(f"Year range  : {df['released_year'].min()} – {df['released_year'].max()}")
print(f"Pre-2020    : {(df['released_year'] < 2020).sum()}")
print(f"2020+       : {(df['released_year'] >= 2020).sum()}")
year_counts = df["released_year"].value_counts().sort_index()
print("\nTop 10 years by count:")
print(year_counts.sort_values(ascending=False).head(10))
print()


# ── 3. Correlations with streams ─────────────────────────────────────────────
print("=== CORRELATIONS WITH STREAMS ===")
corr_cols = [
    "in_spotify_playlists", "in_spotify_charts",
    "in_apple_playlists",   "in_apple_charts",
    "in_deezer_playlists",  "in_deezer_charts",
    "in_shazam_charts", "bpm", "artist_count", "released_year",
    "danceability_%", "valence_%",  "energy_%",      "acousticness_%",
    "instrumentalness_%", "liveness_%", "speechiness_%",
]
corrs        = df[corr_cols + ["streams"]].corr()["streams"].drop("streams")
corrs_sorted = corrs.abs().sort_values(ascending=False)
print("Top 10 drivers (by absolute correlation):")
for col in corrs_sorted.head(10).index:
    print(f"  {col:25s}  r = {corrs[col]:+.3f}")
print()


# ── 4. Artist influence ───────────────────────────────────────────────────────
print("=== ARTIST INFLUENCE ===")
solo   = df[df["artist_count"] == 1]["streams"].dropna()
collab = df[df["artist_count"] > 1]["streams"].dropna()
print(f"Solo   n={len(solo):3d}  median={solo.median():,.0f}  mean={solo.mean():,.0f}")
print(f"Collab n={len(collab):3d}  median={collab.median():,.0f}  mean={collab.mean():,.0f}")

artist_streams = []
for _, row in df.iterrows():
    if pd.notna(row["streams"]) and pd.notna(row["artist(s)_name"]):
        for a in str(row["artist(s)_name"]).split(","):
            artist_streams.append({"artist": a.strip(), "streams": row["streams"]})
adf = pd.DataFrame(artist_streams)
top_artists = adf.groupby("artist")["streams"].agg(["count", "sum"]) \
                 .sort_values("sum", ascending=False).head(10)
print("\nTop 10 artists by total streams:")
print(top_artists.assign(sum=lambda x: x["sum"].apply(lambda v: f"{v:,.0f}")))
print()


# ── 5. Temporal trends ────────────────────────────────────────────────────────
print("=== TEMPORAL TRENDS ===")
yr_med = df[df["released_year"] >= 2018].groupby("released_year")["streams"] \
           .agg(["count", "median"]).round(0)
yr_med["median"] = yr_med["median"].apply(
    lambda x: f"{x:,.0f}" if pd.notna(x) else "NaN"
)
print("Median streams by release year (2018+):")
print(yr_med)
print()

month_med = df.groupby("released_month")["streams"].agg(["count", "median"]).round(0)
month_med["median"] = month_med["median"].apply(
    lambda x: f"{x:,.0f}" if pd.notna(x) else "NaN"
)
print("Median streams by release month:")
print(month_med)
print()


# ── 6. Key & Mode ─────────────────────────────────────────────────────────────
print("=== KEY & MODE ===")
mode_grp = df.groupby("mode")["streams"].agg(["count", "median", "mean"]).round(0)
print(mode_grp)
print()
key_grp = df.groupby("key")["streams"].agg(["count", "median"]) \
             .sort_values("count", ascending=False).head(5)
print("Top 5 keys:")
print(key_grp)
