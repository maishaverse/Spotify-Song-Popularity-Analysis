"""
Spotify Dataset — Chart Data Preparation & Plot Generation
Author: Maisha Khatoon

Purpose:
    Aggregate the cleaned dataset into chart-ready structures,
    save them as chart_data.json, and generate matplotlib PNG charts.
    An optional interactive chart-viewer dialog is shown when running
    this script directly (suppressed in pipeline mode via AUTOMATION_MODE).

Input:  outputs/spotify_clean.csv
        outputs/rf_importance.csv
Output: outputs/chart_data.json
        outputs/*.png  (6 chart files)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── PATHS ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load & filter ─────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(OUTPUT_DIR, "spotify_clean.csv"))
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

df_clean = df[df["released_year"] <= 2023].copy()
ref_date = pd.Timestamp("2024-01-01")
df_clean["days_since_release"] = (
    (ref_date - df_clean["release_date"]).dt.days.clip(lower=1)
)
df_clean["streams_per_day"] = df_clean["streams"] / df_clean["days_since_release"]

data = {}

# ── Overview stats ────────────────────────────────────────────────────────────
data["overview"] = {
    "n_songs":         int(len(df_clean)),
    "n_artists":       int(df_clean["artist(s)_name"].nunique()),
    "year_min":        int(df_clean["released_year"].min()),
    "year_max":        int(df_clean["released_year"].max()),
    "median_streams_M": round(df_clean["streams"].median() / 1e6, 0),
    "total_streams_B":  round(df_clean["streams"].sum() / 1e9, 1),
    "songs_over_1B":   int((df_clean["streams"] > 1e9).sum()),
    "songs_over_100M": int((df_clean["streams"] > 1e8).sum()),
}

# ── Playlist quartile vs streams ──────────────────────────────────────────────
df_clean["playlist_q"] = pd.qcut(
    df_clean["in_spotify_playlists"], 4,
    labels=["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"]
)
pq = df_clean.groupby("playlist_q", observed=True)["streams"].median()
data["playlist_quartile"] = {
    "labels": [str(x) for x in pq.index.tolist()],
    "values": [round(v / 1e6, 0) for v in pq.values],
}

# ── RF feature importance ─────────────────────────────────────────────────────
imp = pd.read_csv(os.path.join(OUTPUT_DIR, "rf_importance.csv")).head(10)
display_names = {
    "in_spotify_playlists": "Spotify Playlists",
    "log_days":             "Time Since Release",
    "in_deezer_playlists":  "Deezer Playlists",
    "in_spotify_charts":    "Spotify Charts",
    "in_deezer_charts":     "Deezer Charts",
    "in_apple_playlists":   "Apple Playlists",
    "in_apple_charts":      "Apple Charts",
    "danceability_%":       "Danceability",
    "bpm":                  "BPM",
    "valence_%":            "Valence",
    "artist_count":         "Artist Count",
    "is_major":             "Major Mode",
}
data["feature_importance"] = {
    "labels": [display_names.get(f, f) for f in imp["feature"].tolist()][::-1],
    "values": [round(v, 3)            for v in imp["importance"].tolist()][::-1],
}

# ── Cross-platform reach ──────────────────────────────────────────────────────
df_clean["platforms"] = (
    (df_clean["in_spotify_playlists"] > 0).astype(int)
    + (df_clean["in_apple_playlists"]  > 0).astype(int)
    + (df_clean["in_deezer_playlists"] > 0).astype(int)
)
plat = df_clean.groupby("platforms")["streams"].median()
data["platform_reach"] = {
    "labels": [f"{i} Platform" + ("s" if i != 1 else "") for i in plat.index],
    "values": [round(v / 1e6, 0) for v in plat.values],
}

# ── Streams by release year ───────────────────────────────────────────────────
yr = df_clean[df_clean["released_year"] >= 2015].groupby("released_year")["streams"].median()
data["streams_by_year"] = {
    "labels": [str(int(y)) for y in yr.index],
    "values": [round(v / 1e6, 0) for v in yr.values],
}

# ── Streams by release month ──────────────────────────────────────────────────
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
mn = df_clean.groupby("released_month")["streams"].median()
data["streams_by_month"] = {
    "labels": [month_names[int(m) - 1] for m in mn.index],
    "values": [round(v / 1e6, 0) for v in mn.values],
}

# ── Solo vs collab ────────────────────────────────────────────────────────────
data["solo_vs_collab"] = {
    "labels": ["Solo Artist", "Collaboration"],
    "values": [
        round(df_clean[df_clean["artist_count"] == 1]["streams"].median() / 1e6, 0),
        round(df_clean[df_clean["artist_count"] >  1]["streams"].median() / 1e6, 0),
    ],
}

# ── Top 10 artists by total streams ──────────────────────────────────────────
artist_rows = []
for _, row in df_clean.iterrows():
    if pd.notna(row["streams"]) and pd.notna(row["artist(s)_name"]):
        for a in str(row["artist(s)_name"]).split(","):
            artist_rows.append({"artist": a.strip(), "streams": row["streams"]})
adf   = pd.DataFrame(artist_rows)
top_a = adf.groupby("artist")["streams"].sum().sort_values(ascending=False).head(10)
data["top_artists"] = {
    "labels": top_a.index.tolist()[::-1],
    "values": [round(v / 1e9, 1) for v in top_a.values][::-1],
}

# ── Top trending songs (streams per day, last 18 months) ─────────────────────
trendy     = df_clean[
    (df_clean["days_since_release"] > 0) &
    (df_clean["days_since_release"] <= 540)
].dropna(subset=["streams_per_day"])
top_trendy = trendy.nlargest(10, "streams_per_day")
data["top_trendy"] = [
    {
        "track":              str(r["track_name"])[:35],
        "artist":             str(r["artist(s)_name"])[:30],
        "streams_per_day_M":  round(r["streams_per_day"] / 1e6, 2),
        "total_streams_M":    round(r["streams"] / 1e6, 0),
    }
    for _, r in top_trendy.iterrows()
]

# ── Audio feature correlations ────────────────────────────────────────────────
audio_cols = [
    "danceability_%", "valence_%",  "energy_%",     "acousticness_%",
    "instrumentalness_%", "liveness_%", "speechiness_%",
]
ac = (
    df_clean[audio_cols + ["streams"]]
    .corr()["streams"].drop("streams")
    .abs().sort_values(ascending=True)
)
data["audio_correlations"] = {
    "labels": [c.replace("_%", "").title() for c in ac.index],
    "values": [round(v, 3) for v in ac.values],
}

# ── Model results (from 03_model.py run) ─────────────────────────────────────
data["model"] = {
    "lr_r2":         0.696,
    "rf_r2":         0.853,
    "training_rows": 558,
}

# ── Data quality summary ──────────────────────────────────────────────────────
data["data_challenges"] = {
    "raw_songs_rows":      949,
    "raw_nodes_rows":      954,
    "duplicates_songs":    136,
    "duplicates_nodes":    138,
    "encoding_errors":     88,
    "month_format_issues": True,
    "corrupted_streams_row": 1,
    "final_rows":          809,
    "missing_key":         80,
    "missing_audio_features": 44,
}

# ── Save JSON ─────────────────────────────────────────────────────────────────
json_path = os.path.join(OUTPUT_DIR, "chart_data.json")
with open(json_path, "w") as f:
    json.dump(data, f, indent=2)

print("Chart data saved.")
print(f"  Q4 vs Q1 playlist median : {data['playlist_quartile']['values'][3]}M vs "
      f"{data['playlist_quartile']['values'][0]}M")
print(f"  Multi-platform vs single : {data['platform_reach']['values'][-1]}M vs "
      f"{data['platform_reach']['values'][0]}M")
print(f"  Top trending song        : {data['top_trendy'][0]['track']} — "
      f"{data['top_trendy'][0]['streams_per_day_M']}M/day")


# ── Generate matplotlib PNGs ─────────────────────────────────────────────────
def bar(labels, values, title, ylabel="", rotate=False, fname=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(values)), values, color="C0")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45 if rotate else 0, ha="right")
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150)
    else:
        plt.show()
    plt.close(fig)

def barh(labels, values, title, xlabel="", fname=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(range(len(values)), values, color="C0")
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150)
    else:
        plt.show()
    plt.close(fig)

def line(labels, values, title, ylabel="", rotate=False, fname=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(len(values)), values, marker="o", linewidth=2, color="C1")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45 if rotate else 0, ha="right")
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150)
    else:
        plt.show()
    plt.close(fig)

def area(labels, values, title, ylabel="", rotate=False, fname=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(values))
    ax.fill_between(x, values, alpha=0.3, color="C2")
    ax.plot(x, values, marker="o", color="C2", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if rotate else 0, ha="right")
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150)
    else:
        plt.show()
    plt.close(fig)

def pie(labels, values, title, fname=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=140,
           textprops={"fontsize": 9})
    ax.set_title(title)
    ax.axis("equal")
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150)
    else:
        plt.show()
    plt.close(fig)


pq_d = data["playlist_quartile"]
bar(pq_d["labels"], pq_d["values"],
    "Median Streams by Spotify Playlist Quartile", "Streams (M)",
    rotate=True, fname=os.path.join(OUTPUT_DIR, "playlist_quartile.png"))

fi_d = data["feature_importance"]
barh(fi_d["labels"], fi_d["values"],
     "Feature Importance (RF)", xlabel="Importance",
     fname=os.path.join(OUTPUT_DIR, "feature_importance.png"))

sy_d = data["streams_by_year"]
line(sy_d["labels"], sy_d["values"],
     "Median Streams by Release Year (M)", "Streams (M)",
     rotate=True, fname=os.path.join(OUTPUT_DIR, "streams_by_year.png"))

sm_d = data["streams_by_month"]
area(sm_d["labels"], sm_d["values"],
     "Median Streams by Release Month (M)", "Streams (M)",
     fname=os.path.join(OUTPUT_DIR, "streams_by_month.png"))

ta_d = data["top_artists"]
barh(ta_d["labels"], ta_d["values"],
     "Top Artists (Total Streams, B)", xlabel="Billions",
     fname=os.path.join(OUTPUT_DIR, "top_artists.png"))

pr_d = data["platform_reach"]
pie(pr_d["labels"], pr_d["values"],
    "Platform Reach (Median Streams)",
    fname=os.path.join(OUTPUT_DIR, "platform_reach.png"))

print("\nCharts generated.")


# ── Optional interactive viewer (suppressed in pipeline mode) ─────────────────
def show_image_dialog(image_files, titles=None, top_trendy=None):
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:
        print("Tkinter not available.")
        return

    try:
        from PIL import Image, ImageTk
        _HAS_PIL = True
    except Exception:
        Image = ImageTk = None
        _HAS_PIL = False

    root = tk.Tk()
    root.title("Chart Viewer")
    root.geometry("900x700")

    imgs = []
    for p in image_files:
        try:
            imgs.append(ImageTk.PhotoImage(Image.open(p)) if _HAS_PIL
                        else tk.PhotoImage(file=p))
        except Exception:
            imgs.append(None)

    idx = 0
    frame = ttk.Frame(root, padding=8)
    frame.pack(fill="both", expand=True)
    title_var = tk.StringVar(value=(titles[0] if titles else ""))
    ttk.Label(frame, textvariable=title_var, font=(None, 12, "bold")).pack()
    img_label = ttk.Label(frame)
    img_label.pack()

    def show(i):
        nonlocal idx
        idx = i % len(imgs)
        title_var.set(titles[idx] if titles else "")
        if imgs[idx]:
            img_label.config(image=imgs[idx])
            img_label.image = imgs[idx]

    btn_frame = ttk.Frame(frame)
    btn_frame.pack(pady=6)
    ttk.Button(btn_frame, text="Previous",
               command=lambda: show((idx - 1) % len(imgs))).pack(side="left", padx=4)
    ttk.Button(btn_frame, text="Next",
               command=lambda: show((idx + 1) % len(imgs))).pack(side="left", padx=4)
    ttk.Button(btn_frame, text="Close", command=root.destroy).pack(side="left", padx=4)

    if top_trendy:
        right = ttk.Frame(root, padding=8)
        right.pack(side="right", fill="y")
        ttk.Label(right, text="Top trending songs", font=(None, 10, "bold")).pack()
        txt = tk.Text(right, width=50, height=15)
        for t in top_trendy:
            txt.insert("end",
                f"{t['track']} — {t['artist']} — "
                f"{t['streams_per_day_M']:.2f}M/d — {t['total_streams_M']:,}M\n")
        txt.config(state="disabled")
        txt.pack()

    show(0)
    root.mainloop()


if __name__ == "__main__" and not os.environ.get("AUTOMATION_MODE"):
    img_files = [
        os.path.join(OUTPUT_DIR, "playlist_quartile.png"),
        os.path.join(OUTPUT_DIR, "feature_importance.png"),
        os.path.join(OUTPUT_DIR, "platform_reach.png"),
        os.path.join(OUTPUT_DIR, "streams_by_year.png"),
        os.path.join(OUTPUT_DIR, "streams_by_month.png"),
        os.path.join(OUTPUT_DIR, "top_artists.png"),
    ]
    titles = [
        "Playlist Quartile", "Feature Importance", "Platform Reach",
        "Streams by Year", "Streams by Month", "Top Artists",
    ]
    show_image_dialog(img_files, titles=titles, top_trendy=data.get("top_trendy"))
