"""
Spotify Dataset — Predictive Modelling
Author: Maisha Khatoon

Purpose:
    Train two complementary models to quantify what drives streams:
      A) Linear Regression on log-streams — interpretable coefficients
      B) Random Forest — non-linear feature importance + predictive power

Input:  outputs/spotify_clean.csv
Output: outputs/lr_coefficients.csv
        outputs/rf_importance.csv
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ── PATHS ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load & filter ─────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(OUTPUT_DIR, "spotify_clean.csv"))
df = df[df["released_year"] <= 2023].copy()
df = df.dropna(subset=["streams"])

# ── Feature engineering ───────────────────────────────────────────────────────
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

# Data-driven reference date: one day after the latest release in the dataset
max_release = df["release_date"].max()
ref_date = (
    (max_release + pd.Timedelta(days=1)).normalize()
    if pd.notna(max_release)
    else pd.Timestamp("2024-01-01")
)
print(f"[info] ref_date = {ref_date.date()}")

df["days_since_release"] = (ref_date - df["release_date"]).dt.days.clip(lower=1)
df["log_streams"]        = np.log10(df["streams"].clip(lower=1))
df["log_days"]           = np.log10(df["days_since_release"])
df["is_solo"]            = (df["artist_count"] == 1).astype(int)
df["is_major"]           = (df["mode"] == "Major").astype(int)

features = [
    "in_spotify_playlists", "in_apple_playlists", "in_deezer_playlists",
    "in_spotify_charts",    "in_apple_charts",    "in_deezer_charts",
    "artist_count", "log_days",
    "danceability_%", "valence_%",  "energy_%",      "acousticness_%",
    "instrumentalness_%", "liveness_%", "speechiness_%",
    "bpm", "is_major",
]

data = df.dropna(subset=features + ["log_streams"]).copy()
print(f"[model] Training rows after dropna: {len(data)}")

X = data[features].values
y = data["log_streams"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── A. Linear Regression (standardised features) ──────────────────────────────
scaler = StandardScaler()
Xtr_s  = scaler.fit_transform(X_train)
Xte_s  = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(Xtr_s, y_train)
y_pred_lr = lr.predict(Xte_s)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"\n=== Linear Regression ===")
print(f"R² on test set        : {r2_lr:.3f}")
print(f"MAE (log10 streams)   : {mean_absolute_error(y_test, y_pred_lr):.3f}")

coef_df = pd.DataFrame({"feature": features, "coef": lr.coef_})
coef_df["abs"] = coef_df["coef"].abs()
coef_df = coef_df.sort_values("abs", ascending=False)
print("\nStandardised coefficients (sorted by magnitude):")
for _, row in coef_df.iterrows():
    print(f"  {row['feature']:25s}  {row['coef']:+.3f}")

# ── B. Random Forest ──────────────────────────────────────────────────────────
print(f"\n=== Random Forest ===")
rf = RandomForestRegressor(
    n_estimators=300, max_depth=10, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"R² on test set        : {r2_rf:.3f}")

imp_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_})
imp_df = imp_df.sort_values("importance", ascending=False)
print("\nFeature importance:")
for _, row in imp_df.iterrows():
    print(f"  {row['feature']:25s}  {row['importance']:.3f}")

# ── Save outputs ─────────────────────────────────────────────────────────────
coef_df.to_csv(os.path.join(OUTPUT_DIR, "lr_coefficients.csv"), index=False)
imp_df.to_csv(os.path.join(OUTPUT_DIR,  "rf_importance.csv"),   index=False)
print(f"\n[save] Model outputs written to {OUTPUT_DIR}")

print(f"\n=== TAKEAWAYS ===")
print("Both models agree: playlist inclusion (Spotify, Apple, Deezer) is the dominant predictor.")
print("Audio features (danceability, valence, energy) have weak predictive power.")
print("Time since release matters — older songs have accumulated more streams.")
print("Charts presence matters less than playlist presence.")
