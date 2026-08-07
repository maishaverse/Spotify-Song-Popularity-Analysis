"""
Spotify Dataset -- HTML Report Generator
Author: Maisha Khatoon

Purpose:
    Reads all outputs from scripts 01-04 and generates one fully
    self-contained HTML dashboard. Every chart is embedded as base64
    so the report opens in any browser with no external dependencies.

Input:  outputs/chart_data.json
        outputs/lr_coefficients.csv
        outputs/*.png  (from 04_prep_chart_data.py)
Output: outputs/Reports/Spotify_Analysis_Report_YYYY-MM-DD.html
"""

import os
import json
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend -- must come before pyplot import
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime

# -- PATHS ---------------------------------------------------------------------
ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(ROOT_DIR, "outputs")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "Reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

CHART_DATA_JSON = os.path.join(OUTPUT_DIR, "chart_data.json")
LR_CSV          = os.path.join(OUTPUT_DIR, "lr_coefficients.csv")
RF_CSV          = os.path.join(OUTPUT_DIR, "rf_importance.csv")

CHART_FILES = {
    "playlist_quartile": os.path.join(OUTPUT_DIR, "playlist_quartile.png"),
    "feature_importance": os.path.join(OUTPUT_DIR, "feature_importance.png"),
    "streams_by_year":   os.path.join(OUTPUT_DIR, "streams_by_year.png"),
    "streams_by_month":  os.path.join(OUTPUT_DIR, "streams_by_month.png"),
    "top_artists":       os.path.join(OUTPUT_DIR, "top_artists.png"),
    "platform_reach":    os.path.join(OUTPUT_DIR, "platform_reach.png"),
}

# -- COLOUR PALETTE ------------------------------------------------------------
C_DEEP  = "#1A4D45"
C_MID   = "#3A7D6F"
C_AMBER = "#E8A33D"
C_AMBRDK= "#C97F1F"
C_LIGHT = "#8FB8AD"
C_TEXT  = "#1A1A1A"
C_MUTED = "#6B6B6B"
C_CREAM = "#F5EFE3"
C_RED   = "#B84C3D"

# -- CHART HELPERS -------------------------------------------------------------

def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64

def _style(ax, fig):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=C_TEXT, labelsize=9)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

def make_audio_corr_chart(labels, values):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = [C_AMBER if v == max(values) else C_MID for v in values]
    bars = ax.barh(labels, values, color=colors, height=0.55)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9, color=C_TEXT)
    ax.set_xlabel("|Correlation with streams|", color=C_MUTED, fontsize=10)
    ax.set_title("Absolute correlation of audio features with streams",
                 color=C_TEXT, fontsize=11, pad=8)
    ax.set_xlim(0, 0.22)
    _style(ax, fig)
    fig.tight_layout()
    return fig_to_b64(fig)

def make_solo_collab_chart(labels, values):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    bars = ax.bar(labels, values, color=[C_DEEP, C_MID], width=0.5)
    ax.bar_label(bars, fmt="%.0fM", padding=5, fontsize=10, color=C_TEXT)
    ax.set_ylabel("Median streams (M)", color=C_MUTED, fontsize=10)
    ax.set_title("Solo vs Collaboration", color=C_TEXT, fontsize=11, pad=8)
    _style(ax, fig)
    fig.tight_layout()
    return fig_to_b64(fig)

def safe_b64(key):
    try:
        return img_to_b64(CHART_FILES[key])
    except Exception as e:
        print(f"  [warn] Could not load chart '{key}': {e}")
        return ""

def img_tag(b64, alt="chart", width="100%"):
    if not b64:
        return (f"<p style='color:{C_MUTED};font-size:12px;'>"
                f"[Chart not available -- run step 04 first]</p>")
    return (f'<img src="data:image/png;base64,{b64}" alt="{alt}" '
            f'style="max-width:{width};height:auto;display:block;">')

def r2_color(val):
    if val >= 0.8: return "#17a74b"
    if val >= 0.6: return C_AMBRDK
    return C_RED

# -- HTML BUILDER --------------------------------------------------------------

def build_html(data, lr_df, rf_df):

    ov  = data["overview"]
    dc  = data["data_challenges"]
    pq  = data["playlist_quartile"]
    pr  = data["platform_reach"]
    ac  = data["audio_correlations"]
    sc  = data["solo_vs_collab"]
    tt  = data["top_trendy"]
    md  = data["model"]

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("  [05] Generating audio-correlation chart...")
    b64_audio = make_audio_corr_chart(ac["labels"], ac["values"])
    print("  [05] Generating solo/collab chart...")
    b64_solo  = make_solo_collab_chart(sc["labels"], sc["values"])

    print("  [05] Embedding PNG charts...")
    b64_pq = safe_b64("playlist_quartile")
    b64_fi = safe_b64("feature_importance")
    b64_sy = safe_b64("streams_by_year")
    b64_sm = safe_b64("streams_by_month")
    b64_ta = safe_b64("top_artists")
    b64_pr = safe_b64("platform_reach")

    # -- Dynamic HTML blocks ---------------------------------------------------

    challenge_rows = "".join(
        f"<tr><td style='font-weight:bold'>{i}</td><td>{d}</td>"
        f"<td style='color:#17a74b'>{f}</td></tr>"
        for i, d, f in [
            ("Merge key mismatch",
             'Song_Id stored as "4,137" in one sheet, "0000003788" in the other',
             "Stripped commas + leading zeros; joined on normalised integer string"),
            ("Duplicate rows",
             f"~{dc['duplicates_songs']} in Songs sheet, ~{dc['duplicates_nodes']} in Nodes",
             "Deduplicated on Song_Id, keeping first occurrence"),
            ("Mixed month formats",
             'released_month had both numeric (7) and text ("July") in the same column',
             "Normalised all values to integers 1-12"),
            ("Corrupted streams cell",
             "Row 570: audio features pasted into the streams column",
             "Flagged as NaN, rest of row retained"),
            ("Encoding errors",
             f'{dc["encoding_errors"]} rows with broken UTF-8 in artist/track names',
             "Stripped replacement characters"),
            ("Missing values",
             f'{dc["missing_key"]} missing key, {dc["missing_audio_features"]} missing audio features',
             "Documented; handled via dropna in model"),
        ]
    )

    trendy_rows = "".join(
        f"<tr>"
        f"<td style='font-weight:bold;color:{C_AMBRDK}'>{i}</td>"
        f"<td style='font-weight:bold'>{t['track']}</td>"
        f"<td>{t['artist']}</td>"
        f"<td style='font-weight:bold'>{t['streams_per_day_M']:.2f}M</td>"
        f"<td>{int(t['total_streams_M']):,}M</td>"
        f"</tr>"
        for i, t in enumerate(tt, 1)
    )

    lr_rows = ""
    for _, row in lr_df.head(10).iterrows():
        color = "#17a74b" if row["coef"] > 0 else C_RED
        lr_rows += (
            f"<tr>"
            f"<td>{row['feature']}</td>"
            f"<td style='font-weight:bold;color:{color}'>{row['coef']:+.3f}</td>"
            f"</tr>"
        )

    rec_cards = "".join(
        f'<div class="rec-card">'
        f'<div class="rec-num">{num}</div>'
        f'<div class="rec-title">{title}</div>'
        f'<div class="rec-body">{body}</div>'
        f'</div>'
        for num, title, body in [
            ("01", "Treat playlist placement as a strategic lever",
             "Playlist inclusion is the strongest predictor -- 58% of RF feature importance. "
             "Editorial playlist curation should be treated as a primary popularity-shaping "
             "tool, not just discovery."),
            ("02", "Push for cross-platform reach early",
             f"Songs on all 3 platforms have a median of {pr['values'][-1]:.0f}M streams -- "
             f"about {int(pr['values'][-1] / max(pr['values'][0], 1))}x single-platform songs. "
             "Coordinating multi-DSP launches amplifies stream momentum."),
            ("03", "Build a stream-velocity trend score",
             "Streams-per-day (not total streams) reveals what is actually trending now. "
             "Use a velocity-based score to surface emerging songs in Discover Weekly "
             "and editorial decisions."),
            ("04", "Keep audio features in recommendation -- not popularity modelling",
             "Audio features show |r| less than 0.15 with streams. They are useful for "
             "taste-matching but are weak popularity predictors."),
        ]
    )

    stat_cards = "".join(
        f'<div class="stat-card"><div class="stat-val">{v}</div>'
        f'<div class="stat-label">{l}</div></div>'
        for v, l in [
            (f"{ov['n_songs']:,}",                       "Unique songs analyzed"),
            (f"{ov['n_artists']:,}",                     "Unique artists"),
            (f"{ov['year_min']}&ndash;{ov['year_max']}", "Release year range"),
            (f"{ov['median_streams_M']:.0f}M",           "Median streams per song"),
        ]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Spotify Popularity Analysis</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:Arial,sans-serif; background:#f2f2f2; color:{C_TEXT}; font-size:14px; }}
  .top-bar {{ background:{C_DEEP}; color:white; padding:18px 32px;
    display:flex; justify-content:space-between; align-items:center; }}
  .top-bar h1 {{ font-size:20px; font-weight:bold; }}
  .top-bar .meta {{ font-size:11px; color:{C_LIGHT}; text-align:right; line-height:1.6; }}
  .container {{ max-width:1200px; margin:0 auto; padding:24px 16px; }}
  .section {{ background:white; border-radius:6px; box-shadow:0 1px 4px rgba(0,0,0,.1);
    margin-bottom:24px; padding:24px; }}
  .section-title {{ font-size:16px; font-weight:bold; color:{C_DEEP};
    border-left:4px solid {C_AMBER}; padding-left:12px; margin-bottom:10px; }}
  .section-sub {{ font-size:12px; color:{C_MUTED}; margin-bottom:16px; line-height:1.5; }}
  .stat-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }}
  .stat-card {{ background:{C_CREAM}; border-radius:6px; padding:20px 14px; text-align:center; }}
  .stat-val {{ font-size:34px; font-weight:bold; color:{C_DEEP}; }}
  .stat-label {{ font-size:12px; color:{C_MUTED}; margin-top:6px; }}
  .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; align-items:start; }}
  .one-third-two {{ display:grid; grid-template-columns:1fr 2fr; gap:20px; align-items:start; }}
  .insight-box {{ background:{C_DEEP}; color:white; border-radius:6px;
    padding:18px; font-size:13px; line-height:1.7; }}
  .insight-box b {{ color:{C_AMBER}; }}
  .insight-box i {{ color:{C_LIGHT}; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ background:{C_DEEP}; color:white; padding:10px 12px; text-align:left; font-size:12px; }}
  td {{ padding:9px 12px; border-bottom:1px solid #eee; vertical-align:top; }}
  tr:nth-child(even) td {{ background:#fafafa; }}
  .model-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:20px; }}
  .model-card {{ background:{C_CREAM}; border-radius:6px; padding:20px; }}
  .model-card .mname {{ font-size:14px; font-weight:bold; color:{C_DEEP}; margin-bottom:8px; }}
  .model-card .mr2 {{ font-size:40px; font-weight:bold; line-height:1.1; }}
  .model-card .mdesc {{ font-size:12px; color:{C_MUTED}; margin-top:8px; line-height:1.5; }}
  .rec-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .rec-card {{ background:{C_CREAM}; border-radius:6px; padding:20px; }}
  .rec-num {{ font-size:30px; font-weight:bold; color:{C_AMBER}; margin-bottom:8px; }}
  .rec-title {{ font-size:13px; font-weight:bold; color:{C_DEEP}; margin-bottom:8px; }}
  .rec-body {{ font-size:13px; color:{C_TEXT}; line-height:1.5; }}
  footer {{ background:white; border-top:1px solid #ddd;
    padding:16px 32px; font-size:12px; color:{C_MUTED}; }}
  details summary {{ cursor:pointer; color:{C_DEEP}; font-weight:bold; padding:8px 0; }}
  details div {{ padding:10px 0 0 12px; line-height:1.8; }}
  @media (max-width:800px) {{
    .stat-grid,.two-col,.one-third-two,.model-grid,.rec-grid {{ grid-template-columns:1fr; }}
  }}
</style>
</head>
<body>

<div class="top-bar">
  <h1>Spotify Popularity Analysis</h1>
  <div class="meta">Generated: {generated_at}<br>Author: Maisha Khatoon</div>
</div>

<div class="container">

  <div class="section">
    <div class="section-title">Dataset Overview</div>
    <div class="section-sub">After cleaning and merging both sheets.
      Popularity = streams (total listener behaviour).
      For trending detection, streams-per-day since release is used to control
      for the accumulation advantage older songs have over newer releases.</div>
    <div class="stat-grid">{stat_cards}</div>
  </div>

  <div class="section">
    <div class="section-title">Data Manipulation Challenges</div>
    <div class="section-sub">
      Six issues resolved before analysis.
      Raw: <strong>{dc['raw_songs_rows']}</strong> + <strong>{dc['raw_nodes_rows']}</strong>
      rows &rarr; Final: <strong>{dc['final_rows']}</strong> validated songs.
    </div>
    <table>
      <thead><tr><th>Issue</th><th>What was found</th><th>How it was resolved</th></tr></thead>
      <tbody>{challenge_rows}</tbody>
    </table>
  </div>

  <div class="section">
    <div class="section-title">Popularity Driver 1 &mdash; Audio Features Are Weak Predictors</div>
    <div class="two-col">
      <div>{img_tag(b64_audio, "Audio correlations")}</div>
      <div class="insight-box">
        <b>What this tells us</b><br><br>
        Every audio feature &mdash; danceability, energy, valence, acousticness &mdash;
        shows <b>near-zero correlation</b> with stream count (|r| &lt; 0.15).<br><br>
        The musical <i>content</i> of a song is a weak signal.
        What matters is where it <i>lives</i>.<br><br>
        <b>Distribution beats composition.</b>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Popularity Driver 2 &mdash; Playlist Inclusion Is Dominant</div>
    <div class="section-sub">
      Songs in the top playlist quartile have ~10x the median streams of the bottom
      (<strong>{pq['values'][3]:.0f}M vs {pq['values'][0]:.0f}M</strong>).
      Random Forest assigns <strong>58%</strong> of feature importance to Spotify
      playlist count alone.
    </div>
    <div class="two-col">
      <div>{img_tag(b64_pq, "Playlist quartile")}</div>
      <div>{img_tag(b64_fi, "Feature importance")}</div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Popularity Driver 3 &mdash; Cross-Platform Reach Amplifies</div>
    <div class="two-col">
      <div>{img_tag(b64_pr, "Platform reach")}</div>
      <div class="insight-box">
        <b>Distribution amplifies</b><br><br>
        Songs on <b>all three platforms</b> have a median of
        <b>{pr['values'][-1]:.0f}M streams</b> &mdash; about
        <b>{int(pr['values'][-1] / max(pr['values'][0], 1))}x</b>
        the songs that only reach one platform.<br><br>
        Getting a song onto editorial playlists across platforms
        is a high-leverage action.
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Temporal Trends</div>
    <div class="section-sub">Older songs accumulate more streams over time (longer window).
      January releases peak &mdash; likely New-Year listening behaviour and editorial bias.</div>
    <div class="two-col">
      <div>{img_tag(b64_sy, "Streams by year")}</div>
      <div>{img_tag(b64_sm, "Streams by month")}</div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Artist Influence</div>
    <div class="section-sub">Solo songs outperform collaborations at the median
      (legacy catalog hits are mostly solo). Top 10 artists account for ~20%
      of credited streams.</div>
    <div class="one-third-two">
      <div>{img_tag(b64_solo, "Solo vs collab")}</div>
      <div>{img_tag(b64_ta, "Top artists")}</div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Predictive Model</div>
    <div class="section-sub">Two models for complementary purposes.
      80/20 train/test split, random_state=42, {md['training_rows']} training rows.</div>
    <div class="model-grid">
      <div class="model-card">
        <div class="mname">Linear Regression</div>
        <div class="mr2" style="color:{r2_color(md['lr_r2'])};">R&sup2; = {md['lr_r2']:.3f}</div>
        <div class="mdesc">Standardised log-streams target &mdash;
          used for interpretability of coefficients.</div>
      </div>
      <div class="model-card">
        <div class="mname">Random Forest &nbsp;(300 trees, max depth 10)</div>
        <div class="mr2" style="color:{r2_color(md['rf_r2'])};">R&sup2; = {md['rf_r2']:.3f}</div>
        <div class="mdesc">Non-linear ensemble &mdash;
          used for predictive power and feature importance.</div>
      </div>
    </div>
    <div class="insight-box" style="margin-bottom:20px;">
      <b>Both models agree:</b> Playlist inclusion is the strongest predictor.
      Time since release is second. Audio features contribute weakly.<br><br>
      <i>Caveat: Playlist inclusion is partly a consequence of popularity, not just a cause.
      The relationship is bidirectional &mdash; popular songs get added to more playlists,
      which drives more streams.</i>
    </div>
    <table>
      <thead><tr><th>Feature (Linear Regression &mdash; top 10)</th>
        <th>Standardised Coefficient</th></tr></thead>
      <tbody>{lr_rows}</tbody>
    </table>
  </div>

  <div class="section">
    <div class="section-title">Trending Songs &mdash; Top 10 by Stream Velocity</div>
    <div class="section-sub">Songs released in the last 18 months, ranked by
      <strong>streams per day since release</strong>. Controls for accumulation bias.</div>
    <table>
      <thead><tr><th>#</th><th>Track</th><th>Artist(s)</th>
        <th>Streams / Day</th><th>Total Streams</th></tr></thead>
      <tbody>{trendy_rows}</tbody>
    </table>
  </div>

  <div class="section">
    <div class="section-title">Recommendations</div>
    <div class="rec-grid">{rec_cards}</div>
  </div>

</div>

<footer>
  <details>
    <summary>Methodology &amp; Definitions</summary>
    <div>
      <strong>Popularity metric:</strong> Streams (total) for historical analysis;
        streams-per-day for trending detection.<br>
      <strong>Trending window:</strong> Songs released within the last 18 months
        (&le;540 days).<br>
      <strong>Filters:</strong> Release year &le; 2023; future-dated rows excluded.<br>
      <strong>Train/test split:</strong> 80/20, random_state=42,
        {md['training_rows']} training rows.<br>
      <strong>LR target:</strong> log10(streams) &mdash; log-transform applied
        for right-skewed distribution.<br>
      <strong>Limitations:</strong> Playlist inclusion and streams are associated,
        not definitively causal. Dataset contains only songs that achieved meaningful
        streaming (survivorship bias).
    </div>
  </details>
  <p style="margin-top:12px;">
    Spotify Popularity Analysis &nbsp;|&nbsp; Maisha Khatoon
    &nbsp;|&nbsp; {generated_at}
  </p>
</footer>

</body>
</html>"""


# -- MAIN ----------------------------------------------------------------------

def main():
    print("[05] Loading data...")
    with open(CHART_DATA_JSON) as f:
        data = json.load(f)
    lr_df = pd.read_csv(LR_CSV).sort_values("abs", ascending=False)
    rf_df = pd.read_csv(RF_CSV).sort_values("importance", ascending=False)

    print("[05] Building report...")
    html = build_html(data, lr_df, rf_df)

    out_path = os.path.join(
        REPORTS_DIR,
        f"Spotify_Analysis_Report_{datetime.now().strftime('%Y-%m-%d')}.html"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[05] Report saved: {out_path}")
    return out_path


if __name__ == "__main__":
    main()
