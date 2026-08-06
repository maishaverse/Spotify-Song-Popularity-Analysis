# What Makes a Song Popular on Spotify?
### An end-to-end Data Analysis Pipeline - Cleaning, EDA, Predictive modelling, and Automated HTML reporting

---

## Key Findings

| Finding | Result |
|---|---|
| Songs in the **top playlist quartile** vs bottom | **1,214M vs 119M** median streams (~10x gap) |
| Songs on **all 3 platforms** (Spotify + Apple + Deezer) vs 1 | **4.7x** more median streams |
| Combined RF importance of all **audio features** | **< 15%** — distribution beats composition |
| Random Forest R² on held-out test set | **0.853** |

> **The headline:** What a song *sounds like* barely predicts its popularity. Where it *lives* (playlists, platforms) explains almost everything.

---

## Project Overview

This project analyzes the Spotify top songs dataset to answer two questions:

1. **What factors drive a song's streaming popularity?**
2. **Which songs are trending right now — and how do you measure that fairly?**

The pipeline runs end-to-end from a raw Excel file to a fully self-contained HTML dashboard, with no manual steps required once the data is in place.

---

## Results

### Popularity is driven by distribution, not content

Spotify playlist count alone accounts for **58% of Random Forest feature importance**. Every audio feature combined — danceability, energy, valence, acousticness — contributes less than 15%.

### Trending songs need a different metric

Raw stream counts are biased toward older songs that have had years to accumulate. This project uses **streams per day since release** to surface what is actually gaining momentum *now*, not what has been popular for a decade.

**Top 5 trending songs (streams/day, last 18 months):**

| Track | Artist | Streams / Day |
|---|---|---|
| Flowers | Miley Cyrus | 3.72M |
| Kill Bill | SZA | 2.99M |
| Unholy (feat. Kim Petras) | Sam Smith, Kim Petras | 2.64M |
| Ella Baila Sola | Eslabon Armado, Peso Pluma | 2.49M |
| Anti-Hero | Taylor Swift | 2.29M |

---

## Data Manipulation Challenges

The raw dataset required significant cleaning before analysis was possible:

| Issue | Detail | Resolution |
|---|---|---|
| Merge key mismatch | Song_Id stored as `"4,137"` in one sheet, `"0000003788"` in the other | Stripped commas + leading zeros before joining |
| Duplicate rows | ~136 duplicates in Songs sheet, ~138 in Nodes | Deduplicated on Song_Id, keeping first occurrence |
| Mixed month formats | `released_month` had both numeric (`7`) and text (`"July"`) values | Normalised all values to integers 1–12 |
| Corrupted streams cell | Row 570 had audio features pasted into the streams column | Flagged as NaN, retained rest of row |
| Encoding errors | 88 rows with broken UTF-8 in artist/track names | Stripped replacement characters |
| Missing values | 80 missing `key`, 44 missing audio features, 1 missing streams | Documented; handled via dropna in model |

Raw input: **949 + 954 rows** across two sheets → Final: **809 merged, validated songs**

---

## Project Structure

```
spotify-popularity-analysis/
├── main.py                  # Single entry point — runs the full pipeline
├── 01_clean_data.py         # Data cleaning and merge
├── 02_eda.py                # Exploratory data analysis (console output)
├── 03_model.py              # Linear regression + Random Forest
├── 04_prep_chart_data.py    # Chart data prep + matplotlib PNG generation
├── 05_generate_report.py    # Self-contained HTML dashboard generator
├── requirements.txt         # Python dependencies
├── data/
│   └── README.md            # Where to get the dataset
└── outputs/                 # All generated files land here
    ├── spotify_clean.csv    # Cleaned dataset (generated)
    ├── chart_data.json      # Aggregated chart inputs (generated)
    ├── *.png                # Matplotlib charts (generated, deleted after report)
    └── Reports/
        └── Spotify_Analysis_Report_YYYY-MM-DD.html  # Final dashboard
```

---

## Pipeline

```
01_clean_data.py   →   spotify_clean.csv
        ↓
02_eda.py          →   console diagnostics
        ↓
03_model.py        →   lr_coefficients.csv, rf_importance.csv
        ↓
04_prep_chart_data.py  →  chart_data.json + PNG charts
        ↓
05_generate_report.py  →  Reports/Spotify_Analysis_Report_YYYY-MM-DD.html
```

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/maishaverse/spotify-song-popularity-analysis.git
cd spotify-popularity-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Place `spotify_data.xlsx` in the `data/` folder.
See [`data/README.md`](data/README.md) for dataset details.

### 4. Run the full pipeline

```bash
python main.py
```

The HTML report is saved to:
```
outputs/Reports/Spotify_Analysis_Report_YYYY-MM-DD.html
```

Open it in any browser. It is fully self-contained — all charts are embedded.

### Run individual steps

```bash
python 01_clean_data.py         # Clean and merge the raw data
python 03_model.py              # Train models (requires cleaned data)
python 05_generate_report.py    # Generate report (requires all prior outputs)
```

---

## Tools & Libraries

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| pandas | Data manipulation and cleaning |
| NumPy | Numerical operations |
| scikit-learn | Linear Regression, Random Forest, train/test split |
| matplotlib | Chart generation |
| openpyxl | Excel file reading |

---

## Model Details

Two models were trained for complementary purposes:

**Linear Regression** (R² = 0.696)
- Target: log₁₀(streams) — log-transformed to address right skew
- Features standardised (z-scored) for comparable coefficients
- Used for interpretability: coefficients show direction and relative magnitude of each driver

**Random Forest** (R² = 0.853, 300 trees, max depth 10)
- Same feature set, non-linear ensemble
- Used for predictive accuracy and robust feature importance ranking
- Both models agree: Spotify playlist count is the dominant predictor

**Important caveat:** Playlist inclusion is partly a *consequence* of popularity, not just a cause. The relationship is bidirectional — popular songs get added to more playlists, which drives more streams. Without temporal playlist-add data, the models cannot fully separate cause from effect.

---

## Author

**Maisha Khatoon**
Data Analyst — Python · SQL · Power BI · DAX

[LinkedIn](https://linkedin.com/in/maisha-khatoon) · [GitHub](https://github.com/maishaverse)
