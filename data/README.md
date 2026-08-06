# Dataset

The raw dataset is not included in this repository.

## What you need

Place a file named `spotify_data.xlsx` in this `data/` folder before running the pipeline.

The file should contain two sheets:

| Sheet | Rows | Contents |
|---|---|---|
| `Song_Deatils` | ~949 | Track metadata, release date, platform playlist/chart counts, streams, BPM, key, mode |
| `Node_Deatils` | ~954 | Audio feature percentages: danceability, valence, energy, acousticness, instrumentalness, liveness, speechiness |

The merge key between both sheets is `Song_Id` (present in both, but formatted differently — the cleaning script handles this automatically).

## Where to get a compatible dataset

A very similar dataset is available on Kaggle:

**[Top Spotify Songs 2023 — by Nidula Elgiriyewithana](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023)**

Note: Column names or sheet structure may differ slightly from the version used in this project. You may need to adjust column references in `01_clean_data.py` to match your file.
