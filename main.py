"""
Spotify Popularity Analysis — Pipeline Runner
Runs all 5 steps in sequence from a single entry point.

Steps:
    01  Clean data        -> outputs/spotify_clean.csv
    02  EDA               -> console diagnostics only
    03  Model             -> outputs/lr_coefficients.csv, rf_importance.csv
    04  Prep chart data   -> outputs/chart_data.json + PNG charts
    05  Generate report   -> outputs/Reports/Spotify_Analysis_Report_YYYY-MM-DD.html

Usage:
    python main.py

To skip a step, comment it out in PIPELINE_STEPS below.
"""

import os
import sys
import subprocess
from datetime import datetime

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

PIPELINE_STEPS = [
    "01_clean_data.py",
    "02_eda.py",
    "03_model.py",
    "04_prep_chart_data.py",
    "05_generate_report.py",
]

# Suppress the interactive chart-viewer dialog in step 04 when running headless
env = os.environ.copy()
env["AUTOMATION_MODE"] = "1"


def run_step(script_name: str, step_num: int, total: int) -> bool:
    print(f"\n{'='*60}")
    print(f"  STEP {step_num}/{total} — {script_name}")
    print(f"{'='*60}")

    script_path = os.path.join(SCRIPTS_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"  [ERROR] Script not found: {script_path}")
        return False

    result = subprocess.run([sys.executable, script_path], env=env)

    if result.returncode != 0:
        print(f"\n  [FAILED] {script_name} exited with code {result.returncode}.")
        return False

    print(f"  [OK] {script_name} complete.")
    return True


def main():
    start = datetime.now()
    print(f"\n{'='*60}")
    print(f"  Spotify Popularity Analysis Pipeline")
    print(f"  Started : {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Steps   : {len(PIPELINE_STEPS)}")
    print(f"{'='*60}")

    for i, script in enumerate(PIPELINE_STEPS, 1):
        success = run_step(script, i, len(PIPELINE_STEPS))
        if not success:
            print(f"\n[PIPELINE HALTED] at step {i}: {script}")
            sys.exit(1)

    elapsed = (datetime.now() - start).seconds
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
