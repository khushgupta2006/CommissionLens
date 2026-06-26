"""
CommissionLens — Main Pipeline
Runs: fetch data → engineer features → train models → generate report.

Usage:
    python pipeline.py                  # 30 fund pairs (quick)
    python pipeline.py --funds 100      # 100 pairs (recommended)
    python pipeline.py --skip-fetch     # reuse existing data, retrain only
"""

import argparse
import sys
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def run_pipeline(n_funds: int = 30, skip_fetch: bool = False):
    print("\n" + "█"*55)
    print("  CommissionLens — Full Pipeline")
    print("█"*55 + "\n")

    # Step 1: Fetch data 
    if skip_fetch and (DATA_DIR / "nav_history.parquet").exists():
        print("⏩  Skipping data fetch (--skip-fetch flag).\n")
        import pandas as pd
        nav_df = pd.read_parquet(DATA_DIR / "nav_history.parquet")
        nifty_df = pd.read_csv(DATA_DIR / "nifty50.csv", parse_dates=["date"])
        macro_df = pd.read_csv(DATA_DIR / "macro.csv", parse_dates=["date"])
    else:
        print("STEP 1 / 3: Fetching Data ")
        from data_fetcher import fetch_all_data
        result = fetch_all_data(max_funds=n_funds, save=True)
        nav_df = result["nav"]
        nifty_df = result["nifty"]
        macro_df = result["macro"]

    # Step 2: Feature engineering
    print("\nSTEP 2 / 3: Engineering Features ")
    from feature_engineering import build_feature_table
    features = build_feature_table(nav_df, nifty_df, macro_df, save=True)

    if len(features) < 20:
        print("\n⚠️  Very few feature rows generated. "
              "Try increasing --funds or check data quality.")
        print("   Proceeding anyway with what we have...\n")

    # Step 3: Train models 
    print("\nSTEP 3 / 3: Training Models ")
    from model_training import run_training_pipeline
    run_training_pipeline(run_sip_backtest=True)

    print("\n" + "█"*55)
    print("  ✅ Pipeline complete!")
    print("█"*55)
    print("""
Next steps:
  streamlit run app.py          # Launch the dashboard
  jupyter notebook notebooks/   # Explore the notebook
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CommissionLens Pipeline")
    parser.add_argument("--funds", type=int, default=30,
                        help="Number of fund pairs to fetch (default: 30)")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip data fetch, use existing data")
    args = parser.parse_args()

    run_pipeline(n_funds=args.funds, skip_fetch=args.skip_fetch)
