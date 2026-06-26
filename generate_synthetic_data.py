"""
CommissionLens — Synthetic Data Generator
Generates realistic NAV, benchmark, and macro data that mirrors the real
AMFI/NSE structure. Use this when APIs are unavailable (e.g. restricted
network) or for rapid testing before connecting real data.

Run:  python generate_synthetic_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────
N_FUND_PAIRS = 110          # number of direct/regular fund pairs
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
SEED = 42

RNG = np.random.default_rng(SEED)

# Indian AMCs
AMCS = [
    "SBI Mutual Fund", "HDFC Mutual Fund", "ICICI Prudential Mutual Fund",
    "Nippon India Mutual Fund", "Axis Mutual Fund", "Kotak Mutual Fund",
    "Mirae Asset Mutual Fund", "DSP Mutual Fund", "Franklin Templeton",
    "Tata Mutual Fund", "Edelweiss Mutual Fund", "Motilal Oswal",
]

FUND_CATEGORIES = [
    "Large Cap Fund", "Mid Cap Fund", "Small Cap Fund", "Flexi Cap Fund",
    "Multi Cap Fund", "ELSS", "Focused Fund", "Value Fund",
    "Large & Mid Cap Fund", "Contra Fund",
]

OPTIONS = ["Growth"]


def generate_fund_pairs() -> pd.DataFrame:
    pairs = []
    reg_code_start = 100000
    dir_code_start = 200000

    for i in range(N_FUND_PAIRS):
        amc = AMCS[i % len(AMCS)]
        cat = FUND_CATEGORIES[i % len(FUND_CATEGORIES)]
        option = "Growth"
        name_base = f"{amc.split()[0]} {cat} #{i+1}"
        reg_code = str(reg_code_start + i)
        dir_code = str(dir_code_start + i)
        pairs.append({
            "fund_family": f"{name_base} - Regular Plan - {option}",
            "regular_code": reg_code,
            "direct_code": dir_code,
            "amc": amc,
            "category": cat,
        })

    return pd.DataFrame(pairs)


def simulate_nav_series(
    start_nav: float,
    annual_return: float,
    annual_vol: float,
    start: str,
    end: str,
) -> pd.Series:
    """GBM simulation for NAV."""
    dates = pd.date_range(start, end, freq="B")
    dt = 1 / 252
    daily_ret = annual_return * dt
    daily_vol = annual_vol * np.sqrt(dt)
    shocks = RNG.normal(daily_ret, daily_vol, len(dates))
    nav = start_nav * np.exp(np.cumsum(shocks))
    return pd.Series(nav, index=dates)


def generate_nav_history(pairs: pd.DataFrame) -> pd.DataFrame:
    print(f"Generating NAV history for {len(pairs)} fund pairs...")

    # Nifty50-like market factor (same across funds)
    market_dates = pd.date_range(START_DATE, END_DATE, freq="B")
    dt = 1 / 252
    market_ret = 0.12 * dt
    market_vol = 0.18 * np.sqrt(dt)
    market_shocks = RNG.normal(market_ret, market_vol, len(market_dates))
    market_factor = np.exp(np.cumsum(market_shocks))

    records = []

    for _, row in pairs.iterrows():
        # Fund characteristics
        alpha_annual = RNG.uniform(-0.02, 0.06)     # fund alpha
        expense_reg = RNG.uniform(0.008, 0.022)      # regular plan TER
        expense_dir = expense_reg - RNG.uniform(0.005, 0.013)  # direct cheaper
        beta = RNG.uniform(0.80, 1.15)
        idio_vol = RNG.uniform(0.04, 0.10)
        start_nav = RNG.uniform(10, 200)

        idio_shocks_reg = RNG.normal(0, idio_vol * np.sqrt(dt), len(market_dates))
        idio_shocks_dir = idio_shocks_reg.copy()

        # Regular plan NAV
        reg_returns = (
            beta * market_shocks
            + alpha_annual * dt
            - expense_reg * dt
            + idio_shocks_reg
        )
        reg_nav = start_nav * np.exp(np.cumsum(reg_returns))

        # Direct plan NAV (same alpha, lower expense)
        dir_returns = (
            beta * market_shocks
            + alpha_annual * dt
            - expense_dir * dt
            + idio_shocks_dir
        )
        dir_nav = start_nav * np.exp(np.cumsum(dir_returns))

        for plan_type, nav_arr, code in [
            ("regular", reg_nav, row["regular_code"]),
            ("direct",  dir_nav, row["direct_code"]),
        ]:
            df_nav = pd.DataFrame({
                "date": market_dates,
                "nav": nav_arr,
                "scheme_code": code,
                "plan_type": plan_type,
                "fund_family": row["fund_family"],
                "amc": row["amc"],
            })
            records.append(df_nav)

    nav_df = pd.concat(records, ignore_index=True)
    print(f"  → {len(nav_df):,} NAV rows")
    return nav_df


def generate_nifty50() -> pd.DataFrame:
    print("Generating Nifty50 benchmark...")
    # Anchored to real Nifty50 levels
    anchors = {
        "2019-01-01": 10831, "2020-01-01": 12168, "2020-04-01": 8598,
        "2021-01-01": 13982, "2022-01-01": 17354, "2023-01-01": 18105,
        "2024-01-01": 21741, "2024-12-31": 23644,
    }
    dates = pd.date_range(START_DATE, END_DATE, freq="B")
    anchor_dates = pd.to_datetime(list(anchors.keys()))
    anchor_vals = list(anchors.values())
    anchor_nums = (anchor_dates - anchor_dates[0]).days.values.astype(float)
    date_nums = (dates - anchor_dates[0]).days.values.astype(float)
    date_nums = np.clip(date_nums, anchor_nums[0], anchor_nums[-1])
    base = np.interp(date_nums, anchor_nums, anchor_vals)
    noise = RNG.normal(0, 0.006, len(dates))
    prices = base * np.exp(np.cumsum(noise) * 0.15)
    return pd.DataFrame({"date": dates, "nifty50_close": prices})


def generate_macro() -> pd.DataFrame:
    """Monthly macro data with realistic India values."""
    from data_fetcher import fetch_rbi_macro_data, generate_fii_dii_flows
    print("Generating macro data...")
    macro = fetch_rbi_macro_data()
    fii_dii = generate_fii_dii_flows()
    macro = macro.merge(fii_dii, on="date", how="left")
    return macro


def run_synthetic_generation():
    print("\n" + "="*55)
    print("  CommissionLens — Synthetic Data Generator")
    print("="*55 + "\n")

    pairs = generate_fund_pairs()
    pairs.to_csv(DATA_DIR / "fund_pairs.csv", index=False)
    print(f"  Saved fund_pairs.csv ({len(pairs)} pairs)")

    nav_df = generate_nav_history(pairs)
    nav_df.to_parquet(DATA_DIR / "nav_history.parquet", index=False)
    print(f"  Saved nav_history.parquet")

    nifty = generate_nifty50()
    nifty.to_csv(DATA_DIR / "nifty50.csv", index=False)
    print(f"  Saved nifty50.csv")

    macro = generate_macro()
    macro.to_csv(DATA_DIR / "macro.csv", index=False)
    print(f"  Saved macro.csv")

    print(f"\n✓ Synthetic data ready in {DATA_DIR}/\n")
    return nav_df, nifty, macro


if __name__ == "__main__":
    run_synthetic_generation()
