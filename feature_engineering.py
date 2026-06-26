"""
CommissionLens — Feature Engineering
Computes quarterly fund-level features and labels for ML.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import linregress

DATA_DIR = Path(__file__).parent / "data"


# 1. NAV → Quarterly Returns

def compute_quarterly_returns(nav_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each fund (scheme_code), resample NAV to quarter-end and compute returns.
    Returns quarterly return series per fund.
    """
    nav_df = nav_df.copy()
    nav_df["date"] = pd.to_datetime(nav_df["date"])
    nav_df = nav_df.sort_values(["scheme_code", "date"])

    records = []
    for code, grp in nav_df.groupby("scheme_code"):
        grp = grp.set_index("date")["nav"].resample("QE").last().dropna()
        if len(grp) < 4:
            continue
        ret = grp.pct_change().dropna()
        for dt, r in ret.items():
            records.append({
                "scheme_code": code,
                "quarter_end": dt,
                "quarterly_return": r,
                "nav_end": grp.loc[dt],
            })
    df = pd.DataFrame(records)
    # Attach metadata
    meta = nav_df[["scheme_code", "plan_type", "fund_family", "amc"]].drop_duplicates("scheme_code")
    df = df.merge(meta, on="scheme_code", how="left")
    return df


def compute_benchmark_quarterly(nifty_df: pd.DataFrame) -> pd.DataFrame:
    """Resample Nifty50 to quarter-end returns."""
    nifty_df = nifty_df.copy()
    nifty_df["date"] = pd.to_datetime(nifty_df["date"])
    q = nifty_df.set_index("date")["nifty50_close"].resample("QE").last().dropna()
    ret = q.pct_change().dropna().reset_index()
    ret.columns = ["quarter_end", "benchmark_return"]
    return ret


# 2. Expense Ratio Gap (Direct vs Regular)

def compute_expense_gap(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each fund_family + quarter, compute the implied expense ratio gap
    between regular and direct plans from NAV-based returns.
    Returns: fund_family, quarter_end, expense_gap_annualized
    """
    direct = returns_df[returns_df["plan_type"] == "direct"][[
        "fund_family", "quarter_end", "quarterly_return", "nav_end", "scheme_code"
    ]].rename(columns={
        "quarterly_return": "direct_return",
        "nav_end": "nav_direct",
        "scheme_code": "direct_code"
    })

    regular = returns_df[returns_df["plan_type"] == "regular"][[
        "fund_family", "quarter_end", "quarterly_return", "nav_end", "scheme_code", "amc"
    ]].rename(columns={
        "quarterly_return": "regular_return",
        "nav_end": "nav_regular",
        "scheme_code": "regular_code"
    })

    merged = regular.merge(direct, on=["fund_family", "quarter_end"], how="inner")

    # Expense gap ≈ annualised return difference (direct earns more)
    merged["return_gap_quarterly"] = merged["direct_return"] - merged["regular_return"]
    merged["expense_gap_annualized"] = merged["return_gap_quarterly"] * 4
    merged["expense_gap_annualized"] = merged["expense_gap_annualized"].clip(-0.01, 0.04)

    return merged


# 3. Rolling Fund-Level Features

def compute_rolling_features(nav_df: pd.DataFrame, nifty_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per fund per quarter: rolling Sharpe, beta, information ratio, max drawdown.
    Uses daily NAV and Nifty for each rolling 1-year window.
    """
    nav_df = nav_df.copy()
    nav_df["date"] = pd.to_datetime(nav_df["date"])

    nifty_df = nifty_df.copy()
    nifty_df["date"] = pd.to_datetime(nifty_df["date"])
    nifty_df = nifty_df.sort_values("date")

    results = []
    TRADING_DAYS = 252
    RISK_FREE_ANNUAL = 0.065  # approximate
    risk_free_daily = RISK_FREE_ANNUAL / TRADING_DAYS

    for code, grp in nav_df.groupby("scheme_code"):
        grp = grp[["date", "nav"]].sort_values("date")
        grp["daily_return"] = grp["nav"].pct_change()

        # Merge with benchmark
        merged = grp.merge(
            nifty_df[["date", "nifty50_close"]].rename(columns={"nifty50_close": "bm_price"}),
            on="date", how="inner"
        )
        merged["bm_return"] = merged["bm_price"].pct_change()
        merged = merged.dropna()

        if len(merged) < 60:
            continue

        # Get quarter boundaries from the data
        merged["quarter_end"] = merged["date"].dt.to_period("Q").dt.end_time.dt.normalize()
        quarter_ends = merged["quarter_end"].unique()

        for qe in quarter_ends:
            qe_ts = pd.Timestamp(qe)
            window_start = qe_ts - pd.DateOffset(years=1)
            window = merged[(merged["date"] >= window_start) & (merged["date"] <= qe_ts)]

            if len(window) < 50:
                continue

            fund_ret = window["daily_return"].values
            bm_ret = window["bm_return"].values

            # Sharpe Ratio
            excess = fund_ret - risk_free_daily
            sharpe = (np.mean(excess) / np.std(excess) * np.sqrt(TRADING_DAYS)
                      if np.std(excess) > 0 else np.nan)

            # Beta
            if np.std(bm_ret) > 0:
                beta = np.cov(fund_ret, bm_ret)[0, 1] / np.var(bm_ret)
            else:
                beta = np.nan

            # Information Ratio
            active_ret = fund_ret - bm_ret
            ir = (np.mean(active_ret) / np.std(active_ret) * np.sqrt(TRADING_DAYS)
                  if np.std(active_ret) > 0 else np.nan)

            # Max Drawdown (rolling 1Y)
            nav_series = window["nav"].values
            roll_max = np.maximum.accumulate(nav_series)
            drawdown = (nav_series - roll_max) / roll_max
            max_dd = drawdown.min()

            # Alpha (CAPM)
            expected_ret = risk_free_daily + beta * (np.mean(bm_ret) - risk_free_daily)
            alpha_daily = np.mean(fund_ret) - expected_ret
            alpha_annualized = alpha_daily * TRADING_DAYS

            results.append({
                "scheme_code": code,
                "quarter_end": qe_ts,
                "sharpe_ratio_1y": sharpe,
                "beta_1y": beta,
                "information_ratio_1y": ir,
                "max_drawdown_1y": max_dd,
                "alpha_annualized_1y": alpha_annualized,
            })

    return pd.DataFrame(results)

# 4. AUM Proxy & Turnover

def estimate_aum_proxy(nav_df: pd.DataFrame) -> pd.DataFrame:
    """
    AUM is not in NAV data. We use NAV level as a proxy for fund age/scale,
    and compute a synthetic AUM trend from NAV + growth.
    In production, use mfapi.in scheme details or AMFI monthly AUM reports.
    """
    nav_df = nav_df.copy()
    nav_df["date"] = pd.to_datetime(nav_df["date"])

    aum_rows = []
    rng = np.random.default_rng(42)

    for code, grp in nav_df.groupby("scheme_code"):
        grp = grp.sort_values("date")
        # NAV level roughly correlated with AUM scale
        nav_level = grp["nav"].median()

        # Assign fund to AUM bucket (₹ Crore)
        if nav_level < 20:
            base_aum = rng.uniform(100, 1000)
        elif nav_level < 100:
            base_aum = rng.uniform(1000, 5000)
        else:
            base_aum = rng.uniform(5000, 50000)

        q_dates = grp.set_index("date")["nav"].resample("QE").last().dropna()
        for i, (dt, nav_val) in enumerate(q_dates.items()):
            growth = rng.normal(0.02, 0.08)
            base_aum = max(100, base_aum * (1 + growth))
            aum_rows.append({
                "scheme_code": str(code),
                "quarter_end": dt,
                "aum_crore": base_aum,
                "log_aum": np.log(base_aum + 1),
            })

    return pd.DataFrame(aum_rows)


def estimate_turnover(nav_df: pd.DataFrame) -> pd.DataFrame:
    """
    Portfolio turnover proxy: high volatility of returns relative to benchmark
    suggests more frequent trading.
    """
    nav_df = nav_df.copy()
    nav_df["date"] = pd.to_datetime(nav_df["date"])

    rows = []
    for code, grp in nav_df.groupby("scheme_code"):
        grp = grp.sort_values("date")
        grp["daily_return"] = grp["nav"].pct_change()
        q_grp = grp.set_index("date")["daily_return"].resample("QE")
        for dt, qdata in q_grp:
            qdata = qdata.dropna()
            if len(qdata) < 10:
                continue
            # Turnover proxy = annualised std of daily returns
            turnover_proxy = qdata.std() * np.sqrt(252)
            rows.append({
                "scheme_code": str(code),
                "quarter_end": dt,
                "turnover_proxy": turnover_proxy,
            })
    return pd.DataFrame(rows)

# 5. Manager Tenure Proxy

def estimate_manager_tenure(nav_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate fund manager tenure by measuring NAV history length.
    (Real tenure requires AMFI factsheet scraping — this is a practical proxy.)
    """
    nav_df = nav_df.copy()
    nav_df["date"] = pd.to_datetime(nav_df["date"])

    rows = []
    for code, grp in nav_df.groupby("scheme_code"):
        grp = grp.sort_values("date")
        fund_start = grp["date"].min()
        q_dates = grp.set_index("date")["nav"].resample("QE").last().dropna()
        for dt in q_dates.index:
            tenure_years = (dt - fund_start).days / 365.25
            rows.append({
                "scheme_code": str(code),
                "quarter_end": dt,
                "manager_tenure_years": tenure_years,
            })
    return pd.DataFrame(rows)


# 6. Assemble Master Feature Table

def build_feature_table(
    nav_df: pd.DataFrame,
    nifty_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    save: bool = True
) -> pd.DataFrame:
    """
    Assembles the full feature matrix used for ML.
    Each row = one fund × one quarter.
    """
    print("\nBuilding feature table...")

    # Step 1: Quarterly returns
    print("  Computing quarterly returns...")
    q_returns = compute_quarterly_returns(nav_df)
    bm_returns = compute_benchmark_quarterly(nifty_df)

    # Step 2: Direct/Regular pairs & expense gap
    print("  Computing expense ratio gap...")
    expense_gap = compute_expense_gap(q_returns)

    # Step 3: Rolling features (Sharpe, beta, IR, drawdown)
    print("  Computing rolling features (this may take a minute)...")
    # Use regular plan only for rolling features
    regular_nav = nav_df[nav_df["plan_type"] == "regular"].copy()
    rolling = compute_rolling_features(regular_nav, nifty_df)

    # Step 4: AUM, Turnover, Tenure
    print("  Estimating AUM, turnover, tenure proxies...")
    aum = estimate_aum_proxy(regular_nav)
    turnover = estimate_turnover(regular_nav)
    tenure = estimate_manager_tenure(regular_nav)

    # Step 5: Merge everything
    print("  Merging all features...")

    # Base: expense gap table (one row per regular fund per quarter)
    df = expense_gap.copy()
    df["scheme_code"] = df["regular_code"].astype(str)
    df["quarter_end"] = pd.to_datetime(df["quarter_end"])

    # Benchmark
    bm_returns["quarter_end"] = pd.to_datetime(bm_returns["quarter_end"])
    df = df.merge(bm_returns, on="quarter_end", how="left")

    # Net alpha (regular fund return - benchmark - expense_gap = net alpha)
    df["gross_alpha"] = df["regular_return"] - df["benchmark_return"]
    df["net_alpha"] = df["gross_alpha"] - df["expense_gap_annualized"] / 4  # quarterly

    # Rolling features
    rolling["quarter_end"] = pd.to_datetime(rolling["quarter_end"])
    rolling["scheme_code"] = rolling["scheme_code"].astype(str)
    df = df.merge(rolling, on=["scheme_code", "quarter_end"], how="left")

    # AUM, turnover, tenure
    aum["scheme_code"] = aum["scheme_code"].astype(str)
    aum["quarter_end"] = pd.to_datetime(aum["quarter_end"])
    df = df.merge(aum, on=["scheme_code", "quarter_end"], how="left")

    turnover["scheme_code"] = turnover["scheme_code"].astype(str)
    turnover["quarter_end"] = pd.to_datetime(turnover["quarter_end"])
    df = df.merge(turnover, on=["scheme_code", "quarter_end"], how="left")

    tenure["scheme_code"] = tenure["scheme_code"].astype(str)
    tenure["quarter_end"] = pd.to_datetime(tenure["quarter_end"])
    df = df.merge(tenure, on=["scheme_code", "quarter_end"], how="left")

    # Macro features
    macro_df = macro_df.copy()
    macro_df["date"] = pd.to_datetime(macro_df["date"])
    macro_df["quarter_end"] = macro_df["date"].dt.to_period("Q").dt.end_time.dt.normalize()
    macro_q = macro_df.groupby("quarter_end").last().reset_index()
    macro_q["quarter_end"] = pd.to_datetime(macro_q["quarter_end"])

    df = df.merge(
        macro_q[["quarter_end", "repo_rate", "cpi_yoy", "yield_10y",
                  "yield_spread", "fii_flow", "dii_flow"]],
        on="quarter_end", how="left"
    )

    # Step 6: Target variable
    # Binary: 1 if next quarter net alpha > 0 (commission justified)
    df = df.sort_values(["scheme_code", "quarter_end"])
    df["next_quarter_net_alpha"] = df.groupby("scheme_code")["net_alpha"].shift(-1)
    df["label_commission_justified"] = (df["next_quarter_net_alpha"] > 0).astype(int)

    # Also keep regression target
    df["target_net_alpha"] = df["next_quarter_net_alpha"]

    # Drop last quarter per fund (no next quarter to predict)
    df = df.dropna(subset=["next_quarter_net_alpha"])

    # Reorder columns
    feature_cols = [
        "fund_family", "amc", "scheme_code", "quarter_end",
        "regular_return", "direct_return", "benchmark_return",
        "gross_alpha", "expense_gap_annualized", "net_alpha",
        "sharpe_ratio_1y", "beta_1y", "information_ratio_1y",
        "max_drawdown_1y", "alpha_annualized_1y",
        "aum_crore", "log_aum", "turnover_proxy", "manager_tenure_years",
        "repo_rate", "cpi_yoy", "yield_10y", "yield_spread", "fii_flow", "dii_flow",
        "next_quarter_net_alpha", "target_net_alpha", "label_commission_justified",
    ]
    available = [c for c in feature_cols if c in df.columns]
    df = df[available]

    print(f"  → Feature table: {len(df)} rows × {len(df.columns)} columns")
    print(f"  → Commission justified: {df['label_commission_justified'].mean():.1%}")

    if save:
        df.to_parquet(DATA_DIR / "features.parquet", index=False)
        df.to_csv(DATA_DIR / "features.csv", index=False)
        print("  Saved features.parquet and features.csv")

    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    nav_df = pd.read_parquet(DATA_DIR / "nav_history.parquet")
    nifty_df = pd.read_csv(DATA_DIR / "nifty50.csv", parse_dates=["date"])
    macro_df = pd.read_csv(DATA_DIR / "macro.csv", parse_dates=["date"])

    features = build_feature_table(nav_df, nifty_df, macro_df)
    print(features.head())
