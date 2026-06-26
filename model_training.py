"""
CommissionLens — ML Model Training
XGBoost classifier (commission justified?) + regressor (net alpha prediction).
Includes temporal train/test split, SHAP explainability, and evaluation metrics.
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
from pathlib import Path
from datetime import datetime

import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
REPORTS_DIR = Path(__file__).parent / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "regular_return", "benchmark_return", "gross_alpha",
    "expense_gap_annualized", "net_alpha",
    "sharpe_ratio_1y", "beta_1y", "information_ratio_1y",
    "max_drawdown_1y", "alpha_annualized_1y",
    "log_aum", "turnover_proxy", "manager_tenure_years",
]


# ─────────────────────────────────────────────
# 1. Data Loading & Temporal Split
# ─────────────────────────────────────────────

def load_features(path: Path = None) -> pd.DataFrame:
    path = path or DATA_DIR / "features.parquet"
    if path.exists():
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(DATA_DIR / "features.csv", parse_dates=["quarter_end"])
    df["quarter_end"] = pd.to_datetime(df["quarter_end"])
    return df


def temporal_train_test_split(
    df: pd.DataFrame,
    test_quarters: int = 4,
    val_quarters: int = 4,
) -> tuple:
    """
    Temporal split to avoid look-ahead bias.
    Train → Validation → Test (no shuffle).
    """
    df = df.sort_values("quarter_end")
    all_quarters = df["quarter_end"].sort_values().unique()

    test_cutoff = all_quarters[-test_quarters]
    val_cutoff = all_quarters[-(test_quarters + val_quarters)]

    train = df[df["quarter_end"] < val_cutoff].copy()
    val = df[(df["quarter_end"] >= val_cutoff) & (df["quarter_end"] < test_cutoff)].copy()
    test = df[df["quarter_end"] >= test_cutoff].copy()

    print(f"  Train: {len(train)} rows  ({train['quarter_end'].min().date()} → "
          f"{train['quarter_end'].max().date()})")
    print(f"  Val:   {len(val)} rows  ({val['quarter_end'].min().date()} → "
          f"{val['quarter_end'].max().date()})")
    print(f"  Test:  {len(test)} rows  ({test['quarter_end'].min().date()} → "
          f"{test['quarter_end'].max().date()})")

    return train, val, test


def prepare_arrays(df: pd.DataFrame, feature_cols: list = None):
    """Drop NaN feature rows and return X, y arrays."""
    feature_cols = feature_cols or FEATURE_COLS
    available = [c for c in feature_cols if c in df.columns]
    sub = df[available + ["label_commission_justified", "target_net_alpha"]].dropna()
    X = sub[available].values
    y_cls = sub["label_commission_justified"].values
    y_reg = sub["target_net_alpha"].values
    return X, y_cls, y_reg, available


# ─────────────────────────────────────────────
# 2. XGBoost Classifier — Commission Justified?
# ─────────────────────────────────────────────

def train_classifier(
    X_train, y_train, X_val, y_val,
    feature_names: list
) -> xgb.XGBClassifier:
    """Train XGBoost binary classifier with early stopping on validation AUC."""
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=3,
        min_child_weight=30,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        reg_alpha=0.5,
        reg_lambda=5.0,
        scale_pos_weight=pos_weight,
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"  Best iteration: {model.best_iteration}  |  Val AUC: {model.best_score:.4f}")
    return model


# ─────────────────────────────────────────────
# 3. XGBoost Regressor — Net Alpha Prediction
# ─────────────────────────────────────────────

def train_regressor(
    X_train, y_train, X_val, y_val,
    feature_names: list
) -> xgb.XGBRegressor:
    """Train XGBoost regressor to predict next-quarter net alpha."""
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=3,
        min_child_weight=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=5.0,
        eval_metric="rmse",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"  Best iteration: {model.best_iteration}  |  Val RMSE: {model.best_score:.6f}")
    return model


# ─────────────────────────────────────────────
# 4. Evaluation
# ─────────────────────────────────────────────

def evaluate_classifier(model, X_test, y_test, feature_names, threshold=0.5) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else np.nan
    f1 = f1_score(y_test, preds, zero_division=0)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)

    # Precision at top decile
    top_decile_idx = np.argsort(proba)[-max(1, len(proba) // 10):]
    precision_top10 = y_test[top_decile_idx].mean()

    metrics = {
        "auc_roc": round(auc, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "precision_top_decile": round(precision_top10, 4),
        "n_test": len(y_test),
        "positive_rate": round(y_test.mean(), 4),
    }

    print("\n  ── Classifier Metrics ──────────────────")
    for k, v in metrics.items():
        print(f"  {k:<28} {v}")
    print()
    print(classification_report(y_test, preds, target_names=["Not Justified", "Justified"],
                                 zero_division=0))
    return metrics


def evaluate_regressor(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    # Direction accuracy
    dir_acc = np.mean(np.sign(preds) == np.sign(y_test))

    metrics = {
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "direction_accuracy": round(dir_acc, 4),
        "mean_predicted_alpha": round(preds.mean(), 6),
        "mean_actual_alpha": round(y_test.mean(), 6),
    }

    print("\n  ── Regressor Metrics ───────────────────")
    for k, v in metrics.items():
        print(f"  {k:<28} {v}")
    return metrics


# ─────────────────────────────────────────────
# 5. SHAP Explainability
# ─────────────────────────────────────────────

def compute_shap_values(model, X_test, feature_names: list) -> pd.DataFrame:
    """Compute SHAP values and return summary as DataFrame."""
    print("\n  Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # positive class

    shap_df = pd.DataFrame(shap_vals, columns=feature_names)
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)

    print("\n  ── Top 10 Features by SHAP Importance ──")
    for feat, val in mean_abs.head(10).items():
        print(f"  {feat:<35} {val:.6f}")

    return shap_df


def plot_shap_summary(shap_df: pd.DataFrame, X_test, feature_names: list, save_path: Path):
    """Save SHAP beeswarm summary plot."""
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    mean_abs = shap_df.abs().mean().sort_values(ascending=False).head(12)
    top_features = mean_abs.index.tolist()
    top_idx = [feature_names.index(f) for f in top_features]

    colors = ["#f5a623" if v > 0 else "#4a9eff" for v in shap_df[top_features].mean()]

    ax.barh(range(len(top_features)), mean_abs.values, color=colors, edgecolor="none", alpha=0.85)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, color="white", fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", color="white")
    ax.set_title("Feature Importance — CommissionLens", color="white", fontsize=12, pad=12)
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    ax.axvline(0, color="white", alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close()
    print(f"  SHAP plot saved → {save_path.name}")


# ─────────────────────────────────────────────
# 6. SIP Back-Validation (XIRR simulation)
# ─────────────────────────────────────────────

def xirr_newton(cashflows: list, dates: list, guess: float = 0.1) -> float:
    """Compute XIRR using Newton-Raphson method."""
    t0 = dates[0]
    years = [(d - t0).days / 365.25 for d in dates]
    amounts = cashflows

    for _ in range(100):
        npv = sum(a / (1 + guess) ** t for a, t in zip(amounts, years))
        d_npv = sum(-t * a / (1 + guess) ** (t + 1) for a, t in zip(amounts, years))
        if abs(d_npv) < 1e-12:
            break
        guess -= npv / d_npv
        if guess <= -1:
            guess = -0.999

    return guess


def simulate_sip_backtest(
    df: pd.DataFrame,
    nav_df: pd.DataFrame,
    classifier: xgb.XGBClassifier,
    feature_cols: list,
    monthly_sip: float = 5000,
    start_date: str = "2019-04-01",
    end_date: str = "2023-12-31",
) -> dict:
    """
    Back-validation: Compare XIRR of model-guided direct fund selection
    vs naive regular plan investing.
    """
    print("\n  Running SIP back-validation (2019–2023)...")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    sip_dates = pd.date_range(start, end, freq="MS")

    feature_df = load_features()
    feature_df = feature_df[feature_df["quarter_end"] >= start - pd.DateOffset(months=3)]

    # For each SIP date, pick top-3 model-recommended funds
    nav_df = nav_df.copy()
    nav_df["date"] = pd.to_datetime(nav_df["date"])
    direct_nav = nav_df[nav_df["plan_type"] == "direct"].copy()

    model_units = {}
    naive_units = {}

    for sip_date in sip_dates:
        # Get latest feature data before this date
        q_data = feature_df[feature_df["quarter_end"] < sip_date].copy()
        if q_data.empty:
            continue

        latest = q_data.sort_values("quarter_end").groupby("scheme_code").last().reset_index()
        avail_feats = [c for c in feature_cols if c in latest.columns]
        X = latest[avail_feats].dropna()

        if X.empty:
            continue

        # Reset index so iloc works correctly
        X_reset = X.reset_index(drop=True)
        latest_reset = latest.iloc[X.index].reset_index(drop=True)

        # Model picks top funds
        proba = classifier.predict_proba(X_reset.values)[:, 1]
        top_positions = np.argsort(proba)[-3:]

        sip_amount_per_fund = monthly_sip / max(len(top_positions), 1)

        # Invest in model-selected direct funds
        for idx in top_positions:
            row = latest_reset.iloc[idx]
            family = row["fund_family"]
            # Find the scheme_code for the direct plan of this fund_family
            nav_on_date = direct_nav[
                (direct_nav["fund_family"] == family) &
                (direct_nav["date"] <= sip_date)
            ]
            if nav_on_date.empty:
                continue
            code = nav_on_date.iloc[-1]["scheme_code"]
            nav_val = nav_on_date.iloc[-1]["nav"]
            units = sip_amount_per_fund / nav_val
            model_units[code] = model_units.get(code, 0) + units

        # Naive: invest equally in all regular plans
        all_codes = latest["scheme_code"].unique()
        naive_per_fund = monthly_sip / max(len(all_codes), 1)
        regular_nav = nav_df[nav_df["plan_type"] == "regular"].copy()
        for code in all_codes[:20]:  # limit to 20 for speed
            nav_on_date = regular_nav[
                (regular_nav["scheme_code"] == code) &
                (regular_nav["date"] <= sip_date)
            ]
            if nav_on_date.empty:
                continue
            nav_val = nav_on_date.iloc[-1]["nav"]
            units = naive_per_fund / nav_val
            naive_units[code] = naive_units.get(code, 0) + units

    # Compute final value at end_date
    def compute_value(units_dict: dict, plan_type: str) -> float:
        nav_sub = nav_df[
            (nav_df["plan_type"] == plan_type) &
            (nav_df["date"] <= end)
        ]
        total = 0
        for code, units in units_dict.items():
            final_nav = nav_sub[nav_sub["scheme_code"] == code]
            if final_nav.empty:
                continue
            total += units * final_nav.iloc[-1]["nav"]
        return total

    model_value = compute_value(model_units, "direct")
    naive_value = compute_value(naive_units, "regular")

    total_invested = monthly_sip * len(sip_dates)

    # Approximate XIRR (simplified annualised return)
    years = (end - start).days / 365.25
    model_xirr = ((model_value / total_invested) ** (1 / years) - 1) if total_invested > 0 else 0
    naive_xirr = ((naive_value / total_invested) ** (1 / years) - 1) if total_invested > 0 else 0

    result = {
        "total_invested": round(total_invested, 0),
        "model_final_value": round(model_value, 0),
        "naive_final_value": round(naive_value, 0),
        "model_xirr": round(model_xirr * 100, 2),
        "naive_xirr": round(naive_xirr * 100, 2),
        "excess_return_pct": round((model_xirr - naive_xirr) * 100, 2),
    }

    print(f"\n  ── SIP Back-Validation Results ─────────")
    print(f"  Total Invested:         ₹{result['total_invested']:,.0f}")
    print(f"  Model Portfolio Value:  ₹{result['model_final_value']:,.0f}  "
          f"(XIRR: {result['model_xirr']}%)")
    print(f"  Naive Regular Value:    ₹{result['naive_final_value']:,.0f}  "
          f"(XIRR: {result['naive_xirr']}%)")
    print(f"  Excess Return:          {result['excess_return_pct']}% p.a.")

    return result


# ─────────────────────────────────────────────
# 7. Main Training Pipeline
# ─────────────────────────────────────────────

def run_training_pipeline(run_sip_backtest: bool = True):
    print("\n" + "="*55)
    print("  CommissionLens — ML Training Pipeline")
    print("="*55)

    # Load data
    print("\nLoading feature data...")
    df = load_features()
    print(f"  {len(df)} rows, {len(df.columns)} columns")
    print(f"  Date range: {df['quarter_end'].min().date()} → {df['quarter_end'].max().date()}")

    # Split
    print("\nTemporal train/val/test split...")
    train, val, test = temporal_train_test_split(df)

    # Prepare arrays
    X_train, y_cls_train, y_reg_train, feat_names = prepare_arrays(train)
    X_val, y_cls_val, y_reg_val, _ = prepare_arrays(val)
    X_test, y_cls_test, y_reg_test, _ = prepare_arrays(test)

    print(f"\n  Features used: {feat_names}")

    if len(X_train) < 10:
        print("\n  ⚠️  Too few training samples. Run data_fetcher.py first.")
        return

    # Train classifier
    print("\nTraining classifier (commission justified / not)...")
    classifier = train_classifier(X_train, y_cls_train, X_val, y_cls_val, feat_names)

    # Train regressor
    print("\nTraining regressor (net alpha prediction)...")
    regressor = train_regressor(X_train, y_reg_train, X_val, y_reg_val, feat_names)

    # Evaluate
    print("\nEvaluating on test set...")
    cls_metrics = evaluate_classifier(classifier, X_test, y_cls_test, feat_names)
    reg_metrics = evaluate_regressor(regressor, X_test, y_reg_test)

    # SHAP
    shap_df = compute_shap_values(classifier, X_test, feat_names)
    plot_shap_summary(shap_df, X_test, feat_names, REPORTS_DIR / "shap_importance.png")

    # Save models
    joblib.dump(classifier, MODELS_DIR / "classifier.pkl")
    joblib.dump(regressor, MODELS_DIR / "regressor.pkl")
    joblib.dump(feat_names, MODELS_DIR / "feature_names.pkl")
    print(f"\n  Models saved to {MODELS_DIR}/")

    # Save metrics report
    report = {
        "run_timestamp": datetime.now().isoformat(),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "features": feat_names,
        "classifier_metrics": cls_metrics,
        "regressor_metrics": reg_metrics,
    }

    # SIP backtest
    if run_sip_backtest:
        try:
            nav_df = pd.read_parquet(DATA_DIR / "nav_history.parquet")
            sip_result = simulate_sip_backtest(df, nav_df, classifier, feat_names)
            report["sip_backtest"] = sip_result
        except Exception as e:
            print(f"  SIP backtest skipped: {e}")

    def _json_safe(obj):
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        return obj

    with open(REPORTS_DIR / "model_report.json", "w") as f:
        json.dump(_json_safe(report), f, indent=2)
    print(f"  Metrics saved to {REPORTS_DIR}/model_report.json")

    # Save SHAP feature importance CSV
    mean_shap = shap_df.abs().mean().sort_values(ascending=False)
    mean_shap.to_csv(REPORTS_DIR / "shap_feature_importance.csv", header=["mean_abs_shap"])
    print(f"  SHAP importance saved to {REPORTS_DIR}/shap_feature_importance.csv")

    print("\n✓ Training pipeline complete.\n")
    return classifier, regressor, feat_names, report


if __name__ == "__main__":
    run_training_pipeline()
