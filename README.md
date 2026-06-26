# CommissionLens 🔍
**Commission-Adjusted Alpha Prediction in Indian Mutual Funds**

> *FEC · IIT Guwahati *

---

## Problem Statement

India has over 90 million demat accounts, yet most retail investors are unaware of the
**expense ratio gap** between regular and direct mutual fund plans. Regular plans charge
0.5%–1.5% more annually as distributor commission. Over 20 years on a ₹5,000/month SIP,
this compounding drag can erode **₹8–12 lakh** in wealth.

**CommissionLens answers:** *"Is this fund generating enough alpha to justify its commission cost?"*

---

## Architecture

```
commissionlens/
├── pipeline.py              # Orchestrator — run this first
├── data_fetcher.py          # AMFI / NSE / RBI data collection
├── feature_engineering.py   # Expense gap, Sharpe, beta, IR, macro features
├── model_training.py        # XGBoost + SHAP + SIP back-validation
├── app.py                   # Streamlit dashboard
├── requirements.txt
├── data/                    # Auto-created: nav_history.parquet, features.parquet, etc.
├── models/                  # Auto-created: classifier.pkl, regressor.pkl
├── reports/                 # Auto-created: model_report.json, shap_importance.png
└── notebooks/
    └── commissionlens_notebook.ipynb
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (fetch data + train model)
python pipeline.py --funds 50          # ~10 min, recommended
python pipeline.py --funds 100         # ~20 min, better model

# 3. Launch the dashboard
streamlit run app.py

# 4. Or explore the notebook
jupyter notebook notebooks/commissionlens_notebook.ipynb
```

---

## Goals & Implementation

| Goal | Module | Status |
|------|--------|--------|
| Compute commission-adjusted net alpha | `feature_engineering.py` | ✅ |
| Engineer fund-level features (expense gap, IR, Sharpe, beta, AUM, turnover, tenure) | `feature_engineering.py` | ✅ |
| Incorporate macro regime variables (repo rate, CPI, yield curve, FII/DII) | `data_fetcher.py` + `feature_engineering.py` | ✅ |
| Train ML models predicting next-quarter net alpha | `model_training.py` | ✅ |
| Binary classification: commission justified / not | `model_training.py` | ✅ |
| SHAP feature importance | `model_training.py` | ✅ |
| SIP back-validation (XIRR comparison) | `model_training.py` | ✅ |
| Streamlit dashboard | `app.py` | ✅ |

---

## Data Sources

| Source | What we fetch | API/URL |
|--------|--------------|---------|
| **AMFI** | All fund schemes + live NAV | `amfiindia.com/spages/NAVAll.txt` |
| **mfapi.in** | Full NAV history per fund | `api.mfapi.in/mf/{scheme_code}` |
| **NSE / Stooq** | Nifty50 historical prices | NSE API → Stooq fallback |
| **RBI** | Repo rate history | Hardcoded known policy rate changes |
| **Government data** | CPI YoY inflation | Official MOSPI series (hardcoded) |
| **Debt market** | 10Y Gsec yield, yield spread | Known series (approximate) |
| **SEBI** | FII/DII net equity flows | Known monthly aggregates |

---

## Features Used by the Model

| Feature | Description |
|---------|-------------|
| `expense_gap_annualized` | Direct − Regular plan return difference (annualised) |
| `gross_alpha` | Fund return − Nifty50 benchmark return |
| `net_alpha` | gross_alpha − expense_gap (quarterly) |
| `sharpe_ratio_1y` | Rolling 1-year Sharpe ratio |
| `beta_1y` | Rolling 1-year market beta |
| `information_ratio_1y` | Rolling 1-year IR (active return / tracking error) |
| `max_drawdown_1y` | Worst rolling 1-year drawdown |
| `alpha_annualized_1y` | CAPM alpha annualised |
| `log_aum` | Log of fund AUM in ₹ Crore |
| `turnover_proxy` | Annualised daily return volatility (trading cost proxy) |
| `manager_tenure_years` | Fund history length as tenure proxy |
| `repo_rate` | RBI repo rate at quarter end |
| `cpi_yoy` | CPI YoY inflation |
| `yield_10y` | 10Y Gsec yield |
| `yield_spread` | 10Y − 2Y yield spread (recession signal) |
| `fii_flow` | FII net equity flow (₹ Crore) |
| `dii_flow` | DII net equity flow (₹ Crore) |

---

## Target Variables

- **`label_commission_justified`** (binary, 0/1): Is next quarter's net alpha > 0?
- **`target_net_alpha`** (continuous): Predicted next-quarter net alpha

---

## Concepts Covered

- ✅ Net Alpha = Gross Return − Benchmark − Expense Gap
- ✅ Information Ratio & alpha persistence
- ✅ XIRR for SIP cash flows
- ✅ Temporal train-test split (no look-ahead bias)
- ✅ SHAP feature importance
- ✅ Rupee Cost Averaging & macro regime features
- ✅ Portfolio Turnover & hidden trading costs

---

## Model Performance (typical, 50 fund pairs)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.68–0.75 |
| F1 Score | 0.60–0.70 |
| Precision @ Top 10% | 0.70–0.80 |
| Net Alpha RMSE | 0.008–0.015 |

*Results vary with number of funds and market conditions in the test window.*

---

## Dashboard Pages

1. **Dashboard** — Overview metrics, expense gap distribution, net alpha over time
2. **Fund Scorer** — Enter any fund's features → get commission-justification score (0–100%)
3. **Feature Explorer** — Scatter plots and distributions of all features
4. **SIP Simulator** — XIRR comparison: model vs naive + manual SIP calculator
5. **Model Insights** — Performance metrics + SHAP importance chart
