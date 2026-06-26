"""
CommissionLens — Streamlit Dashboard
Interactive UI: fund score lookup, feature explorer, SIP simulator.
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"
REPORTS_DIR = BASE / "reports"

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CommissionLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Barlow', sans-serif; }
  .main { background-color: #111; color: #e0e0e0; }
  .stApp { background-color: #111; }
  h1 { font-family: 'Share Tech Mono', monospace; color: #f5a623; letter-spacing: 2px; }
  h2, h3 { color: #f0f0f0; }
  .metric-card {
    background: #1e1e1e; border-left: 3px solid #f5a623;
    padding: 12px 16px; border-radius: 4px; margin: 4px 0;
  }
  .justified { color: #4caf50; font-weight: bold; }
  .not-justified { color: #f44336; font-weight: bold; }
  .score-ring {
    font-size: 3rem; font-weight: 700;
    color: #f5a623; text-align: center; font-family: 'Share Tech Mono';
  }
</style>
""", unsafe_allow_html=True)


# ── Load resources ──────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    clf_path = MODELS_DIR / "classifier.pkl"
    reg_path = MODELS_DIR / "regressor.pkl"
    feat_path = MODELS_DIR / "feature_names.pkl"
    if clf_path.exists():
        clf = joblib.load(clf_path)
        reg = joblib.load(reg_path)
        feats = joblib.load(feat_path)
        return clf, reg, feats
    return None, None, None


@st.cache_data
def load_feature_data():
    p = DATA_DIR / "features.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        df["quarter_end"] = pd.to_datetime(df["quarter_end"])
        return df
    p2 = DATA_DIR / "features.csv"
    if p2.exists():
        return pd.read_csv(p2, parse_dates=["quarter_end"])
    return pd.DataFrame()


@st.cache_data
def load_report():
    p = REPORTS_DIR / "model_report.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_shap():
    p = REPORTS_DIR / "shap_feature_importance.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0)
    return pd.DataFrame()


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🔍 CommissionLens")
    st.markdown("**Commission-Adjusted Alpha Prediction**  \n*IIT Guwahati — FEC*")
    st.divider()
    page = st.radio("Navigate", [
        "🏠 Dashboard",
        "🔎 Fund Scorer",
        "📊 Feature Explorer",
        "💰 SIP Simulator",
        "🧠 Model Insights",
    ])
    st.divider()
    st.markdown("**Data Sources**")
    st.markdown("- AMFI NAV API\n- mfapi.in\n- RBI Repo Rate\n- NSE Nifty50\n- CPI / Yield Data")


# ── Load ────────────────────────────────────────────────────────────────────
clf, reg, feat_names = load_models()
features_df = load_feature_data()
report = load_report()
shap_df = load_shap()

models_ready = clf is not None
data_ready = not features_df.empty

# ── Helper ───────────────────────────────────────────────────────────────────
def score_color(score: float) -> str:
    if score >= 0.7:
        return "#4caf50"
    if score >= 0.4:
        return "#f5a623"
    return "#f44336"


def dark_fig(figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    ax.tick_params(colors="#ccc")
    ax.spines[:].set_color("#333")
    ax.xaxis.label.set_color("#ccc")
    ax.yaxis.label.set_color("#ccc")
    ax.title.set_color("#f5a623")
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("# COMMISS**ION**LENS")
    st.markdown("### Commission-Adjusted Alpha Prediction in Indian Mutual Funds")
    st.divider()

    # Status cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Fund Pairs", len(features_df["scheme_code"].unique()) if data_ready else 0,
                  help="Unique direct/regular pairs")
    with col2:
        st.metric("Quarters Covered",
                  features_df["quarter_end"].nunique() if data_ready else 0)
    with col3:
        if data_ready:
            pct = features_df["label_commission_justified"].mean() * 100
            st.metric("% Commission Justified", f"{pct:.1f}%")
        else:
            st.metric("% Commission Justified", "N/A")
    with col4:
        auc = report.get("classifier_metrics", {}).get("auc_roc", "—")
        st.metric("Model AUC-ROC", auc)

    st.divider()

    if data_ready:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Expense Gap Distribution (annualized)")
            fig, ax = dark_fig()
            vals = features_df["expense_gap_annualized"].dropna() * 100
            ax.hist(vals, bins=40, color="#f5a623", edgecolor="none", alpha=0.85)
            ax.axvline(vals.median(), color="white", linestyle="--", linewidth=1,
                       label=f"Median: {vals.median():.2f}%")
            ax.set_xlabel("Expense Gap (%)")
            ax.set_ylabel("Count")
            ax.legend(labelcolor="white", facecolor="#1e1e1e", edgecolor="#333")
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.subheader("Net Alpha Over Time")
            fig, ax = dark_fig()
            monthly = features_df.groupby("quarter_end")["net_alpha"].mean() * 100
            ax.plot(monthly.index, monthly.values, color="#4a9eff", linewidth=1.5)
            ax.axhline(0, color="#f44336", linewidth=0.8, linestyle="--")
            ax.fill_between(monthly.index, monthly.values, 0,
                            where=(monthly.values > 0), alpha=0.2, color="#4caf50")
            ax.fill_between(monthly.index, monthly.values, 0,
                            where=(monthly.values < 0), alpha=0.2, color="#f44336")
            ax.set_ylabel("Avg Net Alpha (%)")
            st.pyplot(fig)
            plt.close()

    else:
        st.info("⚡ Run `python pipeline.py` to fetch data and train the model first.")
        st.code("""# Quick start
python pipeline.py --funds 50  # fetches 50 fund pairs & trains model
# Then relaunch:
streamlit run app.py""", language="bash")

    # Problem statement card
    st.divider()
    st.subheader("Why This Matters")
    st.markdown("""
India has **90 million+ demat accounts**, yet most retail investors don't realise
they're paying **0.5%–1.5% extra annually** in distributor commissions via regular plans.
Over 20 years on a ₹5,000/month SIP, this compounds to **₹8–12 lakh in lost wealth**.

CommissionLens answers: *"Is this fund's alpha large enough to justify its commission cost?"*
— using historical NAV data, macro indicators, and XGBoost with SHAP explainability.
""")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Fund Scorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔎 Fund Scorer":
    st.markdown("## Fund Commission-Justification Scorer")
    st.markdown("Enter a fund's features to get its predicted commission-justification score.")

    if not models_ready:
        st.warning("Models not trained yet. Run `python pipeline.py` first.")
        st.stop()

    # Fund name lookup from feature data
    col_search, col_info = st.columns([2, 1])

    with col_search:
        if data_ready:
            fund_list = features_df["fund_family"].dropna().unique().tolist()
            selected_fund = st.selectbox("Select a Fund (latest quarter)", ["— Manual Input —"] + sorted(fund_list))
        else:
            selected_fund = "— Manual Input —"

    # Pre-fill if fund selected
    prefill = {}
    if selected_fund != "— Manual Input —" and data_ready:
        latest = features_df[features_df["fund_family"] == selected_fund].sort_values("quarter_end").iloc[-1]
        prefill = latest.to_dict()

    st.divider()
    st.markdown("#### Fund Features")

    col1, col2, col3 = st.columns(3)
    with col1:
        val_eg = float(prefill.get("expense_gap_annualized", 0.01)) * 100
        val_eg = max(-1.0, min(5.0, val_eg))
        expense_gap = st.number_input("Expense Gap (annualized %)",
                                       value=val_eg,
                                       min_value=-1.0, max_value=5.0, step=0.01) / 100
        
        val_ga = float(prefill.get("gross_alpha", 0.005)) * 100
        val_ga = max(-20.0, min(20.0, val_ga))
        gross_alpha = st.number_input("Gross Alpha (quarterly %)",
                                       value=val_ga,
                                       min_value=-20.0, max_value=20.0, step=0.01) / 100

        val_rr = float(prefill.get("regular_return", 0.025)) * 100
        val_rr = max(-50.0, min(50.0, val_rr))
        regular_return = st.number_input("Regular Plan Quarterly Return (%)",
                                          value=val_rr,
                                          min_value=-50.0, max_value=50.0, step=0.1) / 100

        val_br = float(prefill.get("benchmark_return", 0.02)) * 100
        val_br = max(-50.0, min(50.0, val_br))
        benchmark_return = st.number_input("Benchmark (Nifty50) Quarterly Return (%)",
                                            value=val_br,
                                            min_value=-50.0, max_value=50.0, step=0.1) / 100

    with col2:
        val_sharpe = float(prefill.get("sharpe_ratio_1y", 0.5))
        val_sharpe = max(-10.0, min(10.0, val_sharpe))
        sharpe = st.number_input("Sharpe Ratio (1Y)",
                                  value=val_sharpe,
                                  min_value=-10.0, max_value=10.0, step=0.05)

        val_beta = float(prefill.get("beta_1y", 0.95))
        val_beta = max(-5.0, min(5.0, val_beta))
        beta = st.number_input("Beta (1Y)",
                                value=val_beta,
                                min_value=-5.0, max_value=5.0, step=0.01)

        val_ir = float(prefill.get("information_ratio_1y", 0.1))
        val_ir = max(-10.0, min(10.0, val_ir))
        ir = st.number_input("Information Ratio (1Y)",
                              value=val_ir,
                              min_value=-10.0, max_value=10.0, step=0.05)

        val_mdd = float(prefill.get("max_drawdown_1y", -0.12)) * 100
        val_mdd = max(-100.0, min(0.0, val_mdd))
        max_dd = st.number_input("Max Drawdown (1Y, negative %)",
                                  value=val_mdd,
                                  min_value=-100.0, max_value=0.0, step=0.5) / 100

    with col3:
        val_aum = float(prefill.get("log_aum", 8.0))
        val_aum = max(0.0, min(20.0, val_aum))
        log_aum = st.number_input("Log(AUM ₹ Crore)",
                                   value=val_aum,
                                   min_value=0.0, max_value=20.0, step=0.1)

        val_to = float(prefill.get("turnover_proxy", 0.18))
        val_to = max(0.0, min(5.0, val_to))
        turnover = st.number_input("Turnover Proxy",
                                    value=val_to,
                                    min_value=0.0, max_value=5.0, step=0.01)

        val_tenure = float(prefill.get("manager_tenure_years", 3.0))
        val_tenure = max(0.0, min(50.0, val_tenure))
        tenure = st.number_input("Manager Tenure (years)",
                                  value=val_tenure,
                                  min_value=0.0, max_value=50.0, step=0.5)

        val_repo = float(prefill.get("repo_rate", 6.5))
        val_repo = max(0.0, min(20.0, val_repo))
        repo_rate = st.number_input("RBI Repo Rate (%)",
                                     value=val_repo,
                                     min_value=0.0, max_value=20.0, step=0.25)

    st.markdown("#### Macro Features")
    m1, m2, m3 = st.columns(3)
    with m1:
        val_cpi = float(prefill.get("cpi_yoy", 5.2))
        val_cpi = max(0.0, min(20.0, val_cpi))
        cpi = st.number_input("CPI YoY (%)", value=val_cpi,
                               min_value=0.0, max_value=20.0, step=0.1)
    with m2:
        val_y10 = float(prefill.get("yield_10y", 7.1))
        val_y10 = max(0.0, min(20.0, val_y10))
        yield_10y = st.number_input("10Y Gsec Yield (%)", value=val_y10,
                                     min_value=0.0, max_value=20.0, step=0.05)

        val_ys = float(prefill.get("yield_spread", 0.3))
        val_ys = max(-5.0, min(5.0, val_ys))
        yield_spread = st.number_input("Yield Spread (10Y-2Y %)", value=val_ys,
                                        min_value=-5.0, max_value=5.0, step=0.05)
    with m3:
        val_fii = float(prefill.get("fii_flow", 5000.0))
        val_fii = max(-500000.0, min(500000.0, val_fii))
        fii = st.number_input("FII Net Flow (₹ Cr)", value=val_fii,
                               min_value=-500000.0, max_value=500000.0, step=500.0)

        val_dii = float(prefill.get("dii_flow", 3000.0))
        val_dii = max(-500000.0, min(500000.0, val_dii))
        dii = st.number_input("DII Net Flow (₹ Cr)", value=val_dii,
                               min_value=-500000.0, max_value=500000.0, step=500.0)

    net_alpha = gross_alpha - expense_gap / 4  # quarterly
    alpha_1y = net_alpha * 4

    if st.button("🔍 Score This Fund", type="primary"):
        input_map = {
            "regular_return": regular_return,
            "benchmark_return": benchmark_return,
            "gross_alpha": gross_alpha,
            "expense_gap_annualized": expense_gap,
            "net_alpha": net_alpha,
            "sharpe_ratio_1y": sharpe,
            "beta_1y": beta,
            "information_ratio_1y": ir,
            "max_drawdown_1y": max_dd,
            "alpha_annualized_1y": alpha_1y,
            "log_aum": log_aum,
            "turnover_proxy": turnover,
            "manager_tenure_years": tenure,
            "repo_rate": repo_rate,
            "cpi_yoy": cpi,
            "yield_10y": yield_10y,
            "yield_spread": yield_spread,
            "fii_flow": fii,
            "dii_flow": dii,
        }

        X = np.array([[input_map.get(f, 0.0) for f in feat_names]])
        proba = clf.predict_proba(X)[0][1]
        predicted_alpha = reg.predict(X)[0]

        st.divider()
        r1, r2, r3 = st.columns(3)
        with r1:
            color = score_color(proba)
            st.markdown(f"""
            <div class="metric-card">
                <div style="color:#aaa;font-size:12px;">COMMISSION JUSTIFICATION SCORE</div>
                <div class="score-ring" style="color:{color}">{proba*100:.0f}%</div>
                <div style="text-align:center;color:{color};margin-top:4px">
                    {'✅ JUSTIFIED' if proba >= 0.5 else '❌ NOT JUSTIFIED'}
                </div>
            </div>""", unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color:#aaa;font-size:12px;">PREDICTED NEXT-QTR NET ALPHA</div>
                <div class="score-ring" style="color:{'#4caf50' if predicted_alpha > 0 else '#f44336'}">
                    {predicted_alpha*100:+.2f}%
                </div>
                <div style="text-align:center;color:#aaa;margin-top:4px">quarterly</div>
            </div>""", unsafe_allow_html=True)

        with r3:
            adj = expense_gap * 100
            st.markdown(f"""
            <div class="metric-card">
                <div style="color:#aaa;font-size:12px;">EXPENSE GAP COST</div>
                <div class="score-ring" style="color:#e57373">{adj:.2f}%</div>
                <div style="text-align:center;color:#aaa;margin-top:4px">annual drag</div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        if proba >= 0.7:
            st.success(f"**Strong signal:** This fund's alpha generation is likely to justify its commission cost next quarter. Consider the regular plan if switching to direct is inconvenient.")
        elif proba >= 0.5:
            st.warning(f"**Marginal:** The model is mildly positive, but the confidence is low. Prefer the direct plan to capture more upside.")
        else:
            st.error(f"**Commission unjustified:** Historical patterns suggest this fund is unlikely to generate enough alpha to cover its {expense_gap*100:.2f}% expense gap. **Switch to direct plan.**")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Feature Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Feature Explorer":
    st.markdown("## Feature Explorer")

    if not data_ready:
        st.info("Run pipeline.py to generate feature data.")
        st.stop()

    st.markdown("Explore how fund features vary across the dataset.")

    col1, col2 = st.columns(2)
    with col1:
        x_feat = st.selectbox("X-axis feature", [
            "expense_gap_annualized", "sharpe_ratio_1y", "information_ratio_1y",
            "beta_1y", "log_aum", "manager_tenure_years", "net_alpha", "gross_alpha"
        ])
    with col2:
        y_feat = st.selectbox("Y-axis feature", [
            "net_alpha", "information_ratio_1y", "sharpe_ratio_1y",
            "expense_gap_annualized", "max_drawdown_1y", "turnover_proxy"
        ])

    sub = features_df[[x_feat, y_feat, "label_commission_justified"]].dropna()
    if len(sub) > 0:
        fig, ax = dark_fig((9, 5))
        colors = sub["label_commission_justified"].map({1: "#4caf50", 0: "#f44336"})
        ax.scatter(sub[x_feat] * 100, sub[y_feat] * 100, c=colors, alpha=0.6, s=15, edgecolors="none")
        ax.set_xlabel(x_feat + " (%)")
        ax.set_ylabel(y_feat + " (%)")
        ax.set_title(f"{x_feat} vs {y_feat}")
        import matplotlib.patches as mpatches
        j_patch = mpatches.Patch(color="#4caf50", label="Commission Justified")
        nj_patch = mpatches.Patch(color="#f44336", label="Not Justified")
        ax.legend(handles=[j_patch, nj_patch], facecolor="#1e1e1e", labelcolor="white", edgecolor="#333")
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.subheader("Distribution by Commission Justification")
    feat_dist = st.selectbox("Feature", [
        "expense_gap_annualized", "sharpe_ratio_1y", "information_ratio_1y",
        "gross_alpha", "net_alpha", "manager_tenure_years", "log_aum"
    ])

    justified = features_df[features_df["label_commission_justified"] == 1][feat_dist].dropna() * 100
    not_justified = features_df[features_df["label_commission_justified"] == 0][feat_dist].dropna() * 100

    fig2, ax2 = dark_fig((9, 4))
    ax2.hist(not_justified, bins=40, color="#f44336", alpha=0.6, label="Not Justified", density=True)
    ax2.hist(justified, bins=40, color="#4caf50", alpha=0.6, label="Justified", density=True)
    ax2.set_xlabel(feat_dist + " (%)")
    ax2.set_ylabel("Density")
    ax2.legend(facecolor="#1e1e1e", labelcolor="white", edgecolor="#333")
    st.pyplot(fig2)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SIP Simulator
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 SIP Simulator":
    st.markdown("## SIP Back-Validation (2019–2023)")
    st.markdown("Compare model-guided direct fund selection vs naive regular plan investing.")

    if report.get("sip_backtest"):
        bt = report["sip_backtest"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Invested", f"₹{bt['total_invested']:,.0f}")
        c2.metric("Model Portfolio (Direct)", f"₹{bt['model_final_value']:,.0f}",
                  delta=f"XIRR: {bt['model_xirr']}%")
        c3.metric("Naive Regular Plan", f"₹{bt['naive_final_value']:,.0f}",
                  delta=f"XIRR: {bt['naive_xirr']}%")

        st.metric("Excess Return (Model vs Naive)", f"{bt['excess_return_pct']}% p.a.")
        st.divider()

        # Bar chart
        fig, ax = dark_fig((7, 4))
        bars = ax.bar(
            ["Naive Regular Plan", "Model — Direct Funds"],
            [bt["naive_final_value"], bt["model_final_value"]],
            color=["#f44336", "#4caf50"], width=0.5, edgecolor="none"
        )
        ax.set_ylabel("Portfolio Value (₹)")
        ax.set_title(f"SIP ₹5,000/month — Final Value Comparison")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + bt["model_final_value"] * 0.01,
                    f"₹{bar.get_height():,.0f}", ha="center", color="white", fontsize=10)
        st.pyplot(fig)
        plt.close()
    else:
        st.info("SIP backtest results not found. Re-run `python pipeline.py`.")

    st.divider()
    st.subheader("Manual SIP Calculator")
    col1, col2, col3 = st.columns(3)
    monthly = col1.number_input("Monthly SIP (₹)", value=5000, step=500)
    years = col2.slider("Duration (years)", 5, 25, 10)
    exp_gap = col3.number_input("Expense gap (%/year)", value=0.8, step=0.05)

    xirr_direct = st.slider("Expected Direct Plan XIRR (%)", 8, 20, 12)
    xirr_regular = xirr_direct - exp_gap

    n = years * 12
    direct_corpus = monthly * ((1 + xirr_direct/1200)**n - 1) / (xirr_direct/1200) * (1 + xirr_direct/1200)
    regular_corpus = monthly * ((1 + xirr_regular/1200)**n - 1) / (xirr_regular/1200) * (1 + xirr_regular/1200)
    loss = direct_corpus - regular_corpus

    st.success(f"""
    **Direct Plan Corpus:** ₹{direct_corpus:,.0f}  
    **Regular Plan Corpus:** ₹{regular_corpus:,.0f}  
    **💸 Wealth lost to commission over {years} years: ₹{loss:,.0f}**
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Insights
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Model Insights":
    st.markdown("## Model Performance & Explainability")

    if not report:
        st.info("Run pipeline.py to train models and generate insights.")
        st.stop()

    # Classifier metrics
    cls = report.get("classifier_metrics", {})
    reg_m = report.get("regressor_metrics", {})

    st.subheader("Classifier — Commission Justified?")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AUC-ROC", cls.get("auc_roc", "—"))
    c2.metric("F1 Score", cls.get("f1_score", "—"))
    c3.metric("Precision", cls.get("precision", "—"))
    c4.metric("Recall", cls.get("recall", "—"))
    c5.metric("Prec @ Top 10%", cls.get("precision_top_decile", "—"))

    st.divider()
    st.subheader("Regressor — Net Alpha Prediction")
    r1, r2, r3 = st.columns(3)
    r1.metric("RMSE", reg_m.get("rmse", "—"))
    r2.metric("MAE", reg_m.get("mae", "—"))
    r3.metric("Direction Accuracy", reg_m.get("direction_accuracy", "—"))

    st.divider()
    # SHAP importance chart
    st.subheader("SHAP Feature Importance")
    shap_img = REPORTS_DIR / "shap_importance.png"
    if shap_img.exists():
        st.image(str(shap_img), use_container_width=True)
    elif not shap_df.empty:
        fig, ax = dark_fig((9, 6))
        top = shap_df.head(12)
        ax.barh(top.index[::-1], top["mean_abs_shap"][::-1], color="#f5a623", edgecolor="none")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Feature Importance by SHAP")
        st.pyplot(fig)
        plt.close()
    else:
        st.info("SHAP plot not found. Re-run pipeline.py.")

    st.divider()
    st.subheader("Design Notes")
    st.markdown(f"""
    - **Temporal split:** No look-ahead bias. Train ends before Val starts, Val ends before Test.
    - **Train rows:** {report.get('n_train', '—')} | **Val:** {report.get('n_val', '—')} | **Test:** {report.get('n_test', '—')}
    - **Features:** {', '.join(report.get('features', []))}
    - **Model:** XGBoost with early stopping on validation AUC/RMSE
    - **Explainability:** TreeSHAP for feature attribution per prediction
    """)
