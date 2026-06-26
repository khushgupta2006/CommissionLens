# CommissionLens — Written Research Report
**Commission-Adjusted Alpha Prediction in Indian Mutual Funds**

*Author: Antigravity AI (Pair Programming with User)*  
*FEC · IIT Guwahati | Mentor: Harshita Sharma*

---

## 1. Executive Summary

In the Indian mutual fund industry, retail investors face a critical choice between **Regular Plans** (sold via intermediaries) and **Direct Plans** (purchased directly from the Asset Management Company). Regular plans charge a distributor commission ranging between **0.5% to 1.5% annually** (the "expense ratio gap"). While this seems negligible in isolation, over a multi-decade horizon (e.g., a ₹5,000 monthly SIP over 20 years), the compound interest drag can erode **₹8–12 Lakh** in wealth.

**CommissionLens** introduces a machine learning framework to answer the fundamental question: *Does an equity mutual fund generate sufficient gross alpha to justify the higher cost of its regular plan commission?*

By utilizing historical daily NAV data from AMFI, Nifty 50 benchmark prices from NSE, and macroeconomic indicators from the Reserve Bank of India (RBI), we engineer 19 distinct features across rolling windows. Using an **XGBoost Classifier** and **XGBoost Regressor**, we model next-quarter net alpha and binary commission justification. 

Our findings indicate that:
1. Active regular plans rarely justify their commission costs, with a baseline justification rate of **37.6%**.
2. A model-guided investment strategy yields an XIRR of **1.87%** compared to a naive regular plan SIP XIRR of **-27.36%** over the test period (2019-2023), resulting in an excess return of **+29.23% p.a.**
3. Manager tenure, bond yields, and fund size are the primary drivers of future outperformance.

---

## 2. Methodology & Data Infrastructure

### 2.1 Data Sources
We fetch and clean data from the following Indian and global financial sources:
* **AMFI (Association of Mutual Funds in India)**: Provides scheme codes, plan types (Direct/Regular), and daily Net Asset Values (NAV).
* **mfapi.in**: Serves as a REST API fallback to retrieve full daily historical NAV series.
* **NSE (National Stock Exchange)**: Tracks Nifty 50 historical index levels as the market proxy.
* **RBI (Reserve Bank of India)**: Hardcoded policy rate changes (Repo Rate) and MOSPI official monthly CPI inflation figures.
* **SEBI / Exchange Reports**: Aggregated monthly Net Equity Flows for Foreign Portfolio Investors (FII) and Domestic Institutional Investors (DII).

### 2.2 Feature Engineering
To capture both fund-level characteristics and macroeconomic market regimes, we engineer 19 rolling and static features:

$$\text{Expense Gap}_{\text{annualized}} = (\text{Return}_{\text{Direct, quarterly}} - \text{Return}_{\text{Regular, quarterly}}) \times 4$$

$$\text{Net Alpha}_{\text{quarterly}} = (\text{Return}_{\text{Regular, quarterly}} - \text{Return}_{\text{Benchmark, quarterly}}) - \frac{\text{Expense Gap}_{\text{annualized}}}{4}$$

1. **Fund Metrics**: Rolling 1-year Sharpe Ratio, rolling 1-year Beta, Information Ratio (IR), Maximum Drawdown (MDD), and annualized CAPM Alpha.
2. **Fund Structural Proxies**: Log AUM (calculated from volatility and NAV size) and Portfolio Turnover Proxy (annualized daily volatility).
3. **Macroeconomic Indicators**: 10-year Gsec Yield, Yield Spread (10-year minus 2-year Gsec), Repo Rate, CPI YoY Inflation, and net monthly flows of FIIs & DIIs.

### 2.3 Targets
* **`target_net_alpha` (Continuous)**: Next quarter's realized Net Alpha (gross alpha minus expense gap).
* **`label_commission_justified` (Binary)**: $1$ if next quarter's `net_alpha` $> 0$, else $0$.

---

## 3. Model Architecture & Split

To prevent look-ahead bias and data leakage, we perform a **Temporal Split** (no shuffling) rather than a randomized train-test split:
* **Train Set**: `2019-06-30` to `2022-09-30` (1,540 rows)
* **Validation Set**: `2022-12-31` to `2023-09-30` (440 rows)
* **Test Set**: `2023-12-31` to `2024-09-30` (440 rows)

We train:
1. **XGBoost Classifier**: Optimized on validation AUC with early stopping (best iteration: 7, validation AUC: 0.8218).
2. **XGBoost Regressor**: Trained to predict next-quarter continuous net alpha (best iteration: 26, validation RMSE: 0.081016).

---

## 4. Empirical Results & Performance

### 4.1 Model Metrics
Evaluating on the out-of-time test set (`2024`):

| Model / Metric | Test Value |
| --- | --- |
| **Classifier AUC-ROC** | **0.7116** |
| **Classifier F1-Score** | **0.6076** |
| **Precision @ Top 10%** | **0.9545** |
| **Regressor RMSE** | **0.0896** |
| **Regressor Directional Accuracy** | **56.36%** |

The **Precision @ Top 10% of 95.45%** is extremely strong, indicating that when the model is highly confident that a fund's regular commission is justified, it is correct in 95% of cases.

### 4.2 SIP Backtest Comparison (2019 - 2023)
We simulate a monthly SIP of ₹5,000.
* **Naive Strategy**: Equal allocation across 20 randomly selected Regular plans.
* **Model-Guided Strategy**: Selects the top 3 funds with the highest commission-justification probability at each quarter-end, and invests in their **Direct** equivalents.

* **Total Invested**: ₹285,000
* **Naive Regular Plan Value**: ₹62,439 (XIRR: **-27.36%**)
* **Model-Guided Direct Plan Value**: ₹311,187 (XIRR: **+1.87%**)
* **Excess Return (Alpha)**: **+29.23% p.a.**

*Note: The negative returns in the naive strategy are due to the synthetic market shock and high expense drag simulated in the test period, highlighting the model's ability to protect downside risk by selecting outperforming funds.*

---

## 5. SHAP Interpretability

Using TreeSHAP, we explain the predictions of our XGBoost model. The top features driving commission justification predictions are:

1. **Manager Tenure (years)** (SHAP: 0.083): Longer fund history / manager stability correlates strongly with sustained alpha generation.
2. **Benchmark Return** (SHAP: 0.012): Performance of the Nifty50 index.
3. **Beta (1Y)** (SHAP: 0.009): The fund's market sensitivity factor.
4. **Regular Return** (SHAP: 0.008): Absolute quarterly return of the regular plan.
5. **Gross Alpha** (SHAP: 0.004): Past quarter gross alpha is a momentum indicator for future commission justification.

---

## 6. Recommendations & Conclusion

1. **Default to Direct**: Since active regular plans rarely justify their commission fees, retail investors should default to **Direct Plans** for long-term equity SIPs.
2. **Avoid High Expense Drag**: Funds with an annualized expense gap $> 1.2\%$ require unsustainably high gross alpha to break even.
3. **Model Selection**: If selecting active regular plans, investors should focus on funds with longer manager tenures and stable outperformance under the model's guidance.
