# Institutional Quantitative Terminal

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![GARCH](https://img.shields.io/badge/Volatility-GARCH(1%2C1)-green.svg)
![LLM Integration](https://img.shields.io/badge/AI-Gemini_3.6--Flash-purple.svg)

An enterprise-grade financial analytics and risk management dashboard designed for institutional portfolio construction, dynamic volatility forecasting, historical scenario stress-testing, and automated technical audits.

---

## Technical Architecture & Core Modules

### 1. Dynamic Volatility Modeling (GARCH 1,1)
- **Model:** Generalized Autoregressive Conditional Heteroskedasticity ($\text{GARCH}(1,1)$).
- **Implementation:** Fits historical log returns via `arch` library to capture variance targeting and volatility clustering:
  $$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
- **Purpose:** Replaces static historical volatility with forward-looking conditional variance forecasts to optimize tail-risk estimation.

### 2. Tail-Risk Metrics & Monte Carlo Simulation
- **Metrics:** Calculates 95% Parametric and Empirical Value-at-Risk (VaR) alongside Conditional Value-at-Risk (CVaR/Expected Shortfall).
- **Simulation Engine:** $N=1,000$+ iteration Geometric Brownian Motion (GBM) Monte Carlo paths utilizing GARCH conditional variance parameters for dynamic drift-diffusion projection.

### 3. Historical Macroeconomic Crisis Stress-Testing
Evaluates real-time portfolio resilience under historical tail-event drawdowns:
- **2008 Global Financial Crisis (GFC):** High-volatility shock modeling.
- **2020 COVID Liquidity Crunch:** Rapid asset correlation collapse.
- **1970s Stagflation Regime:** Persistent structural inflation pressure.

### 4. Machine Learning & LLM Technical Audit
- **Engine:** Google Gemini (`3.6-flash`).
- **Functionality:** Ingests live technical indicators (SMA 50/200 crossovers, RSI, GARCH forecasts, VaR statistics) to generate automated institutional risk reports and exposure recommendations.

---

## Tech Stack

- **Frontend / Dashboard Framework:** Streamlit
- **Quantitative Modeling & Math:** `scipy`, `arch`, `numpy`, `pandas`
- **Data Acquisition:** `yfinance`
- **Data Visualization:** `plotly`
- **LLM Integration:** `google-genai` (Gemini 3.6-Flash)

---

## Local Setup & Installation

1. Clone repository:
   ```bash
   git clone [https://github.com/adarshrah2009-ship-it/quant-terminal.git](https://github.com/adarshrah2009-ship-it/quant-terminal.git)
   cd quant-terminal