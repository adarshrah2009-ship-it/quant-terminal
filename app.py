import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from arch import arch_model
import google.generativeai as genai
from datetime import date
# ==========================================
# 1. QUANT & MATHEMATICAL HELPER FUNCTIONS
# ==========================================

def get_garch_forecast_volatility(returns):
    """
    Fits a GARCH(1,1) model on historical daily returns to forecast dynamic volatility.
    Falls back to simple standard deviation if numerical optimization fails.
    """
    try:
        # Scale returns to percentages for numerical stability during optimization
        scaled_returns = returns.dropna() * 100
        am = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='normal')
        res = am.fit(disp='off')
        
        # Forecast 1-step ahead conditional variance and convert back to annualized standard deviation
        forecast_var = res.forecast(horizon=1).variance.iloc[-1, 0]
        annualized_garch_vol = (np.sqrt(forecast_var) / 100) * np.sqrt(252)
        return float(annualized_garch_vol)
    except Exception:
        # Fallback to standard historical annualized volatility
        return float(returns.std() * np.sqrt(252))

def calculate_technical_indicators(df):
    """Calculates 14-day RSI and ADX trend strength."""
    df = df.copy()
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ADX (14)
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    
    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), 
                         np.maximum(df['High'] - df['High'].shift(1), 0), 0)
    df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), 
                         np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)

    tr14 = df['TR'].rolling(14).sum()
    plus_di14 = 100 * (df['+DM'].rolling(14).sum() / tr14)
    minus_di14 = 100 * (df['-DM'].rolling(14).sum() / tr14)
    
    dx = 100 * (abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14))
    df['ADX'] = dx.rolling(14).mean()
    
    return df

def calculate_var_cvar(returns, confidence_level=0.95):
    """Calculates Parametric Value at Risk (VaR) and Expected Shortfall (CVaR)."""
    mu = returns.mean()
    sigma = returns.std()
    alpha = 1 - confidence_level
    var = -(mu + sigma * norm.ppf(alpha))
    cvar = -(mu - sigma * (norm.pdf(norm.ppf(alpha)) / alpha))
    return var * 100, cvar * 100

# ==========================================
# 2. STREAMLIT APP CONFIGURATION & SIDEBAR
# ==========================================

st.set_page_config(page_title="Institutional Quant & AI Terminal", layout="wide")
st.title("Institutional Quant & AI Terminal")

st.sidebar.header("Control Panel")
# Change from selectbox to text_input for free search capability
selected_asset = st.sidebar.text_input("Enter Asset Ticker (e.g., RELIANCE.NS, BTC-USD, AAPL)", value="RELIANCE.NS").upper().strip()
data_horizon = st.sidebar.selectbox("Data Horizon", ["6m", "1y", "2y", "5y"], index=2)
api_key = st.sidebar.text_input("Gemini API Key", type="password")

# Fetch Market Data
data = yf.download(selected_asset, period=data_horizon, interval="1d")
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = calculate_technical_indicators(data)
data['Returns'] = data['Close'].pct_change()
latest_price = float(data['Close'].iloc[-1])
latest_rsi = float(data['RSI'].dropna().iloc[-1]) if not data['RSI'].dropna().empty else 50.0
latest_adx = float(data['ADX'].dropna().iloc[-1]) if not data['ADX'].dropna().empty else 20.0

var_95, cvar_95 = calculate_var_cvar(data['Returns'].dropna(), 0.95)
garch_vol = get_garch_forecast_volatility(data['Returns'].dropna())

# Reset session memory when ticker changes to calculate realistic default price targets
if "last_ticker" not in st.session_state or st.session_state.last_ticker != selected_asset:
    st.session_state.last_ticker = selected_asset
    st.session_state.target_price_input = float(latest_price * 1.05)

# ==========================================
# 3. INTERACTIVE DASHBOARD TABS
# ==========================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Market Matrix",
    "📊 Backtest Engine", 
    "🤖 AI Signal Engine",
    "🛡️ Risk & Position Sizer",
    "🎲 Monte Carlo Engine"
])

# ----- TAB 1: MARKET MATRIX -----
with tab1:
    st.subheader(f"Price Action & Technical Analysis: {selected_asset}")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'],
        name="OHLC"
    ))
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

# ----- TAB 2: BACKTEST ENGINE -----
with tab2:
    st.subheader("SMA Crossover Strategy Backtest")
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['SMA_50'] = data['Close'].rolling(50).mean()
    
    data['Signal'] = 0
    data['Signal'] = np.where(data['SMA_20'] > data['SMA_50'], 1, 0)
    data['Strat_Returns'] = data['Signal'].shift(1) * data['Returns']
    
    cum_bench = (1 + data['Returns']).cumprod()
    cum_strat = (1 + data['Strat_Returns']).cumprod()
    
    fig_backtest = go.Figure()
    fig_backtest.add_trace(go.Scatter(x=data.index, y=cum_bench, name="Buy & Hold"))
    fig_backtest.add_trace(go.Scatter(x=data.index, y=cum_strat, name="SMA Crossover Strategy"))
    fig_backtest.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig_backtest, use_container_width=True)

# ----- TAB 3: AI SIGNAL ENGINE -----
with tab3:
    st.subheader("Gemini LLM Technical Audit Engine")
    if st.button("Generate AI Market Intelligence"):
        if not api_key:
            st.warning("Please input your Gemini API Key in the sidebar.")
        else:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-3.6-flash')
                
                today_str = date.today().strftime("%B %d, %Y")
                
                prompt = f"""
                Act as an institutional quantitative analyst writing a formal memo.
                
                MEMORANDUM DETAILS:
                - DATE: {today_str}
                - ASSET TICKER: {selected_asset}
                - CURRENT PRICE: {latest_price:,.2f}
                - RSI (14): {latest_rsi:.1f}
                - ADX TREND STRENGTH: {latest_adx:.1f}
                - 1-DAY 95% Value at Risk (VaR): -{var_95:.2f}%
                - FORECASTED ANNUALIZED GARCH(1,1) VOLATILITY: {garch_vol * 100:.2f}%

                Write a structured memorandum including:
                1. MEMORANDUM HEADER (Include TO, FROM, DATE: {today_str}, ASSET)
                2. Signal Recommendation (BUY, SELL, or HOLD) with Confidence %
                3. Key Technical Rationale (Synthesizing ADX, RSI, and GARCH Volatility)
                4. Risk Mitigation Warning (Interpret VaR in plain language without unformatted dollar signs)
                """
                
                response = model.generate_content(prompt)
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Failed to query Gemini API: {e}")
# ----- TAB 4: RISK & POSITION SIZER -----
with tab4:
    st.subheader("Quantitative Risk Assessment & Capital Allocation")
    
    # Core Risk Metrics Display
    col1, col2, col3 = st.columns(3)
    col1.metric("1-Day 95% VaR", f"-{var_95:.2f}%")
    col2.metric("1-Day 95% Expected Shortfall (CVaR)", f"-{cvar_95:.2f}%")
    col3.metric("GARCH(1,1) Dynamic Volatility", f"{garch_vol * 100:.2f}%")
    
    st.markdown("---")
    
    # Section A: Position Sizing Calculator
    st.subheader("Institutional Position Sizing Calculator")
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        account_size = st.number_input("Portfolio Equity ($)", value=100000, step=5000)
    with col_b:
        max_risk_pct = st.slider("Max Portfolio Capital Risk (%)", 0.5, 5.0, 1.0) / 100
    with col_c:
        stop_loss_pct = st.slider("Stop-Loss Distance (%)", 1.0, 20.0, 5.0) / 100
    
    dollar_risk = account_size * max_risk_pct
    position_size = dollar_risk / stop_loss_pct
    max_leverage = position_size / account_size
    
    st.success(
        f"**Recommended Position Size:** ${position_size:,.2f} "
        f"*(Implied Leverage: {max_leverage:.2f}x | Capital at Risk: ${dollar_risk:,.2f})*"
    )
    
    st.markdown("---")
    
    # Section B: Historical Crisis Stress-Testing Matrix
    st.subheader("🛡️ Macroeconomic Crisis Stress-Test Engine")
    st.caption("Evaluates projected portfolio drawdowns under actual historical liquidity and volatility shocks.")
    
    # Stress test scenario parameters
    scenarios = {
        "2008 Lehman Liquidity Crash": {"price_shock": -0.20, "vol_multiplier": 2.5},
        "2020 COVID-19 Liquidity Shock": {"price_shock": -0.30, "vol_multiplier": 3.0},
        "1970s Style Stagflation Shock": {"price_shock": -0.15, "vol_multiplier": 1.8}
    }
    
    selected_scenario = st.selectbox("Select Historical Crisis Scenario", list(scenarios.keys()))
    scenario_params = scenarios[selected_scenario]
    
    # Calculate stressed metrics
    stressed_price = latest_price * (1 + scenario_params["price_shock"])
    stressed_vol = garch_vol * scenario_params["vol_multiplier"]
    stressed_portfolio_loss = position_size * scenario_params["price_shock"]
    portfolio_equity_drawdown = (abs(stressed_portfolio_loss) / account_size) * 100
    
    # Display scenario results in structured metrics
    stress_col1, stress_col2, stress_col3 = st.columns(3)
    stress_col1.metric("Stressed Asset Price", f"${stressed_price:,.2f}", f"{scenario_params['price_shock']*100:.1f}% Shock")
    stress_col2.metric("Stressed GARCH Volatility", f"{stressed_vol * 100:.2f}%", f"{scenario_params['vol_multiplier']}x Spike")
    stress_col3.metric("Projected Equity Drawdown", f"-${abs(stressed_portfolio_loss):,.2f}", f"-{portfolio_equity_drawdown:.2f}% Equity")
    
    # Scenario Interpretation Box
    if portfolio_equity_drawdown > (max_risk_pct * 100 * 2):
        st.error(
            f"**CRITICAL TAIL-RISK WARNING:** Under the **{selected_scenario}**, your position size of "
            f"${position_size:,.2f} creates an unsustainable equity loss of **{portfolio_equity_drawdown:.2f}%**. "
            "Consider reducing position sizing or adding protective put options."
        )
    else:
        st.info(
            f"**STRESS TEST PASSED:** The allocation withstands the **{selected_scenario}** within acceptable equity parameters."
        )
# ----- TAB 5: MONTE CARLO ENGINE (GARCH-POWERED) -----
with tab5:
    st.subheader("GARCH(1,1) Stochastic Monte Carlo Simulation Engine")
    
    simulations = st.slider("Number of Simulations", 100, 2000, 500)
    time_horizon = st.slider("Forecast Horizon (Days)", 10, 252, 30)
    target_price = st.number_input("Target Price Threshold ($)", key="target_price_input")
    
    if st.button("Run Stochastic Simulation"):
        dt = 1 / 252
        daily_drift = (data['Returns'].mean() - 0.5 * (garch_vol ** 2)) * dt
        daily_vol = garch_vol * np.sqrt(dt)
        
        simulation_matrix = np.zeros((time_horizon, simulations))
        simulation_matrix[0] = latest_price
        
        for t in range(1, time_horizon):
            random_shocks = np.random.standard_normal(simulations)
            simulation_matrix[t] = simulation_matrix[t-1] * np.exp(daily_drift + daily_vol * random_shocks)
        
        # Plot Simulations
        fig_mc = go.Figure()
        for i in range(min(simulations, 100)):
            fig_mc.add_trace(go.Scatter(y=simulation_matrix[:, i], mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
        fig_mc.update_layout(template="plotly_dark", height=450, title="Monte Carlo Price Paths")
        st.plotly_chart(fig_mc, use_container_width=True)
        
        final_prices = simulation_matrix[-1]
        
        # Adjust logic to calculate correct probability based on target direction (above or below current market price)
        if target_price >= latest_price:
            success_prob = (np.sum(final_prices >= target_price) / simulations) * 100
            st.info(f"Probability of reaching or exceeding **${target_price:,.2f}** within {time_horizon} days: **{success_prob:.1f}%**")
        else:
            success_prob = (np.sum(final_prices <= target_price) / simulations) * 100
            st.info(f"Probability of dropping to or below **${target_price:,.2f}** within {time_horizon} days: **{success_prob:.1f}%**")