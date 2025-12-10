import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Page Configuration & Style

st.set_page_config(page_title="Stock Analysis Suite", layout="wide", page_icon=":bar_chart:")

st.markdown("""
    <style>
        .main {background-color: #f5f7fa;}
        .block-container {padding-top: 2rem;}
        .stDataFrame {background-color: #fff;}
        .stMetric {background-color: #f0f2f6;}
        .stAlert {border-radius: 0.5rem;}
        .stButton>button {background-color: #0056b3; color: white;}
        .stSlider > div {color: #0056b3;}
        .st-bb {background-color: #f0f2f6;}
        .st-cb {background-color: #f0f2f6;}
    </style>
""", unsafe_allow_html=True)

# Helper Functions

def get_index_data():
    indices = {'S&P 500 Index': '^GSPC', 'Dow Jones Industrial Average': '^DJI', 'NASDAQ Composite Index': '^IXIC'}
    data = {}
    for name, ticker in indices.items():
        try:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period="2d")
            if len(hist) >= 2:
                last, prev = hist['Close'].iloc[-1], hist['Close'].iloc[-2]
                change = last - prev
                pct_change = change / prev * 100
                data[name] = {'price': last, 'change': change, 'pct_change': pct_change}
        except Exception:
            data[name] = {'price': None, 'change': None, 'pct_change': None}
    return data

def get_vix():
    try:
        vix = yf.Ticker('^VIX')
        hist = vix.history(period="2d")
        if len(hist) >= 2:
            last, prev = hist['Close'].iloc[-1], hist['Close'].iloc[-2]
            change = last - prev
            pct_change = change / prev * 100
            return {'price': last, 'change': change, 'pct_change': pct_change}
    except Exception:
        pass
    return None

def get_financial_metrics(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.get_info()
        if not info or 'regularMarketPrice' not in info:
            return None
        pe = info.get('trailingPE')
        roe = info.get('returnOnEquity')
        roa = info.get('returnOnAssets')
        eps = info.get('trailingEps')
        pb = info.get('priceToBook')
        fcf = None
        wc = None
        try:
            cashflow = t.get_cashflow()
            if not cashflow.empty:
                op_cf = cashflow.loc['Total Cash From Operating Activities'].iloc[0]
                capex = cashflow.loc['Capital Expenditures'].iloc[0]
                fcf = op_cf + capex
        except Exception:
            pass
        try:
            balance = t.get_balance_sheet()
            if not balance.empty:
                ca = balance.loc['Total Current Assets'].iloc[0]
                cl = balance.loc['Total Current Liabilities'].iloc[0]
                wc = ca - cl
        except Exception:
            pass
        metrics = {
            "P/E Ratio": pe,
            "ROE": roe,
            "ROA": roa,
            "EPS": eps,
            "Price-to-Book": pb,
            "Free Cash Flow": fcf,
            "Working Capital": wc
        }
        return metrics
    except Exception:
        return None

def download_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data

def portfolio_optimization(returns, bounds, target_return=None):
    n = returns.shape[1]
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    def portfolio_perf(weights):
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return ret, vol

    def min_volatility(weights):
        return portfolio_perf(weights)[1]

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    if target_return is not None and target_return > 0:
        constraints.append({'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target_return})
    result = minimize(
        min_volatility,
        n * [1. / n,],
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result, mean_returns, cov_matrix

def perform_pca(prices, n_components=2):
    returns = prices.pct_change().dropna()
    pca = PCA(n_components=n_components)
    pca.fit(returns)
    return pca, returns

# Sidebar Branding

with st.sidebar:
    st.title("Pro Stock Analysis")
    st.markdown("A modern, interactive dashboard that leverages Linear Algebra.")
    page = st.radio("Navigation", [
        "Market Overview",
        "Stock Financial Metrics",
        "Portfolio Optimization",
        "Stock PCA Analysis",
        "Linear Regression",
    ])
    st.markdown("---")
    st.caption("Powered by [Streamlit](https://streamlit.io), [Yahoo Finance](https://finance.yahoo.com), and [LinearAlgebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)")

# Pages

# Market Overview 
if page == "Market Overview":
    st.title("General Stock Market Overview")
    st.markdown("#### Live summary of major US indices, sector performance, and volatility.")

    st.subheader("Major Indices")
    data = get_index_data()
    cols = st.columns(len(data))
    for i, (name, d) in enumerate(data.items()):
        if d['price'] is not None:
            cols[i].metric(
                label=f"{name}",
                value=f"{d['price']:.2f}",
                delta=f"{d['change']:.2f} ({d['pct_change']:.2f}%)",
                help=f"Change from previous close"
            )
        else:
            cols[i].write(f"{name}: Data unavailable")

    st.subheader("Volatility Index (VIX)")
    vix = get_vix()
    if vix:
        st.metric(
            label="CBOE Volatility Index (VIX)",
            value=f"{vix['price']:.2f}",
            delta=f"{vix['change']:.2f} ({vix['pct_change']:.2f}%)",
            help="VIX measures market volatility expectations"
        )
    else:
        st.write("VIX data unavailable.")

    st.info(
        "The indices show the performance of the overall US stock market. "
        "VIX measures market volatility expectations. "
    )

# Stock Financial Metrics 
elif page == "Stock Financial Metrics":
    st.title("Key Financial Metrics")
    st.markdown("#### Enter a stock ticker to view essential financial ratios and figures.")
    ticker = st.text_input("Stock Ticker (e.g., GOOG, MSFT, TSLA):", "GOOG")
    if ticker:
        metrics = get_financial_metrics(ticker)
        if metrics:
            st.subheader(f"Financial Metrics for **{ticker.upper()}**")
            mcols = st.columns(2)
            for idx, (k, v) in enumerate(metrics.items()):
                if k in ["Free Cash Flow", "Working Capital"] and (v is None or v == "N/A"):
                    continue
                mcols[idx % 2].markdown(f"<div style='font-size:1.1em'><b>{k}:</b> {v if v is not None else 'N/A'}</div>", unsafe_allow_html=True)
            st.markdown("""
<div style='margin-top:1.5em;'>
<b>Metric Definitions:</b>
<ul>
<li><b>P/E Ratio</b>: Price/Earnings, a valuation metric.</li>
<li><b>ROE</b>: Return on Equity, profitability on shareholder equity.</li>
<li><b>ROA</b>: Return on Assets, profitability on total assets.</li>
<li><b>EPS</b>: Earnings per Share.</li>
<li><b>Price-to-Book</b>: Market value vs. book value.</li>
<li><b>Free Cash Flow</b>: Cash left after capital expenditures.</li>
<li><b>Working Capital</b>: Current assets minus current liabilities.</li>
</ul>
</div>
            """, unsafe_allow_html=True)
        else:
            st.warning(
                "Could not retrieve financial data for this ticker. "
                "This may be due to data source limitations. Try another ticker or check your internet connection."
            )

# Portfolio Optimization
elif page == "Portfolio Optimization":
    st.title("Portfolio Optimization")
    st.markdown("#### Optimize your portfolio weights based on historical returns and your constraints.")

    tickers = st.text_input("Enter stock tickers (comma separated):", "AAPL,MSFT,GOOG")
    start = st.date_input("Start Date", pd.to_datetime("2025-01-01"))
    end = st.date_input("End Date", pd.to_datetime("today"))
    tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if len(tickers) >= 2:
        c1, c2, c3 = st.columns(3)
        min_weight = c1.slider("Minimum weight per stock", 0.0, 1.0, 0.0, 0.01)
        max_weight = c2.slider("Maximum weight per stock", 0.0, 1.0, 1.0, 0.01)
        target_ret = c3.number_input("Target annualized return (leave 0 for min volatility)", 0.0)
        data = download_prices(tickers, start, end)
        returns = data.pct_change().dropna()
        bounds = tuple((min_weight, max_weight) for _ in tickers)
        if not returns.empty:
            result, mean_returns, cov_matrix = portfolio_optimization(returns, bounds, target_return=target_ret if target_ret > 0 else None)
            st.subheader("Covariance Matrix of Asset Returns")
            st.dataframe(cov_matrix.style.background_gradient(cmap="Blues"), use_container_width=True)
            st.markdown("""
**What does this mean?**

The covariance matrix shows how the returns of each pair of assets move together.  
- **Diagonal values** are the variances (risk) of each asset.
- **Off-diagonal values** are covariances:  
    - Positive: the two assets move together.
    - Negative: the assets move in opposite directions.  
                        
Diversification benefits come from combining assets with low or negative covariances, reducing overall portfolio risk.
            """)
            if result.success:
                weights = result.x
                exp_ret, exp_vol = np.dot(weights, mean_returns), np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                st.subheader("Optimal Portfolio Weights")
                wcols = st.columns(len(tickers))
                for i, (t, w) in enumerate(zip(tickers, weights)):
                    wcols[i].metric(label=f"{t}", value=f"{w:.2%}")
                st.success(f"**Expected Annual Return:** {exp_ret:.2%}  \n**Expected Annual Volatility:** {exp_vol:.2%}")
            else:
                st.warning("Optimization failed. Try adjusting your constraints.")
        else:
            st.warning("Not enough data for the selected stocks and date range.")
    else:
        st.info("Enter at least two tickers for portfolio optimization.")

# Stock PCA Analysis
elif page == "Stock PCA Analysis":
    st.title("Stock PCA Factor Analysis")
    st.markdown(
        "Principal Component Analysis (PCA) helps identify the main factors influencing a stock's historical returns. "
        "It does **not** forecast future prices, but shows how much of the stock's movement is explained by market trends."
    )

    ticker = st.text_input("Enter a stock ticker for PCA analysis:", "GOOG")
    start = st.date_input("Start Date for PCA", pd.to_datetime("2025-01-01"), key="pca_start")
    end = st.date_input("End Date for PCA", pd.to_datetime("today"), key="pca_end")

    if ticker:
        pca_tickers = {
            ticker.upper(): "Selected Stock",
            "^GSPC": "S&P 500 Index",
            "^IXIC": "NASDAQ Composite Index",
            "^DJI": "Dow Jones Industrial Average"
        }
        tickers_list = list(pca_tickers.keys())

        data = download_prices(tickers_list, start, end)
        if data is not None and not data.empty:
            data = data.dropna(axis=1, how='all').dropna()
            data = data.rename(columns={k: v if k != ticker.upper() else ticker.upper() for k, v in pca_tickers.items()})
            present_cols = data.columns.tolist()
            returns = data.pct_change().dropna()
            if ticker.upper() in returns.columns:
                st.subheader(f"Historical Returns for {ticker.upper()}")
                st.line_chart(returns[ticker.upper()])
            else:
                st.warning(f"No return data for {ticker.upper()} in the selected date range. Available: {present_cols}")

            if returns.shape[1] >= 2:
                pca = PCA(n_components=min(2, returns.shape[1]))
                pca.fit(returns)
                st.subheader("PCA Components (Loadings)")
                readable_names = [pca_tickers.get(col, col) if col != ticker.upper() else ticker.upper() for col in returns.columns]
                loadings_df = pd.DataFrame(
                    pca.components_,
                    columns=readable_names,
                    index=[f"PC{i+1}" for i in range(pca.n_components_)]
                )
                st.dataframe(loadings_df.style.background_gradient(cmap="Purples"), use_container_width=True)
                st.info(
                    "The first principal component (PC1) typically represents the overall market movement. "
                    f"If {ticker.upper()}'s returns load heavily on PC1, its price is largely explained by market trends. "
                    "Subsequent components may capture sector movements."
                )
            else:
                st.warning("Not enough assets with data for PCA analysis. Try a different ticker or date range.")
        else:
            st.warning("No price data found for the selected tickers and date range.")

# Linear Regression and Least Squares
elif page == "Linear Regression":
    st.title("Linear Regression & Least Squares")
    st.markdown(
        "Fit a linear model to predict a stock's returns from other stocks. "
    )

    tickers = st.text_input("Enter stock tickers (comma separated, at least 2):", "AAPL,MSFT,GOOG")
    start = st.date_input("Start Date", pd.to_datetime("2025-01-01"), key="lr_start")
    end = st.date_input("End Date", pd.to_datetime("today"), key="lr_end")
    tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if len(tickers) >= 2:
        data = download_prices(tickers, start, end)
        returns = data.pct_change().dropna()
        target = st.selectbox("Select target variable:", tickers)
        predictors = [t for t in tickers if t != target]
        selected_predictors = st.multiselect("Select predictor variables:", predictors, default=predictors)
        if selected_predictors:
            X = returns[selected_predictors].values
            y = returns[target].values
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
            # Least squares solution: beta = (X^T X)^-1 X^T y
            beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            y_pred = X_with_intercept @ beta
            residuals = y - y_pred
            r2 = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)
            st.subheader("Regression Coefficients")
            coef_table = pd.DataFrame({
                "Variable": ["Intercept"] + selected_predictors,
                "Coefficient": beta
            })
            st.dataframe(coef_table, use_container_width=True)
            st.success(f"R² (explained variance): {r2:.3f}")

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(returns.index, y, label="Actual Returns", color='blue')
            ax.plot(returns.index, y_pred, label="Predicted Returns", color='orange', alpha=0.7)
            ax.set_title("Actual vs Predicted Returns Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Returns")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.info("Select at least one predictor variable.")
    else:
        st.info("Enter at least two tickers.")

