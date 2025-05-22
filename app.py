import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import time

# ✅ Safe fetch function with retry logic
def safe_fetch(callable_fn, *args, retries=3, delay=3, **kwargs):
    for i in range(retries):
        try:
            return callable_fn(*args, **kwargs)
        except Exception as e:
            if "Too Many Requests" in str(e):
                time.sleep(delay)
            else:
                raise e
    raise Exception("❌ Still rate-limited after multiple attempts.")

# ✅ Load CSV with companies and sectors
df = pd.read_csv("company_data_with_ns.csv2")  # Make sure this file exists
companies = df.groupby("Sector")["Symbol"].apply(list).to_dict()

# ✅ Streamlit layout
st.set_page_config(page_title="📈 Stock Market Dashboard", layout="wide")
st.title("📈 Stock Market Dashboard (Optimized - No News)")

# ✅ Select sector and company
selected_sector = st.selectbox("Select a sector", list(companies.keys()))
selected_company = st.selectbox("Select a company", companies[selected_sector])

# ✅ Cached ticker object
@st.cache_resource(ttl=3600)
def get_ticker(symbol):
    return yf.Ticker(symbol)

# ✅ Display stock metrics
def display_stock_metrics(stock_obj, symbol):
    try:
        stock_info = safe_fetch(lambda: stock_obj.info)
        st.subheader(f"📊 Stock Info: {symbol}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Price", f"{stock_info.get('currentPrice', 'N/A')} ₹")
            st.metric("Open Price", f"{stock_info.get('open', 'N/A')} ₹")
            st.metric("Previous Close", f"{stock_info.get('previousClose', 'N/A')} ₹")
            st.metric("Market Cap", f"{stock_info.get('marketCap', 'N/A'):,}" if stock_info.get('marketCap') else 'N/A')
            st.metric("Dividend Yield", f"{stock_info.get('dividendYield', 'N/A')}")
        with col2:
            st.metric("P/E Ratio", f"{stock_info.get('trailingPE', 'N/A')}")
            st.metric("Day Low", f"{stock_info.get('dayLow', 'N/A')} ₹")
            st.metric("Day High", f"{stock_info.get('dayHigh', 'N/A')} ₹")
            st.metric("EPS (TTM)", f"{stock_info.get('earningsPerShare', 'N/A')}")
    except Exception as e:
        if "Too Many Requests" in str(e):
            st.warning("⚠️ Too many requests to Yahoo Finance. Please wait and try again.")
        else:
            st.error(f"❌ Error fetching stock info for {symbol}: {e}")

# ✅ Display stock info
stock = get_ticker(selected_company)
display_stock_metrics(stock, selected_company)

# ✅ Candlestick chart with SMA
st.subheader("🕯️ Candlestick Chart with 20 & 50-day SMA - Last 6 Months")
fig = go.Figure()
try:
    hist = safe_fetch(stock.history, period="6mo")
    if not hist.empty:
        hist['SMA20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA50'] = hist['Close'].rolling(window=50).mean()

        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist['Open'], high=hist['High'],
            low=hist['Low'], close=hist['Close'], name=f"{selected_company} Candlestick",
            increasing_line_color='green', decreasing_line_color='red', opacity=0.5
        ))
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['SMA20'], mode='lines', name="SMA 20"
        ))
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['SMA50'], mode='lines', name="SMA 50"
        ))
    else:
        st.warning(f"❗ No historical data for {selected_company}")
except Exception as e:
    if "Too Many Requests" in str(e):
        st.warning("⚠️ Too many requests to Yahoo Finance. Please wait and try again.")
    else:
        st.error(f"❌ Error fetching history for {selected_company}: {e}")

fig.update_layout(
    title=f"{selected_company} - Last 6 Months",
    xaxis_title="Date",
    yaxis_title="Price (₹)",
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# ✅ Price prediction using Linear Regression
st.subheader("📈 Simple Stock Price Prediction (Next 5 Days)")
try:
    hist = safe_fetch(stock.history, period="1y")
    if hist.empty:
        st.warning(f"Not enough data for prediction: {selected_company}")
    else:
        hist = hist.reset_index()
        hist['Date_ordinal'] = pd.to_datetime(hist['Date']).map(datetime.toordinal)

        X = hist['Date_ordinal'].values.reshape(-1, 1)
        y = hist['Close'].values

        model_lr = LinearRegression()
        model_lr.fit(X, y)

        last_date = hist['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, 6)]
        future_ordinal = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
        preds = model_lr.predict(future_ordinal)

        pred_df = pd.DataFrame({"Date": future_dates, "Predicted Close Price": preds})
        st.write(f"**{selected_company}** - Predicted Close Prices for Next 5 Days")
        st.dataframe(pred_df.style.format({"Predicted Close Price": "{:.2f} ₹"}))
except Exception as e:
    if "Too Many Requests" in str(e):
        st.warning("⚠️ Too many requests to Yahoo Finance. Please wait and try again.")
    else:
        st.error(f"❌ Prediction failed for {selected_company}: {e}")
