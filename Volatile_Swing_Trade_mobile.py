import numpy as np
import pandas as pd
import requests
from scipy.signal import find_peaks
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Volatility Swing Trader", layout="wide")

st.title("📊 Volatility Swing Trader & Risk Manager")


@st.cache_data
def fetch_sp500_tickers():
  # Direct pull from reliable Wikipedia source table via pandas
  try:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df["Symbol"].tolist()
    # Clean up any ticker formatting (e.g., BRK.B -> BRK-B for Yahoo Finance)
    tickers = [str(t).replace(".", "-").strip() for t in tickers]
    if len(tickers) > 400:
      return tickers
  except Exception:
    pass

  # Backup source if Wikipedia is blocked
  try:
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    df = pd.read_csv(url)
    tickers = df["Symbol"].dropna().tolist()
    return [str(t).replace(".", "-").strip() for t in tickers]
  except Exception:
    pass

  return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD"]


# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("1. Market Selection")
scan_target = st.sidebar.radio(
    "Scan Target:", ["Full S&P 500", "Custom Watchlist"]
)

if scan_target == "Custom Watchlist":
  watchlist_input = st.sidebar.text_input(
      "Enter Tickers (comma-separated)",
      "NVDA, TSLA, META, AAPL, MSFT, AMZN, AMD, NFLX, PLTR, AVGO",
  )
  tickers = [t.strip().upper() for t in watchlist_input.split(",")]
else:
  tickers = fetch_sp500_tickers()

bypass_filters = st.sidebar.checkbox("Bypass Filters", value=False)

st.sidebar.header("2. Strategy Criteria Filters")
fast_p = st.sidebar.number_input("Fast MA Period", value=20)
slow_p = st.sidebar.number_input("Slow MA Period", value=50)
min_dist_pct = st.sidebar.number_input(
    "Min Price Distance to Fast MA (%)", value=-2.0
)
max_dist_pct = st.sidebar.number_input(
    "Max Price Distance to Fast MA (%)", value=3.0
)
atr_mult = st.sidebar.number_input("Stop Loss ATR Multiplier", value=1.5)
risk_budget = st.sidebar.number_input("Max Trade Risk Cap ($)", value=35.0)
min_rr_req = st.sidebar.number_input("Min Target R:R Ratio", value=2.0)
min_mcap_req = st.sidebar.number_input("Min Market Cap ($B)", value=20.0)
min_vol_req = st.sidebar.number_input("Min ATR Volatility (%)", value=2.5)
min_volume_m = st.sidebar.number_input("Min Avg Volume (M)", value=5.0)
min_rvol_req = st.sidebar.number_input("Min Relative Volume (RVOL)", value=1.2)
require_trend = st.sidebar.checkbox(
    "Require Uptrend (Fast MA > Slow MA)", value=True
)

if st.sidebar.button("🔍 Run Screener Scan", type="primary"):
  st.write(f"Scanning target list of {len(tickers)} stocks...")
  progress_bar = st.progress(0)
  results = []
  end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
  start_date = (pd.Timestamp.today() - pd.Timedelta(days=365)).strftime(
      "%Y-%m-%d"
  )

  batch_size = 50
  for i in range(0, len(tickers), batch_size):
    batch = tickers[i : i + batch_size]
    progress_bar.progress(min((i + batch_size) / len(tickers), 1.0))

    try:
      data = yf.download(
          batch,
          start=start_date,
          end=end_date,
          progress=False,
          auto_adjust=True,
          group_by="ticker",
          threads=True,
      )

      if data.empty:
        continue

      for ticker in batch:
        try:
          if len(batch) == 1:
            df = data.copy()
          else:
            if isinstance(data.columns, pd.MultiIndex):
              if ticker not in data.columns.levels[0]:
                continue
              df = data[ticker].dropna(how="all").copy()
            else:
              if ticker not in data.columns:
                continue
              df = data[[ticker]].copy()

          df = df.dropna(how="all")
          if df.empty or len(df) < max(fast_p, slow_p):
            continue

          df.columns = [str(c).strip().capitalize() for c in df.columns]
          if (
              "Close" not in df.columns
              or "High" not in df.columns
              or "Volume" not in df.columns
          ):
            continue

          df["FastMA"] = df["Close"].rolling(window=int(fast_p)).mean()
          df["SlowMA"] = df["Close"].rolling(window=int(slow_p)).mean()

          df["PrevClose"] = df["Close"].shift(1)
          df["TR"] = np.maximum(
              df["High"] - df["Low"],
              np.maximum(
                  np.abs(df["High"] - df["PrevClose"]),
                  np.abs(df["Low"] - df["PrevClose"]),
              ),
          )
          df["ATR"] = df["TR"].rolling(window=14).mean()
          df["AvgVol20"] = df["Volume"].rolling(window=20).mean() / 1e6
          df["VolAvg20Raw"] = df["Volume"].rolling(window=20).mean()
          df["RVOL"] = df["Volume"] / df["VolAvg20Raw"]

          df = df.dropna(
              subset=["Close", "FastMA", "SlowMA", "ATR", "AvgVol20", "RVOL"]
          )
          if df.empty:
            continue

          latest = df.iloc[-1]
          price = float(latest["Close"])
          fast_ma_val = float(latest["FastMA"])
          slow_ma_val = float(latest["SlowMA"])
          atr = float(latest["ATR"])
          avg_vol_m = float(latest["AvgVol20"])
          rvol = float(latest["RVOL"])

          dist_from_fast_pct = ((price - fast_ma_val) / fast_ma_val) * 100.0
          volatility_pct = (atr / price) * 100.0

          mcap_b = 50.0
          try:
            tk_obj = yf.Ticker(ticker)
            mcap_val = tk_obj.info.get("marketCap", 0)
            if mcap_val:
              mcap_b = mcap_val / 1e9
          except Exception:
            pass

          if not bypass_filters:
            if require_trend and not (fast_ma_val > slow_ma_val):
              continue
            if not (min_dist_pct <= dist_from_fast_pct <= max_dist_pct):
              continue
            if mcap_b > 0 and mcap_b < min_mcap_req:
              continue
            if volatility_pct < min_vol_req:
              continue
            if avg_vol_m < min_volume_m:
              continue
            if rvol < min_rvol_req:
              continue

          stop_distance = atr_mult * atr
          stop_loss = price - stop_distance
          shares = max(1, int(risk_budget // stop_distance))
          actual_risk = shares * stop_distance
          take_profit = price + (min_rr_req * stop_distance)
          actual_rr = (take_profit - price) / stop_distance

          results.append({
              "Ticker": ticker,
              "Price": round(price, 2),
              "M-Cap ($B)": round(mcap_b, 1),
              "Avg Vol (M)": round(avg_vol_m, 1),
              "RVOL": round(rvol, 2),
              "Stop Loss": round(stop_loss, 2),
              "Take Profit": round(take_profit, 2),
              "R:R Ratio": f"1:{actual_rr:.2f}",
              "Shares": shares,
              "Max Risk ($)": round(actual_risk, 2),
          })
        except Exception:
          continue
    except Exception:
      continue

  if results:
    res_df = pd.DataFrame(results)
    st.success(
        f"Scan complete! Scanned S&P 500 and found {len(res_df)} matching"
        " setups."
    )
    st.dataframe(res_df, use_container_width=True)
  else:
    st.warning("Scan finished, but no setups matched your criteria.")