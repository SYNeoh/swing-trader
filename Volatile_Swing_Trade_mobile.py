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
  try:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df["Symbol"].tolist()
    tickers = [str(t).replace(".", "-").strip() for t in tickers]
    if len(tickers) > 400:
      return tickers
  except Exception:
    pass

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
          if stop_distance <= 0 or np.isnan(stop_distance):
            stop_distance = price * 0.02

          stop_loss = price - stop_distance
          shares = max(1, int(risk_budget // stop_distance))
          actual_risk = shares * stop_distance

          take_profit = price + (min_rr_req * stop_distance)
          actual_rr = (take_profit - price) / stop_distance

          peaks, _ = find_peaks(df["High"].values, distance=15)
          overhead_peaks = sorted(
              [
                  p
                  for p in df["High"].iloc[peaks].values
                  if p > (price * 1.005)
              ]
          )

          resistance_status = "Clear Target"
          if overhead_peaks:
            nearest_peak = overhead_peaks[0]
            if nearest_peak < take_profit:
              resistance_status = f"⚠️ Block @ ${nearest_peak:.2f}"
            else:
              resistance_status = f"Peak @ ${nearest_peak:.2f}"

          results.append({
              "Ticker": ticker,
              "Price": f"${price:.2f}",
              "M-Cap ($B)": f"${mcap_b:.1f}B",
              "Avg Vol (M)": f"{avg_vol_m:.1f}M",
              "RVOL": f"{rvol:.2f}x",
              "Fast MA": f"${fast_ma_val:.2f}",
              "Slow MA": f"${slow_ma_val:.2f}",
              "Dist (%)": f"{dist_from_fast_pct:+.2f}%",
              "Volatility (%)": f"{volatility_pct:.2f}%",
              "Stop Loss ($)": f"${stop_loss:.2f}",
              "Take Profit ($)": f"${take_profit:.2f}",
              "R:R Ratio": f"1:{actual_rr:.2f}",
              "Shares": int(shares),
              "Max Risk ($)": f"${actual_risk:.2f}",
              "Resistance": resistance_status,
          })
        except Exception:
          continue
    except Exception:
      continue

  if results:
    st.session_state["res_df"] = pd.DataFrame(results)
    first_row = st.session_state["res_df"].iloc[0]
    st.session_state["calc_tk"] = str(first_row["Ticker"])
    st.session_state["calc_en"] = float(
        str(first_row["Price"]).replace("$", "").replace(",", "")
    )
    st.session_state["calc_st"] = float(
        str(first_row["Stop Loss ($)"]).replace("$", "").replace(",", "")
    )
    st.session_state["calc_tp"] = float(
        str(first_row["Take Profit ($)"]).replace("$", "").replace(",", "")
    )
    st.session_state["calc_sh"] = int(first_row["Shares"])
    st.session_state["last_selected_row"] = 0
    st.success(
        f"Scan complete! Scanned and found"
        f" {len(st.session_state['res_df'])} matching setups."
    )
  else:
    st.session_state["res_df"] = pd.DataFrame()
    st.warning("Scan finished, but no setups matched your criteria.")

# --- DISPLAY TABLE & HANDLE ROW SELECTION ---
if "res_df" in st.session_state and not st.session_state["res_df"].empty:
  st.markdown("### Screener Results (Click a row to select setup)")
  event = st.dataframe(
      st.session_state["res_df"],
      key="screener_table",
      selection_mode="single-row",
      on_select="rerun",
      use_container_width=True,
  )

  if "last_selected_row" not in st.session_state:
    st.session_state["last_selected_row"] = 0

  df_display = st.session_state["res_df"]

  selected_rows = []
  if event:
    try:
      selected_rows = event.selection.rows
    except Exception:
      try:
        selected_rows = event["selection"]["rows"]
      except Exception:
        pass

  if selected_rows and selected_rows[0] != st.session_state["last_selected_row"]:
    st.session_state["last_selected_row"] = selected_rows[0]
    row_idx = selected_rows[0]
    selected_row_data = df_display.iloc[row_idx]

    st.session_state["calc_tk"] = str(selected_row_data["Ticker"])
    st.session_state["calc_en"] = float(
        str(selected_row_data["Price"]).replace("$", "").replace(",", "")
    )
    st.session_state["calc_st"] = float(
        str(selected_row_data["Stop Loss ($)"])
        .replace("$", "")
        .replace(",", "")
    )
    st.session_state["calc_tp"] = float(
        str(selected_row_data["Take Profit ($)"])
        .replace("$", "")
        .replace(",", "")
    )
    st.session_state["calc_sh"] = int(selected_row_data["Shares"])
    st.rerun()

  if "calc_tk" not in st.session_state:
    st.session_state["calc_tk"] = str(df_display.iloc[0]["Ticker"])
    st.session_state["calc_en"] = float(
        str(df_display.iloc[0]["Price"]).replace("$", "").replace(",", "")
    )
    st.session_state["calc_st"] = float(
        str(df_display.iloc[0]["Stop Loss ($)"]).replace("$", "").replace(",", "")
    )
    st.session_state["calc_tp"] = float(
        str(df_display.iloc[0]["Take Profit ($)"]).replace("$", "").replace(",", "")
    )
    st.session_state["calc_sh"] = int(df_display.iloc[0]["Shares"])

  # --- SECTION 3: INTEGRATED EXECUTION PLAN & CALCULATOR ---
  st.markdown("---")
  st.subheader("3. Selected Setup Trade Execution Plan (Auto-Risk Lock)")


  # Callback: When Entry changes, automatically adjust Stop Loss to maintain Max Risk Budget ($35)
  def update_on_entry():
    new_entry = st.session_state["calc_en"]
    current_shares = st.session_state["calc_sh"]
    if current_shares > 0:
      # Required risk per share to match risk_budget (e.g. 35) exactly
      risk_per_share_needed = risk_budget / current_shares
      st.session_state["calc_st"] = new_entry - risk_per_share_needed


  col1, col2, col3, col4, col5 = st.columns(5)
  with col1:
    calc_ticker = st.text_input("Ticker", key="calc_tk")
  with col2:
    calc_entry = st.number_input(
        "Entry Price ($)",
        format="%.2f",
        key="calc_en",
        on_change=update_on_entry,
    )
  with col3:
    calc_stop = st.number_input("Stop Loss ($)", format="%.2f", key="calc_st")
  with col4:
    calc_target = st.number_input(
        "Take Profit ($)", format="%.2f", key="calc_tp"
    )
  with col5:
    calc_shares = st.number_input(
        "Position (Shares)", min_value=1, key="calc_sh"
    )

  try:
    risk_per_share = calc_entry - calc_stop
    if risk_per_share <= 0:
      st.error("⚠️ Error: Stop loss must be lower than Entry price.")
    else:
      reward_per_share = calc_target - calc_entry
      total_risk = calc_shares * risk_per_share
      total_reward = calc_shares * reward_per_share
      rr_ratio = reward_per_share / risk_per_share
      total_capital = calc_shares * calc_entry

      # Exactly 6 lines of clean plain text with uniform sizing and zero asterisks
      st.markdown(f"Ticker: {calc_ticker.upper()}")
      st.markdown(f"Capital Required: ${total_capital:,.2f}")
      st.markdown(f"Shares: {calc_shares}")
      st.markdown(
          f"Total Downside Risk: ${total_risk:,.2f} (${risk_per_share:.2f}/share)"
      )
      st.markdown(f"Total Potential Gain: ${total_reward:,.2f}")
      st.markdown(f"Risk / Reward Ratio: 1 : {rr_ratio:.2f}")
  except Exception:
    st.info("Fill out the execution fields above to preview your risk metrics.")