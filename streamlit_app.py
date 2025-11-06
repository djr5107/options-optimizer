# streamlit_app.py
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Options Screener", layout="wide")
st.title("High-Probability Options Screener")
st.caption("Works on AAPL, SPY, NVDA, TSLA, etc. | Paper trade only.")

# --- Helpers ---
def get_price(ticker):
    try:
        return yf.Ticker(ticker).fast_info["lastPrice"]
    except:
        return None

def get_risk_free():
    try:
        return yf.Ticker("^TNX").fast_info["lastPrice"] / 100
    except:
        return 0.05

def get_div_yield(ticker):
    try:
        return yf.Ticker(ticker).info.get("dividendYield", 0) or 0
    except:
        return 0

def bs_delta(S, K, T, r, sigma, q=0, type_="call"):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return np.exp(-q*T) * norm.cdf(d1) if type_ == "call" else -np.exp(-q*T) * norm.cdf(-d1)

def get_atm_iv(calls, puts, S):
    strikes = np.unique(np.concatenate([calls.strike.values, puts.strike.values]))
    atm = strikes[np.argmin(np.abs(strikes - S))]
    row = calls[calls.strike == atm]
    if row.empty: row = puts[puts.strike == atm]
    return row.impliedVolatility.iloc[0] if not row.empty else 0.3

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", "AAPL").upper()
    strategy = st.selectbox("Strategy", ["Short Strangle", "Iron Condor"])
    min_pop = st.slider("Min POP (%)", 50, 95, 70)
    max_dte = st.slider("Max DTE", 7, 90, 45)
    min_credit = st.number_input("Min Credit ($)", 0.05, 5.0, 0.20)
    delta_target = st.slider("Short Delta", 0.05, 0.40, 0.16)

# --- Scan ---
if st.button("Scan Now"):
    with st.spinner("Loading..."):
        S = get_price(ticker)
        if not S:
            st.error("Cannot fetch price. Check ticker.")
            st.stop()

        r = get_risk_free()
        q = get_div_yield(ticker)

        # Get expirations
        try:
            expirations = yf.Ticker(ticker).options
        except:
            st.error("No options chain found.")
            st.stop()

        expirations = [e for e in expirations if 0 < (datetime.strptime(e, "%Y-%m-%d") - datetime.now()).days <= max_dte]
        if not expirations:
            st.error("No expirations in range.")
            st.stop()

        results = []
        debug = st.empty()

        for exp in expirations:
            dte = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
            T = dte / 365.25
            try:
                chain = yf.Ticker(ticker).option_chain(exp)
            except:
                continue

            calls = chain.calls
            puts = chain.puts

            if calls.empty or puts.empty:
                continue

            iv = get_atm_iv(calls, puts, S)

            # Filter liquid options
            calls = calls[(calls.bid > 0.01) & (calls.ask > 0.01) & (calls.impliedVolatility > 0.01)]
            puts = puts[(puts.bid > 0.01) & (puts.ask > 0.01) & (puts.impliedVolatility > 0.01)]
            if calls.empty or puts.empty: continue

            # === SHORT STRANGLE ===
            if strategy == "Short Strangle":
                # Find ~0.16 delta put and call
                put_deltas = puts.strike.apply(lambda k: abs(bs_delta(S, k, T, r, iv, q, "put") - delta_target))
                call_deltas = calls.strike.apply(lambda k: abs(bs_delta(S, k, T, r, iv, q, "call") + delta_target))
                if put_deltas.empty or call_deltas.empty: continue

                short_put = puts.loc[put_deltas.idxmin()]
                short_call = calls.loc[call_deltas.idxmin()]

                credit = short_put.bid + short_call.bid
                if credit < min_credit: continue

                be_low = short_put.strike - credit
                be_high = short_call.strike + credit
                pop = norm.cdf((S - be_low)/(S*iv*np.sqrt(T))) + norm.cdf((be_high - S)/(S*iv*np.sqrt(T)))

                if pop < min_pop / 100: continue

                results.append({
                    "Exp": exp, "DTE": dte, "Strategy": "Short Strangle",
                    "Strikes": f"{short_put.strike}P / {short_call.strike}C",
                    "Credit": round(credit, 2), "POP (%)": round(pop*100, 1)
                })

            # === IRON CONDOR ===
            if strategy == "Iron Condor":
                put_deltas = puts.strike.apply(lambda k: abs(bs_delta(S, k, T, r, iv, q, "put") - delta_target))
                call_deltas = calls.strike.apply(lambda k: abs(bs_delta(S, k, T, r, iv, q, "call") + delta_target))
                if put_deltas.empty or call_deltas.empty: continue

                short_put = puts.loc[put_deltas.idxmin()]
                short_call = calls.loc[call_deltas.idxmin()]

                long_put = puts[puts.strike < short_put.strike]
                long_call = calls[calls.strike > short_call.strike]
                if long_put.empty or long_call.empty: continue
                long_put = long_put.iloc[-1]
                long_call = long_call.iloc[0]

                credit = short_put.bid + short_call.bid - long_put.ask - long_call.ask
                if credit < min_credit: continue

                width = min(short_put.strike - long_put.strike, short_call.strike - long_call.strike)
                max_loss = width - credit
                rr = credit / max_loss if max_loss > 0 else 0

                results.append({
                    "Exp": exp, "DTE": dte, "Strategy": "Iron Condor",
                    "Strikes": f"{long_put.strike}P/{short_put.strike}P - {short_call.strike}C/{long_call.strike}C",
                    "Credit": round(credit, 2), "Max Loss": round(max_loss, 2),
                    "RR": round(rr, 2), "POP (%)": 85
                })

        # --- Results ---
        if results:
            df = pd.DataFrame(results).sort_values("Credit", ascending=False)
            st.success(f"Found {len(df)} setups")
            st.dataframe(df.head(10), use_container_width=True)

            top = df.iloc[0]
            st.metric(f"**Top: {top['Strategy']}**", f"${top['Credit']} credit", f"POP {top['POP (%)']}%")

            # Payoff Chart
            fig = go.Figure()
            x = np.linspace(S*0.7, S*1.3, 200)
            payoff = np.full_like(x, top['Credit'])

            if "Iron Condor" in top['Strategy']:
                strikes = top['Strikes'].replace(" ", "").split("-")
                low = float(strikes[0].split("/")[0][:-1])
                high = float(strikes[1].split("/")[1][:-1])
                payoff = np.where(x <= low, top['Credit'],
                         np.where(x >= high, top['Credit'],
                         top['Credit'] - np.maximum(0, low - x) - np.maximum(0, x - high)))

            fig.add_trace(go.Scatter(x=x, y=payoff, mode='lines', name='P&L'))
            fig.add_vline(x=S, line_dash="dash", line_color="green")
            fig.update_layout(title="Payoff at Expiration", xaxis_title="Price", yaxis_title="P&L")
            st.plotly_chart(fig, use_container_width=True)

        else:
            debug.error(f"No data for {ticker}. Try:\n"
                        "- NVDA, TSLA, AMD (high IV)\n"
                        "- SPY, QQQ (liquid)\n"
                        "- Lower Min POP to 60%\n"
                        "- Increase Max DTE to 60")
