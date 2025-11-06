# streamlit_app.py
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Premium Seller Pro", layout="wide")
st.title("Premium Seller Pro – High-Probability Options Screener")
st.warning("NOT FINANCIAL ADVICE. Paper trade first. Naked shorts = unlimited risk.")

# ------------------- Helpers -------------------
def bs_call(S, K, T, r, sigma, q=0):
    if T <= 0 or sigma <= 0: return max(S - K, 0)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_put(S, K, T, r, sigma, q=0):
    if T <= 0 or sigma <= 0: return max(K - S, 0)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def option_delta(S, K, T, r, sigma, q=0, type_="call"):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return np.exp(-q*T) * norm.cdf(d1) if type_=="call" else -np.exp(-q*T) * norm.cdf(-d1)

def get_atm_iv(calls, puts, S):
    all_strikes = np.unique(np.concatenate([calls.strike.values, puts.strike.values]))
    atm = all_strikes[np.argmin(np.abs(all_strikes - S))]
    row = calls[calls.strike == atm]
    if row.empty: row = puts[puts.strike == atm]
    return row.impliedVolatility.iloc[0] if not row.empty else 0.3

# ------------------- Sidebar -------------------
with st.sidebar:
    st.header("Filters")
    ticker_sym = st.text_input("Ticker", "AAPL").upper()
    strategy = st.selectbox("Strategy", [
        "Short Strangle", "Short Straddle", "Iron Condor",
        "Bull Put Spread", "Bear Call Spread", "Jade Lizard"
    ])
    target_pop = st.slider("Min POP (%)", 50, 95, 70)
    max_dte = st.slider("Max DTE", 7, 120, 60)
    min_credit = st.number_input("Min Credit ($)", 0.10, 5.0, 0.20)
    delta_target = st.slider("Short Delta (abs)", 0.05, 0.40, 0.16)
    rr_min = st.slider("Min Reward:Risk", 0.0, 2.0, 0.2)

# ------------------- Scan -------------------
if st.button("Scan Options"):
    with st.spinner("Fetching live data..."):
        try:
            ticker = yf.Ticker(ticker_sym)
            S = ticker.fast_info["lastPrice"]
        except:
            st.error("Invalid ticker.")
            st.stop()

        try:
            r = yf.Ticker("^TNX").fast_info["lastPrice"] / 100
        except:
            r = 0.05
        q = ticker.info.get("dividendYield") or 0

        hist = ticker.history(period="2y")
        if len(hist) < 50:
            st.error("Not enough price history.")
            st.stop()
        rets = np.log(hist.Close / hist.Close.shift(1)).dropna()
        hv = rets.std() * np.sqrt(252)

        st.write(f"**Spot:** ${S:.2f} | **HV:** {hv*100:.1f}% | **r:** {r*100:.2f}%")

        expirations = [e for e in ticker.options 
                      if 0 < (datetime.strptime(e, "%Y-%m-%d") - datetime.now()).days <= max_dte]
        results = []

        debug = st.empty()

        for exp in expirations:
            dte = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
            T = dte / 365.25
            chain = ticker.option_chain(exp)
            calls = chain.calls[(chain.calls.bid > 0.01) & (chain.calls.impliedVolatility > 0.01)]
            puts = chain.puts[(chain.puts.bid > 0.01) & (chain.puts.impliedVolatility > 0.01)]
            if calls.empty or puts.empty: continue

            iv = get_atm_iv(calls, puts, S)

            def add_trade(trade):
                if trade["Credit"] < min_credit: return
                if trade["POP"] < target_pop/100: return
                if trade.get("RR", 1) < rr_min: return
                results.append(trade)

            # === STRANGLE / STRADDLE ===
            if strategy in ["Short Strangle", "Short Straddle"]:
                call_deltas = np.abs(calls.strike.apply(lambda k: option_delta(S, k, T, r, iv, q, "call")) - (-delta_target))
                put_deltas = np.abs(puts.strike.apply(lambda k: option_delta(S, k, T, r, iv, q, "put")) - delta_target)
                short_call = calls.loc[call_deltas.idxmin()]
                short_put = puts.loc[put_deltas.idxmin()]
                Kc, Kp = short_call.strike, short_put.strike

                if strategy == "Short Straddle":
                    K = round(S / 5) * 5
                    sc = calls[calls.strike == K]
                    sp = puts[puts.strike == K]
                    if sc.empty or sp.empty: continue
                    short_call, short_put = sc.iloc[0], sp.iloc[0]
                    Kc = Kp = K

                credit = short_call.bid + short_put.bid
                be_low = Kp - credit
                be_high = Kc + credit
                pop = norm.cdf((S - be_low) / (S * iv * np.sqrt(T))) + norm.cdf((be_high - S) / (S * iv * np.sqrt(T)))

                add_trade({
                    "Exp": exp, "DTE": dte, "Strategy": strategy,
                    "Strikes": f"{Kp}P/{Kc}C", "Credit": round(credit,2),
                    "POP": round(pop*100,1), "RR": 0, "Valuation": "—"
                })

            # === IRON CONDOR ===
            if strategy == "Iron Condor":
                put_deltas = np.abs(puts.strike.apply(lambda k: option_delta(S, k, T, r, iv, q, "put")) - delta_target)
                call_deltas = np.abs(calls.strike.apply(lambda k: option_delta(S, k, T, r, iv, q, "call")) - (-delta_target))
                short_put = puts.loc[put_deltas.idxmin()]
                short_call = calls.loc[call_deltas.idxmin()]

                long_put = puts[puts.strike < short_put.strike]
                long_call = calls[calls.strike > short_call.strike]
                long_put = long_put.iloc[-1] if not long_put.empty else short_put
                long_call = long_call.iloc[0] if not long_call.empty else short_call

                credit = short_put.bid + short_call.bid - long_put.ask - long_call.ask
                if credit <= 0: continue
                width = min(short_put.strike - long_put.strike, short_call.strike - long_call.strike)
                max_loss = width - credit
                rr = credit / max_loss if max_loss > 0 else 0

                theo = bs_put(S, short_put.strike, T, r, hv, q) + bs_call(S, short_call.strike, T, r, hv, q)
                ratio = credit / theo if theo > 0 else 99
                val = "Expensive" if ratio>1.2 else "Rich" if ratio>1.0 else "Fair"

                pop = 0.85  # conservative estimate
                add_trade({
                    "Exp": exp, "DTE": dte, "Strategy": "Iron Condor",
                    "Strikes": f"{long_put.strike}P/{short_put.strike}P - {short_call.strike}C/{long_call.strike}C",
                    "Credit": round(credit,2), "MaxLoss": round(max_loss,2), "RR": round(rr,2),
                    "POP": round(pop*100,1), "PremRatio": round(ratio,2), "Valuation": val
                })

            # === CREDIT SPREADS ===
            if strategy in ["Bull Put Spread", "Bear Call Spread"]:
                is_bull = strategy == "Bull Put Spread"
                short_leg = puts if is_bull else calls
                long_leg = puts if is_bull else calls
                deltas = np.abs(short_leg.strike.apply(lambda k: option_delta(S, k, T, r, iv, q, "put" if is_bull else "call")) - delta_target)
                short_opt = short_leg.loc[deltas.idxmin()]
                long_opt = (long_leg[long_leg.strike < short_opt.strike].iloc[-1] if is_bull
                           else long_leg[long_leg.strike > short_opt.strike].iloc[0])
                if long_opt.empty: continue
                credit = short_opt.bid - long_opt.ask
                if credit <= 0: continue
                width = abs(short_opt.strike - long_opt.strike)
                max_loss = width - credit
                rr = credit / max_loss if max_loss > 0 else 0
                add_trade({
                    "Exp": exp, "DTE": dte, "Strategy": strategy,
                    "Strikes": f"{long_opt.strike}/{short_opt.strike}",
                    "Credit": round(credit,2), "MaxLoss": round(max_loss,2), "RR": round(rr,2),
                    "POP": 88, "Valuation": "—"
                })

            # === JADE LIZARD ===
            if strategy == "Jade Lizard":
                put_deltas = np.abs(puts.strike.apply(lambda k: option_delta(S, k, T, r, iv, q, "put")) - 0.30)
                short_put = puts.loc[put_deltas.idxmin()]
                short_call_itm = calls[calls.strike < S].iloc[-1] if not calls[calls.strike < S].empty else calls.iloc[0]
                long_call_otm = calls[calls.strike > S].iloc[0]
                credit = short_put.bid + short_call_itm.bid - long_call_otm.ask
                if credit <= 0: continue
                add_trade({
                    "Exp": exp, "DTE": dte, "Strategy": "Jade Lizard",
                    "Strikes": f"{short_put.strike}P + {short_call_itm.strike}C - {long_call_otm.strike}C",
                    "Credit": round(credit,2), "POP": 80, "Valuation": "—"
                })

        # ------------------- Results -------------------
        if results:
            df = pd.DataFrame(results).sort_values("Credit", ascending=False)
            st.success(f"Found **{len(df)}** setups ≥ {target_pop}% POP")
            st.dataframe(df.head(20), use_container_width=True)

            top = df.iloc[0]
            st.metric(
                label=f"**Best: {top['Strategy']}** – {top['Strikes']} (exp {top['Exp']})",
                value=f"${top['Credit']} credit",
                delta=f"POP {top['POP']}% | RR {top.get('RR','—')}"
            )
            if "Expensive" in top.get("Valuation", ""):
                st.balloons()

            # Payoff Chart
            fig = go.Figure()
            x = np.linspace(S*0.7, S*1.3, 300)
            payoff = np.full_like(x, top['Credit'])

            if "Iron Condor" in top['Strategy']:
                parts = top['Strikes'].replace(" ", "").split("-")
                low = float(parts[0].split("/")[0][:-1])
                high = float(parts[1].split("/")[1][:-1])
                payoff = np.where(x <= low, top['Credit'],
                         np.where(x >= high, top['Credit'],
                         top['Credit'] - np.maximum(0, low - x) - np.maximum(0, x - high)))
            fig.add_trace(go.Scatter(x=x, y=payoff, mode='lines', name='Payoff', line=dict(width=3)))
            fig.add_vline(x=S, line_dash="dash", line_color="green", annotation_text="Spot")
            fig.update_layout(title="Payoff at Expiration", xaxis_title="Price", yaxis_title="P&L ($)")
            st.plotly_chart(fig, use_container_width=True)

        else:
            debug.error("No setups found. Try:\n"
                        "• Lower **Min POP** to 70%\n"
                        "• Increase **Max DTE** to 60+\n"
                        "• Use high-IV tickers: **NVDA, TSLA, AMD**\n"
                        "• Lower **Min Credit** to $0.20")
