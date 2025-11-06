# streamlit_app.py
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Premium Seller Pro", layout="wide")
st.title("Premium Seller Pro – High-Probability Options Strategies")

st.warning("NOT FINANCIAL ADVICE. Options carry unlimited risk. Paper trade first.")

# ------------------- Helpers -------------------
def bs_call(S, K, T, r, sigma, q=0):
    if T <= 0 or sigma <= 0: return max(S - K * np.exp(-r*T), 0)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_put(S, K, T, r, sigma, q=0):
    if T <= 0 or sigma <= 0: return max(K*np.exp(-r*T) - S, 0)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def option_delta(S, K, T, r, sigma, q=0, type_="call"):
    if T <= 0 or sigma <= 0: return 1 if (type_=="call" and S>K) else -1 if (type_=="put" and K>S) else 0
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return np.exp(-q*T) * norm.cdf(d1) if type_=="call" else -np.exp(-q*T) * norm.cdf(-d1)

def get_atm_iv(calls, puts, S):
    # Closest strike to spot
    all_strikes = pd.concat([calls['strike'], puts['strike']]).unique()
    atm = min(all_strikes, key=lambda x: abs(x - S))
    row = calls[calls.strike == atm]
    if row.empty: row = puts[puts.strike == atm]
    return row.iloc[0].impliedVolatility if not row.empty else 0.3

# ------------------- Sidebar Inputs -------------------
with st.sidebar:
    st.header("Inputs")
    ticker_sym = st.text_input("Ticker", "AAPL").upper()
    strategy = st.selectbox("Strategy", [
        "Short Strangle", "Short Straddle", "Iron Condor",
        "Bull Put Spread", "Bear Call Spread", "Jade Lizard"
    ])
    target_pop = st.slider("Min POP (%)", 70, 98, 90)
    max_dte = st.slider("Max DTE", 7, 120, 45)
    min_credit = st.number_input("Min Credit ($)", 0.10, 10.0, 0.50)
    delta_target = st.slider("Short Delta (abs)", 0.05, 0.30, 0.16)
    rr_min = st.slider("Min Reward:Risk", 0.1, 2.0, 0.3)

# ------------------- Main Logic -------------------
if st.button("Scan Chains"):
    with st.spinner("Fetching live data..."):
        ticker = yf.Ticker(ticker_sym)
        try:
            S = ticker.fast_info["lastPrice"]
        except:
            st.error("Invalid ticker.")
            st.stop()

        # Rates
        try:
            r = yf.Ticker("^TNX").fast_info["lastPrice"] / 100
        except:
            r = 0.05
        q = ticker.info.get("dividendYield") or 0

        # HV (2y)
        hist = ticker.history(period="2y")
        if len(hist) < 50:
            st.error("Not enough history.")
            st.stop()
        rets = np.log(hist.Close / hist.Close.shift(1))
        hv = rets.std() * np.sqrt(252)

        st.write(f"**Spot:** ${S:.2f} | **HV:** {hv*100:.1f}% | **r:** {r*100:.2f}%")

        expirations = [e for e in ticker.options if (datetime.strptime(e, "%Y-%m-%d") - datetime.now()).days <= max_dte]
        results = []

        for exp in expirations:
            dte = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
            T = dte / 365.25
            chain = ticker.option_chain(exp)
            calls = chain.calls[(chain.calls.bid > 0.01) & (chain.calls.impliedVolatility > 0.01)]
            puts = chain.puts[(chain.puts.bid > 0.01) & (chain.puts.impliedVolatility > 0.01)]
            if calls.empty or puts.empty: continue

            iv = get_atm_iv(calls, puts, S)

            # ----- Strategy Builders -----
            def add_trade(trade):
                if trade["Credit"] < min_credit: return
                if trade["POP"] < target_pop/100: return
                if trade.get("RR", 1) < rr_min: return
                results.append(trade)

            # 1. Short Strangle
            if strategy in ["Short Strangle", "Short Straddle"]:
                short_call = calls.iloc[np.argmin(np.abs(calls.impliedVolatility * calls.bid - delta_target * S))]  # approx delta
                short_put = puts.iloc[np.argmin(np.abs(puts.impliedVolatility * puts.bid - delta_target * S))]
                Kc, Kp = short_call.strike, short_put.strike
                if strategy == "Short Straddle": Kp = Kc = round(S/5)*5
                credit = short_call.bid + short_put.bid
                max_loss = float('inf')
                pop = norm.cdf((Kp - credit - S)/ (S * iv * np.sqrt(T))) + norm.cdf(-(Kc + credit - S)/ (S * iv * np.sqrt(T)))
                add_trade({
                    "Exp": exp, "DTE": dte, "Strategy": strategy,
                    "Strikes": f"{Kp}P/{Kc}C", "Credit": round(credit,2),
                    "POP": round(pop*100,1), "RR": 0,
                    "Valuation": "—"
                })

            # 2. Iron Condor
            if strategy == "Iron Condor":
                # Short wings ~0.16Δ, long wings ~0.05Δ
                short_put = puts.iloc[np.argmin(np.abs([abs(option_delta(S, k, T, r, iv, q, "put")) - delta_target for k in puts.strike]))]
                short_call = calls.iloc[np.argmin(np.abs([abs(option_delta(S, k, T, r, iv, q, "call")) - delta_target for k in calls.strike]))]
                long_put = puts[puts.strike < short_put.strike].iloc[-1] if not puts[puts.strike < short_put.strike].empty else short_put
                long_call = calls[calls.strike > short_call.strike].iloc[0] if not calls[calls.strike > short_call.strike].empty else short_call

                credit = short_put.bid + short_call.bid - long_put.ask - long_call.ask
                width = min(short_put.strike - long_put.strike, short_call.strike - long_call.strike)
                max_loss = width - credit
                rr = credit / max_loss if max_loss > 0 else 0
                pop = 0.85  # approximate
                theo = bs_put(S, short_put.strike, T, r, hv, q) + bs_call(S, short_call.strike, T, r, hv, q)
                ratio = credit / theo if theo > 0 else 99
                val = "Expensive" if ratio>1.2 else "Rich" if ratio>1.0 else "Fair"
                add_trade({
                    "Exp": exp, "DTE": dte, "Strategy": "Iron Condor",
                    "Strikes": f"{long_put.strike}P/{short_put.strike}P - {short_call.strike}C/{long_call.strike}C",
                    "Credit": round(credit,2), "MaxLoss": round(max_loss,2), "RR": round(rr,2),
                    "POP": round(pop*100,1), "PremRatio": round(ratio,2), "Valuation": val
                })

            # 3. Bull Put / Bear Call Spreads
            if strategy in ["Bull Put Spread", "Bear Call Spread"]:
                is_bull = strategy == "Bull Put Spread"
                short_leg = puts if is_bull else calls
                long_leg = puts if is_bull else calls
                short_opt = short_leg.iloc[np.argmin(np.abs([abs(option_delta(S, k, T, r, iv, q, "put" if is_bull else "call")) - delta_target for k in short_leg.strike]))]
                long_opt = long_leg[long_leg.strike < short_opt.strike].iloc[-1] if is_bull else long_leg[long_leg.strike > short_opt.strike].iloc[0]
                if long_opt.empty: continue
                credit = short_opt.bid - long_opt.ask
                width = abs(short_opt.strike - long_opt.strike)
                max_loss = width - credit
                rr = credit / max_loss if max_loss > 0 else 0
                add_trade({
                    "Exp": exp, "DTE": dte, "Strategy": strategy,
                    "Strikes": f"{long_opt.strike}/{short_opt.strike}",
                    "Credit": round(credit,2), "MaxLoss": round(max_loss,2), "RR": round(rr,2),
                    "POP": 88, "Valuation": "—"
                })

            # 4. Jade Lizard (Short Put + Short Call ITM + Long Call OTM)
            if strategy == "Jade Lizard":
                short_put = puts.iloc[np.argmin(np.abs([abs(option_delta(S, k, T, r, iv, q, "put")) - 0.30 for k in puts.strike]))]
                short_call_itm = calls[calls.strike < S].iloc[-1] if not calls[calls.strike < S].empty else calls.iloc[0]
                long_call_otm = calls[calls.strike > S].iloc[0]
                credit = short_put.bid + short_call_itm.bid - long_call_otm.ask
                add_trade({
                    "Exp": exp, "DTE": dte, "Strategy": "Jade Lizard",
                    "Strikes": f"{short_put.strike}P + {short_call_itm.strike}C - {long_call_otm.strike}C",
                    "Credit": round(credit,2), "POP": 80, "Valuation": "—"
                })

        # ------------------- Results -------------------
        if results:
            df = pd.DataFrame(results)
            df = df.sort_values("Credit", ascending=False)
            st.success(f"Found **{len(df)}** setups ≥ {target_pop}% POP")
            st.dataframe(df.head(20), use_container_width=True)

            top = df.iloc[0]
            st.metric(
                label=f"**Best: {top['Strategy']}** – {top['Strikes']} (exp {top['Exp']})",
                value=f"${top['Credit']} credit",
                delta=f"POP {top['POP']}% | RR {top.get('RR','—')}"
            )
            if top.get("Valuation") and "Expensive" in top["Valuation"]:
                st.balloons()

            # Plot payoff
            if "MaxLoss" in top:
                fig = go.Figure()
                x = np.linspace(S*0.7, S*1.3, 200)
                payoff = np.where(x < top['Strikes'].split('/')[0], top['Credit'], 
                                np.where(x > top['Strikes'].split('/')[-1], top['Credit'], 
                                         top['Credit'] - abs(x - S)))
                fig.add_trace(go.Scatter(x=x, y=payoff, mode='lines', name='Payoff'))
                fig.add_vline(x=S, line_dash="dash", line_color="green")
                fig.update_layout(title="Payoff at Expiration", xaxis_title="Stock Price", yaxis_title="P&L")
                st.plotly_chart(fig(fig, use_container_width=True)
        else:
            st.error("No setups found. Try lowering POP, increasing DTE, or using high-IV tickers (NVDA, TSLA).")
