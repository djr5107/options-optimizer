import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime

# Black-Scholes functions
def bs_call(S, K, T, r, sigma, q=0):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma, q=0):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def calc_d2(S, K, T, r, sigma, q=0):
    if T <= 0 or sigma <= 0:
        return np.inf if S > K else -np.inf
    return (np.log(S / K) + (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

# Streamlit app
st.title("High-Probability Options Seller Tool (Short Strangle/Straddle)")
st.warning("WARNING: Options trading is risky. No guarantees. Use paper trading. Naked shorts have unlimited risk.")

ticker_sym = st.text_input("Stock Ticker", "AAPL").upper()
strategy = st.selectbox("Strategy", ["Short Strangle (flexible strikes)", "Short Straddle (same strike only)"])
min_pop_pct = st.slider("Minimum POP (%)", 70, 99, 95)
max_dte = st.slider("Max Days to Expiration", 7, 180, 60)

if st.button("Run Analysis"):
    with st.spinner("Fetching data & scanning chains..."):
        ticker = yf.Ticker(ticker_sym)
        try:
            S = ticker.fast_info["lastPrice"]
        except:
            st.error("Invalid ticker or no data.")
            st.stop()

        # Risk-free rate (^TNX = 10yr Treasury)
        try:
            r = yf.Ticker("^TNX").fast_info["lastPrice"] / 100
        except:
            r = 0.05
        q = ticker.info.get("dividendYield", 0) or 0

        # Historical Volatility (2y annualized)
        hist = ticker.history(period="2y")
        if len(hist) < 100:
            st.error("Not enough history for HV.")
            st.stop()
        hist["ret"] = np.log(hist["Close"] / hist["Close"].shift(1))
        hv = hist["ret"].std() * np.sqrt(252)

        st.write(f"**Current Price:** ${S:.2f} | **HV:** {hv*100:.1f}% | **r:** {r*100:.2f}% | **q:** {q*100:.2f}%")

        expirations = ticker.options
        results = []

        for exp in expirations:
            exp_dt = datetime.strptime(exp, "%Y-%m-%d")
            dte = (exp_dt - datetime.now()).days
            if dte > max_dte or dte <= 0:
                continue
            T = dte / 365.25

            chain = ticker.option_chain(exp)
            calls = chain.calls[(chain.calls.bid > 0.01) & (chain.calls.impliedVolatility > 0.01)]
            puts = chain.puts[(chain.puts.bid > 0.01) & (chain.puts.impliedVolatility > 0.01)]
            if calls.empty or puts.empty:
                continue

            # ATM IV for POP (market's vol forecast)
            strikes = sorted(set(calls.strike) | set(puts.strike))
            atm_strike = min(strikes, key=lambda x: abs(x - S))
            atm_row = calls[calls.strike == atm_strike] if atm_strike in calls.strike.values else puts[puts.strike == atm_strike]
            if atm_row.empty:
                atm_row = calls.iloc[[np.argmin(np.abs(calls.strike - S))]]
            sigma = atm_row.iloc[0].impliedVolatility  # ATM IV

            for _, put_row in puts.iterrows():
                Kp = put_row.strike
                put_bid = put_row.bid
                for _, call_row in calls.iterrows():
                    Kc = call_row.strike
                    call_bid = call_row.bid
                    if Kc <= Kp:
                        continue

                    # Force straddle if selected
                    if strategy == "Short Straddle (same strike only)" and abs(Kp - Kc) > 0.01:
                        continue

                    credit = put_bid + call_bid
                    if credit < 0.10:  # minimum worthwhile
                        continue

                    BE_low = Kp - credit
                    BE_high = Kc + credit
                    if BE_low >= BE_high:
                        continue

                    # POP using ATM IV
                    d2_low = calc_d2(S, BE_low, T, r, sigma, q)
                    d2_high = calc_d2(S, BE_high, T, r, sigma, q)
                    pop = norm.cdf(d2_low) - norm.cdf(d2_high)

                    if pop < min_pop_pct / 100:
                        continue

                    # Theoretical fair value using HV
                    theo_call = bs_call(S, Kc, T, r, hv, q)
                    theo_put = bs_put(S, Kp, T, r, hv, q)
                    theo_credit = theo_call + theo_put
                    premium_ratio = credit / theo_credit if theo_credit > 0 else np.inf

                    expected_move_pct = sigma * np.sqrt(T) * 100

                    results.append({
                        "Expiration": exp,
                        "DTE": dte,
                        "Put Strike": Kp,
                        "Call Strike": Kc,
                        "Credit": round(credit, 2),
                        "POP (%)": round(pop * 100, 1),
                        "Prem Ratio": round(premium_ratio, 2),
                        "Valuation": "Expensive (sell)" if premium_ratio > 1.2 else "Rich (sell)" if premium_ratio > 1.0 else "Fair" if 0.8 <= premium_ratio <= 1.0 else "Cheap (avoid selling)",
                        "ATM IV (%)": round(sigma * 100, 1),
                        "Exp Move (±%)": round(expected_move_pct, 1),
                        "Breakevens": f"${BE_low:.2f} – ${BE_high:.2f}"
                    })

        if results:
            df = pd.DataFrame(results)
            df = df.sort_values("Credit", ascending=False)
            st.success(f"Found {len(df)} setups ≥ {min_pop_pct}% POP!")
            st.dataframe(df.head(20), use_container_width=True)

            st.write("**Top pick:** Highest credit while meeting criteria.")
            top = df.iloc[0]
            st.metric(f"Best Credit: ${top['Credit']} ({top['Put Strike']}P / {top['Call Strike']}C, exp {top['Expiration']})",
                      f"POP {top['POP (%)']}% | {top['Valuation']} (ratio {top['Prem Ratio']})")

            if top["Prem Ratio"] > 1.0:
                st.balloons()
                st.write("🎉 Options look **expensive**—great for selling!")
        else:
            st.error("No setups found meeting criteria. Try lowering min POP, increasing max DTE, or high-IV stock (e.g., TSLA, NVDA during earnings). Short straddles rarely hit 95% POP.")
