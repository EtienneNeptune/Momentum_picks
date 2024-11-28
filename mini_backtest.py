import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

st.title("Momentum Picks")

# Paramètres de backtest
st.sidebar.header("Paramètres du Backtest")
num_ranges = st.sidebar.number_input("Nombre de plages de dates", min_value=1, max_value=100, value=1)

date_ranges = []
tickers = []
weights = []

for i in range(int(num_ranges)):
    st.sidebar.subheader(f"Plage {i + 1}")
    start_date = st.sidebar.date_input(f"Début de la plage {i + 1}")
    end_date = st.sidebar.date_input(f"Fin de la plage {i + 1}")
    selected_tickers = st.sidebar.text_input(f"Tickers pour la plage {i + 1} (séparés par des virgules)")
    ticker_weights = st.sidebar.text_input(f"Poids pour chaque ticker pour la plage {i + 1} (séparés par des virgules, somme = 1)", "1")

    date_ranges.append((start_date, end_date))
    tickers.append(selected_tickers.split(","))
    weights.append([float(w) for w in ticker_weights.split(",")])


# Fonction pour récupérer les données
@st.cache_data
def fetch_data(tickers, start, end):
    return yf.download(tickers, start=start, end=end)

portfolio_returns = []
portfolio_perf_per_period = []
bench_perf_per_period = []
sp500_returns = []

for i, (start, end) in enumerate(date_ranges):
    if not tickers[i]:
        continue
    # Récupérer les données pour les tickers et calculer les rendements pondérés
    datas = fetch_data(tickers[i], start, end)
    stocks = pd.concat([datas["Close"]])
    # stocks.index = pd.to_datetime(stocks.index)
    returns = stocks.pct_change().dropna()
    #weighted_returns = (returns * weights[i]).sum(axis=1) if daily rebalanced
    cumulative_values = (1 + returns).cumprod()
    st.write(cumulative_values)

    portfolio_value = cumulative_values.dot(weights[i])
    weighted_returns = portfolio_value.pct_change().fillna((portfolio_value.iloc[0]-1)/1)
    portfolio_returns.append(weighted_returns)

    portfolio_perf = np.prod(1+weighted_returns)-1
    portfolio_perf_per_period.append(portfolio_perf)

    st.write(portfolio_perf)
    # Récupérer les rendements du benchmark
    sp500_data = fetch_data("^GSPC", start, end)["Adj Close"].pct_change().dropna()
    st.write(sp500_data)
    sp500_returns.append(sp500_data)
    sp500_perf = np.prod(1 + sp500_data) - 1
    st.write(sp500_perf)
    bench_perf_per_period.append(sp500_perf[0])


# Consolidation des rendements
portfolio = pd.concat(portfolio_returns).sort_index()
benchmark = pd.concat(sp500_returns).sort_index()

# Performance cumulée
portfolio_cum = (1 + portfolio).cumprod()
benchmark_cum = (1 + benchmark).cumprod()

# Calcul des indicateurs financiers
st.sidebar.header("Analyse des résultats")

# Rendement moyen et volatilité annualisés
mean_return = portfolio.mean() * 252
volatility = portfolio.std() * np.sqrt(252)

# Ratio de Sharpe
sharpe_ratio = mean_return / volatility

# VaR (95 %) historical method
var_95 = np.percentile(portfolio, 5)

# daily VaR (95 %) using parametric method
z_score = norm.ppf(0.05)
daily_parametric_VaR = (portfolio.mean() + z_score * portfolio.std())

# monthly VaR (95 %) using parametric method
monthly_parametric_VaR = (portfolio.mean() + z_score * (portfolio.std()*np.sqrt(20)))

# Drawdown maximal
drawdown = portfolio_cum / portfolio_cum.cummax() - 1
max_drawdown = drawdown.min()

# Bêta (covariance / variance benchmark)
covariance = pd.concat([portfolio,benchmark],axis=1).cov().dropna()
benchmark_variance = benchmark.var()
beta = covariance.iloc[0,1] / benchmark_variance[0]

# Affichage des résultats
st.subheader("Analyse des résultats")
st.write(f"Rendement annuel moyen : {mean_return:.2%}")
st.write(f"Volatilité annuelle : {volatility:.2%}")
st.write(f"Ratio de Sharpe : {sharpe_ratio:.2f}")
st.write(f"DAILY Value at Risk (95 %), historical : {var_95:.2%}")
st.write(f"DAILY Value at Risk (95 %), parametric: {daily_parametric_VaR:.2%}")
st.write(f"MONTHLY Value at Risk (95 %), parametric: {monthly_parametric_VaR:.2%}")
st.write(f"Drawdown maximal : {max_drawdown:.2%}")
st.write(f"Bêta : {beta:.2f}")

# Graphique performance cumulée
st.subheader("Performance du Portefeuille vs S&P500")
fig, ax = plt.subplots()
portfolio_cum.plot(ax=ax, label="Portefeuille", color="blue")
benchmark_cum.plot(ax=ax, label="S&P500", color="orange")
ax.legend()
ax.set_title("Performance cumulée")
ax.set_ylabel("Valeur cumulée")
ax.grid()
st.pyplot(fig)

# Graphique rendements mensuels
st.subheader("Rendements Portefeuille")
labels = [f"{start} - {end}" for start, end in date_ranges]
fig, ax = plt.subplots()
plt.bar(labels, portfolio_perf_per_period, color='blue')
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Rendements sur les différentes périodes")
ax.set_ylabel("Rendements")
ax.grid(axis="y")
plt.xticks(rotation=45, ha="right")
st.pyplot(fig)

# Graphique rendements mensuels
st.subheader("Rendements Bench")
labels = [f"{start} - {end}" for start, end in date_ranges]
fig, ax = plt.subplots()
plt.bar(labels, bench_perf_per_period, color='red')
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Rendements sur les différentes périodes")
ax.set_ylabel("Rendements")
ax.grid(axis="y")
plt.xticks(rotation=45, ha="right")
st.pyplot(fig)

# Histogramme rendements journalier
st.subheader("Histogramme des Rendements Journaliers du portefeuille")
fig, ax = plt.subplots()
portfolio.hist(ax=ax, bins=20, color="blue", alpha=0.7)
plt.axvline(var_95, color='red', linestyle='--', label=f'VaR (95%): {var_95:.2%}')
ax.set_title("Distribution des rendements journaliers")
ax.set_xlabel("Rendements journaliers")
ax.set_ylabel("Fréquence")
st.pyplot(fig)
