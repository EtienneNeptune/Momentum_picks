import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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
def fetch_data(tickers, start, end, adj_close=True):
    # Download the data
    data = yf.download(tickers, start=start, end=end)

    # Select the appropriate price column
    if adj_close:
        data = data['Adj Close']
    else:
        data = data['Close']

    data = data.dropna()

    # Check and convert to USD if needed
    for ticker in tickers:
        stock_info = yf.Ticker(ticker).info
        currency = stock_info.get('currency', 'USD')  # Default to 'USD' if currency not found

        if currency != 'USD':
            # Fetch the exchange rate data (FX: Currency/USD)
            fx_ticker = f"{currency}=X"

            st.write(ticker, fx_ticker)

            fx_data = yf.download(fx_ticker, start=start, end=end)['Adj Close']

            # Interpolate missing FX data to align with stock data
            fx_data = fx_data.reindex(data.index).ffill()

            # Convert the stock prices to USD
            data[ticker] = data[ticker] / fx_data.iloc[:, 0]

    return data

portfolio_returns = []
portfolio_perf_per_period = []
bench_perf_per_period = []
sp500_returns = []

for i, (start, end) in enumerate(date_ranges):
    if not tickers[i]:
        continue
    # Récupérer les données pour les tickers et calculer les rendements pondérés
    returns = fetch_data(tickers[i], start, end).pct_change().dropna()
    #weighted_returns = (returns * weights[i]).sum(axis=1) if daily rebalanced
    cumulative_values = (1 + returns).cumprod()
    st.write((1-cumulative_values)*100)

    portfolio_value = cumulative_values.dot(weights[i])
    weighted_returns = portfolio_value.pct_change().fillna((portfolio_value.iloc[0]-1)/1)
    portfolio_returns.append(weighted_returns)

    portfolio_perf = np.prod(1+weighted_returns)-1
    portfolio_perf_per_period.append(portfolio_perf)

    # Récupérer les rendements du benchmark
    sp500_data = fetch_data("^SP500TR", start, end).pct_change().dropna()
    sp500_returns.append(sp500_data)
    sp500_perf = np.prod(1 + sp500_data) - 1
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

bench_mean_return = benchmark.mean() * 252
bench_volatility = benchmark.std() * np.sqrt(252)

# Ratio de Sharpe
sharpe_ratio = mean_return / volatility
bench_sharpe_ratio = bench_mean_return / bench_volatility

# VaR (95 %) historical method
var_95 = np.percentile(portfolio, 5)
bench_var_95 = np.percentile(benchmark, 5)

# daily VaR (95 %) using parametric method
z_score = norm.ppf(0.05)
daily_parametric_VaR = (portfolio.mean() + z_score * portfolio.std())
bench_daily_parametric_VaR = (benchmark.mean() + z_score * benchmark.std())

# monthly VaR (95 %) using parametric method
monthly_parametric_VaR = (portfolio.mean() + z_score * (portfolio.std()*np.sqrt(20)))
bench_monthly_parametric_VaR = (benchmark.mean() + z_score * (benchmark.std()*np.sqrt(20)))

# Drawdown maximal
drawdown = portfolio_cum / portfolio_cum.cummax() - 1
max_drawdown = drawdown.min()

bench_drawdown = benchmark_cum / benchmark_cum.cummax() - 1
bench_max_drawdown = bench_drawdown.min()

# Bêta (covariance / variance benchmark)
covariance = pd.concat([portfolio,benchmark],axis=1).cov().dropna()
benchmark_variance = benchmark.var()
beta = covariance.iloc[0,1] / benchmark_variance[0]

# Affichage des résultats
st.subheader("Analyse des résultats du portefeuille")
st.write(f"Rendement annuel moyen : {mean_return:.2%}")
st.write(f"Volatilité annuelle : {volatility:.2%}")
st.write(f"Ratio de Sharpe : {sharpe_ratio:.2f}")
st.write(f"DAILY Value at Risk (95 %), historical : {var_95:.2%}")
st.write(f"DAILY Value at Risk (95 %), parametric: {daily_parametric_VaR:.2%}")
st.write(f"MONTHLY Value at Risk (95 %), parametric: {monthly_parametric_VaR:.2%}")
st.write(f"Drawdown maximal : {max_drawdown:.2%}")
st.write(f"Bêta : {beta:.2f}")

st.subheader("Analyse des résultats du benchmark")
st.write(f"Rendement annuel moyen : {bench_mean_return[0]:.2%}")
st.write(f"Volatilité annuelle : {bench_volatility[0]:.2%}")
st.write(f"Ratio de Sharpe : {bench_sharpe_ratio[0]:.2f}")
st.write(f"DAILY Value at Risk (95 %), historical : {bench_var_95:.2%}")
st.write(f"DAILY Value at Risk (95 %), parametric: {bench_daily_parametric_VaR[0]:.2%}")
st.write(f"MONTHLY Value at Risk (95 %), parametric: {bench_monthly_parametric_VaR[0]:.2%}")
st.write(f"Drawdown maximal : {bench_max_drawdown[0]:.2%}")
st.write(f"Bêta : "+str(1))

# Graphique performance cumulée
st.subheader("Performance du Portefeuille vs S&P500")
fig = go.Figure()

aligned_bench = benchmark_cum.reindex(portfolio_cum.index).iloc[:, 0]
# Tracer les courbes du portefeuille et du benchmark
fig.add_trace(go.Scatter(x=portfolio_cum.index, y=portfolio_cum.values, mode='lines', name="Portefeuille", line=dict(color='blue')))
fig.add_trace(go.Scatter(x=aligned_bench.index, y=aligned_bench.values, mode='lines', name="S&P500", line=dict(color='red')))

# Ajouter titre et labels
fig.update_layout(
    title="Performance cumulée",
    xaxis_title="Date",
    yaxis_title="Valeur cumulée",
    template="plotly_dark"
)

# Affichage dans Streamlit
st.plotly_chart(fig)

# Graphique rendements mensuels (Portefeuille)
st.subheader("Rendements trimestriels du portefeuille")
fig = go.Figure()

# Tracer les rendements mensuels
labels = [f"{start} - {end}" for start, end in date_ranges]
fig.add_trace(go.Bar(
    x=labels, y=portfolio_perf_per_period, marker=dict(color='blue'),
    hovertemplate="Période: %{x}<br>Rendement: %{y:.2%}<extra></extra>"
))

# Ajouter ligne pour zéro
fig.add_trace(go.Scatter(x=labels, y=[0]*len(labels), mode='lines', line=dict(color='black', dash='dash'), name="Zero"))

# Ajouter titre et labels
fig.update_layout(
    title="Rendements sur les différentes périodes",
    xaxis_title="Périodes",
    yaxis_title="Rendements",
    template="plotly_dark",
    xaxis_tickangle=-45
)

# Affichage dans Streamlit
st.plotly_chart(fig)

# Graphique rendements mensuels (Benchmark)
st.subheader("Rendements trimestriels du benchmark")
fig = go.Figure()

# Tracer les rendements mensuels
fig.add_trace(go.Bar(
    x=labels, y=bench_perf_per_period, marker=dict(color='red'),
    hovertemplate="Période: %{x}<br>Rendement: %{y:.2%}<extra></extra>"
))

# Ajouter ligne pour zéro
fig.add_trace(go.Scatter(x=labels, y=[0]*len(labels), mode='lines', line=dict(color='black', dash='dash'), name="Zero"))

# Ajouter titre et labels
fig.update_layout(
    title="Rendements sur les différentes périodes",
    xaxis_title="Périodes",
    yaxis_title="Rendements",
    template="plotly_dark",
    xaxis_tickangle=-45
)

# Affichage dans Streamlit
st.plotly_chart(fig)

# Histogramme des rendements journaliers (Portefeuille)
st.subheader("Histogramme des Rendements Journaliers du portefeuille")

# Tracer l'histogramme
fig = px.histogram(portfolio, nbins=20, color_discrete_sequence=['blue'])

# Ajouter la VaR (Value at Risk)
fig.add_vline(x=float(var_95), line=dict(color='red', dash='dash'), annotation_text=f"VaR (95%): {var_95:.2%}", annotation_position="top right")

# Ajouter titre et labels
fig.update_layout(
    title="Distribution des rendements journaliers",
    xaxis_title="Rendements journaliers",
    yaxis_title="Fréquence",
    template="plotly_dark"
)

# Affichage dans Streamlit
st.plotly_chart(fig)
