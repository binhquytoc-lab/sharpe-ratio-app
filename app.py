import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ====================================
# TITLE
# ====================================

st.title("Vietnam Stock Sharpe Ratio")

st.write("Portfolio Performance Analysis")

# ====================================
# STOCK INPUT
# ====================================

stock1 = st.text_input(
    "Stock 1",
    "VCB.VN"
)

stock2 = st.text_input(
    "Stock 2",
    "FPT.VN"
)

stock3 = st.text_input(
    "Stock 3",
    "HPG.VN"
)

# ====================================
# WEIGHTS
# ====================================

w1 = st.number_input(
    "Weight 1",
    min_value=0.0,
    max_value=1.0,
    value=0.4
)

w2 = st.number_input(
    "Weight 2",
    min_value=0.0,
    max_value=1.0,
    value=0.3
)

w3 = st.number_input(
    "Weight 3",
    min_value=0.0,
    max_value=1.0,
    value=0.3
)

# ====================================
# DATES
# ====================================

start_date = st.text_input(
    "Start Date",
    "2022-01-01"
)

end_date = st.text_input(
    "End Date",
    "2025-01-01"
)

# ====================================
# RISK FREE RATE
# ====================================

risk_free_rate = st.number_input(
    "Risk Free Rate",
    value=0.04
)

# ====================================
# BUTTON
# ====================================

if st.button("Calculate Sharpe Ratio"):

    weights = np.array([w1, w2, w3])

    # Check weights
    if round(weights.sum(), 5) != 1:

        st.error("Total weights must equal 1")

    else:

        try:

            stocks = [
                stock1,
                stock2,
                stock3
            ]

            # Download data
            data = yf.download(
                stocks,
                start=start_date,
                end=end_date
            )

            # Close prices
            close_prices = data['Close']

            # Daily returns
            returns = close_prices.pct_change().dropna()

            # Annual returns
            annual_returns = returns.mean() * 252

            # Covariance matrix
            cov_matrix = returns.cov() * 252

            # Portfolio return
            portfolio_return = np.dot(
                weights,
                annual_returns
            )

            # Portfolio volatility
            portfolio_volatility = np.sqrt(
                np.dot(
                    weights.T,
                    np.dot(cov_matrix, weights)
                )
            )

            # Sharpe Ratio
            sharpe_ratio = (
                portfolio_return - risk_free_rate
            ) / portfolio_volatility

            # ====================================
            # RESULTS
            # ====================================

            st.success("Calculation Successful!")

            st.subheader("Results")

            st.write(
                "Portfolio Return:",
                round(float(portfolio_return), 4)
            )

            st.write(
                "Portfolio Volatility:",
                round(float(portfolio_volatility), 4)
            )

            st.write(
                "Sharpe Ratio:",
                round(float(sharpe_ratio), 4)
            )

            # ====================================
            # CHART
            # ====================================

            st.subheader("Closing Price Chart")

            st.line_chart(close_prices)

        except Exception as e:

            st.error(str(e))
