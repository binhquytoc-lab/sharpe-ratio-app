import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ====================================
# TITLE
# ====================================

st.title("Tính chỉ số Sharpe Ratio của danh mục đầu tư_TS. VŨ ĐỨC BÌNH")

st.write("Điền các mã cổ phiếu và tỷ trọng đầu tư trong DMĐT chứng khoán")

# ====================================
# STOCK INPUT
# ====================================

stock1 = st.text_input("Stock 1", "VCB.VN")
stock2 = st.text_input("Stock 2", "FPT.VN")
stock3 = st.text_input("Stock 3", "HPG.VN")

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
# DATE INPUT
# ====================================

start_date = st.text_input(
    "Start Date",
    "2022-01-01"
)

end_date = st.text_input(
    "End Date",
    "2026-05-10"
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

    try:

        # Stocks
        stocks = [
            stock1.strip(),
            stock2.strip(),
            stock3.strip()
        ]

        # Weights
        weights = np.array([
            w1,
            w2,
            w3
        ])

        # ====================================
        # CHECK WEIGHTS
        # ====================================

        if round(weights.sum(), 5) != 1:

            st.error("Total weights must equal 1")

        else:

            # ====================================
            # DOWNLOAD DATA
            # ====================================

            data = yf.download(
                stocks,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False
            )

            # ====================================
            # GET CLOSE PRICE
            # ====================================

            if 'Close' in data.columns:

                close_prices = data['Close']

            else:

                close_prices = data

            # ====================================
            # HANDLE SINGLE STOCK
            # ====================================

            if isinstance(close_prices, pd.Series):

                close_prices = close_prices.to_frame()

            # ====================================
            # REMOVE MISSING VALUES
            # ====================================

            close_prices = close_prices.dropna()

            # ====================================
            # CHECK EMPTY DATA
            # ====================================

            if close_prices.empty:

                st.error("No data found. Check stock symbols.")

            else:

                # ====================================
                # DAILY RETURNS
                # ====================================

                returns = close_prices.pct_change().dropna()

                # ====================================
                # CHECK RETURNS
                # ====================================

                if returns.empty:

                    st.error("Return data is empty.")

                else:

                    # ====================================
                    # ANNUAL RETURN
                    # ====================================

                    annual_returns = returns.mean() * 252

                    # ====================================
                    # COVARIANCE MATRIX
                    # ====================================

                    cov_matrix = returns.cov() * 252

                    # ====================================
                    # PORTFOLIO RETURN
                    # ====================================

                    portfolio_return = np.dot(
                        weights,
                        annual_returns
                    )

                    # ====================================
                    # VOLATILITY
                    # ====================================

                    portfolio_volatility = np.sqrt(
                        np.dot(
                            weights.T,
                            np.dot(cov_matrix, weights)
                        )
                    )

                    # ====================================
                    # CHECK ZERO VOLATILITY
                    # ====================================

                    if portfolio_volatility == 0:

                        st.error("Portfolio volatility is zero.")

                    else:

                        # ====================================
                        # SHARPE RATIO
                        # ====================================

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

                        st.subheader("HÌNH BIỂU DIỄN GIÁ ĐÓNG CỬA CỦA CÁC CỔ PHIẾU TRONG DANH MỤC ĐẦU TƯ")

                        st.line_chart(close_prices)

    except Exception as e:

        st.error(f"Error: {str(e)}")
