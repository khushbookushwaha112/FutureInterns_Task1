import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
import os

st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")
st.title("📊 AI-Based Sales Forecasting — Task 1")

st.markdown("Upload CSV with sales data. System automatically detects date & sales columns.")

uploaded = st.file_uploader("Upload CSV", type=['csv'])

DEMO_CSV = "mock_kaggle.csv"
DEMO_FORECAST = "forecast_output.csv"


# ⭐ SAFE CLEAN FUNCTION (no error)
def clean_df(df):
    df = df.copy()

    # Auto-detect date column
    possible_date_cols = ["data", "date", "Date", "DATE"]
    date_col = None
    for col in possible_date_cols:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        st.error("❌ Date column not found! Your CSV must contain a date column such as 'date' or 'data'.")
        st.stop()

    # Auto-detect sales column
    possible_sales = ["venda", "sales", "Sales", "Sale", "amount"]
    sales_col = None
    for col in possible_sales:
        if col in df.columns:
            sales_col = col
            break

    if sales_col is None:
        st.error("❌ Sales column not found! Your CSV must contain a numeric sales column like 'venda' or 'sales'.")
        st.stop()

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')
    df = df.dropna(subset=[date_col, sales_col])

    df = df.rename(columns={date_col: "ds", sales_col: "y"})
    df = df[["ds", "y"]].sort_values("ds")

    return df


# ⭐ FORECAST FUNCTION (NO ERROR)
def make_forecast(ts):
    m = Prophet()
    m.fit(ts)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    return forecast


# ⭐ MAIN APP
if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    df_clean = clean_df(df)
    forecast = make_forecast(df_clean)

    st.subheader("Sales Forecast (Next 30 Days)")

    fig = px.line(forecast, x='ds', y='yhat', title="Predicted Sales")
    st.plotly_chart(fig)

else:
    st.info("No CSV uploaded. Showing demo dataset prediction.")

    if os.path.exists(DEMO_CSV) and os.path.exists(DEMO_FORECAST):
        df_demo = pd.read_csv(DEMO_CSV)
        forecast = pd.read_csv(DEMO_FORECAST)

        st.subheader("Demo Data")
        st.dataframe(df_demo.head())

        fig = px.line(forecast, x='ds', y='yhat', title="Predicted Sales (Demo)")
        st.plotly_chart(fig)
    else:
        st.warning("Demo files not found. Upload your CSV.")
