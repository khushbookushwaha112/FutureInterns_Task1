import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
import os

st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")
st.title("📊 Sales Forecasting Dashboard — Future Interns Task 1")

st.markdown("Upload your CSV (columns: `data`, `venda`) OR use default dataset provided below.")

uploaded = st.file_uploader("Upload sales CSV", type=['csv'])

def clean_df(df):
    df['data'] = pd.to_datetime(df['data'], errors='coerce')
    df['venda'] = pd.to_numeric(df['venda'], errors='coerce')
    df = df.dropna(subset=['data','venda']).sort_values('data')
    return df

# If user uploads CSV
if uploaded:
    df = pd.read_csv(uploaded)
    df = clean_df(df)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    ts = df.rename(columns={'data':'ds','venda':'y'})[['ds','y']]

    if st.button("Generate Forecast"):
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        m.fit(ts)

        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)

        merged = forecast[['ds','yhat']].merge(ts, on='ds', how='left')
        fig = px.line(merged, x='ds', y=['y','yhat'], title="Actual vs Predicted Sales")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast Sample")
        st.dataframe(forecast.tail())

else:
    # Load demo data from repo
    st.info("No file uploaded. Showing demo forecast from repository.")

    if os.path.exists("mock_kaggle.csv") and os.path.exists("forecast_output.csv"):
        df = pd.read_csv("mock_kaggle.csv")
        df = clean_df(df)

        st.subheader("Demo Dataset Preview")
        st.dataframe(df.head())

        forecast = pd.read_csv("forecast_output.csv")
        forecast['ds'] = pd.to_datetime(forecast['ds'])

        merged = forecast[['ds','yhat']].merge(
            df.rename(columns={'data':'ds','venda':'y'})[['ds','y']], on='ds', how='left'
        )

        fig = px.line(merged, x='ds', y=['y','yhat'], title="Demo Actual vs Forecast")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast (last rows)")
        st.dataframe(forecast.tail())

    else:
        st.error("Demo files not found. Please upload your CSV.")
