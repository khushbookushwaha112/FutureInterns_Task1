import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")
st.title("📊 Sales Forecasting Dashboard — Future Interns Task 1")

st.write("This app shows demo forecast generated using Prophet in Google Colab.")

# Demo dataset
df = pd.read_csv("mock_kaggle.csv")
df['data'] = pd.to_datetime(df['data'])

# Load precomputed forecast
forecast = pd.read_csv("forecast_output.csv")
forecast['ds'] = pd.to_datetime(forecast['ds'])

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Forecast Preview")
st.dataframe(forecast.tail())

# Merge for visualization
merged = forecast[['ds','yhat']].merge(
    df.rename(columns={'data':'ds','venda':'y'})[['ds','y']],
    on='ds', how='left'
)

fig = px.line(
    merged,
    x='ds',
    y=['y','yhat'],
    title="Actual vs Predicted Sales",
    labels={'ds': 'Date', 'value': 'Sales'}
)
st.plotly_chart(fig, use_container_width=True)

        st.dataframe(forecast.tail())

    else:
        st.error("Demo files not found. Please upload your CSV.")
