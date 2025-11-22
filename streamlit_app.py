import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI Sales Forecasting — Task 1", layout="wide")
st.title("📈 AI-Based Sales Forecasting — Task 1")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

def clean_df(df):

    df = df.copy()
    
    # 1️⃣ Automatically detect date column
    date_cols = ["date", "data", "Date", "Data", "ds"]

    found_date = None
    for c in df.columns:
        if c in date_cols:
            found_date = c
            break

    if not found_date:
        st.error("❌ Date column not found! CSV must contain a date column (date/data/ds).")
        return None

    # Rename to standard name
    df.rename(columns={found_date: "date"}, inplace=True)

    # Convert to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 2️⃣ Detect sales/target column (y, sales, value, etc.)
    target_cols = ["y", "sales", "Sale", "value"]

    found_target = None
    for c in df.columns:
        if c in target_cols:
            found_target = c
            break

    # If Prophet output is uploaded (yhat exists)
    if "yhat" in df.columns:
        df["prediction"] = df["yhat"]
        return df

    if not found_target:
        st.error("❌ Sales column not found! CSV must contain target column such as y/sales.")
        return None

    df.rename(columns={found_target: "y"}, inplace=True)

    return df


if uploaded:

    df = pd.read_csv(uploaded)

    cleaned = clean_df(df)

    if cleaned is not None:
        st.success("✅ File processed successfully!")
        st.subheader("Cleaned Data Preview")
        st.dataframe(cleaned.head())

else:
    st.info("Upload your CSV to begin forecasting.")
