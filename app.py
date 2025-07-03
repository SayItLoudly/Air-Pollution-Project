# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set Streamlit config
st.set_page_config(page_title="Air Quality Visualizer", layout="wide")

# Load the cleaned dataset
df = pd.read_csv("cleaned_data.csv", parse_dates=['sampling_date'])
df = df.dropna(subset=['sampling_date'])
df['sampling_date'] = pd.to_datetime(df['sampling_date'], errors='coerce')
df.set_index('sampling_date', inplace=True)

# Sidebar UI
st.sidebar.title("🧪 Air Pollution Dashboard")
city = st.sidebar.selectbox("Select City", sorted(df['location'].dropna().unique()))
pollutant = st.sidebar.selectbox("Select Pollutant", ['so2', 'no2', 'rspm', 'spm', 'all'])
min_date = df.index.min().date()
max_date = df.index.max().date()
date_range = st.sidebar.slider("Select Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date), format="YYYY-MM")

# Filter data
df_city = df[df['location'] == city]
df_city = df_city.sort_index()
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])
df_city = df_city[(df_city.index >= start_date) & (df_city.index <= end_date)]

# Resample
df_monthly = df_city.select_dtypes(include=['number']).resample('M').mean().reset_index()
df_monthly['year'] = df_monthly['sampling_date'].dt.year
df_monthly['month'] = df_monthly['sampling_date'].dt.month

# Plotting
st.title("📊 Air Quality Visualizer")
if pollutant != 'all':
    st.subheader(f"📈 Trend of {pollutant.upper()} in {city}")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df_monthly, x='sampling_date', y=pollutant, ax=ax, marker='o')
    ax.set_title(f"{pollutant.upper()} Trend in {city}")
    ax.set_ylabel(f"{pollutant.upper()} Level")
    ax.set_xlabel("Date")
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.subheader(f"📉 All Pollutants in {city}")
    fig, ax = plt.subplots(figsize=(12, 6))
    for p in ['so2', 'no2', 'rspm', 'spm']:
        sns.lineplot(data=df_monthly, x='sampling_date', y=p, label=p, ax=ax, marker='o')
    ax.set_title(f"All Pollutants in {city}")
    ax.set_ylabel("Pollution Level")
    ax.set_xlabel("Date")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Summary stats
if st.checkbox("📋 Show Summary Statistics"):
    st.write(df_city[['so2', 'no2', 'rspm', 'spm']].describe())

# ----------------- PREDICTION PANEL -------------------
st.header("🤖 Predict Future Pollutant Level")
predict_city = st.selectbox("Select City for Prediction", sorted(df['location'].dropna().unique()), key="predict_city")
predict_pollutant = st.selectbox("Select Pollutant to Predict", ['so2', 'no2', 'rspm', 'spm'], key="predict_pollutant")

# Filter data
data = df[df['location'] == predict_city].copy()
data['year'] = data.index.year
data['month'] = data.index.month

# Drop missing target
data = data.dropna(subset=[predict_pollutant])

# Use other pollutants as features + time
features = ['year', 'month'] + [col for col in ['so2', 'no2', 'rspm', 'spm'] if col != predict_pollutant]
data = data.dropna(subset=features)  # Ensure all feature columns have no NaN

if data.empty:
    st.warning("Not enough data available to train prediction model. Please try another city or pollutant.")
else:
    X = data[features]
    y = data[predict_pollutant]

    if len(X) < 2:
        st.warning("Not enough data points to train the model. Please change your selection.")
    else:
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Predict & evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.markdown(f"**Model Accuracy**: R² = `{r2:.2f}`, MSE = `{mse:.2f}`")

        # Forecast input
        future_year = st.slider("Select Year to Predict", min_value=int(df.index.min().year), max_value=2030, value=2025)
        future_month = st.slider("Select Month to Predict", 1, 12, 1)

        # Create dummy input from most recent values
        recent = data.sort_index().iloc[-1]
        future_input = pd.DataFrame({
            'year': [future_year],
            'month': [future_month],
            features[2]: [recent[features[2]]],
            features[3]: [recent[features[3]]],
            features[4]: [recent[features[4]]],
        })

        future_prediction = model.predict(future_input)[0]

        st.success(f"Predicted {predict_pollutant.upper()} Level in {predict_city} for {future_month}/{future_year} is **{future_prediction:.2f} µg/m³**")
