import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load trained Random Forest model
best_rf = joblib.load("car_price_best_rf_model.pkl")

# -----------------------------
# Streamlit app title and config
st.set_page_config(page_title="Used Car Selling Price Prediction", layout="wide")
st.title("🚗 Used Car Selling Price Predictor")
st.markdown("Predict the estimated selling price of a used car based on its features.")

# -----------------------------
# Sidebar for user inputs
st.sidebar.header("Enter Car Details")

# Free-text inputs for categorical features with many options
make = st.sidebar.text_input("Make (e.g., BMW, Kia)")
body = st.sidebar.text_input("Body Type (e.g., SUV, Sedan)")
state = st.sidebar.text_input("State (e.g., CA, TX)")
color = st.sidebar.text_input("Exterior Color (e.g., white, black)")
interior = st.sidebar.text_input("Interior Color (e.g., black, beige)")

# Dropdown for limited-option categorical features
transmission = st.sidebar.selectbox("Transmission", ["automatic", "manual"])

# Sliders / numeric inputs
year = st.sidebar.slider("Year of Manufacture", 1990, 2026, 2015)
odometer = st.sidebar.number_input("Odometer (km)", min_value=0, max_value=1000000, value=50000, step=1000)
condition = st.sidebar.slider("Condition (1-5)", 1.0, 5.0, 4.0, step=0.1)

# -----------------------------
# Predict button
if st.button("Predict Selling Price"):

    # Feature engineering
    car_age = 2015 - year
    price_per_km = 0 if odometer == 0 else 0  # placeholder if needed later

    # Input dataframe
    df_input = pd.DataFrame({
        'year': [year],
        'make': [make],
        'body': [body],
        'transmission': [transmission],
        'state': [state],
        'odometer': [odometer],
        'condition': [condition],
        'color': [color],
        'interior': [interior],
        'car_age': [car_age],
        'price_per_km': [price_per_km],
    })

    # One-hot encode categorical features
    cat_cols = ['make','body','transmission','state','color','interior']
    df_input = pd.get_dummies(df_input, columns=cat_cols, drop_first=True)

    # Align input with trained model
    df_input = df_input.reindex(columns=best_rf.feature_names_in_, fill_value=0)

    # Predict
    predicted_price = best_rf.predict(df_input)[0]

    # Display result in a card-style box
    st.markdown(
        f"""
        <div style="padding: 20px; background-color: #e0f7fa; border-radius: 10px; text-align: center;">
            <h2>💰 Predicted Selling Price</h2>
            <h1>${predicted_price:,.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Feature importance checkbox
if st.checkbox("Show Top 10 Feature Importances"):
    feat_imp = pd.DataFrame({
        'Feature': best_rf.feature_names_in_,
        'Importance': best_rf.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(10)

    st.subheader("Top 10 Feature Importances")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis', ax=ax)
    st.pyplot(fig)

# -----------------------------
# Page styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f9fc;
        font-family: 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load trained Random Forest model
best_rf = joblib.load("car_price_best_rf_model.pkl")

# -----------------------------
# Streamlit app title and config
st.set_page_config(page_title="Used Car Selling Price Prediction", layout="wide")
st.title("🚗 Used Car Selling Price Predictor")
st.markdown("Predict the estimated selling price of a used car based on its features.")

# -----------------------------
# Sidebar for user inputs
st.sidebar.header("Enter Car Details")

# Free-text inputs for categorical features with many options
make = st.sidebar.text_input("Make (e.g., BMW, Kia)")
body = st.sidebar.text_input("Body Type (e.g., SUV, Sedan)")
state = st.sidebar.text_input("State (e.g., CA, TX)")
color = st.sidebar.text_input("Exterior Color (e.g., white, black)")
interior = st.sidebar.text_input("Interior Color (e.g., black, beige)")

# Dropdown for limited-option categorical features
transmission = st.sidebar.selectbox("Transmission", ["automatic", "manual"])

# Sliders / numeric inputs
year = st.sidebar.slider("Year of Manufacture", 1990, 2026, 2015)
odometer = st.sidebar.number_input("Odometer (km)", min_value=0, max_value=1000000, value=50000, step=1000)
condition = st.sidebar.slider("Condition (1-5)", 1.0, 5.0, 4.0, step=0.1)

# -----------------------------
# Predict button
if st.button("Predict Selling Price"):

    # Feature engineering
    car_age = 2015 - year
    price_per_km = 0 if odometer == 0 else 0  # placeholder if needed later

    # Input dataframe
    df_input = pd.DataFrame({
        'year': [year],
        'make': [make],
        'body': [body],
        'transmission': [transmission],
        'state': [state],
        'odometer': [odometer],
        'condition': [condition],
        'color': [color],
        'interior': [interior],
        'car_age': [car_age],
        'price_per_km': [price_per_km],
    })

    # One-hot encode categorical features
    cat_cols = ['make','body','transmission','state','color','interior']
    df_input = pd.get_dummies(df_input, columns=cat_cols, drop_first=True)

    # Align input with trained model
    df_input = df_input.reindex(columns=best_rf.feature_names_in_, fill_value=0)

    # Predict
    predicted_price = best_rf.predict(df_input)[0]

    # Display result in a card-style box
    st.markdown(
        f"""
        <div style="padding: 20px; background-color: #e0f7fa; border-radius: 10px; text-align: center;">
            <h2>💰 Predicted Selling Price</h2>
            <h1>${predicted_price:,.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Feature importance checkbox
if st.checkbox("Show Top 10 Feature Importances"):
    feat_imp = pd.DataFrame({
        'Feature': best_rf.feature_names_in_,
        'Importance': best_rf.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(10)

    st.subheader("Top 10 Feature Importances")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis', ax=ax)
    st.pyplot(fig)

# -----------------------------
# Page styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f9fc;
        font-family: 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)
