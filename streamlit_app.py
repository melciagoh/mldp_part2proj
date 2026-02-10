import joblib
import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# Load trained Random Forest model
best_rf = joblib.load("car_price_best_rf_model.pkl")

# -----------------------------
# Streamlit app title
st.set_page_config(page_title="Used Car Selling Price Prediction", layout="wide")
st.title("🚗 Used Car Selling Price Predictor")
st.markdown("""
Predict the estimated selling price of a used car based on its features.
""")

# -----------------------------
# Sidebar for user inputs
st.sidebar.header("Enter Car Details")

# Example options; adjust based on your dataset
makes = df_clean['make'].unique().tolist()
bodies = df_clean['body'].unique().tolist()
transmissions = df_clean['transmission'].unique().tolist()
states = df_clean['state'].unique().tolist()
colors = df_clean['color'].unique().tolist()
interiors = df_clean['interior'].unique().tolist()
conditions = sorted(df_clean['condition'].unique().tolist())

# User input widgets
year = st.sidebar.slider("Year of Manufacture", 1990, 2015, 2010)
make = st.sidebar.selectbox("Make", makes)
body = st.sidebar.selectbox("Body Type", bodies)
transmission = st.sidebar.selectbox("Transmission", transmissions)
state = st.sidebar.selectbox("State", states)
odometer = st.sidebar.number_input("Odometer (km)", min_value=0, max_value=500000, value=50000, step=1000)
condition = st.sidebar.slider("Condition (1-5)", 1.0, 5.0, 4.0, step=0.1)
color = st.sidebar.selectbox("Exterior Color", colors)
interior = st.sidebar.selectbox("Interior Color", interiors)

# -----------------------------
# Predict button
if st.button("Predict Selling Price"):
    
    # Create input dataframe
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
        # Include car_age and price_per_km
        'car_age': [2015 - year],
        'price_per_km': [0 if odometer == 0 else odometer / 1],  # placeholder; adjust formula if needed
    })
    
    # One-hot encoding for categorical columns
    cat_cols = ['make','body','transmission','state','color','interior']
    df_input = pd.get_dummies(df_input, columns=cat_cols, drop_first=True)
    
    # Align input with trained model columns
    df_input = df_input.reindex(columns=best_rf.feature_names_in_, fill_value=0)
    
    # Predict
    predicted_price = best_rf.predict(df_input)[0]
    
    st.success(f"💰 Predicted Selling Price: ${predicted_price:,.2f}")

# -----------------------------
# Optional: Display feature importance
if st.checkbox("Show Feature Importance"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
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
        background-color: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)
