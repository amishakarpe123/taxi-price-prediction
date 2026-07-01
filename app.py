import streamlit as st
import pickle
import numpy as np

# Set page layout and title
st.set_page_config(page_title="Trip Fare Predictor", page_icon="🚖", layout="centered")

# Custom CSS for a clean, modern look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        width: 100%;
        border-radius: 8px;
        font-size: 18px;
        height: 50px;
    }
    .result-box {
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #2e7d32;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚖 Trip Fare Prediction App")
st.write("Enter the trip details below to estimate the total fare.")

# Load the model securely
@st.cache_resource
def load_model():
    # Make sure 'model (9).pkl' is in the same directory as this script
    with open("model (9).pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ 'model (9).pkl' file not found! Please place it in the same directory as app.py.")
    st.stop()

# Form layout split into two columns
col1, col2 = st.columns(2)

with col1:
    trip_distance = st.number_input("Trip Distance (km)", min_value=0.0, value=5.0, step=0.1)
    
    # Mapping categorical values to numeric values if your model encoded them as such
    time_of_day = st.selectbox("Time of Day", options=["Morning", "Afternoon", "Evening", "Night"], index=0)
    time_mapping = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
    
    day_of_week = st.selectbox("Day of Week", options=["Weekday", "Weekend"], index=0)
    day_mapping = {"Weekday": 0, "Weekend": 1}
    
    passenger_count = st.number_input("Passenger Count", min_value=1, max_value=8, value=1)
    
    traffic_cond = st.selectbox("Traffic Conditions", options=["Low", "Medium", "High"], index=1)
    traffic_mapping = {"Low": 0, "Medium": 1, "High": 2}

with col2:
    weather = st.selectbox("Weather", options=["Clear", "Rainy", "Snowy", "Foggy"], index=0)
    weather_mapping = {"Clear": 0, "Rainy": 1, "Snowy": 2, "Foggy": 3}
    
    base_fare = st.number_input("Base Fare ($)", min_value=0.0, value=2.5, step=0.5)
    per_km_rate = st.number_input("Per Km Rate ($)", min_value=0.0, value=1.5, step=0.1)
    per_minute_rate = st.number_input("Per Minute Rate ($)", min_value=0.0, value=0.3, step=0.05)
    trip_duration = st.number_input("Trip Duration (Minutes)", min_value=0.0, value=15.0, step=1.0)

st.markdown("---")

# Predict button
if st.button("Calculate Estimated Fare"):
    # Convert encoded variables to their numeric equivalents
    features = np.array([[
        trip_distance,
        time_mapping[time_of_day],
        day_mapping[day_of_week],
        passenger_count,
        traffic_mapping[traffic_cond],
        weather_mapping[weather],
        base_fare,
        per_km_rate,
        per_minute_rate,
        trip_duration
    ]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    # Display the output cleanly
    st.markdown(
        f'<div class="result-box">💰 Estimated Fare: ${prediction:.2f}</div>', 
        unsafe_allow_html=True
    )
