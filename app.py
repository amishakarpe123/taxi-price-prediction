import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
from streamlit_lottie import st_lottie

# 1. Page Configuration
st.set_page_config(
    page_title="Taxi/Ride Fare Predictor",
    page_icon="🚖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Function to Load Lottie Animations
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# Load a clean ride-sharing/taxi animation
lottie_taxi = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_vyb8gqvl.json")

# 3. Load the Pickled Model Safely
@st.cache_resource
def load_model():
    try:
        with open("model (9).pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("⚠️ 'model (9).pkl' not found. Please ensure the file is in the same directory as app.py.")
        return None

model = load_model()

# 4. App Header & Animation
st.title("🚖 Fare Estimation Engine")
st.markdown("Enter the trip details below to predict the estimated fare in real-time.")

if lottie_taxi:
    st_lottie(lottie_taxi, height=200, key="taxi_animation")
else:
    st.write("---")

# 5. User Input Form
st.subheader("📋 Trip Specifications")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        trip_distance = st.number_input("Trip Distance (km)", min_value=0.0, max_value=500.0, value=5.0, step=0.1)
        # Note: If your model expects encoded numerical values for categorical features, 
        # ensure these selections map exactly to how you encoded them during training.
        time_of_day = st.selectbox("Time of Day", options=[0, 1, 2, 3], format_func=lambda x: ["Morning", "Afternoon", "Evening", "Night"][x])
        day_of_week = st.selectbox("Day of Week", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x])
        passenger_count = st.number_input("Passenger Count", min_value=1, max_value=8, value=1, step=1)
        traffic_conditions = st.selectbox("Traffic Conditions", options=[0, 1, 2], format_func=lambda x: ["Low", "Medium", "High"][x])

    with col2:
        weather = st.selectbox("Weather Conditions", options=[0, 1, 2], format_func=lambda x: ["Clear", "Rainy", "Snowy"][x])
        base_fare = st.number_input("Base Fare ($)", min_value=0.0, value=2.50, step=0.50)
        per_km_rate = st.number_input("Per Km Rate ($)", min_value=0.0, value=1.20, step=0.10)
        per_minute_rate = st.number_input("Per Minute Rate ($)", min_value=0.0, value=0.35, step=0.05)
        trip_duration = st.number_input("Trip Duration (Minutes)", min_value=0.0, max_value=1440.0, value=15.0, step=1.0)

    # Submit Button
    submit_btn = st.form_submit_button("Calculate Estimated Fare")

# 6. Prediction Logic
if submit_btn:
    if model is not None:
        # Construct features array in the exact order your model expects:
        # [Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions, Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes]
        features = np.array([[
            trip_distance,
            time_of_day,
            day_of_week,
            passenger_count,
            traffic_conditions,
            weather,
            base_fare,
            per_km_rate,
            per_minute_rate,
            trip_duration
        ]], dtype=float)
        
        try:
            # Generate prediction
            prediction = model.predict(features)[0]
            
            # Display results beautifully
            st.success("🎉 Estimation Complete!")
            st.metric(label="Predicted Total Fare", value=f"${prediction:,.2f}")
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("💡 Double-check that your model does not require text strings or explicit one-hot encoded arrays.")
