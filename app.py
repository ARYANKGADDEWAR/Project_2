import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

model = joblib.load("yield_model.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")
state_encoder = joblib.load("state_encoder.pkl")
season_encoder = joblib.load("season_encoder.pkl")

st.title("🌾 AI Crop Yield Prediction System")

crop = st.selectbox("Crop", crop_encoder.classes_)
year = st.number_input("Year", 1997, 2035, 2026)
season = st.selectbox("Season", season_encoder.classes_)
state = st.selectbox("State", state_encoder.classes_)

area = st.number_input("Area")
rainfall = st.number_input("Annual Rainfall")
fertilizer = st.number_input("Fertilizer")
pesticide = st.number_input("Pesticide")

if st.button("Predict Yield"):

    crop_encoded = crop_encoder.transform([crop])[0]
    season_encoded = season_encoder.transform([season])[0]
    state_encoded = state_encoder.transform([state])[0]

    features = np.array([[crop_encoded,
                          year,
                          season_encoded,
                          state_encoded,
                          area,
                          rainfall,
                          fertilizer,
                          pesticide]])

    prediction = model.predict(features)

    st.success(
        f"Predicted Yield: {prediction[0]:.2f}"
    )
