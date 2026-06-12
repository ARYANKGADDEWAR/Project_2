
import streamlit as st
import joblib
import numpy as np

# Set page configuration for better UI/UX
st.set_page_config(
    page_title="Enhanced Crop Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and encoders with caching
@st.cache_resource
def load_resources_in_app(): # Renamed to avoid conflict if executed in Colab
    model = joblib.load("yield_model.pkl")
    crop_encoder = joblib.load("crop_encoder.pkl")
    state_encoder = joblib.load("state_encoder.pkl")
    season_encoder = joblib.load("season_encoder.pkl")
    return model, crop_encoder, state_encoder, season_encoder

model, crop_encoder, state_encoder, season_encoder = load_resources_in_app()

# --- Sidebar for additional info ---
st.sidebar.title("About This App")
st.sidebar.info(
    "This application predicts crop yield based on various environmental and agricultural factors. "
    "It utilizes a Random Forest Regressor model for accurate predictions."
)
st.sidebar.markdown("---")
st.sidebar.subheader("Contact")
st.sidebar.write("For support, please contact: your.email@example.com") # Placeholder

# --- Main Title and Description ---
st.title("🌾 Enhanced AI Crop Yield Prediction System")
st.markdown("""
Welcome to the **Enhanced Crop Yield Prediction System**! This tool helps farmers and agricultural analysts
estimate crop yield by leveraging machine learning. Input key factors like crop type, year, season,
state, area, rainfall, fertilizer, and pesticide usage to get an accurate prediction.
Use the chatbot at the bottom for quick information!
""")

st.markdown("---")

# --- Input Section with improved layout ---
st.header("Prediction Inputs")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Crop & Location Details")
    crop = st.selectbox("Select Crop", crop_encoder.classes_, key="crop_input")
    year = st.number_input("Crop Year", min_value=1997, max_value=2035, value=2026, key="year_input")
    season = st.selectbox("Select Season", season_encoder.classes_, key="season_input")
    state = st.selectbox("Select State", state_encoder.classes_, key="state_input")

with col2:
    st.subheader("Environmental & Agricultural Factors")
    area = st.number_input("Area (in hectares)", min_value=0.0, value=10.0, key="area_input")
    rainfall = st.number_input("Annual Rainfall (in mm)", min_value=0.0, value=1000.0, key="rainfall_input")
    fertilizer = st.number_input("Fertilizer (in kg/hectare)", min_value=0.0, value=50.0, key="fertilizer_input")
    pesticide = st.number_input("Pesticide (in kg/hectare)", min_value=0.0, value=20.0, key="pesticide_input")

st.markdown("---")

# --- Prediction Button and Output ---
if st.button("Predict Yield", help="Click to get the crop yield prediction"):
    try:
        crop_encoded = crop_encoder.transform([crop])[0]
        season_encoded = season_encoder.transform([season])[0]
        state_encoded = state_encoder.transform([state])[0]

        features = np.array([[
            crop_encoded,
            year,
            season_encoded,
            state_encoded,
            area,
            rainfall,
            fertilizer,
            pesticide
        ]])

        prediction = model.predict(features)
        st.success(f"### Predicted Yield: **{prediction[0]:.2f} units** 📈")
        st.balloons()
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")

# --- Chatbot Integration ---
st.header("💬 Crop Yield Assistant Chatbot")
st.write("Ask me questions about crop yield prediction, the model, or general farming advice.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Type your question here..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate assistant response (simple rule-based)
    response = ""
    prompt_lower = prompt.lower()
    if any(keyword in prompt_lower for keyword in ["hello", "hi", "hey"]):
        response = "Hello! How can I help you with crop yield prediction today?"
    elif any(keyword in prompt_lower for keyword in ["yield", "predict", "forecast"]):
        response = "This app predicts crop yield based on factors like crop type, year, season, state, area, rainfall, fertilizer, and pesticide. Just input the values above and click 'Predict Yield'!"
    elif any(keyword in prompt_lower for keyword in ["model", "algorithm", "how it works"]):
        response = "The prediction is made using a Random Forest Regressor model, which is a powerful ensemble machine learning algorithm. It learns patterns from historical data to make predictions."
    elif any(keyword in prompt_lower for keyword in ["factors", "what affects"]):
        response = "Several factors significantly affect crop yield, including the specific crop type, the year, chosen season, geographical state, cultivated area, amount of rainfall, and the quantities of fertilizers and pesticides applied."
    elif any(keyword in prompt_lower for keyword in ["fertilizer", "pesticide"]):
        response = "Fertilizers supply essential nutrients to enrich soil and boost plant growth, while pesticides are used to protect crops from damage by pests and diseases. Both are critical for maximizing yield, but their application should be carefully managed."
    elif "thank" in prompt_lower:
        response = "You're welcome! Feel free to ask if you have more questions."
    else:
        response = "I'm sorry, I can only answer questions related to crop yield prediction at the moment. Please try rephrasing your question or asking about the app's functionality."

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
