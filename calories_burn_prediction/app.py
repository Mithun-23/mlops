import streamlit as st
import pandas as pd
import mlflow.pyfunc
import os

st.set_page_config(page_title="Calories Burn Predictor", page_icon="🔥", layout="centered")

# Load model
@st.cache_resource
def load_ml_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "api", "model")
    try:
        model = mlflow.pyfunc.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_ml_model()

st.title("Calories Burn Predictor 🔥")
st.write("Enter your workout details below to predict the calories burned.")

# Creating two columns for better UI layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (Years)", min_value=12, max_value=100, value=28, step=1)
    gender_input = st.selectbox("Gender", ["Female", "Male"])
    gender = gender_input.lower()
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=165.0, step=0.5)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=65.0, step=0.5)

with col2:
    duration = st.number_input("Duration (minutes)", min_value=1, max_value=300, value=45, step=1)
    heart_rate = st.number_input("Average Heart Rate (bpm)", min_value=40, max_value=220, value=145, step=1)
    body_temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=39.2, step=0.1)

if st.button("Predict Calories Burned", type="primary"):
    if model:
        # Create input DataFrame matching exactly the expected features
        input_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Height": height,
            "Weight": weight,
            "Duration": duration,
            "Heart_Rate": heart_rate,
            "Body_Temp": body_temp
        }])
        
        try:
            prediction = model.predict(input_data)
            predicted_cals = round(float(prediction[0]), 2)
            st.success(f"### Predicted Calories Burned: {predicted_cals} kcal")
            st.balloons()
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.error("Model is not loaded. Please check the deployment logs.")
