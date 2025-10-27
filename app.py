import streamlit as st
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# Show current directory (for debugging)
st.write("Current working directory:", os.getcwd())
st.write("Files in directory:", os.listdir())

# --- Title ---
st.title("📊 Customer Churn Prediction App")

# --- Safe Model & Object Loading ---
@st.cache_resource
def load_model_and_encoders():
    try:
        model = tf.keras.models.load_model("model.keras", compile=False)
    except Exception as e:
        st.error(f"❌ Error loading model.h5: {e}")
        model = None

    try:
        with open("label_encoder_gender.pkl", "rb") as f:
            label_encoder_gender = pickle.load(f)
        with open("onehot_encoder_geo.pkl", "rb") as f:
            onehot_encoder_geo = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error loading encoders/scaler: {e}")
        return None, None, None, None

    return model, label_encoder_gender, onehot_encoder_geo, scaler


model, label_encoder_gender, onehot_encoder_geo, scaler = load_model_and_encoders()

if model and label_encoder_gender and onehot_encoder_geo and scaler:
    # --- Sidebar Inputs ---
    st.sidebar.header("🧾 Input Customer Details")

    geography = st.sidebar.selectbox("Geography", onehot_encoder_geo.categories_[0])
    gender = st.sidebar.selectbox("Gender", label_encoder_gender.classes_)
    age = st.sidebar.slider("Age", 18, 92, 35)
    balance = st.sidebar.number_input("Balance", min_value=0.0, step=100.0)
    credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=900, step=1)
    estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, step=100.0)
    tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 3)
    num_of_products = st.sidebar.slider("Number of Products", 1, 4, 1)
    has_cr_card = st.sidebar.selectbox("Has Credit Card?", [0, 1])
    is_active_member = st.sidebar.selectbox("Is Active Member?", [0, 1])

    # --- Prepare Input Data ---
    try:
        input_data = pd.DataFrame({
            "CreditScore": [credit_score],
            "Gender": [label_encoder_gender.transform([gender])[0]],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary],
        })

        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
        )

        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
        input_data_scaled = scaler.transform(input_data)

        # --- Prediction ---
        prediction = model.predict(input_data_scaled)
        prediction_proba = float(prediction[0][0])

        # --- Output ---
        st.subheader("🔮 Prediction Result")
        st.write(f"**Churn Probability:** {prediction_proba:.2f}")

        if prediction_proba > 0.5:
            st.error("⚠️ The customer is **likely to churn.**")
        else:
            st.success("✅ The customer is **not likely to churn.**")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

else:
    st.stop()
