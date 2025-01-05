import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

logistic_model = joblib.load("logistic_regression_model.pkl")
knn_model = joblib.load("KNNCluster.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Flood Prediction from Climate using Logistic Regression")

st.write("Application for predicting flood with specific provided provided ")

st.subheader("Predict Flood")
with st.form(key="prediction_form"):
    Tn_input = st.text_input("Tn (Minimum Temperature)")
    Tx_input = st.text_input("Tx (Maximum Temperature)")
    Tavg_input = st.text_input("Tavg (Average Temperature)")
    RH_avg_input = st.text_input("RH_avg (Average Relative Humidity)")
    RR_input = st.text_input("RR (Rainfall)")
    ff_x_input = st.text_input("ff_x (Max Wind Speed)")
    ddd_x_input = st.text_input("ddd_x (Wind Direction at max speed)")
    ff_avg_input = st.text_input("ff_avg (Average Wind Speed)")
    
    submit_button = st.form_submit_button(label="Predict")
    
    if submit_button:
        try:
            input_data = pd.DataFrame({
                "Tn": [float(Tn_input)],
                "Tx": [float(Tx_input)],
                "Tavg": [float(Tavg_input)],
                "RH_avg": [float(RH_avg_input)],
                "RR": [float(RR_input)],
                "ff_x": [float(ff_x_input)],
                "ddd_x": [float(ddd_x_input)],
                "ff_avg": [float(ff_avg_input)]
            })
            
            # Apply the same scaling transformation
            scaled_input_data = scaler.transform(input_data)
            
            # Make predictions
            prediction = logistic_model.predict(scaled_input_data)
            result = "Flood" if prediction[0] == 1 else "No Flood"
            st.write(f"Prediction: {result}")
        except ValueError:
            st.write("Please enter valid numbers for all fields.")
