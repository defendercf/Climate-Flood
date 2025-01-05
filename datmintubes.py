import streamlit as st
import pandas as pd
import joblib 

model = joblib.load("logistic_regression_model.pkl")

st.title('Flood Prediction from Climate using Logistic Regression')
st.write("Application for predicting flood with specific provided data.")

st.subheader('Predict Flood')
with st.form(key='prediction_form'):
    Tn_input = st.number_input('Tn (Minimum Temperature)', value=20)
    Tx_input = st.number_input('Tx (Maximum Temperature)', value=30)
    Tavg_input = st.number_input('Tavg (Average Temperature)', value=25)
    RH_avg_input = st.number_input('RH_avg (Average Relative Humidity)', value=80)
    RR_input = st.number_input('RR (Rainfall)', value=200)
    ff_x_input = st.number_input('ff_x (Max Wind Speed)', value=5)
    ddd_x_input = st.number_input('ddd_x (Wind Direction)', value=180)
    ff_avg_input = st.number_input('ff_avg (Average Wind Speed)', value=3)
    
    submit_button = st.form_submit_button(label='Predict')
    
    if submit_button:
        input_data = pd.DataFrame({
            'Tn': [Tn_input],
            'Tx': [Tx_input],
            'Tavg': [Tavg_input],
            'RH_avg': [RH_avg_input],
            'RR': [RR_input],
            'ff_x': [ff_x_input],
            'ddd_x': [ddd_x_input],
            'ff_avg': [ff_avg_input]
        })
        
        prediction = model.predict(input_data)
        result = 'Flood' if prediction[0] == 1 else 'No Flood'
        st.write(f'Prediction: {result}')
