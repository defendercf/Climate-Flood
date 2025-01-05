import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

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
        # Create a DataFrame from the input
        input_data = pd.DataFrame({
            'Tn': [Tn_input],
            'Tx': [Tx_input],
            'Tavg': [Tavg_input],
            'RH_avg': [RH_avg_input],
            'RR': [RR_input],  # Exclude from scaling
            'ff_x': [ff_x_input],
            'ddd_x': [ddd_x_input],
            'ff_avg': [ff_avg_input]
        })
        
        # Separate the features to scale and the feature to leave unscaled
        features_to_scale = input_data.drop(columns=['RR'])
        unscaled_feature = input_data['RR'].values.reshape(-1, 1)
        
        # Apply the scaling transformation
        scaled_features = scaler.transform(features_to_scale)
        
        # Combine scaled and unscaled features
        combined_features = pd.DataFrame(
            np.hstack((scaled_features, unscaled_feature)),
            columns=features_to_scale.columns.tolist() + ['RR']
        )
        
        # Ensure the order of columns matches the model's expectation
        combined_features = combined_features[['Tn', 'Tx', 'Tavg', 'RH_avg', 'ff_x', 'ddd_x', 'ff_avg', 'RR']]
        
        # Make a prediction
        prediction = model.predict(combined_features)
        result = 'Flood' if prediction[0] == 1 else 'No Flood'
        st.write(f'Prediction: {result}')
