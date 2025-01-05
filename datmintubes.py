import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logistic_model = joblib.load("logistic_regression_model.pkl")
knn_model = joblib.load("KKNCluster.pkl")
scaler = joblib.load("scaler.pkl")

st.title('Flood Prediction from Climate using Logistic Regression')
st.write("Application for predicting flood with specific provided data.")

st.subheader('Predict Flood')
with st.form(key='input_form'):
    Tn_input = st.text_input('Tn (Minimum Temperature)')
    Tx_input = st.text_input('Tx (Maximum Temperature)')
    Tavg_input = st.text_input('Tavg (Average Temperature)')
    RH_avg_input = st.text_input('RH_avg (Average Relative Humidity)')
    RR_input = st.text_input('RR (Rainfall)')
    ff_x_input = st.text_input('ff_x (Max Wind Speed)')
    ddd_x_input = st.text_input('ddd_x (Wind Direction at max speed)')
    ff_avg_input = st.text_input('ff_avg (Average Wind Speed)')
    
    submit_button = st.form_submit_button(label='Submit')
    
    if submit_button:
        try:
            input_data = pd.DataFrame({
                'Tn': [float(Tn_input)],
                'Tx': [float(Tx_input)],
                'Tavg': [float(Tavg_input)],
                'RH_avg': [float(RH_avg_input)],
                'RR': [float(RR_input)],
                'ff_x': [float(ff_x_input)],
                'ddd_x': [float(ddd_x_input)],
                'ff_avg': [float(ff_avg_input)]
            })
            
            scaled_input_data = scaler.transform(input_data)
            
            prediction = logistic_model.predict(scaled_input_data)
            flood_result = 'Flood' if prediction[0] == 1 else 'No Flood'
            st.write(f'Flood Prediction: {flood_result}')
            
            cluster = knn_model.predict(scaled_input_data)
            st.write(f'KNN Cluster: {cluster[0]}')

            pca = PCA(n_components=2)
            X_train_scaled = scaler.transform(X)
            X_pca = pca.fit_transform(X_train_scaled)
            input_pca = pca.transform(scaled_input_data)

            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', s=100, alpha=0.6)
            plt.scatter(input_pca[0, 0], input_pca[0, 1], color='red', s=200, label='Input Data', edgecolor='k')
            plt.title('KNN Clustering Visualization')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.legend()
            st.pyplot(plt)
            
        except ValueError:
            st.error("Please enter valid numbers for all fields.")
