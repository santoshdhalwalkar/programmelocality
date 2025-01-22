import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and preprocessing objects
model = joblib.load('xgb_locality_model.pkl')
encoders = joblib.load('encoders.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app title
st.title("Locality Prediction App")
st.write("Enter the details below to predict the Locality ID.")

# Input fields for user input
state_name = st.text_input("State Name", value="California")
state_code = st.text_input("State Code", value="CA")
county_id = st.number_input("County ID", value=1234)
zip_code = st.number_input("Zip Code", value=90210)
location_cluster = st.number_input("Location Cluster", value=2)

# Predict button
if st.button("Predict Locality ID"):
    try:
        # Prepare the input data as a DataFrame
        input_data = pd.DataFrame([{
            "StateName": state_name,
            "StateCode": state_code,
            "CountyID": county_id,
            "ZipCode": zip_code,
            "LocationCluster": location_cluster
        }])
        
        # Preprocess the input data
        for col, encoder in encoders.items():
            if col in input_data:
                input_data[col] = encoder.transform(input_data[col])
        
        # Scale numeric features
        numeric_features = ['CountyID', 'ZipCode', 'LocationCluster']
        input_data[numeric_features] = scaler.transform(input_data[numeric_features])
        
        # Make predictions
        prediction = model.predict(input_data)
        
        # Display the prediction
        st.success(f"The predicted Locality ID is: {int(prediction[0])}")
    except Exception as e:
        st.error(f"Error: {str(e)}")