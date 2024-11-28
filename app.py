import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import joblib
# Load the saved model
model = joblib.load(r"C:\Users\rolka\myenv\Car_price_predict.pkl")
# App Title
st.title('Car Price Prediction')

# Define the prediction function
def predict_price(year, km_driven, fuel, seller_type, transmission, owner):
    # Create a DataFrame for the input data
    data = pd.DataFrame({
        'year': [year],
        'km_driven': [km_driven],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner]
    })
    
    # Make predictions using the model
    sc=StandardScaler()

    data=sc.fit_transform(data)
    prediction = model.predict(data)
    return prediction[0]  # Return the first value in the prediction array

# User Inputs
year = st.number_input('Year of Manufacture', min_value=2000, max_value=2023, step=1)
km_driven = st.number_input('Kilometers Driven', min_value=0, step=500)
fuel = st.selectbox('Fuel Type: 0(CNG), 1(Disel), 2(Electric),3(LPG),4(Petrol)', [0,1,2,3,4])
seller_type = st.selectbox('Seller Type: Individual , Dealer', [0,1])
transmission = st.selectbox('Transmission Manual, Automatic', [0,1])
owner = st.selectbox('Owner Type: First Owner, Second Owner, Third Owner ,Fourth & Above Owner,Test Drive Car', [0,1,2,3,4])

# Predict Button
if st.button('Predict Price'):
    result = predict_price(year, km_driven, fuel, seller_type, transmission, owner)
    st.success(f"Predicted Selling Price: ₹ {result:,.2f}")
