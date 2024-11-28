import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


model = joblib.load(r'C:\Users\rolka\myenv\Car_price_predict.pkl')

def predict_price(year, km_driven, fuel, seller_type, transmission, owner):
    data = pd.DataFrame({
        'year': [year],
        'km_driven': [km_driven],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner]
    })

    sc=StandardScaler()

    data=sc.fit_transform(data)

    prediction = model.predict(data)
    return prediction[0]



# Streamlit web app layout
st.title('Car Price Prediction App')
st.write("Predict the selling price of a used car based on its characteristics")

# User input fields
year = st.number_input('Year of Manufacture', min_value=1990, max_value=2024, value=2015)
km_driven = st.number_input('Kilometers Driven', min_value=0, max_value=1000000, value=50000)
fuel = st.selectbox('Fuel Type CNG, Diesel,Electric,LPG,Petrol', [0,1,2,3,4])
seller_type = st.selectbox('Seller Type Dealer, Individual ,Trust Dealer', [0,1,2])
transmission = st.selectbox('Transmission Automatic,Manual', [0,1])
owner = st.selectbox('Owner 1st Owner, 4th & above owner, 2nd owner, Test Drive, 3rd Owner', [0,1,2,3,4])

# Predict button
if st.button('Predict Price'):
    result = predict_price(year, km_driven, fuel, seller_type, transmission, owner)
    st.success(f"The predicted selling price of the car is: ₹ {np.round(result, 2)}")