import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
from datetime import datetime
import base64

# Load dataset
data = pd.read_csv("car.csv")
data = data.dropna()
categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission']
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)
X = data.drop(['Selling_Price', 'Car_Name'], axis=1)
y = data['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
feature_columns = X.columns

# Streamlit page
st.set_page_config(page_title="Car Selling Price Predictor", page_icon="🚗", layout="wide")

# Add background image from local or URL
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1571607380987-80d2cf44f3be?auto=format&fit=crop&w=1470&q=80");
             background-size: cover;
             background-position: center;
             background-attachment: fixed;
         }}
         .card {{
             background-color: rgba(255, 255, 255, 0.85);
             padding: 20px;
             border-radius: 15px;
             box-shadow: 2px 2px 12px rgba(0,0,0,0.2);
             margin-bottom: 20px;
         }}
         .stButton>button {{
             background-color: #ff4b4b;
             color: white;
             font-size: 16px;
             height: 50px;
             border-radius: 10px;
             border: none;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

st.title("🚗 Car Selling Price Prediction")
st.markdown("Enter your car details to predict its selling price in **lakhs**.")
st.info("Car Year will be used to calculate Years of Service automatically.")

# Input Card
st.markdown('<div class="card">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    fuel = st.selectbox("Fuel Type", ['Petrol','Diesel','CNG'])
    kms_driven = st.number_input("Kilometers Driven", min_value=0)
with col2:
    seller_type = st.selectbox("Seller Type", ['Dealer','Individual'])
    owner = st.number_input("Number of Previous Owners", min_value=0)
with col3:
    transmission = st.selectbox("Transmission", ['Manual','Automatic'])
    car_year = st.number_input("Car Year (e.g., 2015)", min_value=1990, max_value=datetime.now().year)
present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1)
st.markdown('</div>', unsafe_allow_html=True)

years = datetime.now().year - car_year

# Prediction Card
st.markdown('<div class="card">', unsafe_allow_html=True)
if st.button("Predict Selling Price"):
    input_dict = {
        'Year': [years],
        'Present_Price': [present_price],
        'Kms_Driven': [kms_driven],
        'Owner': [owner],
        'Fuel_Type_Diesel': [1 if fuel=='Diesel' else 0],
        'Fuel_Type_CNG': [1 if fuel=='CNG' else 0],
        'Seller_Type_Individual': [1 if seller_type=='Individual' else 0],
        'Transmission_Manual': [1 if transmission=='Manual' else 0]
    }
    input_df = pd.DataFrame(input_dict)
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]
    input_scaled = scaler.transform(input_df)
    price = rf_model.predict(input_scaled)
    st.metric(label="💰 Predicted Selling Price", value=f"₹{price[0]:.2f} lakhs")
    st.balloons()
st.markdown('</div>', unsafe_allow_html=True)
