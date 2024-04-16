import streamlit as st
import numpy as np
import pickle

import os

# Load the model
model = pickle.load(open('saved_models/.model1.pkl.swp', 'rb'))

st.title('House Pricing Prediction App')

# Defining the user inputs
bedrooms = st.number_input('Number of Bedrooms', min_value=0, max_value=10, value=5)
bathrooms = st.number_input('Number of Bathrooms', min_value=0, max_value=10, value=3)
toilets = st.number_input('Number of Toilets', min_value=0, max_value=10, value=3)
parking_space = st.number_input('Number of Parking Spaces', min_value=0, max_value=10, value=2)

# Prediction logic
if st.button('Predict Price'):
    features = np.array([[bedrooms, bathrooms, toilets, parking_space]])
    price = model.predict(features)
    st.write(f'Estimated Price: ${price[0]:,.2f}')
