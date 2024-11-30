import os
import time

import requests
from dotenv import load_dotenv

import streamlit as st

load_dotenv(".env.local")

# API URL
base_url = f"http://{os.getenv('FASTAPI_HOST')}:{os.getenv('FASTAPI_PORT')}"
health_url = f"{base_url}/ping"
predict_url = f"{base_url}/predict"
train_url = f"{base_url}/train"

st.title("NYC Ridge Prediction Model")


# Function to check API connection
def health_check(url):
    try:
        return requests.get(url).status_code == 200
    except requests.RequestException:
        return False


with st.spinner("Checking API connection..."):
    while not health_check(health_url):
        time.sleep(3)

st.success("Connected")

# Input sliders
st.header("Enter Pickup Information")
pickup_day = st.slider("Pickup Day", min_value=1, max_value=31, step=1)
pickup_month = st.slider("Pickup Month", min_value=1, max_value=12, step=1)
pickup_hour = st.slider("Pickup Hour", min_value=0, max_value=23, step=1)

col1, col2 = st.columns(2)

# Predict button
with col1:
    if st.button("Predict", type="primary", use_container_width=True):
        input_data = {
            "pickup_day": pickup_day,
            "pickup_month": pickup_month,
            "pickup_hour": pickup_hour,
        }

        try:
            response = requests.post(predict_url, json=input_data)
            if response.status_code == 200:
                res = response.json()
                st.toast(
                    f"Prediction: {res['prediction']} {res['unit']} \n Model: {res['model']}"
                )
            else:
                st.toast("Error: Could not get prediction")
        except requests.exceptions.RequestException:
            st.toast("Error: API not reachable")

# Train button
with col2:
    if st.button("Train Model", use_container_width=True):
        try:
            response = requests.get(train_url)
            if response.status_code == 200:
                res = response.json()
                st.toast(f"{res['message']}")
            else:
                st.toast("Error: Could not train model")
        except requests.exceptions.RequestException:
            st.toast("Error: API not reachable")
