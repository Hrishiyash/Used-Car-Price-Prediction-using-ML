import streamlit as st
import pandas as pd
import joblib

# Load cleaned data
car = pd.read_csv('cleaned_car_data.csv')

# Load model
model = joblib.load('LinearRegressionModel.joblib')

# Streamlit page config
st.set_page_config(
    page_title="Used Car Price Predictor upto 2019",
    page_icon="🚗",
    layout="centered"
)