import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(BASE_DIR, 'cleaned_car_data.csv')

model_path = os.path.join(BASE_DIR, 'LinearRegressionModel.joblib')

car = pd.read_csv(csv_path)

model = joblib.load(model_path)