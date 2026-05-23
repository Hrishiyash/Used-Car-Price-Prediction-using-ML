import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(BASE_DIR, 'cleaned_car_data.csv')

model_path = os.path.join(BASE_DIR, 'LinearRegressionModel.joblib')

car = pd.read_csv(csv_path)

model = joblib.load(model_path)

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Price Predictor")

st.write("Predict the estimated resale price of a used car till 2019.")

name = st.selectbox(
    'Car Name',
    sorted(car['name'].unique())
)

year = st.selectbox(
    'Year',
    sorted(car['year'].unique(), reverse=True)
)

fuel_type = st.selectbox(
    'Fuel Type',
    car['fuel_type'].unique()
)

kms_driven = st.number_input(
    'Kilometers Driven',
    min_value=0,
    step=1000
)

if st.button('Predict Price'):

    input_df = pd.DataFrame(
        [[name, year, kms_driven, fuel_type]],
        columns=['name', 'year', 'kms_driven', 'fuel_type']
    )

    prediction = model.predict(input_df)

    st.success(
        f"Estimated Price: ₹ {int(prediction[0]):,}"
    )