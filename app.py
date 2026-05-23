import streamlit as st
import pandas as pd
import pickle

# Load cleaned data
car = pd.read_csv('cleaned_car_data.csv')

# Load model
with open('LinearRegressionModel.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit page config
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# Title
st.title("🚗 Car Price Predictor")

st.write("Predict the estimated resale price of a used car till 2019.")

# Inputs

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

# Predict button
if st.button('Predict Price'):

    try:

        input_df = pd.DataFrame(
            [[name, year, kms_driven, fuel_type]],
            columns=['name','year', 'kms_driven', 'fuel_type']
        )

        prediction = model.predict(input_df)

        st.success(
            f"Estimated Price: ₹ {int(prediction[0]):,}"
        )

    except Exception as e:
        st.error(f"Error: {e}")