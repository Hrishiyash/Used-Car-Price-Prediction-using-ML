import streamlit as st
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# Load dataset
car = pd.read_csv('cleaned_car_data.csv')

# Features and target
X = car[['name', 'year', 'kms_driven', 'fuel_type']]
y = car['Price']

# Preprocessing
ohe = OneHotEncoder(handle_unknown='ignore')

ohe.fit(X[['name', 'fuel_type']])

column_trans = make_column_transformer(
    (
        OneHotEncoder(
            categories=ohe.categories_,
            handle_unknown='ignore'
        ),
        ['name', 'fuel_type']
    ),
    remainder='passthrough'
)

# Model pipeline
lr = LinearRegression()

pipe = make_pipeline(column_trans, lr)

# Train model
pipe.fit(X, y)

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

# Prediction
if st.button('Predict Price'):

    input_df = pd.DataFrame(
        [[name, year, kms_driven, fuel_type]],
        columns=['name', 'year', 'kms_driven', 'fuel_type']
    )

    prediction = pipe.predict(input_df)

    st.success(
        f"Estimated Price: ₹ {int(prediction[0]):,}"
    )