# app.py
import streamlit as st
import pickle
import pandas as pd

# ---------------------- LOAD MODEL ---------------------- #
model = pickle.load(open('best_model.pkl', 'rb'))
encoder_categories = pickle.load(open('encoder_categories.pkl', 'rb'))

st.title("Car Price Prediction App")

# ---------------------- INPUTS ---------------------- #
# Extract categorical options
name_cats, company_cats, fuel_cats = encoder_categories

name = st.selectbox("Car Name", sorted(name_cats))
company = st.selectbox("Car Company", sorted(company_cats))
fuel_type = st.selectbox("Fuel Type", sorted(fuel_cats))

year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, value=2015)
kms_driven = st.number_input("Kms Driven", min_value=0, max_value=3000000, value=50000)

# ---------------------- PREDICTION ---------------------- #
if st.button("Predict Price"):
    input_df = pd.DataFrame([[name, company, fuel_type, year, kms_driven]],
                            columns=['name', 'company', 'fuel_type', 'year', 'kms_driven'])
    pred = model.predict(input_df)[0]
    st.success(f"Estimated Car Price: ₹ {int(pred):,}")
