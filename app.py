import streamlit as st
import joblib
import numpy as np

# Load trained objects
knn = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Cancer Prediction using Gene Expression")

g1 = st.number_input("Gene 1 Expression")
g2 = st.number_input("Gene 2 Expression")

if st.button("Predict"):
    sample = scaler.transform([[g1, g2]])
    prediction = knn.predict(sample)

    if prediction[0] == 1:
        st.error("Cancer Detected")
    else:
        st.success("No Cancer Detected")
