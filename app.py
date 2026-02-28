import streamlit as st
import pickle
import numpy as np

st.title("🏦 Bank Customer Intelligence System")

# Load model
model = pickle.load(open("kmeans_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.header("Enter Customer Details")

credit_limit = st.number_input("Credit Limit")
transactions = st.number_input("Total Transactions")
utilization = st.number_input("Utilization Ratio")

if st.button("Predict Segment"):

    data = np.array([[credit_limit,
                      transactions,
                      utilization]])

    data = scaler.transform(data)
    cluster = model.predict(data)

    st.success(f"Customer belongs to Cluster {cluster[0]}")