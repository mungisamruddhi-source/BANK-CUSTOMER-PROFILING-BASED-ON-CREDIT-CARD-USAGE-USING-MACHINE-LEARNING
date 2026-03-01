import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# ---------- Load Models ----------
BASE_DIR = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE_DIR,"model","kmeans_model.pkl"),"rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR,"model","scaler.pkl"),"rb"))
segment_map = pickle.load(open(os.path.join(BASE_DIR,"model","segment_map.pkl"),"rb"))
recommendation_map = pickle.load(open(os.path.join(BASE_DIR,"model","recommendation_map.pkl"),"rb"))

# ---------- UI ----------
st.title("🏦 Bank Customer Segmentation System")

st.write("Enter Customer Details")

recency = st.number_input("Months on Book",0,100)
frequency = st.number_input("Transaction Count",0,200)
monetary = st.number_input("Transaction Amount",0.0,50000.0)

# ---------- Prediction ----------
if st.button("Predict Customer Segment"):

    sample = pd.DataFrame(
        [[recency, frequency, monetary]],
        columns=["Recency", "Frequency", "Monetary"]
    )

    scaled = scaler.transform(sample)
    cluster = model.predict(scaled)[0]

    segment = segment_map[cluster]

    # Business refinement
    if segment == "Risk Customers":
        if frequency > 20 or monetary > 1500:
            segment = "Low Engagement Customers"

    # ✅ Customer Recommendation Mapping
    recommendation_map = {
        "High Value Customers": "Provide premium banking services",
        "Regular Customers": "Offer loyalty rewards",
        "At Risk Customers": "Run retention campaigns",
        "Low Engagement Customers": "Send engagement offers"
    }

    # ✅ Safe Recommendation Fetch (100% crash-proof)
    recommendation = recommendation_map.get(
        segment,
        "General customer engagement strategy recommended."
    )

    st.success(f"Customer Segment: {segment}")
    st.info(f"Recommendation: {recommendation}")