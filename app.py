import streamlit as st
import pickle
import numpy as np

# -------------------------
# Page Setup
# -------------------------
st.set_page_config(
    page_title="Bank Customer Intelligence",
    page_icon="🏦",
    layout="centered"
)

# -------------------------
# Load Model
# -------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -------------------------
# Title Section
# -------------------------
st.markdown("""
<h1 style='text-align:center'>
🏦 Bank Customer Intelligence System
</h1>
<p style='text-align:center'>
Customer Profiling Based on Credit Card Usage
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------------
# Customer Input Section
# -------------------------
st.markdown("## 📋 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    credit_limit = st.number_input(
        "Credit Limit",
        min_value=0.0,
        placeholder="Enter credit limit"
    )

with col2:
    total_transactions = st.number_input(
        "Total Transactions",
        min_value=0.0,
        placeholder="Enter total transactions"
    )

utilization_ratio = st.slider(
    "Utilization Ratio",
    0.0, 1.0, 0.5
)

# -------------------------
# Segment Meaning
# -------------------------
segment_info = {

0:{
"name":"High Value Customer 💎",
"action":"Provide premium rewards and credit upgrades."
},

1:{
"name":"Low Engagement Customer 💤",
"action":"Run activation campaigns."
},

2:{
"name":"Risk Customer ⚠️",
"action":"Apply retention and EMI offers."
},

3:{
"name":"Regular Customer 👍",
"action":"Provide loyalty rewards and cashback."
}
}

# -------------------------
# Predict Button
# -------------------------
predict = st.button("🔍 Analyze Customer")

# -------------------------
# Prediction Result
# -------------------------
if predict:

    if credit_limit == 0 or total_transactions == 0:
        st.warning("⚠️ Please enter customer details.")
    else:

        input_data = np.array([[credit_limit,
                                total_transactions,
                                utilization_ratio]])

        scaled_data = scaler.transform(input_data)

        cluster = model.predict(scaled_data)[0]

        segment = segment_info[cluster]

        st.markdown("---")
        st.markdown("## 🧠 Customer Intelligence Result")

        st.success(f"Customer Segment: {segment['name']}")

        st.info(f"Recommended Action: {segment['action']}")