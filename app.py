import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="🏦 Customer Churn Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Load Model
# ----------------------------
model = pickle.load(open("random_forest_churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("Churn_Modelling.csv")

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>

.stApp{
    background-color:#F4F8FB;
}

h1,h2,h3{
    color:#003366;
    font-family:Arial;
}

[data-testid="stSidebar"]{
    background:#0B3C5D;
}

[data-testid="stSidebar"] *{
    color:white;
}

.stButton>button{
    width:100%;
    height:55px;
    border:none;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
    color:white;
    background:linear-gradient(90deg,#2563EB,#1D4ED8);
}

.stButton>button:hover{
    background:#003366;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:12px;
    padding:18px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.15);
}

hr{
    border:1px solid #D6E4F0;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.image(
    "https://img.icons8.com/color/96/bank-building.png",
    width=90
)

st.sidebar.title("🏦 Banking Dashboard")
# ----------------------------
# Navigation
# ----------------------------
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "📊 Prediction", "ℹ About"]
)

# =====================================================
# HOME PAGE
# =====================================================

if page == "🏠 Home":

    st.title("🏦 Customer Churn Prediction System")

    st.markdown(
        "<h4 style='color:#2563EB;'>AI Powered Banking Analytics Dashboard</h4>",
        unsafe_allow_html=True
    )

    st.image(
        "https://images.unsplash.com/photo-1554224155-6726b3ff858f?w=1600",
        use_container_width=True
    )

    st.write("")

    # KPI Cards
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            label="👥 Customers",
            value=f"{len(df):,}"
        )

    with c2:
        st.metric(
            label="🤖 Model",
            value="Random Forest"
        )

    with c3:
        st.metric(
            label="🎯 Accuracy",
            value="86%"
        )

    with c4:
        st.metric(
            label="📊 Features",
            value="11"
        )

    st.write("---")
    col1, col2 = st.columns([2,1])

    with col1:

        st.subheader("About the Project")

        st.write("""
Customer Churn Prediction helps banks identify customers who are likely to leave.

This application uses a trained **Random Forest Machine Learning Model** to predict whether a customer will stay or exit based on their banking details.

### Features

✔ Customer Churn Prediction

✔ Banking Analytics Dashboard

✔ Random Forest Algorithm

✔ Instant Prediction

✔ Interactive Interface

✔ Simple and User Friendly
""")

    with col2:

        st.info("""
### Project Information

🏦 Domain : Banking

🤖 Model : Random Forest

📊 Dataset : 10,000 Customers

🎯 Target : Customer Churn

💻 Framework : Streamlit
""")

    st.write("---")

    st.subheader("📈 Dataset Overview")

    st.dataframe(df.head(), use_container_width=True)
    # =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "📊 Prediction":

    st.title("📊 Customer Churn Prediction")

    st.markdown(
        "<h4 style='color:#2563EB;'>Enter Customer Details</h4>",
        unsafe_allow_html=True
    )

    st.write("")

    with st.container():

        left, right = st.columns(2)

        # ------------------------
        # Left Column
        # ------------------------

        with left:

            credit_score = st.number_input(
                "💳 Credit Score",
                min_value=300,
                max_value=900,
                value=600
            )

            age = st.number_input(
                "🎂 Age",
                min_value=18,
                max_value=100,
                value=30
            )

            tenure = st.number_input(
                "📅 Tenure (Years)",
                min_value=0,
                max_value=10,
                value=5
            )

            balance = st.number_input(
                "💰 Account Balance",
                min_value=0.0,
                value=50000.0,
                step=1000.0
            )

            num_products = st.selectbox(
                "🏦 Number of Products",
                [1,2,3,4]
            )
                    # ------------------------
        # Right Column
        # ------------------------

        with right:

            has_card = st.selectbox(
                "💳 Has Credit Card",
                ["Yes","No"]
            )

            active_member = st.selectbox(
                "🟢 Active Member",
                ["Yes","No"]
            )

            salary = st.number_input(
                "💼 Estimated Salary",
                min_value=0.0,
                value=50000.0,
                step=1000.0
            )

            country = st.selectbox(
                "🌍 Country",
                ["France","Germany","Spain"]
            )

            gender = st.selectbox(
                "👤 Gender",
                ["Male","Female"]
            )

    st.write("")

    st.markdown("---")

    st.subheader("Customer Summary")
    col1, col2, col3 = st.columns(3)

    col1.metric("Credit Score", credit_score)
    col2.metric("Age", age)
    col3.metric("Balance", f"₹{balance:,.0f}")

    st.write("")

    predict = st.button("🔍 Predict Customer Churn")

    germany = 1 if country == "Germany" else 0
    spain = 1 if country == "Spain" else 0
    male = 1 if gender == "Male" else 0

    has_card = 1 if has_card == "Yes" else 0
    active_member = 1 if active_member == "Yes" else 0

    if predict:

        data = np.array([[
            credit_score,
            age,
            tenure,
            balance,
            num_products,
            has_card,
            active_member,
            salary,
            germany,
            spain,
            male
        ]])

        data = scaler.transform(data)
                # ----------------------------
        # Predict
        # ----------------------------
        prediction = model.predict(data)
        probability = model.predict_proba(data)

        stay_prob = probability[0][0] * 100
        churn_prob = probability[0][1] * 100

        st.write("")
        st.markdown("---")
        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "🟢 Stay Probability",
                f"{stay_prob:.2f}%"
            )

        with col2:
            st.metric(
                "🔴 Churn Probability",
                f"{churn_prob:.2f}%"
            )

        st.progress(int(churn_prob))
        if prediction[0] == 1:

            st.error("🚨 High Risk Customer")

            st.markdown("""
### ⚠ Customer is likely to EXIT

The customer has a high probability of leaving the bank.

### Recommended Actions

✔ Contact customer immediately

✔ Offer cashback or discount

✔ Assign Relationship Manager

✔ Provide loyalty rewards

✔ Offer premium banking services

""")

        else:

            st.success("🎉 Customer is likely to STAY")

            st.markdown("""
### ✅ Low Risk Customer

The customer is likely to continue banking.

### Recommended Actions

✔ Continue regular engagement

✔ Offer Premium Credit Card

✔ Increase Reward Points

✔ Provide Exclusive Banking Offers

✔ Maintain Customer Satisfaction

""")
    st.write("---")

    st.subheader("📈 Risk Level")

    if churn_prob < 30:

            st.success("🟢 LOW RISK")

    elif churn_prob < 70:

            st.warning("🟡 MEDIUM RISK")

    else:

            st.error("🔴 HIGH RISK")
            st.write("")

            st.subheader("📋 Customer Summary")

            summary = {
            "Credit Score": credit_score,
            "Age": age,
            "Country": country,
            "Gender": gender,
            "Balance": balance,
            "Estimated Salary": salary,
            "Products": num_products,
            "Prediction": "Exit" if prediction[0] == 1 else "Stay"
        }

            st.table(pd.DataFrame(summary.items(),
                              columns=["Parameter", "Value"]))
            # =====================================================
# ABOUT PAGE
# =====================================================

elif page == "ℹ About":

    st.title("ℹ About Customer Churn Prediction System")

    st.write("")

    col1, col2 = st.columns([2, 1])

    with col1:

        st.markdown("""
## 🏦 Project Overview

The **Customer Churn Prediction System** is a Machine Learning application that predicts whether a customer is likely to leave the bank.

The prediction is performed using a **Random Forest Classifier**, which analyzes customer information such as:

- Credit Score
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Status
- Active Membership
- Estimated Salary
- Country
- Gender

The application helps banks identify customers at risk of leaving and supports better customer retention strategies.
""")

    with col2:

        st.info("""
### 📌 Project Details

🏦 Domain
Banking

🤖 Algorithm
Random Forest

📊 Dataset
Churn Modelling

💻 Framework
Streamlit

🐍 Language
Python
""")
    st.write("---")

    st.subheader("🛠 Technologies Used")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.success("""
🐍 Python

✔ NumPy

✔ Pandas
""")

    with c2:
        st.success("""
🤖 Machine Learning

✔ Scikit-Learn

✔ Pickle
""")

    with c3:
        st.success("""
🌐 Frontend

✔ Streamlit

✔ HTML/CSS
""")
    st.write("---")

    st.subheader("🎯 Project Features")

    st.markdown("""
✅ Customer Churn Prediction

✅ Banking Dashboard

✅ Machine Learning Model

✅ Interactive User Interface

✅ Probability Prediction

✅ Risk Analysis

✅ Customer Summary

✅ User Friendly Interface
""")
    st.write("---")

    st.subheader("👩‍💻 Developer")

    st.info("""
**Name:** Pooja Dnyaneshwar Bhumkar

**Project:** Customer Churn Prediction System

**Technology:** Python | Streamlit | Machine Learning

**Algorithm:** Random Forest Classifier
""")
    st.write("---")

    st.markdown(
        """
<div style="text-align:center;
padding:20px;
background:#003366;
border-radius:10px;
color:white;">

<h3>🏦 Customer Churn Prediction System</h3>

<p>
Developed using Python • Streamlit • Scikit-Learn
</p>

<p>
© 2026 All Rights Reserved
</p>

</div>
""",
        unsafe_allow_html=True
    )
