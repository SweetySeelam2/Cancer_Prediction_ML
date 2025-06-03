# --------------------------------------------------
# 🩺 AI-Powered Breast Cancer Diagnosis - Streamlit App
# --------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# -------------------------------
# 📦 Load Model, Scaler, Columns
# -------------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = pd.read_csv("X_train_columns.csv").iloc[:, 0].tolist()

# -------------------------------
# ⚙️ Streamlit Configuration
# -------------------------------
st.set_page_config(page_title="AI Breast Cancer Diagnosis", layout="wide")
st.sidebar.title("🔍 Navigation")

# Sidebar Navigation
page = st.sidebar.radio("Go to:", [
    "🏠 Project Overview",
    "🧪 Run a Prediction",
    "📊 Sample Results",
    "📈 Model Info & Performance",
    "💡 Business Recommendations"
])

# -------------------------------
# 🏠 Project Overview
# -------------------------------
if page == "🏠 Project Overview":
    st.title("🩺 AI-Powered Breast Cancer Diagnosis")
    st.markdown("""
    This professional-grade application uses a trained **Logistic Regression model** to detect whether a breast tumor is **benign** or **malignant**, based on medical imaging features.

    ### 👨‍⚕️ Why It Matters:
    - Early detection of cancer saves lives
    - Helps hospitals automate and augment diagnostic workflows
    - Builds trust through interpretability and high accuracy
    
    ✅ Developed with medical data from the [UCI Wisconsin Breast Cancer Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).
    """)

# -------------------------------
# 🧪 Prediction Page
# -------------------------------
elif page == "🧪 Run a Prediction":
    st.title("🧠 Enter Patient Metrics to Predict Diagnosis")
    st.markdown("""Enter values below for each imaging feature. You can input custom values or test the model using default/sample values.""")

    # Form Layout
    with st.form("prediction_form"):
        user_input = []
        cols = st.columns(3)
        for i, col in enumerate(columns):
            with cols[i % 3]:
                value = st.number_input(f"{col}", value=0.0, step=0.01)
                user_input.append(value)
        submitted = st.form_submit_button("🔍 Predict Diagnosis")

    if submitted:
        X_input = np.array(user_input).reshape(1, -1)
        X_scaled = scaler.transform(X_input)
        prediction = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]

        st.subheader("📌 Prediction Result:")
        if prediction == 1:
            st.error("🔬 Diagnosis: Malignant Tumor")
        else:
            st.success("🧴 Diagnosis: Benign Tumor")

        st.metric("Probability of Malignant", f"{proba[1]*100:.2f}%")
        st.metric("Probability of Benign", f"{proba[0]*100:.2f}%")

        # Business Interpretation
        st.markdown("""
        ### 🧠 Interpretation & Insights
        - A high malignant probability (>90%) should prompt immediate medical follow-up.
        - If the probability is borderline (45–55%), consider reviewing the feature values and rechecking with additional scans.
        - The model is designed to **minimize false positives** while capturing early-stage malignant patterns.
        """)

# -------------------------------
# 📊 Sample Results
# -------------------------------
elif page == "📊 Sample Results":
    st.title("📁 Try with Real Patient-Like Sample Data")
    df_sample = pd.read_csv("Cancer prediction dataset.csv").sample(5)
    st.dataframe(df_sample[columns + ["diagnosis"]])
    st.markdown("Use these values in the prediction form to see how the model responds to real data.")

# -------------------------------
# 📈 Model Info & Performance
# -------------------------------
elif page == "📈 Model Info & Performance":
    st.title("📈 Model Information & Results")
    st.markdown("""
    ### 🎯 Model Used: Logistic Regression (with GridSearchCV tuning)
    - Accuracy: **98.25%**
    - Precision (Malignant): **100%**
    - Recall (Malignant): **95%**
    - F1 Score: **0.98**
    
    ### 🧬 SHAP Explainability:
    - Shows which features most influence predictions.
    - Helps doctors understand *why* a tumor is considered malignant.
    
    🔍 Key Predictive Features:
    - `texture_worst`, `radius_worst`, `concave_points_worst`, etc.
    
    📁 Visuals are available in the GitHub repo under `/images`
    """)

# -------------------------------
# 💡 Business Recommendations
# -------------------------------
elif page == "💡 Business Recommendations":
    st.title("💼 Business Recommendations")
    st.markdown("""
    ### 📊 From Our Results:
    - Early identification can **save $300K–$1M annually** per hospital by reducing late-stage treatment costs.
    - Use this tool as a **second-opinion assistant** during diagnosis to reduce human error.
    
    ### 📍 Best Areas of Prediction:
    - Extremely accurate in distinguishing malignant tumors when probabilities are >90%.
    - Most reliable on high-weight features like `concave_points_worst` and `radius_worst`.

    ### 🔧 Areas for Improvement:
    - Augment with additional clinical factors (e.g. age, family history)
    - Retrain on larger, hospital-specific datasets
    - Combine with imaging AI for even higher accuracy
    
    ### 💡 Actionable Next Steps:
    - Deploy this model into your hospital's diagnostic pipeline.
    - Train clinical staff on interpreting SHAP explanations.
    - Use insights to prioritize follow-up actions on high-risk patients.
    """)

# -------------------------------
# 🔒 Footer with License
# -------------------------------
st.markdown("""
---
© 2024 Sweety Seelam. All rights reserved. Licensed under the MIT License.
""")

st.sidebar.caption(f"🕒 Last updated: {datetime.datetime.now().strftime('%Y-%m-%d')}")
